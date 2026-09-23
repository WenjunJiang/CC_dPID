"""
raytune_benchmark.py

Unified Ray Tune benchmark for prompt-injection detection with two modes:
  - "search": Run hyperparameter search with Ray Tune, then automatically evaluate best per model
  - "evaluate": Load existing results and evaluate best per model (no search)

Pipeline (after HP search)
--------------------------
  1. Retrain best config on train_ds
  2. Temperature scaling   — calibrate on cal_ds  (minimise NLL)
  3. Threshold tuning      — sweep on cal_ds       (maximise F1)
  4. Final evaluation      — apply T + threshold on test_ds
  5. Print results table

Note: Temperature scaling and threshold tuning both use the calibration set to avoid data leakage.

Data contract
-------------
  load_and_preprocess_data_with_caching() must return four
  HuggingFace Dataset objects, each with columns:
    input_ids, attention_mask, [token_type_ids], labels  (0/1)

Models
------
  google/mobilebert-uncased                       (no cased upstream)
  bert-base-cased
  roberta-base                                    (BPE → effectively cased)
  protectai/deberta-v3-base-prompt-injection-v2
  distilbert-base-cased

Install
-------
    pip install "ray[tune]" optuna torch transformers accelerate peft \
                scikit-learn pandas pyarrow numpy sentencepiece psutil tabulate

Usage
-----
    # Run hyperparameter search
    python raytune_benchmark.py mode=search training-mode=ft
    
    # Evaluate existing results and find best per model
    python raytune_benchmark.py --mode=evaluate --ray-results-dir ray_results
    
    # Evaluate with PEFT
    python raytune_benchmark.py --mode evaluate --ray-results-dir ray_results_peft --training-mode peft
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf
import json
import logging
import os
import ssl
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_softmax, softmax
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, EvalPrediction,
    Trainer, TrainerCallback, TrainerState, TrainingArguments,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from datasets import load_from_disk
import random

from torchao.quantization import quantize_
from torchao.quantization.qat import QATConfig, IntxFakeQuantizeConfig

# Disable SSL verification for HuggingFace downloads (needed for corporate proxies)
ssl._create_default_https_context = ssl._create_unverified_context

logging.getLogger("transformers").setLevel(logging.ERROR)

# ── CHANGED: RANDOM_STATE now read from cfg.seed in main() ──
# Kept as module-level default for backward compat in functions
# that aren't called from main() yet.
# RANDOM_STATE = 42
_BORDER = "!" * 64

_MINIMIZE_METRICS = {"val_loss"}

def _metric_mode(metric: str) -> str:
    return "min" if metric in _MINIMIZE_METRICS else "max"

_FAILED_REPORT = {
    "val_loss":      float("inf"),
    "val_accuracy":  0.0,
    "val_f1":        0.0,
    "val_precision": 0.0,
    "val_recall":    0.0,
    "val_fpr":       0.0,
    "val_fnr":       1.0,
}

def _report_to_tune(metrics: dict, **kwargs):
    """
    Report metrics to Ray Tune. Handles API differences across Ray versions.

    Ray < 2.0: tune.report(val_f1=0.9, ...) kwargs
    Ray 2.0-2.6: ray.train.report(metrics={...}) ray.train
    Ray 2.7-2.40: tune.report(metrics={...}) metrics keyword
    Ray 2.40+: tune.report({...}) positional dict
    Ray 2.54+: ??? keeps changing

    This wrapper tries each style until one works.
    """


    # Try 1: positional dict (Ray 2.40+)
    try:
        tune.report(metrics, **kwargs)
        return
    except TypeError:
        pass

    # Try 2: metrics= keyword (Ray 2.7-2.40)
    try:
        tune.report(metrics=metrics, **kwargs)
        return
    except TypeError:
        pass

    # Try 3: kwargs style (Ray < 2.0)
    try:
        tune.report(**metrics, **kwargs)
        return
    except TypeError:
        pass

    # Try 4: ray.train.report (Ray 2.0-2.6)
    try:
        import ray.train
        ray.train.report(metrics=metrics, **kwargs)
        return
    except (TypeError, ImportError, AttributeError):
        pass

    raise RuntimeError(
        f"Could not report metrics to Ray Tune. "
        f"Ray version: {__import__('ray').__version__}. "
        f"Please check the Ray Tune API for your version."
    )

# =============================================================================
# §1  Models
# =============================================================================

@dataclass
class ModelSpec:
    family: str
    peft_target_modules: list[str] = field(default_factory=lambda: ["query", "value"])
    model_path: Optional[str] = None

# MODELS and EXPECTED_MODELS will be loaded from JSON in main()
MODELS: dict[str, ModelSpec] = {}
EXPECTED_MODELS: list[str] = []

# Metrics and their optimization directions
METRICS = {
    "val_accuracy": "max",
    "val_f1": "max",
    "val_precision": "max",
    "val_recall": "max",
    "val_loss": "min",
}

# Metrics where lower is better; all others are maximised.
_MINIMIZE_METRICS = {"val_loss"}

def build_model_registry_from_yaml(models_cfg) -> dict[str, ModelSpec]:
    models = {}
    for model_id, spec_cfg in models_cfg.specs.items():
        if not spec_cfg.get("enabled", True):
            continue
        models[model_id] = ModelSpec(
            family=spec_cfg.family,
            peft_target_modules=list(
                spec_cfg.get("peft_target_modules", ["query", "value"])
            ),
        )
    return models

# =============================================================================
# §2  Data loading
# =============================================================================

def _read_arrow_dir(path: str) -> "datasets.Dataset":
    """
    Read Arrow files from a directory.

    Supports two layouts:
    1. HuggingFace save_to_disk() format (has dataset_info.json)
    2. Directory of raw *.arrow files
    """
    from datasets import Dataset, concatenate_datasets
    import pyarrow as pa

    p = Path(path)

    # HuggingFace dataset format
    if (p / "dataset_info.json").exists() or (p / "state.json").exists():
        return load_from_disk(str(p))

    # Raw .arrow files in directory
    arrow_files = sorted(p.glob("*.arrow"))
    if arrow_files:
        tables = []
        for f in arrow_files:
            with pa.memory_mapped_file(str(f), "r") as source:
                tables.append(pa.ipc.open_stream(source).read_all())
        merged = pa.concat_tables(tables)
        return Dataset(merged)

    raise FileNotFoundError(
        f"No Arrow data found at '{path}'. Expected either:\n"
        f" - HuggingFace dataset dir (with dataset_info.json)\n"
        f" - Directory containing *.arrow files"
    )

def _read_csv(path: str) -> "datasets.Dataset":
    from datasets import Dataset
    import pandas as pd
    return Dataset.from_pandas(pd.read_csv(path))


def _read_parquet(path: str) -> "datasets.Dataset":
    from datasets import Dataset
    import pandas as pd
    return Dataset.from_pandas(pd.read_parquet(path))


def _read_iceberg(table_name: str, iceberg_cfg: dict) -> "datasets.Dataset":
    from datasets import Dataset
    from pyiceberg.catalog import load_catalog
    catalog = load_catalog(
        iceberg_cfg["catalog_name"],
        **{
            "type": iceberg_cfg["catalog_type"],
            "uri": iceberg_cfg["catalog_uri"],
            "warehouse": iceberg_cfg["warehouse"],
        },
    )
    table = catalog.load_table(table_name)
    df = table.scan().to_pandas()
    return Dataset.from_pandas(df)


def _load_single_source(source_cfg: dict, iceberg_cfg: dict = None) -> "datasets.Dataset":
    """
    Load one data source based on explicit 'format' field.

    Supported formats:
    arrow → path to dir (HF dataset or raw *.arrow files) or single .arrow file
    csv → path to .csv file
    parquet → path to .parquet file
    iceberg → table name, requires iceberg_cfg
    """
    import datasets

    name = source_cfg.get("name", "?")
    fmt = source_cfg.get("format")
    path = source_cfg.get("path")
    table = source_cfg.get("table")

    if not fmt:
        raise ValueError(
            f"Source '{name}' is missing 'format' field. "
            f"Set format to one of: arrow, csv, parquet, iceberg"
        )

    if fmt == "arrow":
        if not path:
            raise ValueError(f"Source '{name}' (format=arrow) requires 'path'.")
        p = Path(path)
        if p.is_dir():
            print(f" Loading '{name}' from Arrow dir: {path}")
            ds = _read_arrow_dir(path)
        elif p.exists() and p.suffix == ".arrow":
            print(f" Loading '{name}' from Arrow file: {path}")
            import pyarrow as pa
            with pa.memory_mapped_file(str(p), "r") as source:
                ds = datasets.Dataset(pa.ipc.open_stream(source).read_all())
        elif p.exists():
            print(f" Loading '{name}' from Arrow dir: {path}")
            ds = _read_arrow_dir(path)
        else:
            raise FileNotFoundError(f"Source '{name}': path not found: {path}")

    elif fmt == "csv":
        if not path:
            raise ValueError(f"Source '{name}' (format=csv) requires 'path'.")
        print(f" Loading '{name}' from CSV: {path}")
        ds = _read_csv(path)

    elif fmt == "parquet":
        if not path:
            raise ValueError(f"Source '{name}' (format=parquet) requires 'path'.")
        print(f" Loading '{name}' from Parquet: {path}")
        ds = _read_parquet(path)

    elif fmt == "iceberg":
        if not table:
            raise ValueError(f"Source '{name}' (format=iceberg) requires 'table'.")
        if not iceberg_cfg:
            raise ValueError(
                f"Source '{name}' (format=iceberg) requires data.iceberg "
                f"config in config.yaml."
            )
        print(f" Loading '{name}' from Iceberg: {table}")
        ds = _read_iceberg(table, iceberg_cfg)

    else:
        raise ValueError(
            f"Source '{name}' has unknown format '{fmt}'. "
            f"Supported: arrow, csv, parquet, iceberg"
        )

    n = len(ds)
    cols = list(ds.column_names)
    print(f" → {n} rows, columns: {cols}")
    return ds

def _compute_data_cache_key(data_sources: list, split_cfg: dict) -> str:
    """
    Deterministic cache key from source configs + split config.
    Same sources + same splits → same key. Any change → cache miss.
    """
    import hashlib
    parts = []
    for src in sorted(data_sources, key=lambda s: s.get("name", "")):
        if not src.get("enabled", True):
            continue
        parts.append(f"{src.get('name')}:{src.get('format')}:{src.get('path')}:{src.get('table')}")
    parts.append(f"split:{split_cfg}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]

def load_and_preprocess_data_with_caching(
    data_sources: list,
    tokenizer,
    max_seq_length: int,
    cache_dir: str = ".cache/tokenized",
    data_cache_dir: str = ".cache/pooled",
    iceberg_cfg: dict = None,
    split_cfg: dict = None,
):
    """
    Load sources → pool → split → tokenize, with two cache layers.

    Layer 1 (data_cache_dir): pooled + split + column-normalized datasets.
            Shared across all models. Keyed by source configs + split ratios.
            Skips Arrow/CSV/Iceberg loading on cache hit.

    Layer 2 (cache_dir):      tokenized datasets, per model.
            Keyed by model name + max_seq_length.
            Skips tokenization on cache hit.

    Parameters
    ----------
    data_sources      List of dicts: {name, format, path, table, enabled}
    tokenizer         HuggingFace tokenizer (model-specific).
    max_seq_length    Maximum token length for truncation.
    cache_dir         Per-model tokenized cache directory.
    data_cache_dir    Shared pooled+split cache directory.
    iceberg_cfg       Optional Iceberg connection config.
    split_cfg         Optional split config {train, val, cal, test, split_seed}.

    Returns
    -------
    train_ds, val_ds, cal_ds, test_ds — four HuggingFace Dataset objects.
    """
    from datasets import concatenate_datasets, DatasetDict

    # ── Defaults ──
    if split_cfg is None:
        split_cfg = {"train": 0.80, "val": 0.10, "cal": 0.05,
                     "test": 0.05, "split_seed": 99}

    # ── Layer 2 check: tokenized cache (fastest) ──
    model_name = tokenizer.name_or_path.replace("/", "_")
    tok_cache_key = f"{model_name}_{max_seq_length}"
    tok_cache_path = Path(cache_dir) / tok_cache_key

    if tok_cache_path.exists():
        try:
            print(f"  ⚡ Loading tokenized cache: {tok_cache_path}")
            cached = load_from_disk(str(tok_cache_path))
            return cached["train"], cached["val"], cached["cal"], cached["test"]
        except Exception as exc:
            print(f"  ⚠ Tokenized cache corrupted, regenerating: {exc}")
            import shutil
            shutil.rmtree(tok_cache_path, ignore_errors=True)

    # ── Layer 1 check: pooled+split cache ──
    data_cache_key = _compute_data_cache_key(data_sources, split_cfg)
    data_cache_path = Path(data_cache_dir) / data_cache_key

    if data_cache_path.exists():
        try:
            print(f"  ⚡ Loading pooled+split cache: {data_cache_path}")
            cached = load_from_disk(str(data_cache_path))
            train_ds = cached["train"]
            val_ds = cached["val"]
            cal_ds = cached["cal"]
            test_ds = cached["test"]
            print(f"    train={len(train_ds)} val={len(val_ds)} "
                  f"cal={len(cal_ds)} test={len(test_ds)}")
        except Exception as exc:
            print(f"  ⚠ Pooled cache corrupted, regenerating: {exc}")
            import shutil
            shutil.rmtree(data_cache_path, ignore_errors=True)
            data_cache_path = None  # force reload below

    if not data_cache_path or not data_cache_path.exists():
        # ── Load from sources (slow path) ──
        print("\n  Loading data sources …")
        all_datasets = []
        for src in data_sources:
            if not src.get("enabled", True):
                print(f"  Skipping disabled source: {src.get('name', '?')}")
                continue
            ds = _load_single_source(src, iceberg_cfg)
            all_datasets.append(ds)

        if not all_datasets:
            raise RuntimeError("No enabled data sources found.")

        # ── Merge ──
        if len(all_datasets) == 1:
            pooled = all_datasets[0]
        else:
            common_cols = set(all_datasets[0].column_names)
            for ds in all_datasets[1:]:
                common_cols &= set(ds.column_names)
            common_cols = sorted(common_cols)
            aligned = [ds.select_columns(common_cols) for ds in all_datasets]
            pooled = concatenate_datasets(aligned)

        print(f"  Pooled: {len(pooled)} rows, columns: {list(pooled.column_names)}")

        # ── Detect and normalize label column ──
        label_col = None
        for candidate in ("label", "labels", "is_injection", "Label"):
            if candidate in pooled.column_names:
                label_col = candidate
                break
        if label_col is None:
            raise ValueError(
                f"No label column found. Available: {pooled.column_names}. "
                f"Expected one of: label, labels, is_injection"
            )
        if label_col != "label":
            pooled = pooled.rename_column(label_col, "label")

        # ── Detect and normalize text column ──
        text_col = None
        for candidate in ("text", "prompt", "content", "sentence"):
            if candidate in pooled.column_names:
                text_col = candidate
                break
        if text_col is None:
            raise ValueError(
                f"No text column found. Available: {pooled.column_names}. "
                f"Expected one of: text, prompt, content, sentence"
            )
        if text_col != "text":
            pooled = pooled.rename_column(text_col, "text")

        # ── Cast label to ClassLabel (required by stratify_by_column) ──
        from datasets import ClassLabel
        num_classes = len(set(pooled["label"]))
        pooled = pooled.cast_column(
            "label", ClassLabel(num_classes=num_classes)
        )

        # ── Split ──
        split_seed = split_cfg.get("split_seed", 99)
        test_frac = split_cfg.get("test", 0.05)
        cal_frac = split_cfg.get("cal", 0.05)
        val_frac = split_cfg.get("val", 0.10)

        split1 = pooled.train_test_split(
            test_size=test_frac, seed=split_seed, stratify_by_column="label",
        )
        rest, test_ds = split1["train"], split1["test"]

        cal_of_rest = cal_frac / (1 - test_frac)
        split2 = rest.train_test_split(
            test_size=cal_of_rest, seed=split_seed, stratify_by_column="label",
        )
        rest2, cal_ds = split2["train"], split2["test"]

        val_of_rest = val_frac / (1 - test_frac - cal_frac)
        split3 = rest2.train_test_split(
            test_size=val_of_rest, seed=split_seed, stratify_by_column="label",
        )
        train_ds, val_ds = split3["train"], split3["test"]

        print(f"  Split: train={len(train_ds)} val={len(val_ds)} "
              f"cal={len(cal_ds)} test={len(test_ds)}")

        # ── Save Layer 1 cache ──
        save_path = Path(data_cache_dir) / _compute_data_cache_key(data_sources, split_cfg)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dd = DatasetDict({"train": train_ds, "val": val_ds,
                          "cal": cal_ds, "test": test_ds})
        dd.save_to_disk(str(save_path))
        print(f"  💾 Saved pooled+split cache: {save_path}")

    # ── Tokenize ──
    print(f"  Tokenizing (max_seq_length={max_seq_length}) …")

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True,
            padding=False, max_length=max_seq_length,
        )

    # Rename label → labels (HuggingFace Trainer expects "labels")
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    cal_ds = cal_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    cal_ds = cal_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    # Keep only model-relevant columns
    keep_cols = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    for ds_name, ds in [("train", train_ds), ("val", val_ds),
                        ("cal", cal_ds), ("test", test_ds)]:
        drop = [c for c in ds.column_names if c not in keep_cols]
        if ds_name == "train":
            train_ds = ds.remove_columns(drop)
        elif ds_name == "val":
            val_ds = ds.remove_columns(drop)
        elif ds_name == "cal":
            cal_ds = ds.remove_columns(drop)
        else:
            test_ds = ds.remove_columns(drop)

    # ── Save Layer 2 cache ──
    tok_cache_path.parent.mkdir(parents=True, exist_ok=True)
    dd = DatasetDict({"train": train_ds, "val": val_ds,
                      "cal": cal_ds, "test": test_ds})
    dd.save_to_disk(str(tok_cache_path))
    print(f"  💾 Saved tokenized cache: {tok_cache_path}")

    return train_ds, val_ds, cal_ds, test_ds

# =============================================================================
# §3  Pre-flight check
# =============================================================================

def _load_error_reason(exc: Exception) -> str:
    msg = str(exc).lower()
    if any(k in msg for k in ("401", "403", "gated", "token", "authentication")):
        return "Auth required — run: huggingface-cli login  OR  export HF_TOKEN=<token>"
    if any(k in msg for k in ("404", "not found", "does not exist")):
        return "Model not found on Hub — check model ID"
    if any(k in msg for k in ("connection", "timeout", "network")):
        return "Network error — check internet connection"
    return f"{type(exc).__name__}: {str(exc)[:150]}"

def preflight_check(model_ids: list[str]) -> list[str]:
    """Downloads only config.json per model; filters out unavailable ones."""
    print(f"\n{'='*64}\n  PRE-FLIGHT CHECK  ({len(model_ids)} models)\n{'='*64}")
    available = []
    for model_id in model_ids:
        # actual_model_id = _get_actual_model_id(model_id)
        try:
            AutoConfig.from_pretrained(model_id)
            available.append(model_id)
            print(f"  ✓  {model_id} (actual: {model_id})")
        except Exception as exc:
            print(f"\n{'!'*64}\n  ⚠  SKIPPING  {model_id}\n"
                  f"     {_load_error_reason(exc)}\n{'!'*64}\n")
    print(f"\n  {len(available)}/{len(model_ids)} available\n{'='*64}\n")
    if not available:
        raise RuntimeError("No models passed preflight. Check network / HF_TOKEN.")
    return available

# =============================================================================
# §4  Model loading & LoRA
# =============================================================================
class ModelLoadError(RuntimeError):
    pass

def prepare_tokenizer(model_id: str) -> Any:
    """Load and prepare tokenizer with pad_token fix for decoder-only models."""
    # actual_model_id = _get_actual_model_id(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    # Fix for decoder-only models (Pythia, OPT, SmolLM2, etc.)
    # These models don't have a pad_token by default
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"  Set pad_token = eos_token for {model_id}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token = '[PAD]'
            print(f"  Added [PAD] special token for {model_id}")
    
    return tokenizer


def load_model(
    model_id: str,
    classifier_dropout: float,
    *,
    attn_implementation: str = "eager",
):
    """Returns (tokenizer, model). Raises ModelLoadError on any failure.

    `attn_implementation` is keyword-only and defaults to "eager" so that the
    16 existing call sites across 8 scripts keep their exact current behaviour.
    "sdpa" is numerically equivalent for ModernBERT/mmBERT (verified: max|dlogit|
    = 1.1e-5 with identical weights; both the eager and sdpa kernels apply the
    same sliding_window_mask for the local-attention layers) and is ~10% cheaper
    in dispatched ops, so opt into it per experiment via `perf.attn_implementation`.
    Only output_attentions=True genuinely requires eager, and nothing here sets it.
    """
    try:
        tokenizer = prepare_tokenizer(model_id)
    except Exception as exc:
        raise ModelLoadError(
            f"Tokenizer failed [{model_id}]: {_load_error_reason(exc)}"
        ) from exc
    
    try:
        hf_cfg = AutoConfig.from_pretrained(model_id, num_labels=2, problem_type="single_label_classification")
        for attr in ("classifier_dropout", "seq_classif_dropout",
                     "summary_last_dropout", "hidden_dropout_prob"):
            if getattr(hf_cfg, attr, None) is not None:
                setattr(hf_cfg, attr, classifier_dropout)
                break
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, config=hf_cfg, ignore_mismatched_sizes=True,
            # fp32 master weights are required for full fine-tuning at lr~1e-5:
            # bf16 params (8-bit mantissa) would round the update away. This does
            # NOT forbid mixed precision - TrainingArguments(bf16=True) is autocast
            # and keeps these weights fp32.
            torch_dtype=torch.float32,
            attn_implementation=attn_implementation,
        )
        
        # If we added a new pad token, resize model embeddings
        if tokenizer.pad_token is not None and len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
            print(f"  Resized embeddings to include new pad_token for {model_id}")
        
        # CRITICAL: Update model config to know about the pad_token
        model.config.pad_token_id = tokenizer.pad_token_id
    except torch.cuda.OutOfMemoryError as exc:
        raise ModelLoadError(f"CUDA OOM loading [{model_id}]") from exc
    except Exception as exc:
        raise ModelLoadError(
            f"Model failed [{model_id}]: {_load_error_reason(exc)}"
        ) from exc
    return tokenizer, model


def apply_lora(model, spec: ModelSpec, hp: dict) -> Any:
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        raise RuntimeError("Run: pip install peft")
    peft_model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(hp["lora_r"]),
        lora_alpha=float(hp["lora_alpha"]),
        lora_dropout=float(hp["lora_dropout"]),
        target_modules=spec.peft_target_modules,
        bias="none",
    ))
    peft_model.print_trainable_parameters()
    return peft_model


# =============================================================================
# §5  Score calibration
#
# The model's native score is logits[:, 1] - logits[:, 0]. That value is the
# input to calibration, not the decision variable: thresholds, PR curves and
# the shipped rule all live on the CALIBRATED PROBABILITY, sigmoid(a * z + b).
# The Platt intercept is what makes that safe - it shifts every score down by
# roughly log(deploy_ratio), holding the scores clear of the point where a
# float32 probability rounds to exactly 1.0 and the whole head of the ranking
# collapses into one indistinguishable tie. `check_score_resolution` verifies
# that headroom rather than assuming it.
# =============================================================================
def binary_margin(logits: np.ndarray) -> np.ndarray:
    """Model-intrinsic score: the log-odds of the positive class, uncalibrated.

    Accepts two-class logits of shape (N, 2) or an already-reduced (N,) margin.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 2 and logits.shape[1] == 2:
        return logits[:, 1] - logits[:, 0]
    if logits.ndim == 1:
        return logits
    raise ValueError(
        f"Expected two-class logits (N, 2) or a margin (N,), got shape {logits.shape}"
    )


@dataclass
class ThresholdConstraints:
    """Constraints for threshold optimization.

    The candidate thresholds are taken from the data rather than from a fixed
    grid, so every operating point the scores can actually express is
    considered. The previous ``linspace(0.5, 0.95, 50)`` grid could not reach any
    operating point above precision ~0.90, because the probabilities that matter
    are packed into the sliver above 0.95 that the grid never sampled.
    """
    target_precision: float = 0.95
    min_recall: float = 0.9
    beta: float = 0.5
    # Cap on how many candidate thresholds to evaluate. The candidates are
    # sub-sampled uniformly over rank, so coverage stays even across the range.
    max_candidates: int = 4096
    # When no threshold satisfies both constraints, fall back to maximising
    # F-beta over all candidates instead of raising.
    relax_if_infeasible: bool = True


class TemperatureScaler:
    """Single-parameter temperature scaling.

    Retained for backward compatibility. Prefer :class:`PlattCalibrator`: a
    temperature can only rescale, so it cannot represent the intercept shift
    that separates one benign:malicious ratio from another, and fitting it on a
    10:1 view for use at 500:1 asks it to do exactly that.
    """

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Find T in [0.05, 5.0] that minimises cross-entropy on (logits, labels)."""
        def nll(T: float) -> float:
            scaled_lp = log_softmax(logits / T, axis=-1)
            return -float(np.mean(scaled_lp[np.arange(len(labels)), labels]))

        result = minimize_scalar(nll, bounds=(0.05, 5.0), method="bounded")
        self.temperature = float(result.x)
        return self.temperature

    def scale(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for the positive class."""
        scaled = softmax(logits / self.temperature, axis=-1)
        return scaled[:, 1]


class PlattCalibrator:
    """Two-parameter calibration: p = sigmoid(a * z + b), z = logit_1 - logit_0.

    ``a`` plays the role of ``1 / temperature`` and ``b`` carries the prior shift
    that a temperature cannot express. Fit it at the prevalence you deploy at, or
    pass ``sample_weight`` to reweight a calibration view built at a different
    ratio. Threshold on :meth:`scale`, the calibrated probability.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a)
        self.b = float(b)

    @property
    def temperature(self) -> float:
        """Equivalent temperature, for reporting next to legacy artifacts."""
        return 1.0 / self.a if self.a else float("inf")

    @staticmethod
    def _nll(z: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> float:
        per = np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1.0 - y)
        if w is None:
            return float(per.mean())
        return float((w * per).sum() / w.sum())

    @staticmethod
    def _regularized_targets(
        y: np.ndarray,
        w: Optional[np.ndarray],
    ) -> np.ndarray:
        """Platt's (1999) smoothed targets, which keep the fit from diverging.

        On separable data the plain cross-entropy has no minimum: it is driven
        down by sending ``a`` to infinity, which turns the calibrator into a step
        function and pushes every confident score straight past the float32
        limit. Detectors like this one are close to separable on a calibration
        set, so the degenerate solution is the *typical* case, not a corner one.

        Replacing the hard 0/1 targets with
            y+ = (N+ + 1) / (N+ + 2),  y- = 1 / (N- + 2)
        makes the objective finite and the fit well posed.
        """
        weights = np.ones_like(y) if w is None else w
        n_pos = float(weights[y == 1].sum())
        n_neg = float(weights[y == 0].sum())
        return np.where(y == 1,
                        (n_pos + 1.0) / (n_pos + 2.0),
                        1.0 / (n_neg + 2.0))

    @staticmethod
    def _soft_nll(z: np.ndarray, t: np.ndarray, w: Optional[np.ndarray]) -> float:
        per = np.logaddexp(0.0, -z) * t + np.logaddexp(0.0, z) * (1.0 - t)
        if w is None:
            return float(per.mean())
        return float((w * per).sum() / w.sum())

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        max_slope: float = 10.0,
    ) -> Tuple[float, float]:
        """Minimise (weighted) cross-entropy over (a, b) with Platt smoothing.

        ``max_slope`` is a final backstop on ``a``. Smoothed targets normally
        keep the fit well behaved on their own; the clamp only matters if the
        calibration set is so small that even the smoothed optimum is extreme.
        Returns (a, b).
        """
        z = binary_margin(logits)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        t = self._regularized_targets(y, w)

        result = minimize(
            lambda p: self._soft_nll(p[0] * z + p[1], t, w),
            x0=np.array([1.0, 0.0]),
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 4000},
        )
        a, b = float(result.x[0]), float(result.x[1])

        if not np.isfinite(a) or not np.isfinite(b) or abs(a) > max_slope:
            scale = max_slope / abs(a) if np.isfinite(a) and a != 0 else 0.0
            print(f"  ⚠ Platt fit returned a={a:.4g}, b={b:.4g}; clamping the slope to "
                  f"±{max_slope}. A calibration set this separable carries little "
                  f"information about the shape of the curve.")
            a, b = (np.sign(a) * max_slope, b * scale) if scale else (1.0, 0.0)

        self.a, self.b = float(a), float(b)
        return self.a, self.b

    def margin(self, logits: np.ndarray) -> np.ndarray:
        """Calibrated log-odds. Internal to :meth:`scale` - threshold on the
        probability that returns, not on this."""
        return self.a * binary_margin(logits) + self.b

    def scale(self, logits: np.ndarray) -> np.ndarray:
        """Calibrated probabilities. This is the value to threshold on."""
        z = self.margin(logits)
        return np.asarray(1.0 / (1.0 + np.exp(-z)))

    def to_dict(self) -> Dict[str, float]:
        return {"type": "platt", "a": self.a, "b": self.b,
                "equivalent_temperature": self.temperature}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlattCalibrator":
        return cls(a=float(payload["a"]), b=float(payload["b"]))


def check_score_resolution(
    scores: np.ndarray,
    name: str = "scores",
    top_fraction: float = 0.01,
    min_distinct_ratio: float = 0.9,
) -> Dict[str, Any]:
    """Warn if a float32 score has run out of resolution at the top of the ranking.

    Probabilities are safe to threshold on only while the calibrated log-odds
    stay below ~17.3, past which a float32 rounds to exactly 1.0 and every
    affected sample shares one score. A Platt fit keeps the scores in range on
    its own -- the intercept subtracts roughly log(deploy_ratio) from everything
    -- but the headroom is finite, so this checks rather than assumes.

    Returns the measured stats and prints a warning when they look wrong.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_top = max(1, int(len(scores) * top_fraction))
    top = np.sort(scores)[-n_top:]
    n_distinct = int(len(np.unique(top)))
    n_saturated = int((scores >= 1.0).sum())

    stats = {
        "n_saturated": n_saturated,
        "n_top": n_top,
        "n_distinct_in_top": n_distinct,
        "distinct_ratio": n_distinct / n_top,
    }

    if n_saturated:
        print(f"  ⚠ {name}: {n_saturated:,} rows sit at exactly 1.0 and are "
              f"indistinguishable. Every metric above that point is capped at "
              f"whatever fraction of the tie is positive, and no threshold can "
              f"separate them. Refit the calibrator - a Platt intercept fitted at "
              f"the deployment prevalence normally keeps scores clear of this.")
    elif stats["distinct_ratio"] < min_distinct_ratio:
        print(f"  ⚠ {name}: only {n_distinct:,} distinct values among the top "
              f"{n_top:,} scores ({stats['distinct_ratio']:.1%}). The ranking is "
              f"losing resolution where the threshold lives.")

    return stats


def deployment_sample_weight(
    labels: np.ndarray,
    view_ratio: float,
    deploy_ratio: float,
) -> np.ndarray:
    """Weights that turn a ``view_ratio``:1 calibration view into ``deploy_ratio``:1.

    Only the negatives are reweighted; the positives already appear at their
    natural count. Use this instead of rebuilding the view when the benign pool
    is large enough to make a 500:1 materialisation expensive.
    """
    labels = np.asarray(labels).reshape(-1)
    return np.where(labels == 1, 1.0, float(deploy_ratio) / float(view_ratio))

# =============================================================================
# §6  Threshold optimization
# =============================================================================
def fbeta(p: float, r: float, beta: float = 0.5) -> float:
    """Calculate Fβ score."""
    if p <= 0.0 or r <= 0.0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * (p * r) / (b2 * p + r)


def metrics_from_probs(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    threshold: float,
) -> dict:
    """
    Calculate metrics from predicted probabilities and true labels.

    Args:
        y_true: True labels
        prob_pos: Predicted probabilities for positive class
        threshold: Threshold for classification

    Returns:
        Dictionary with metrics
    """
    y_pred = (prob_pos >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(p), "recall": float(r),
    }


def sweep_operating_points(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Every operating point ``scores`` can express, in one pass.

    One point per distinct score, ordered from the highest threshold down. This
    is the same set of points ``precision_recall_curve`` produces, computed here
    so that TP/FP counts and the tie sizes stay available.

    O(n log n) - do not loop a confusion matrix over candidate thresholds; at
    750k rows that is minutes per call instead of milliseconds.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if len(scores) != len(labels):
        raise ValueError(f"scores and labels differ in length: "
                         f"{len(scores)} vs {len(labels)}")
    if len(scores) == 0:
        raise ValueError("No valid threshold found: the score array is empty")

    n_pos = int(labels.sum())
    if n_pos == 0:
        raise ValueError("No positive samples: precision/recall are undefined")

    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]

    # Last index of each run of equal scores: thresholding at that score admits
    # the whole run, so these are exactly the reachable operating points.
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]

    tp = np.cumsum(y)[last]
    n_pred = last + 1
    fp = n_pred - tp
    tie = np.diff(np.r_[-1, last])

    return {
        "threshold": s[last],
        "tp": tp,
        "fp": fp,
        "fn": n_pos - tp,
        "tn": (len(y) - n_pos) - fp,
        "precision": tp / n_pred,
        "recall": tp / n_pos,
        "tie_size": tie,
    }


def _threshold_candidates(scores: np.ndarray, max_candidates: int) -> np.ndarray:
    """Distinct scores, sub-sampled over rank if there are too many.

    Kept for callers that want the candidate list directly. ``optimize_threshold``
    uses :func:`sweep_operating_points` instead, which needs no sub-sampling.
    """
    distinct = np.unique(scores)
    if len(distinct) <= max_candidates:
        return distinct
    idx = np.linspace(0, len(distinct) - 1, max_candidates).round().astype(int)
    return distinct[np.unique(idx)]


def optimize_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    constraints: Optional[ThresholdConstraints] = None,
) -> Tuple[float, dict]:
    """Pick the threshold that maximises F-beta subject to the constraints.

    ``scores`` are the calibrated probabilities from
    :meth:`PlattCalibrator.scale`. The search is scale-agnostic and derives its
    candidates from the data, so no fixed grid can clip the answer.

    Constraints are enforced, not merely declared. If none of the candidates
    satisfy both ``target_precision`` and ``min_recall``, the search relaxes to
    plain F-beta maximisation (unless ``relax_if_infeasible`` is False) and says
    so.
    """
    if constraints is None:
        constraints = ThresholdConstraints()

    sweep = sweep_operating_points(scores, labels)
    precision, recall = sweep["precision"], sweep["recall"]

    b2 = constraints.beta ** 2
    denom = b2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(denom > 0,
                         (1.0 + b2) * precision * recall / np.maximum(denom, 1e-300),
                         0.0)

    feasible = ((precision >= constraints.target_precision)
                & (recall >= constraints.min_recall))

    if feasible.any():
        best = int(np.flatnonzero(feasible)[np.argmax(score[feasible])])
    else:
        if not constraints.relax_if_infeasible:
            raise ValueError(
                f"No threshold reaches precision >= {constraints.target_precision} "
                f"and recall >= {constraints.min_recall}"
            )
        reachable = recall >= constraints.min_recall
        best_p = precision[reachable].max() if reachable.any() else precision.max()
        print(f"  [optimize_threshold] no threshold satisfies precision >= "
              f"{constraints.target_precision} and recall >= {constraints.min_recall} "
              f"(best precision at that recall: {best_p:.4f}); "
              f"relaxing to F{constraints.beta} maximisation")
        best = int(np.argmax(score))

    best_metrics = {
        "tp": int(sweep["tp"][best]),
        "fp": int(sweep["fp"][best]),
        "fn": int(sweep["fn"][best]),
        "tn": int(sweep["tn"][best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }

    # A threshold landing inside a tie means the scores have run out of
    # resolution exactly where the decision is being made.
    tie_size = int(sweep["tie_size"][best])
    if tie_size > 1:
        print(f"  [optimize_threshold] WARNING: {tie_size} samples share the chosen "
              f"threshold {sweep['threshold'][best]!r}. The probabilities have run "
              f"out of float32 resolution here; refit the calibrator at the "
              f"deployment prevalence - see check_score_resolution().")

    return float(sweep["threshold"][best]), best_metrics


# =============================================================================
# §7  Shared metrics helpers
# =============================================================================

def _fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    except Exception:
        return float("nan")

def _fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        _, _, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan")
    except Exception:
        return float("nan")

def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def compute_metrics_fn():
    """Called by HuggingFace Trainer after every eval epoch."""
    def _compute(eval_pred: EvalPrediction) -> dict[str, float]:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        probs = softmax(logits, axis=-1)[:, 1]
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy":  float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
            "recall":    float(recall_score(labels, preds, average="binary", zero_division=0)),
            "f1":        float(f1_score(labels, preds, average="binary", zero_division=0)),
            "fpr":       _fpr(labels, preds),
            "fnr":       _fnr(labels, preds),
            "auc":       _auc(labels, probs),
        }
    return _compute


# =============================================================================
# §8  Ray Tune callback
# =============================================================================

class ReportToRay(TrainerCallback):
    """
    Forward eval metrics + optional checkpoint to Ray Tune after each epoch.

    When save_checkpoints=True, HF Trainer saves via save_strategy="epoch"
    to output_dir/checkpoint-{step}. This callback finds the latest
    checkpoint and reports it to Ray Tune, enabling:
    - Resuming interrupted HPO runs
    - Inspecting intermediate models from any trial
    - ASHA making informed early-stopping decisions
    """

    def __init__(self, save_checkpoints: bool = False, output_dir: str = None):
        self.save_checkpoints = save_checkpoints
        self.output_dir = output_dir

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        ray_m = {k.replace("eval_", "val_"): v
                 for k, v in (metrics or {}).items() if k.startswith("eval_")}
        ray_m["epoch"] = state.epoch

        if self.save_checkpoints and self.output_dir:
            # HF Trainer saves to output_dir/checkpoint-{global_step}
            ckpt_dir = os.path.join(
                self.output_dir, f"checkpoint-{state.global_step}"
            )
            if os.path.isdir(ckpt_dir):
                from ray.train import Checkpoint
                _report_to_tune(
                    ray_m,
                    checkpoint = Checkpoint.from_directory(ckpt_dir),
                )
            else:
                _report_to_tune(ray_m)
        else:
            _report_to_tune(ray_m)

# =============================================================================
# §9 Search spaces
#
#      Two strategies:
#        grid:      pre-generate configs × grid_search(model, config, seed)
#        two_phase: tune.choice(model) + random HP sampling (phase 1),
#                   then multi-seed validation of winner (phase 2)
# =============================================================================

_TUNE_BUILDERS = {
    "uniform":      lambda p: tune.uniform(p["low"], p["high"]),
    "loguniform":   lambda p: tune.loguniform(p["low"], p["high"]),
    "choice":       lambda p: tune.choice(list(p["values"])),
    "randint":      lambda p: tune.randint(p["low"], p["high"]),
}

# Python-native samplers for pre-generating HP configs (grid strategy)
_SAMPLERS = {
    "uniform":    lambda p, rng: float(rng.uniform(p["low"], p["high"])),
    "loguniform": lambda p, rng: float(np.exp(rng.uniform(
                      np.log(p["low"]), np.log(p["high"])))),
    "choice":     lambda p, rng: p["values"][rng.randint(len(p["values"]))],
    "randint":    lambda p, rng: int(rng.randint(p["low"], p["high"])),
}

def _generate_hp_configs(
    search_space_cfg: dict,
    n_configs: int,
    config_seed: int = 0,
) -> list[dict]:
    """Pre-generate N random HP configs from YAML search space."""
    rng = np.random.RandomState(config_seed)
    hp_samples = []

    space_dict = {
        name: {"type": str(spec["type"]), **{k: v for k, v in spec.items() if k != "type"}}
        for name, spec in search_space_cfg.items()
    }

    for i in range(n_configs):
        hp_sample = {"config_id": i}
        for param_name, param_def in space_dict.items():
            sampler = _SAMPLERS.get(param_def["type"])
            if sampler is None:
                raise ValueError(
                    f"Unknown type '{param_def['type']}' for {param_name}. "
                    f"Supported: {list(_SAMPLERS.keys())}"
                )
            hp_sample[param_name] = sampler(param_def, rng)
        hp_samples.append(hp_sample)

    return hp_samples

def build_grid_space(
    training_cfg,
    active_models: list[str],
    seeds: list[int],
    num_configs: int,
    config_seed: int = 0,
) -> dict:
    """
    Grid strategy: pre-generate configs, cross with all models × seeds.

    Guarantees equal trials per model AND same HP configs across seeds.
    Total trials = models × num_configs × seeds.
    """
    search_space_dict = OmegaConf.to_container(
        training_cfg.search_space, resolve=True
    )
    hp_configs = _generate_hp_configs(
        search_space_dict, num_configs, config_seed
    )

    return {
        "model_name": tune.grid_search(active_models),
        "seed":       tune.grid_search(seeds),
        "hp_config":  tune.grid_search(hp_configs),
    }

def build_random_space(
        training_cfg,
        active_models: list[str],
        seed: int,
        search_alg: str = "random",
) -> dict:
    """
    Two-phase strategy (phase 1): random HP sampling with fixed seed.

    When search_alg="random":
    model_name uses grid_search → guaranteed equal trial distribution.
    num_samples in TuneConfig = configs per model.

    When search_alg="optuna":
    model_name uses choice → Optuna controls everything.
    num_samples in TuneConfig = total trials across all models.
    Optuna's TPE sampler handles model selection jointly with HPs.
    """
    # Optuna can't coexist with grid_search — it needs full control
    if search_alg == "optuna":
        model_space = tune.choice(active_models)
    else:
        model_space = tune.grid_search(active_models)

    space = {
        "model_name": model_space,
        "seed": seed,
    }

    search_space_dict = OmegaConf.to_container(
        training_cfg.search_space, resolve=True
    )

    for param_name, param_cfg in search_space_dict.items():
        builder = _TUNE_BUILDERS.get(param_cfg["type"])
        if builder is None:
            raise ValueError(
                f"Unknown search space type '{param_cfg.type}' for {param_name}. "
                f"Supported: {list(_TUNE_BUILDERS.keys())}"
            )
        space[param_name] = builder(param_cfg)

    return space


# =============================================================================
# §9  Ray Tune trainable
# =============================================================================

def trainable(
    trial: dict[str, Any],
    data_sources: List,
    max_seq_length: int,
    training_mode: str,
    save_checkpoints: bool = False,
    split_cfg: dict = None,
    data_cache_dir: str = ".cache/pooled",
    cache_dir: str = ".cache/tokenized",
) -> None:
    """
    Ray Tune trainable. Called once per trial.

    trial:              dict from Ray Tune (NOT Hydra cfg)
    save_checkpoints:   if True, HF Trainer saves after each epoch
    split_cfg:          split ratios {train, val, cal, test, split_seed}
    data_cache_dir:     shared cache for pooled+split data
    cache_dir:          per-model cache for tokenized data
    """
    model_id = trial["model_name"]
    hp       = trial.get("hp_config", trial)
    spec     = MODELS.get(model_id, ModelSpec("unknown"))
    try:
        trial_id = tune.get_context().get_trial_id()
    except Exception:
        trial_id = "local"

    # Load tokenizer + model
    try:
        tokenizer, model = load_model(model_id, float(hp["classifier_dropout"]))
    except ModelLoadError as exc:
        print(f"\n{_BORDER}\n  ✗ LOAD FAILED  (trial {trial_id})  {model_id}"
              f"\n  {exc}\n{_BORDER}\n")
        _report_to_tune({**_FAILED_REPORT, "error":str(exc)})
        return

    # Apply LoRA (PEFT only)
    if training_mode == "peft":
        try:
            model = apply_lora(model, spec, hp)
        except Exception as exc:
            print(f"\n{_BORDER}\n  ✗ LORA FAILED  (trial {trial_id})  {model_id}"
                  f"\n  {exc}\n{_BORDER}\n")
            _report_to_tune({**_FAILED_REPORT, "error":str(exc)})
            return
    # Configuration for QAT (quantization-aware training)
    weight_fq = IntxFakeQuantizeConfig(
        dtype=torch.int8,
        granularity="per_channel",
        is_symmetric=True,
    )
    qat_config = QATConfig(
        weight_config=weight_fq,
        step="prepare",
    )
    quantize_(model, qat_config)


    # Load tokenized data for this model
    try:
        train_ds, val_ds, _cal_ds, _test_ds = load_and_preprocess_data_with_caching(
            data_sources=data_sources,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            cache_dir=cache_dir,
            data_cache_dir=data_cache_dir,
            split_cfg=split_cfg,
        )
    except Exception as exc:
        print(f"\n{_BORDER}\n  ✗ DATA LOAD FAILED  (trial {trial_id})"
              f"\n  {exc}\n{_BORDER}\n")
        _report_to_tune({**_FAILED_REPORT, "error":str(exc)})
        return

    # Train
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_args = TrainingArguments(
                output_dir=tmpdir,
                num_train_epochs=int(hp["num_train_epochs"]),
                per_device_train_batch_size=int(hp["per_device_train_batch_size"]),
                per_device_eval_batch_size=int(hp["per_device_train_batch_size"]) * 2,
                gradient_accumulation_steps=int(hp["gradient_accumulation_steps"]),
                learning_rate=float(hp["learning_rate"]),
                weight_decay=float(hp["weight_decay"]),
                warmup_ratio=float(hp["warmup_ratio"]),
                lr_scheduler_type="linear",
                max_grad_norm=1.0,
                eval_strategy="epoch",
                save_strategy="epoch" if save_checkpoints else "no",
                logging_strategy="epoch",
                load_best_model_at_end=False,
                report_to="none",
                disable_tqdm=True,
                fp16=False, # Necessary for mmBERT
                seed=int(trial["seed"]),
            )
            Trainer(
                model=model,
                args=train_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=compute_metrics_fn(),
                callbacks=[ReportToRay(
                    save_checkpoints=save_checkpoints,
                    output_dir=tmpdir,
                )],
            ).train()

    except torch.cuda.OutOfMemoryError:
        print(f"\n{_BORDER}\n  ✗ CUDA OOM  (trial {trial_id})  {model_id}"
              f"  bs={trial['per_device_train_batch_size']}\n{_BORDER}\n")
        _report_to_tune({**_FAILED_REPORT, "error":"cuda_oom"})
    except Exception as exc:
        print(f"\n{_BORDER}\n  ✗ TRAIN ERROR  (trial {trial_id})  {model_id}"
              f"\n  {exc}\n{_BORDER}\n")
        _report_to_tune({**_FAILED_REPORT, "error":str(exc)})

# =============================================================================
# §10b  Seed aggregation — pick winner by mean across seeds
#
#       Problem:  get_best_result() picks the single luckiest trial.
#                 With 5 seeds × 5 models, the "best" trial might be
#                 a model that scored 0.95 on one seed but 0.88 on others.
#
#       Solution: group trials by model_name, compute mean ± std of the
#                 metric across seeds, pick model with best mean.
#                 Then select that model's best single-trial config
#                 for final retraining.
# =============================================================================

def select_winner_by_mean(
    results,
    metric: str,
    mode: str,
) -> dict:
    """
    Aggregate Ray Tune results across seeds, pick winner by mean.

    Returns dict with:
      winner_model:      model_id with best mean metric
      winner_mean:       mean metric across seeds
      winner_std:        std across seeds
      winner_seeds:      number of seeds that ran for this model
      best_trial_params: Ray Tune trial dict from the best single trial
      best_trial:        trial_id of that trial
      all_models:        list of {model, mean, std, n_seeds} for reporting
    """

    # ── Collect per-model results ──
    model_scores = defaultdict(list)      # model_id → [metric_value, ...]
    model_trials = defaultdict(list)      # model_id → [result, ...]

    for result in results:
        trial_params = result.config      # Ray Tune's per-trial dict (NOT Hydra cfg)
        model_id = trial_params.get("model_name", "unknown")

        # Skip failed trials
        metric_val = result.metrics.get(metric)
        if metric_val is None:
            continue
        if metric == "val_loss" and metric_val == float("inf"):
            continue
        if metric != "val_loss" and metric_val == 0.0:
            continue

        model_scores[model_id].append(metric_val)
        model_trials[model_id].append(result)

    if not model_scores:
        raise RuntimeError("No successful trials found.")

    # ── Compute mean ± std per model ──
    model_stats = []
    for model_id, scores in model_scores.items():
        model_stats.append({
            "model": model_id,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n_seeds": len(scores),
        })

    # ── Sort: best mean first ──
    reverse = (mode == "max")
    model_stats.sort(key=lambda x: x["mean"], reverse=reverse)

    # ── Print comparison table ──
    print(f"\n{'='*64}")
    print(f"  SEED AGGREGATION  (metric={metric}, mode={mode})")
    print(f"{'='*64}")
    for i, s in enumerate(model_stats):
        marker = " ← WINNER" if i == 0 else ""
        short = s["model"].split("/")[-1]
        print(f"  {short:40s}  {s['mean']:.4f} ± {s['std']:.4f}  "
              f"(n={s['n_seeds']}){marker}")
    print(f"{'='*64}\n")

    # ── Pick best single trial from the winning model ──
    winner_model = model_stats[0]["model"]
    winner_trials = model_trials[winner_model]

    if mode == "max":
        best_trial_result = max(winner_trials, key=lambda r: r.metrics.get(metric, 0))
    else:
        best_trial_result = min(winner_trials, key=lambda r: r.metrics.get(metric, float("inf")))

    return {
        "winner_model": winner_model,
        "winner_mean":  model_stats[0]["mean"],
        "winner_std":   model_stats[0]["std"],
        "winner_seeds": model_stats[0]["n_seeds"],
        "best_trial_params": best_trial_result.config,
        "best_trial":   str(best_trial_result.metrics.get("trial_id", "n/a")),
        "all_models":   model_stats,
    }

# =============================================================================
# §11  Final evaluation with calibration + threshold
# =============================================================================
def _extract_predictions(trainer, dataset, name: str):
    """
    Run trainer.predict() and extract (logits, labels) with shape validation.

    Handles every known HF Trainer output format:
    - np.ndarray (N, C) → standard
    - tuple of np.ndarray → (logits, hidden_states, ...)
    - np.ndarray with ndim=3 → (num_outputs, N, C)
    - np.ndarray with dtype=object → numpy-wrapped tuple
    """
    out = trainer.predict(dataset)
    raw = out.predictions
    labels = out.label_ids

    if hasattr(labels, 'copy'):
        labels = labels.copy()

    # # ── Diagnostic print ──
    # print(f" {name} raw predictions: type={type(raw).__name__}", end="")
    # if isinstance(raw, np.ndarray):
    #     print(f" shape={raw.shape} dtype={raw.dtype}")
    # elif isinstance(raw, tuple):
    #     print(f" len={len(raw)}")
    # for i, item in enumerate(raw):
    #     if isinstance(item, np.ndarray):
    #         print(f" [{i}] ndarray shape={item.shape} dtype={item.dtype}")
    #     else:
    #         print(f" [{i}] {type(item).__name__}")
    # else:
    #     print(f" (unknown type)")

    # ── Extract logits ──
    logits = raw

    # Case 1: tuple → take first element
    if isinstance(logits, tuple):
        logits = logits[0]

    # Case 2: numpy object array (wraps a tuple) → extract
    if isinstance(logits, np.ndarray) and logits.dtype == object:
        print(f" ⚠ {name}: object array, converting")
        # Try to stack if elements are arrays
        try:
            logits = np.stack(logits)
        except (ValueError, TypeError):
            logits = logits[0]

    # Case 3: 3D array → take first slice
    if isinstance(logits, np.ndarray) and logits.ndim == 3:
        print(f" ⚠ {name}: 3D array {logits.shape}, taking [0]")
        logits = logits[0]

    # Case 4: 1D array → might be class-1 probabilities already
    if isinstance(logits, np.ndarray) and logits.ndim == 1:
        if logits.shape[0] == len(labels):
            print(f" ⚠ {name}: 1D predictions, treating as class-1 probs")
            # Convert to 2-class logits so downstream code works
            logits = np.stack([1 - logits, logits], axis=-1)

    # Ensure we have a copy
    if hasattr(logits, 'copy'):
        logits = logits.copy()

    # ── Validate ──
    if not isinstance(logits, np.ndarray):
        raise ValueError(
            f"{name}: could not extract logits. "
            f"Got type={type(logits).__name__} from predictions type={type(raw).__name__}"
        )

    if logits.ndim != 2:
        raise ValueError(
            f"{name}: expected 2D logits (N, num_classes), got shape={logits.shape}"
        )

    if logits.shape[0] != len(labels):
        raise ValueError(
            f"{name}: logits ({logits.shape[0]}) and labels ({len(labels)}) "
            f"have different lengths. Raw predictions type={type(raw).__name__}, "
            f"raw shape={raw.shape if hasattr(raw, 'shape') else 'N/A'}"
        )

    print(f" {name}: logits={logits.shape} labels={labels.shape} ✓")
    return logits, np.asarray(labels, dtype=int)

def final_eval_with_calibration(
    best_hp: dict[str, Any],
    best_trial_name: str,
    data_sources: list,
    max_seq_length: int,
    training_mode: str,
    seed: int,
    split_cfg: dict = None,
    output_dir: str = None,
    data_cache_dir: str = ".cache/pooled",
    cache_dir: str = ".cache/tokenized",
    stage: str = "final",# hpo or final
) -> dict[str, Any]:
    """
    Retrain → calibrate → threshold → test.

    best_hp: flattened HP dict from the winning trial (NOT Hydra cfg).
    split_cfg: split ratios {train, val, cal, test, split_seed}.
    data_cache_dir: shared cache for pooled+split data.
    cache_dir: per-model cache for tokenized data.
    """
    model_id = best_hp["model_name"]
    spec     = MODELS.get(model_id, ModelSpec("unknown"))
    # actual_model_id = _get_actual_model_id(model_id)

    print(f"\n  Processing {model_id} (trial: {best_trial_name})...")

    # Clear CUDA cache before loading new model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Load model
    try:
        tokenizer, model = load_model(model_id, float(best_hp["classifier_dropout"]))
    except ModelLoadError as exc:
        print(f"  ERROR: {exc}")
        return {
            "model": model_id,
            "seed": seed,
            "best_trial": best_trial_name,
            "error": str(exc),
        }

    # Apply LoRA if PEFT
    if training_mode == "peft":
        model = apply_lora(model, spec, best_hp)

    # Configuration for QAT (quantization-aware training)
    weight_fq = IntxFakeQuantizeConfig(
        dtype=torch.int8,
        granularity="per_channel",
        is_symmetric=True,
    )
    qat_config = QATConfig(
        weight_config=weight_fq,
        step="prepare",
    )
    quantize_(model, qat_config)

    # Load all splits with the correct tokenizer (model-specific cache)
    train_ds, val_ds, cal_ds, test_ds = load_and_preprocess_data_with_caching(
        data_sources=data_sources,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        cache_dir=cache_dir,
        data_cache_dir=data_cache_dir,
        split_cfg=split_cfg,
    )
    collator = DataCollatorWithPadding(tokenizer)

    # Retrain on train_ds
    print(f"  Retraining {model_id} (seed={seed}) …")
    with tempfile.TemporaryDirectory() as tmpdir:
        train_args = TrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=int(best_hp["num_train_epochs"]),
            per_device_train_batch_size=int(best_hp["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(best_hp["per_device_train_batch_size"]) * 2,
            gradient_accumulation_steps=int(best_hp["gradient_accumulation_steps"]),
            learning_rate=float(best_hp["learning_rate"]),
            weight_decay=float(best_hp["weight_decay"]),
            warmup_ratio=float(best_hp["warmup_ratio"]),
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            fp16=False, # Necessary for mmBERT
            seed=seed,
        )
        trainer = Trainer(
            model=model, args=train_args,
            train_dataset=train_ds,
            processing_class=tokenizer,
            data_collator=collator,
        )
        trainer.train()

        # ----- Extract predictions immmediately, with validation ----
        cal_logits, cal_labels = _extract_predictions(trainer, cal_ds, "cal")
        val_logits, val_labels = _extract_predictions(trainer, val_ds, "val")
        test_logits, test_labels = _extract_predictions(trainer, test_ds, "test")

        # Save the model and tokenizer with the specified path structure
        # Path: {args.output_dir}_seed{RANDOM_STATE}/best_model_{training_mode}/{model_name}/
        try:
            model_save_dir = (
                    Path(output_dir)
                    / f"best_model_{training_mode}"
                    / stage
                    / model_id.replace("/", "_")
                    / f"seed_{seed}"
            )
            model_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Saving model and tokenizer to {model_save_dir}")

            # Save model
            trainer.save_model(model_save_dir)

            # Save tokenizer
            tokenizer.save_pretrained(model_save_dir)

            for split_name, split_ds in [("val", val_ds), ("cal", cal_ds), ("test", test_ds)]:
                split_path = model_save_dir / "datasets" / split_name
                split_path.mkdir(parents=True, exist_ok=True)
                split_ds.save_to_disk(str(split_path))
                print(f"  Saved {split_name} dataset ({len(split_ds)} rows) to {split_path}")

            # Save training arguments
            with open(model_save_dir / "training_args.json", "w") as f:
                json.dump(train_args.to_dict(), f, indent=2)

            print(f"  Model and tokenizer saved successfully to {model_save_dir}")
        except Exception as e:
            print(f"  WARNING: Failed to save model and tokenizer: {e}")

    # Temperature calibration on cal_ds
    print("  Fitting temperature scaler on calibration set...")
    scaler = TemperatureScaler()
    temperature = scaler.fit(cal_logits, cal_labels)
    print(f" Optimal temperature T = {temperature:.4f}")

    # Threshold optimisation on cal_ds (after scaling) - use calibration set to avoid data leakage
    print("  Optimising decision threshold on calibration set...")
    cal_probs = scaler.scale(cal_logits)
    try:
        threshold, cal_metrics = optimize_threshold(cal_probs, cal_labels)
        # Check if required metrics exist before accessing them
        if "precision" in cal_metrics and "recall" in cal_metrics:
            cal_f1 = fbeta(cal_metrics["precision"], cal_metrics["recall"], beta=1.0)  # F1 score
            print(f"  Optimal threshold = {threshold:.4f}  (cal F1 = {cal_f1:.4f})")
        else:
            raise ValueError("Required metrics 'precision' and/or 'recall' are missing from computed metrics")
    except ValueError as e:
        print(f"  ERROR: Failed to optimize threshold: {e}")
        raise

    print(f" Optimal threshold = {threshold:.4f}")

    test_probs = scaler.scale(test_logits)
    test_probs = np.asarray(test_probs, dtype=float).ravel()
    test_labels = np.asarray(test_labels, dtype=int).ravel()
    test_preds = (test_probs >= threshold).astype(int)

    # ── Final shape check before confusion matrix ──
    assert test_labels.shape == test_preds.shape, \
        f"Shape mismatch before confusion_matrix: " \
        f"test_labels={test_labels.shape} test_preds={test_preds.shape}. " \
        f"Check scaler.scale() output shape."

    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()

    result = {
        "model": model_id,
        "seed": seed,
        "best_trial": best_trial_name,
        "temperature": round(temperature, 4),
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(test_labels, test_preds, average="binary", zero_division=0)), 4),
        "recall": round(float(recall_score(test_labels, test_preds, average="binary", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(test_labels, test_preds)), 4),
        "f1": round(float(f1_score(test_labels, test_preds, average="binary", zero_division=0)), 4),
        "fpr": round(_fpr(test_labels, test_preds), 4),
        "fnr": round(_fnr(test_labels, test_preds), 4),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }

    # Save result_per_seed.json
    with open(model_save_dir / "result_per_seed.json", "w") as f:
        json.dump(result, f, indent=2)

    # Clean up GPU memory
    print("  Cleaning up GPU memory...")
    del model, trainer, tokenizer
    
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return result


# =============================================================================
# §12  Results table
# =============================================================================

def _cm_str(row: dict) -> str:
    """Format confusion matrix as a compact string."""
    return f"TP={row['tp']},FN={row['fn']};FP={row['fp']},TN={row['tn']}"


def print_results_table(rows: list[dict]) -> None:
    try:
        from tabulate import tabulate
        headers = [
            "model", "seed", "best_trial", "temperature", "threshold",
            "precision", "recall", "accuracy", "F1", "[TP,FN;FP,TN]", "FPR", "FNR",
        ]
        table = [
            [
                r["model"].split("/")[-1],
                r["seed"],
                r["best_trial"],
                r["temperature"],
                r["threshold"],
                r["precision"],
                r["recall"],
                r["accuracy"],
                r["f1"],
                _cm_str(r),
                r["fpr"],
                r["fnr"],
            ]
            for r in rows
        ]
        print("\n" + tabulate(table, headers=headers, tablefmt="grid"))

    except ImportError:
        col_w = 20
        headers = ["model", "seed", "best_trial", "temp", "thresh",
                   "prec", "recall", "acc", "F1", "[TP,FN;FP,TN]", "FPR", "FNR"]
        print("\n" + " ".join(h.ljust(col_w) for h in headers))
        print("-" * (col_w + 2) * len(headers))
        for r in rows:
            vals = [
                r["model"].split("/")[-1], str(r["seed"]), r["best_trial"],
                str(r["temperature"]), str(r["threshold"]),
                str(r["precision"]), str(r["recall"]),
                str(r["accuracy"]), str(r["f1"]),
                _cm_str(r), str(r["fpr"]), str(r["fnr"]),
            ]
        print(" ".join(v.ljust(col_w) for v in vals))

# =============================================================================
# §14  HPO orchestration + main
# =============================================================================

def _save_checkpoint(output_dir: str, stage: str, data: dict) -> Path:
    """
    Save intermediate results to {output_dir}/checkpoints/{stage}.json.

    Called at every stage boundary. If the pipeline crashes, the last
    checkpoint contains everything needed to resume or analyze.
    """

    cp_dir = Path(output_dir) / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    path = cp_dir / f"{stage}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f" 💾 Checkpoint saved: {path}")
    return path

def _extract_trial_results(results, metric: str) -> list[dict]:
    """Extract per-trial results from Ray Tune into serializable dicts."""
    rows = []
    for result in results:
        trial_params = result.config
    rows.append({
        "model_name": trial_params.get("model_name", "?"),
        "seed": trial_params.get("seed", "?"),
        "config_id": trial_params.get("hp_config", {}).get("config_id", "?"),
        "hp": trial_params.get("hp_config", trial_params),
        "metrics": {k: v for k, v in result.metrics.items()
                    if isinstance(v, (int, float)) and not k.startswith("_")},
        "status": "ok" if result.metrics.get(metric) else "failed",
    })
    return rows

def _make_train_fn(cfg: DictConfig, data_sources: list):
    """Build Ray Tune trainable with resources attached."""
    save_checkpoints = cfg.hpo.get("save_checkpoints", False)

    # Convert OmegaConf -> plan dict for Ray pickling
    split_cfg = OmegaConf.to_container(cfg.data.splitting, resolve=True) if cfg.data.get("splitting") else None

    data_cache_dir = cfg.data.get("data_cache_dir", ".cache/pooled")
    cache_dir = cfg.data.get("cache_dir", ".cache/tokenized")

    train_fn = tune.with_parameters(
        trainable,
        data_sources=data_sources,
        max_seq_length=cfg.data.max_seq_length,
        training_mode=cfg.training.mode,
        save_checkpoints=save_checkpoints,
        split_cfg=split_cfg,
        data_cache_dir=data_cache_dir,
        cache_dir=cache_dir,
    )
    return tune.with_resources(
        train_fn,
        {"cpu": cfg.ray.cpus_per_trial,
         "gpu": cfg.ray.gpus_per_trial, "accelerator_type:H200": cfg.ray.num_nvidia_gpu_per_trial},
    )

def _run_grid_hpo(cfg: DictConfig, active: list[str], train_fn) -> tuple:
    """
    Grid strategy: pre-generate configs × grid(model, config, seed).
    Returns (winner_dict, best_hp_dict).
    """
    seeds = list(cfg.hpo.seeds)
    num_configs = cfg.hpo.num_configs
    metric = cfg.hpo.metric
    mode = _metric_mode(metric)
    output_dir = cfg.ray.output_dir

    space = build_grid_space(
        cfg.training, active, seeds,
        num_configs=num_configs,
        config_seed=cfg.hpo.get("config_seed", 0),
    )
    total_trials = len(active) * num_configs * len(seeds)
    print(f"\nStrategy: GRID  |  {cfg.training.mode.upper()}  |  "
          f"{len(active)} models × {num_configs} configs × {len(seeds)} seeds = "
          f"{total_trials} total trials")
    print(f"Optimising: {metric}  (mode={mode})")

    scheduler = ASHAScheduler(
        max_t=cfg.hpo.max_epochs, grace_period=1, reduction_factor=2,
    )

    results = tune.Tuner(
        train_fn,
        run_config=tune.RunConfig(
            name=f"nano_{cfg.training.mode}_grid",
            storage_path=str(Path(cfg.ray.output_dir).resolve()),
        ),
        tune_config=tune.TuneConfig(
            metric=metric, mode=mode,
            scheduler=scheduler,
            num_samples=1,
            max_concurrent_trials=cfg.ray.max_concurrent_trials,
        ),
        param_space=space,
    ).fit()

    # --- Checkpoint: all trial results ---
    trial_rows = _extract_trial_results(results, metric)
    _save_checkpoint(output_dir, "grid_all_trials", {
        "strategy": "grid",
        "total_trials": len(trial_rows),
        "metric": metric,
        "trials": trial_rows,
    })

    # --- Checkpoint: winner selection ---
    winner = select_winner_by_mean(results, metric=metric, mode=mode)

    # Flatten hp_config for downstream
    raw_trial = winner["best_trial_params"]
    best_hp = {**raw_trial.get("hp_config", {}),
               "model_name": raw_trial["model_name"]}

    _save_checkpoint(output_dir, "grid_winner", {
        "winner_model": winner["winner_model"],
        "winner_mean": winner["winner_mean"],
        "winner_std": winner["winner_std"],
        "winner_seeds": winner["winner_seeds"],
        "best_hp": best_hp,
        "all_models": winner["all_models"],
    })

    return winner, best_hp


def _print_phase1_ranking(p1_results, metric: str, mode: str, winner_model: str):
    """Print per-model best scores from phase 1."""
    model_best = defaultdict(lambda: float("-inf") if mode == "max" else float("inf"))
    for result in p1_results:
        mid = result.config.get("model_name", "?")
        val = result.metrics.get(metric)
        if val is None:
            continue
        if mode == "max" and val > model_best[mid]:
            model_best[mid] = val
        elif mode == "min" and val < model_best[mid]:
            model_best[mid] = val

    ranked = sorted(model_best.items(),
                    key=lambda x: x[1], reverse=(mode == "max"))
    print(f"\n{'='*64}")
    print(f"  PHASE 1 RESULTS  (metric={metric})")
    print(f"{'='*64}")
    for mid, val in ranked:
        marker = " ← WINNER" if mid == winner_model else ""
        print(f"  {mid.split('/')[-1]:40s}  {val:.4f}{marker}")
    print(f"{'='*64}\n")
    return ranked


def _run_two_phase_hpo(
    cfg: DictConfig, active: list[str], train_fn, data_sources: list,
) -> tuple:
    """
    Two-phase strategy:
      Phase 1: explore many configs with one seed.
      Phase 2: validate winner across all seeds.
    Returns (winner_dict, best_hp_dict).
    """
    seeds = list(cfg.hpo.seeds)
    num_configs = cfg.hpo.num_configs
    metric = cfg.hpo.metric
    mode = _metric_mode(metric)
    phase1_seed = cfg.hpo.get("phase1_seed", cfg.hpo.phase1_seed)
    output_dir = cfg.ray.output_dir

    # ── Phase 1: explore ──
    alg_name = cfg.hpo.search_alg  # "random" or "optuna"
    space = build_random_space(
        cfg.training, active, seed=phase1_seed, search_alg=alg_name,
    )

    # num_samples meaning differs by search algorithm:
    # random: num_samples = configs per model (grid_search handles models)
    # optuna: num_samples = total trials (optuna handles everything)
    if alg_name == "optuna":
        total_p1 = len(active) * num_configs
        p1_num_samples = total_p1
    else:
        total_p1 = len(active) * num_configs
        p1_num_samples = num_configs # per model (grid expands)

    print(f"\nStrategy: TWO-PHASE ({alg_name})")
    print(f"  Phase 1: {cfg.training.mode.upper()}  |  "
          f"{len(active)} models × {num_configs} configs × 1 seed ({phase1_seed}) = "
          f"{total_p1} trials")
    print(f"  Phase 2: winner × {len(seeds)} seeds = "
          f"{len(seeds)} validation trials")
    print(f"Optimising: {metric}  (mode={mode})")

    search_alg_obj = None
    if alg_name == "optuna":
        from ray.tune.search.optuna import OptunaSearch
        search_alg_obj = OptunaSearch(metric=metric, mode=mode)

    scheduler = ASHAScheduler(
        max_t=cfg.hpo.max_epochs, grace_period=1, reduction_factor=2,
    )

    p1_results = tune.Tuner(
        train_fn,
        run_config=tune.RunConfig(
            name=f"nano_{cfg.training.mode}_p1",
            storage_path=str(Path(cfg.ray.output_dir).resolve()),
        ),
        tune_config=tune.TuneConfig(
            metric=metric, mode=mode,
            scheduler=scheduler, search_alg=search_alg_obj,
            num_samples=p1_num_samples,
            max_concurrent_trials=cfg.ray.max_concurrent_trials,
        ),
        param_space=space,
    ).fit()

    p1_best = p1_results.get_best_result(metric=metric, mode=mode)
    best_hp = p1_best.config
    winner_model = best_hp["model_name"]

    ranked = _print_phase1_ranking(p1_results, metric, mode, winner_model)


    # --- Checkpoint: phase 1 results ---
    trial_rows = _extract_trial_results(p1_results, metric)
    _save_checkpoint(output_dir, "two_phase_p1", {
        "phase": 1,
        "phase1_seed": phase1_seed,
        "total_trials": len(trial_rows),
        "metric": metric,
        "winner_model": winner_model,
        "best_hp": best_hp,
        "ranking": [{"model": m, "best_score": s} for m, s in ranked],
        "trials": trial_rows,
    })

    # ── Phase 2: validate winner across seeds ──
    print(f"  Phase 2: validating {winner_model} across {len(seeds)} seeds …\n")
    phase2_rows = []
    for s in seeds:
        print(f"  ── Phase 2 seed {s} ──")
        row = final_eval_with_calibration(
            best_hp=best_hp,
            best_trial_name=str(p1_best.metrics.get("trial_id", "n/a")),
            data_sources=data_sources,
            max_seq_length=cfg.data.max_seq_length,
            training_mode=cfg.training.mode,
            seed=s,
            split_cfg=OmegaConf.to_container(cfg.data.splitting, resolve=True)
                if cfg.data.get("splitting") else None,
            output_dir=output_dir,
            stage="hpo",
        )
        phase2_rows.append(row)

        # --- Checkpoint: after each phase 2 seed ---
        _save_checkpoint(output_dir, "two_phase_p2_progress", {
            "phase": 2,
            "winner_model": winner_model,
            "seeds_completed": [r["seed"] for r in phase2_rows],
            "seeds_remaining": [x for x in seeds if x not in [r["seed"] for r in phase2_rows]],
            "results_so_far": phase2_rows,
        })

    p2_accuracy = [r["accuracy"] for r in phase2_rows]
    p2_accuracy_mean = float(np.mean(p2_accuracy))
    p2_accuracy_std = float(np.std(p2_accuracy))

    p2_precisions = [r["precision"] for r in phase2_rows]
    p2_precisions_mean = float(np.mean(p2_precisions))
    p2_precisions_std = float(np.std(p2_precisions))

    p2_recalls = [r["recall"] for r in phase2_rows]
    p2_recalls_mean = float(np.mean(p2_recalls))
    p2_recalls_std = float(np.std(p2_recalls))

    p2_f1s = [r["f1"] for r in phase2_rows]
    p2_f1s_mean = float(np.mean(p2_f1s))
    p2_f1s_std = float(np.std(p2_f1s))

    p2_fprs = [r["fpr"] for r in phase2_rows]
    p2_fprs_mean = float(np.mean(p2_fprs))
    p2_fprs_std = float(np.std(p2_fprs))

    p2_fnrs = [r["fnr"] for r in phase2_rows]
    p2_fnrs_mean = float(np.mean(p2_fnrs))
    p2_fnrs_std = float(np.std(p2_fnrs))

    print(f"\n{'='*64}")
    print(f"  PHASE 2 VALIDATION: {winner_model}")
    print(f"  Accuracy across {len(seeds)} seeds: {p2_accuracy_mean:.4f} ± {p2_accuracy_std:.4f}")
    print(f"  Precision across {len(seeds)} seeds: {p2_precisions_mean:.4f} ± {p2_precisions_std:.4f}")
    print(f"  Recall across {len(seeds)} seeds: {p2_recalls_mean:.4f} ± {p2_recalls_std:.4f}")
    print(f"  F1 across {len(seeds)} seeds: {p2_f1s_mean:.4f} ± {p2_f1s_std:.4f}")
    print(f"  FPR across {len(seeds)} seeds: {p2_fprs_mean:.4f} ± {p2_fprs_std:.4f}")
    print(f"  FNR across {len(seeds)} seeds: {p2_fnrs_mean:.4f} ± {p2_fnrs_std:.4f}")
    print(f"{'='*64}")

    winner = {
        "winner_model": winner_model,
        "winner_mean":  p2_f1s_mean,
        "winner_std":   p2_f1s_std,
        "winner_seeds": len(seeds),
        "best_hp":      best_hp,
        "best_trial":   str(p1_best.metrics.get("trial_id", "n/a")),
        "all_models":   [{"model": m, "best_score": s} for m, s in ranked],
        "phase2_results": phase2_rows,
    }

    # --- Checkpoint: phase 2 complete ---
    _save_checkpoint(output_dir, "two_phase_p2_complete",{
        "winner": winner,
        "best_hp": best_hp,
    })

    return winner, best_hp


def run_hpo(cfg: DictConfig, active: list[str], data_sources: list) -> tuple:
    """
    Dispatch to grid or two-phase HPO strategy.
    Returns (winner_dict, best_hp_dict, strategy_name).
    """
    strategy = cfg.hpo.strategy
    assert strategy in ("grid", "two_phase"), \
        f"Unknown HPO strategy: '{strategy}'. Must be 'grid' or 'two_phase'."

    train_fn = _make_train_fn(cfg, data_sources)

    if strategy == "grid":
        winner, best_hp = _run_grid_hpo(cfg, active, train_fn)
    else:
        winner, best_hp = _run_two_phase_hpo(cfg, active, train_fn, data_sources)

    return winner, best_hp, strategy


def run_final_eval(
    cfg: DictConfig, winner: dict, best_hp: dict, data_sources: list,
) -> tuple:
    """
    Retrain winner with held-out seeds, aggregate results.
    Returns (result_rows, final_mean, final_std).
    """
    final_seeds = list(cfg.final_seeds)
    best_trial = winner["best_trial"]
    output_dir = cfg.ray.output_dir

    print("\n" + "=" * 64)
    print("  POST-HOC CALIBRATION & FINAL EVALUATION")
    print(f"  Winner: {winner['winner_model']}  "
          f"(HPO mean {cfg.hpo.metric}={winner['winner_mean']:.4f} "
          f"± {winner['winner_std']:.4f}, {winner['winner_seeds']} seeds)")
    print(f"  Final retraining with {len(final_seeds)} held-out seeds: {final_seeds}")
    print("=" * 64)

    result_rows = []
    for s in final_seeds:
        print(f"\n  ── Seed {s} ──")
        row = final_eval_with_calibration(
            best_hp=best_hp,
            best_trial_name=best_trial,
            data_sources=data_sources,
            max_seq_length=cfg.data.max_seq_length,
            training_mode=cfg.training.mode,
            seed=s,
            split_cfg=OmegaConf.to_container(cfg.data.splitting, resolve=True)
                if cfg.data.get("splitting") else None,
            output_dir=output_dir,
            stage="final",
        )
        result_rows.append(row)

        # --- Checkpoint: after final seed ---
        _save_checkpoint(output_dir, "final_eval_progress", {
            "winner_model": winner["winner_model"],
            "seed_completed": [r["seed"] for r in result_rows],
            "seeds_remaining": [x for x in final_seeds if x not in [r["seed"] for r in result_rows]],
            "result_rows": result_rows,
        })

    final_accuracy = [r["accuracy"] for r in result_rows]
    final_accuracy_mean = float(np.mean(final_accuracy))
    final_accuracy_std = float(np.std(final_accuracy))

    final_precisions = [r["precision"] for r in result_rows]
    final_precisions_mean = float(np.mean(final_precisions))
    final_precisions_std = float(np.std(final_precisions))

    final_recalls = [r["recall"] for r in result_rows]
    final_recalls_mean = float(np.mean(final_recalls))
    final_recalls_std = float(np.std(final_recalls))

    final_f1s = [r["f1"] for r in result_rows]
    final_f1s_mean = float(np.mean(final_f1s))
    final_f1s_std = float(np.std(final_f1s))

    final_fprs = [r["fpr"] for r in result_rows]
    final_fprs_mean = float(np.mean(final_fprs))
    final_fprs_std = float(np.std(final_fprs))

    final_fnrs = [r["fnr"] for r in result_rows]
    final_fnrs_mean = float(np.mean(final_fnrs))
    final_fnrs_std = float(np.std(final_fnrs))

    print(f"\n{'=' * 64}")
    print(f"  FINAL F1 (across {len(final_seeds)} held-out seeds): ")
    print(f"  Accuracy across {len(final_seeds)} seeds: {final_accuracy_mean:.4f} ± {final_accuracy_std:.4f}")
    print(f"  Precision across {len(final_seeds)} seeds: {final_precisions_mean:.4f} ± {final_precisions_std:.4f}")
    print(f"  Recall across {len(final_seeds)} seeds: {final_recalls_mean:.4f} ± {final_recalls_std:.4f}")
    print(f"  F1 across {len(final_seeds)} seeds: {final_f1s_mean:.4f} ± {final_f1s_std:.4f}")
    print(f"  FPR across {len(final_seeds)} seeds: {final_fprs_mean:.4f} ± {final_fprs_std:.4f}")
    print(f"  FNR across {len(final_seeds)} seeds: {final_fnrs_mean:.4f} ± {final_fnrs_std:.4f}")
    print(f"{'=' * 64}")

    # --- Checkpoint: final evaluation complete ---
    _save_checkpoint(output_dir, "final_eval_progress", {
        "winner_model": winner["winner_model"],
        "final_seeds": final_seeds,
        "f1_mean": final_f1s_mean,
        "f1_std": final_f1s_std,
        "per_seed": result_rows,
    })

    return result_rows, final_f1s_mean, final_f1s_std


def save_summary(
    cfg: DictConfig, strategy: str, winner: dict, best_hp: dict,
    result_rows: list, final_mean: float, final_std: float,
) -> Path:
    """Write nano_summary.json with full audit trail."""
    out = Path(cfg.ray.output_dir) / "nano_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "training_mode":    cfg.training.mode,
        "hpo_strategy":     strategy,
        "winner": {
            "model":     winner["winner_model"],
            "hpo_mean":  winner["winner_mean"],
            "hpo_std":   winner["winner_std"],
            "hpo_seeds": winner["winner_seeds"],
        },
        "final": {
            "seeds":    list(cfg.final_seeds),
            "f1_mean":  final_mean,
            "f1_std":   final_std,
            "per_seed": result_rows,
        },
        "all_models": winner["all_models"],
        "best_hp":    best_hp,
    }, indent=2, default=str))
    return out

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    # ----- Validate mode ------
    assert cfg.mode in ("search", "evaluate"), \
        f"Unknown mode: '{cfg.mode}'. Must be 'search' or 'evaluate'."

    # RANDOM_STATE = cfg.seed
    #
    # # Set seeds for reproducibility
    # random.seed(RANDOM_STATE)
    # np.random.seed(RANDOM_STATE)
    # torch.manual_seed(RANDOM_STATE)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(RANDOM_STATE)
    # print(f"Reproducibility: Using seed {RANDOM_STATE} for Ray Tune, NumPy, random, and PyTorch")

    # --- SETUP ---
    # ---- Build model registry from YAML ---
    print(f"\n{'='*100}")
    print(f"LOADING MODEL SET")
    print(f"{'='*100}")
    try:
        # Mutated in place, not rebound: the module-level name is what the
        # trial functions read, and rebinding it here would need `global`.
        MODELS.clear()
        MODELS.update(build_model_registry_from_yaml(cfg.models))
        EXPECTED_MODELS[:] = list(MODELS.keys())
    except Exception as e:
        print(f"Error loading model set: {e}")
        raise
    # ---- Convert data sources to plain list (Ray needs pickable objects) ---
    try:
        data_sources = OmegaConf.to_container(cfg.data.sources, resolve=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    # --- Pre-flight ---
    active = preflight_check(list(MODELS.keys()))

    # --- HPO ---
    winner, best_hp, strategy = run_hpo(cfg, active, data_sources)

    # --- Final eval on held-out seeds ---
    result_rows, final_mean, final_std = run_final_eval(cfg, winner, best_hp, data_sources)

    # --- Results ----
    print_results_table(result_rows)
    out = save_summary(
        cfg, strategy, winner, best_hp, result_rows, final_mean, final_std,
    )
    print(f"\nSummary saved to: {out} ")


if __name__ == "__main__":
    main()