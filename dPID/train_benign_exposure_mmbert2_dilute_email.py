"""
train_benign_exposure_mmbert2_dilute_email.py

Standalone six-form email-background training for prompt injection detection.
The original train_benign_exposure_mmbert2_dilute_template.py is unchanged.

Payload, template + payload, and the two email + payload forms combined receive
equal shares of the payload-bearing budget. Email-only forms occupy 10% of the
benign budget. Training samples dynamically; validation, calibration, and test
use persisted fixed combinations from isolated email pools.

See email_augmentation/README.md for the full design and download/cache paths.

Usage:
    # Full fine-tuning with the existing FT hyperparameters:
    python train_benign_exposure_mmbert2_dilute_email.py

    # Full fine-tuning with a new hyperparameter search:
    python train_benign_exposure_mmbert2_dilute_email.py hpo.skip=false

    # PEFT (LoRA/QAT):
    python train_benign_exposure_mmbert2_dilute_email.py \
        configs/training/peft_benign_exposure_mmbert2_email.yaml

The inherited FT/PEFT checkpoint compatibility checks remain in effect.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple, Dict, List

from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from scipy.special import softmax
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, EvalPrediction,
    Trainer, TrainerCallback, TrainerState, TrainingArguments,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from datasets import load_from_disk, Dataset, concatenate_datasets
import random

from torchao.quantization import quantize_
from torchao.quantization.qat import QATConfig, IntxFakeQuantizeConfig

# Import reusable components from raytune_benchmark.py
from raytune_benchmark import (
    # Model loading
    apply_lora,
    build_model_registry_from_yaml,
    load_model,
    prepare_tokenizer,
    ModelSpec,
    ModelLoadError,
    
    # Metrics and callbacks
    compute_metrics_fn,
    ReportToRay,
    
    # Search space builders
    _TUNE_BUILDERS,
    _SAMPLERS,
    build_random_space,
    
    # Helpers
    preflight_check,
    _report_to_tune,
    _metric_mode,
    
    # Calibration components
    PlattCalibrator,
    ThresholdConstraints,
    check_score_resolution,
    deployment_sample_weight,
    optimize_threshold,
    metrics_from_probs,
    fbeta,
    _fpr,
    _fnr,
    
    # Prediction extraction
    _extract_predictions,
    
    # Global
    _BORDER,
    _MINIMIZE_METRICS,
)

# PEFT (LoRA) support: adapters are attached on top of the (optionally
# fake-quantized) frozen backbone and are never merged into it.
from peft import PeftModel

# ASL Loss for imbalanced classification
from asl_loss import ASLTrainer

# Template Dilution Collator for augmentation
from template_dilution_collator import (
    TemplateDilutionCollator,
    load_templates,
)
from email_augmentation import load_email_pools, make_email_collator


def _segment_enabled(hp):
    return "segment_tau" in hp


def _resolve_segment_hp(hp, perf_cfg):
    """Map the searched effective batch to one micro-batch/accumulation pair."""
    hp = dict(hp)
    if _segment_enabled(hp):
        effective = int(hp["effective_batch_size"])
        cap = int(_perf_flag(perf_cfg, "max_micro_batch_size", 32))
        micro = min(cap, effective)
        if micro <= 0 or effective % micro:
            raise ValueError("Effective batch must be divisible by the micro-batch size")
        hp.update(per_device_train_batch_size=micro,
                  gradient_accumulation_steps=effective // micro, classifier_dropout=0.0)
    return hp


def _training_components(hp):
    if _segment_enabled(hp):
        from segment_training import SegmentTrainer, segment_metrics
        return SegmentTrainer, segment_metrics(compute_metrics_fn())
    return ASLTrainer, compute_metrics_fn()


def _training_batch(hp, perf_cfg):
    pair = (int(hp["per_device_train_batch_size"]), int(hp["gradient_accumulation_steps"]))
    # Segment batches are already resolved; do not probe the legacy ASLTrainer.
    if _segment_enabled(hp):
        return pair
    return repack_micro_batch(*pair, _perf_flag(perf_cfg, "max_micro_batch_size", 32))


def _evaluation_collator(tokenizer, email_cfg):
    if email_cfg is not None and email_cfg.get("region_supervision", False):
        from segment_scoring import RegionPaddingCollator
        return RegionPaddingCollator(tokenizer)
    return DataCollatorWithPadding(tokenizer)

logging.getLogger("transformers").setLevel(logging.ERROR)

# Training modes. "ft" trains every parameter; "peft" freezes the backbone and
# trains LoRA adapters plus the classification head.
_TRAINING_MODES = ("ft", "peft")

# LoRA hyperparameters. These are deliberately NOT given defaults anywhere: a
# full fine-tuning checkpoint must be rejected, not silently reused with an
# FT-tuned learning rate.
_LORA_KEYS = ("lora_r", "lora_alpha", "lora_dropout")


# =============================================================================
# §0a  Throughput helpers
#
# One training micro-step costs ~76ms of fixed Python-dispatch/kernel-launch
# overhead regardless of how many samples it carries (measured across the 30
# trials in benign_exposure_results_mmbert2_seed42/benign_exposure_hpo/:
# 78.2ms/step at bs=2 vs 87.1ms at bs=8). At bs=2 only ~2ms of the step is
# actual compute, so 97% of an epoch is overhead. Reducing the NUMBER of
# micro-steps is therefore worth far more than making any single one cheaper.
# =============================================================================

def _perf_flag(perf_cfg: Optional[DictConfig], key: str, default):
    """Read a `perf` config knob, tolerating an absent block."""
    if perf_cfg is None:
        return default
    value = perf_cfg.get(key, None)
    return default if value is None else value


_REPACK_SAFE: Optional[bool] = None


def verify_repack_equivalence(tolerance: float = 1e-3) -> bool:
    """Check that folding accumulation into the batch really is gradient-neutral.

    The equivalence depends on the loss being divided by the accumulation count
    exactly once, which is an emergent property of how Trainer and Accelerator
    interact - and that interaction is version-sensitive. transformers 4.51.3 has

        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:

    while 5.9.0 has

        if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and ...

    Both currently leave the division to Accelerator.backward for this model, so
    the repack is safe (verified on both). But if a future version divides in
    Trainer as well, the repack would silently become an N-fold learning-rate
    increase - the loss would still go down, the run would still finish, and only
    the final metrics would be quietly wrong. That is worth 2 seconds to rule out.

    Runs one optimizer step of a tiny ModernBERT (same forward signature, so
    `model_accepts_loss_kwargs` resolves identically) at (1x4) and (4x1) and
    compares gradients.
    """
    global _REPACK_SAFE
    if _REPACK_SAFE is not None:
        return _REPACK_SAFE

    import tempfile
    from transformers import ModernBertConfig, ModernBertForSequenceClassification, TrainerCallback

    cfg = ModernBertConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, num_labels=2, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, cls_token_id=1, sep_token_id=2, classifier_dropout=0.0,
    )
    ds = Dataset.from_dict({
        "input_ids": [[1, 5, 6, 7, 2], [1, 8, 9, 2], [1, 3, 4, 5, 6, 2], [1, 7, 2]],
        "attention_mask": [[1] * 5, [1] * 4, [1] * 6, [1] * 3],
        "labels": [0, 1, 0, 1],
    })

    class _Grab(TrainerCallback):
        def __init__(self): self.g = None
        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            if self.g is None:
                self.g = {n: p.grad.detach().clone()
                          for n, p in model.named_parameters() if p.grad is not None}

    def _grads(bs, accum):
        torch.manual_seed(0)
        model = ModernBertForSequenceClassification(cfg)
        grab = _Grab()
        with tempfile.TemporaryDirectory() as td:
            args = TrainingArguments(
                output_dir=td, per_device_train_batch_size=bs,
                gradient_accumulation_steps=accum, learning_rate=0.0,
                lr_scheduler_type="constant", max_steps=1, seed=0, fp16=False,
                report_to="none", disable_tqdm=True, logging_strategy="no",
                save_strategy="no", use_cpu=True,
            )
            ASLTrainer(
                model=model, args=args, train_dataset=ds,
                data_collator=DataCollatorWithPadding(_repack_probe_tokenizer()),
                callbacks=[grab], gamma_pos=0.5, gamma_neg=1.5, asl_clip=0.02,
                check_finite_every=0,
            ).train()
        return grab.g

    try:
        a, b = _grads(1, 4), _grads(4, 1)
        keys = sorted(set(a) & set(b))
        rel = [(a[n] - b[n]).abs().max().item() / a[n].abs().max().item()
               for n in keys if a[n].abs().max().item() > 1e-12]
        worst = max(rel) if rel else 0.0
        _REPACK_SAFE = worst < tolerance
        if not _REPACK_SAFE:
            print(f"\n{_BORDER}\n  ✗ MICRO-BATCH REPACK DISABLED\n"
                  f"  Folding gradient accumulation into the batch is NOT gradient-neutral\n"
                  f"  under transformers/accelerate as installed (max relative gradient\n"
                  f"  difference {worst:.2e} > {tolerance:.0e}). Training will use the\n"
                  f"  sampled batch/accumulation split unchanged, which is slower but correct.\n"
                  f"{_BORDER}\n")
        else:
            print(f"  Repack equivalence verified (max relative grad diff {worst:.2e})")
    except Exception as exc:
        # Never let a self-test failure take down a trial; just skip the optimization.
        print(f"  ⚠ Repack self-check failed to run ({exc}); leaving batch split unchanged.")
        _REPACK_SAFE = False
    return _REPACK_SAFE


def _repack_probe_tokenizer():
    """Minimal padding-capable tokenizer for the self-check above."""
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models
    tk = Tokenizer(models.WordLevel(vocab={"[PAD]": 0}, unk_token="[PAD]"))
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]")


def repack_micro_batch(
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    cap: int,
) -> Tuple[int, int]:
    """Fold gradient accumulation into the per-device batch, effective batch fixed.

    `(bs=2, accum=8)` and `(bs=16, accum=1)` push the same 16 samples through
    one optimizer step, but the first does 8 forward/backward passes where the
    second does 1 - and each pass costs the same ~76ms of fixed overhead. So the
    second is ~5x faster for an identical gradient.

    Equivalence rests on the loss being divided by the accumulation count exactly
    once. It is, but by a coincidence between two libraries that must be pinned
    (see requirements.txt): ModernBertForSequenceClassification.forward takes
    **kwargs and defines no `accepts_loss_kwargs`, so Trainer sets
    model_accepts_loss_kwargs=True and SKIPS its own `loss /= accum`; accelerate's
    Accelerator.backward then applies it. If either half changes, this repack
    silently becomes an N-fold learning-rate change.

    The optimizer-step count (and hence the linear LR schedule) is unchanged,
    since it depends only on rows / effective_batch.

    One cosmetic side effect: Trainer sums the UNDIVIDED per-micro-step loss into
    its logs, so the reported `train_loss` for a previously accum=N config drops
    by N after the repack. Only the logged number changes - HPO selects on
    val_f1, which is unaffected.

    Verified empirically on mmBERT-small (dropout=0, fixed seed): 2x8 vs 16x1
    gives a median relative gradient difference of 1.5e-5 and 8x2 vs 16x1 gives
    1.6e-6 (fp32 summation-order noise, smaller with fewer partial sums), while a
    genuinely different effective batch gives 0.41.

    Returns:
        (per_device_batch_size, gradient_accumulation_steps), repacked.
    """
    effective = int(per_device_batch_size) * int(gradient_accumulation_steps)
    if effective == int(per_device_batch_size):
        return int(per_device_batch_size), int(gradient_accumulation_steps)
    if not verify_repack_equivalence():
        return int(per_device_batch_size), int(gradient_accumulation_steps)
    per_device = min(effective, int(cap))
    return per_device, max(1, effective // per_device)


def resolve_eval_batch_size(
    perf_cfg: Optional[DictConfig],
    search_space: Optional[DictConfig] = None,
    fallback: int = 64,
) -> int:
    """Eval batch size, decoupled from the sampled train batch size.

    Bound to the search space's LARGEST per-device batch rather than to the
    batch this trial happened to sample, so every trial evaluates at the same
    speed - which also makes time_this_iter_s comparable across trials. Eval runs
    under no_grad and stores no backward activations, so it needs far less memory
    than training at the same batch size.
    """
    if perf_cfg is not None and perf_cfg.get("eval_batch_size", None):
        return int(perf_cfg["eval_batch_size"])
    if search_space is not None:
        values = search_space.get("per_device_train_batch_size", {}).get("values", None)
        if values:
            return int(max(values)) * 2
    return int(fallback)


def sort_dataset_by_length(dataset: Dataset, text_key: str = "text") -> Dataset:
    """Sort by length so evaluation batches are length-homogeneous.

    Padding is to the longest row in the batch, so an unsorted batch is priced by
    its longest member. On the real 76,143-row validation set the mean length is
    55 tokens but an unsorted batch of 64 pads to 370 - a 6.7x waste that grows
    with batch size and cancels most of the gain from batching. Sorted, every
    batch pads to ~55 regardless of batch size.

    Exactly result-neutral: metrics are permutation-invariant, the eval collator
    decides template wrapping from blake2b(seed|id) rather than from position,
    and _extract_predictions reads labels from the same predict() pass, so logits
    and labels cannot drift apart.
    """
    if text_key in dataset.column_names:
        lengths = [len(t) if isinstance(t, str) else 0 for t in dataset[text_key]]
    elif "input_ids" in dataset.column_names:
        lengths = [len(x) for x in dataset["input_ids"]]
    else:
        return dataset

    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    return dataset.select(order)


# =============================================================================
# §0  Model preparation (QAT / LoRA)
# =============================================================================

def _apply_qat(model) -> None:
    """Swap every nn.Linear for an int8 per-channel weight fake-quant module.

    Shared by the HPO trainable and the final training run so the two cannot
    drift apart. In PEFT mode this runs BEFORE the adapters are attached, so the
    fake-quant applies to the frozen backbone only and the fp32 adapter learns to
    compensate for the backbone's quantization error.
    """
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


def _qat_enabled(peft_cfg: Optional[DictConfig]) -> bool:
    """QAT is on by default; only an explicit `peft.qat: false` turns it off."""
    if peft_cfg is None:
        return True
    return bool(peft_cfg.get("qat", True))


def _prepare_model(
    model,
    model_id: str,
    hp: Dict[str, Any],
    training_mode: str,
    models_registry: Optional[Dict[str, ModelSpec]],
    peft_cfg: Optional[DictConfig],
):
    """Apply QAT and (in PEFT mode) LoRA, in that order.

    The order matters: attaching LoRA first and then calling quantize_() would
    fake-quantize lora_A/lora_B as well, which is exactly what we do not want -
    the adapter must stay fp32.
    """
    if _qat_enabled(peft_cfg):
        _apply_qat(model)
        print(f"  QAT: on (int8 per-channel weight fake-quant)")
    else:
        print(f"  QAT: off")

    if training_mode != "peft":
        return model

    if not models_registry or model_id not in models_registry:
        raise ModelLoadError(
            f"PEFT mode requires '{model_id}' in the model registry "
            f"(it supplies peft_target_modules). "
            f"Available: {sorted(models_registry or [])}"
        )

    spec = models_registry[model_id]
    extra = list((peft_cfg or {}).get("extra_modules_to_save", []) or [])
    if extra:
        # peft appends {"classifier", "score"} to modules_to_save for SEQ_CLS,
        # so the classification head stays trainable either way.
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(hp["lora_r"]),
            lora_alpha=float(hp["lora_alpha"]),
            lora_dropout=float(hp["lora_dropout"]),
            target_modules=spec.peft_target_modules,
            bias="none",
            modules_to_save=extra,
        ))
        model.print_trainable_parameters()
    else:
        model = apply_lora(model, spec, hp)

    print(f"  Applied LoRA to {model_id} "
          f"(r={hp['lora_r']}, alpha={hp['lora_alpha']}, "
          f"dropout={hp['lora_dropout']}, "
          f"target_modules={list(spec.peft_target_modules)})")
    return model


def load_trained_model(model_path: str, training_mode: str = "ft"):
    """Load a model saved by train_final_model().

    In PEFT mode the saved root directory holds the BASE only (frozen backbone +
    trained classification head, no LoRA delta) so that it can be quantized
    without quantizing the adapter. The adapter lives in `adapter/` and must be
    attached here, otherwise the returned model silently scores without it.

    Evaluation is fp32: the training-time fake-quant is deliberately not
    reapplied (int8 has no GPU acceleration, so it is not simulated here).
    """
    from segment_scoring import is_segment_checkpoint, load_scoring_model
    if is_segment_checkpoint(model_path):
        return load_scoring_model(model_path, training_mode)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if training_mode == "peft":
        adapter_dir = Path(model_path) / "adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"PEFT mode but no adapter directory at {adapter_dir}. "
                f"The root directory holds the base weights only; without the "
                f"adapter the model has no LoRA delta."
            )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        print(f"  Attached LoRA adapter from {adapter_dir}")
    model.eval()
    return model


# =============================================================================
# §1  Data Loading (specific to benign exposure with 16-file split)
# =============================================================================

def load_split_data(split_dir: str, seed: int) -> Dict[str, Dataset]:
    """
    Load pre-split data from data_split2/{seed}/.
    
    Returns dict with keys for all 16 splits:
        M_core_train, M_core_val, M_core_cal, M_core_test
        M_extra_train, M_extra_val, M_extra_cal, M_extra_test
        B_core_train, B_core_val, B_core_cal, B_core_test
        B_extra_train, B_extra_val, B_extra_cal, B_extra_test
    """
    print("\n" + "=" * 64)
    print("Loading pre-split data (16-file structure)...")
    print("=" * 64)
    
    base_path = Path(split_dir) / str(seed)
    
    if not base_path.exists():
        raise FileNotFoundError(
            f"Split data not found at {base_path}. "
            f"Run split_dataset2.py first to generate the splits."
        )
    
    splits = {}
    split_names = [
        "M_core_train", "M_core_val", "M_core_cal", "M_core_test",
        "M_extra_train", "M_extra_val", "M_extra_cal", "M_extra_test",
        "B_core_train", "B_core_val", "B_core_cal", "B_core_test",
        "B_extra_train", "B_extra_val", "B_extra_cal", "B_extra_test",
    ]
    
    for name in split_names:
        split_path = base_path / name
        if not split_path.exists():
            raise FileNotFoundError(
                f"Required split '{name}' not found at {split_path}. "
                f"Run split_dataset2.py first to generate the splits."
            )
        splits[name] = load_from_disk(str(split_path))
        print(f"  {name}: {len(splits[name])} samples")
    
    return splits


def combine_malicious_pools(M_core: Dataset, M_extra: Dataset) -> Dataset:
    """
    Combine M_core and M_extra into single malicious pool.
    """
    if len(M_core) == 0 and len(M_extra) == 0:
        return Dataset.from_dict({'text': [], 'label': [], 'id': []})
    elif len(M_core) == 0:
        return M_extra
    elif len(M_extra) == 0:
        return M_core
    else:
        return concatenate_datasets([M_core, M_extra])


def combine_benign_pools(B_core: Dataset, B_extra: Dataset) -> Dataset:
    """
    Combine B_core and B_extra into single benign pool.
    """
    if len(B_core) == 0 and len(B_extra) == 0:
        return Dataset.from_dict({'text': [], 'label': [], 'id': []})
    elif len(B_core) == 0:
        return B_extra
    elif len(B_extra) == 0:
        return B_core
    else:
        return concatenate_datasets([B_core, B_extra])


def sample_training_view(
    M_train: Dataset,
    B_train: Dataset,
    benign_to_malicious_ratio: int,
    seed: int,
    email_collator=None,
) -> Dataset:
    """
    Sample training data with configurable benign:malicious ratio.
    
    Args:
        M_train: Combined malicious training samples (M_core + M_extra)
        B_train: Combined benign training samples (B_core + B_extra)
        benign_to_malicious_ratio: How many benign samples per malicious (5, 10, 15, 20, 25, 30)
        seed: Random seed for sampling
    
    Returns:
        Combined dataset with sampled malicious and benign data
    """
    if email_collator is not None:
        M_train = email_collator.filter_payloads(M_train)
        B_train = email_collator.filter_payloads(B_train)
    n_malicious = len(M_train)
    n_benign_needed = n_malicious * benign_to_malicious_ratio
    
    # Shuffle and sample malicious
    M_shuffled = M_train.shuffle(seed=seed)
    
    # Shuffle and sample benign
    B_shuffled = B_train.shuffle(seed=seed + 1000)
    n_benign_actual = min(n_benign_needed, len(B_train))
    
    if n_benign_actual < n_benign_needed:
        print(f"  ⚠ Warning: Requested {n_benign_needed} benign samples, "
              f"but only {len(B_train)} available. Using all.")
    
    B_sampled = B_shuffled.select(range(n_benign_actual))
    
    # Combine
    combined = concatenate_datasets([M_shuffled, B_sampled])
    combined = combined.shuffle(seed=seed + 2000)
    
    print(f"  Sampled training view: {n_malicious} malicious + {n_benign_actual} benign "
          f"(ratio 1:{n_benign_actual/n_malicious:.1f})")
    
    return combined


def build_validation_set(
    M_val: Dataset,
    B_val: Dataset,
    benign_per_malicious: int = 50,
    email_collator=None,
) -> Dataset:
    """
    Build validation set with fixed benign:malicious ratio.
    
    Args:
        M_val: Combined malicious validation samples (M_core + M_extra)
        B_val: Combined benign validation samples (B_core + B_extra)
        benign_per_malicious: Fixed ratio (default 50)
    
    Returns:
        Combined validation dataset
    """
    if email_collator is not None:
        M_val = email_collator.filter_payloads(M_val)
        B_val = email_collator.filter_payloads(B_val)
    n_malicious = len(M_val)
    n_benign_needed = n_malicious * benign_per_malicious
    
    n_benign_actual = min(n_benign_needed, len(B_val))
    
    if n_benign_actual < n_benign_needed:
        print(f"  ⚠ Warning: Requested {n_benign_needed} benign validation samples, "
              f"but only {len(B_val)} available. Using all.")
    
    B_sampled = B_val.select(range(n_benign_actual))
    
    # Combine
    combined = concatenate_datasets([M_val, B_sampled])
    
    print(f"  Validation set: {n_malicious} malicious + {n_benign_actual} benign "
          f"(ratio 1:{n_benign_actual/n_malicious:.1f})")
    
    return combined


# =============================================================================
# §2  Calibration/Test Set Building
# =============================================================================

def build_calibration_view(
    M_cal: Dataset,
    B_cal: Dataset,
    benign_per_malicious: int,
    seed: int,
    email_collator=None,
) -> Dataset:
    """
    Build calibration view with specified benign:malicious ratio.
    
    Used for both temperature scaling (10:1) and threshold optimization (500:1).
    
    Args:
        M_cal: Combined malicious calibration samples (M_core + M_extra)
        B_cal: Combined benign calibration samples (B_core + B_extra)
        benign_per_malicious: Ratio of benign to malicious samples
        seed: Random seed for sampling
    
    Returns:
        Combined calibration dataset
    """
    if email_collator is not None:
        M_cal = email_collator.filter_payloads(M_cal)
        B_cal = email_collator.filter_payloads(B_cal)
    n_malicious = len(M_cal)
    n_benign_needed = n_malicious * benign_per_malicious
    
    # Shuffle and sample
    B_shuffled = B_cal.shuffle(seed=seed + 3000)
    n_benign_actual = min(n_benign_needed, len(B_cal))
    
    if n_benign_actual < n_benign_needed:
        print(f"  ⚠ Warning: Requested {n_benign_needed} benign calibration samples, "
              f"but only {len(B_cal)} available. Using all.")
    
    B_sampled = B_shuffled.select(range(n_benign_actual))
    
    # Combine
    combined = concatenate_datasets([M_cal, B_sampled])
    combined = combined.shuffle(seed=seed + 4000)
    
    print(f"  Calibration view: {n_malicious} malicious + {n_benign_actual} benign "
          f"(ratio 1:{n_benign_actual/n_malicious:.1f})")
    
    return combined


def build_test_set(
    M_test: Dataset,
    B_test: Dataset,
    benign_per_malicious: int = 500,
    seed: int = 42,
    email_collator=None,
) -> Dataset:
    """
    Build test set with deployment-like ratio (500:1).
    
    Args:
        M_test: Combined malicious test samples (M_core + M_extra)
        B_test: Combined benign test samples (B_core + B_extra)
        benign_per_malicious: Ratio of benign to malicious samples (default 500)
        seed: Random seed for sampling
    
    Returns:
        Combined test dataset
    """
    if email_collator is not None:
        M_test = email_collator.filter_payloads(M_test)
        B_test = email_collator.filter_payloads(B_test)
    n_malicious = len(M_test)
    n_benign_needed = n_malicious * benign_per_malicious
    
    # Shuffle and sample
    B_shuffled = B_test.shuffle(seed=seed + 5000)
    n_benign_actual = min(n_benign_needed, len(B_test))
    
    if n_benign_actual < n_benign_needed:
        print(f"  ⚠ Warning: Requested {n_benign_needed} benign test samples, "
              f"but only {len(B_test)} available. Using all.")
    
    B_sampled = B_shuffled.select(range(n_benign_actual))
    
    # Combine
    combined = concatenate_datasets([M_test, B_sampled])
    combined = combined.shuffle(seed=seed + 6000)
    
    print(f"  Test set: {n_malicious} malicious + {n_benign_actual} benign "
          f"(ratio 1:{n_benign_actual/n_malicious:.1f})")
    
    return combined


# =============================================================================
# §3  Tokenization
# =============================================================================

def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_seq_length: int
) -> Dataset:
    """
    Tokenize a single dataset.
    
    Expects 'text' and 'label' columns.
    Returns dataset with 'input_ids', 'attention_mask', 'labels' columns.
    """
    # Rename label → labels (HuggingFace Trainer expects "labels")
    if 'label' in dataset.column_names:
        dataset = dataset.rename_column('label', 'labels')
    
    # Convert labels to int (required for cross-entropy loss)
    if 'labels' in dataset.column_names:
        import datasets as ds
        dataset = dataset.cast_column('labels', ds.Value('int64'))
    
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], 
            truncation=True,
            padding=False, 
            max_length=max_seq_length,
        )
    
    dataset = dataset.map(tokenize_fn, batched=True)
    
    # Keep only model-relevant columns
    keep_cols = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    drop_cols = [c for c in dataset.column_names if c not in keep_cols]
    dataset = dataset.remove_columns(drop_cols)
    
    return dataset


# =============================================================================
# §3  Ray Tune Trainable (modified for 16-file split structure)
# =============================================================================

_FAILED_REPORT = {
    "val_loss":      float("inf"),
    "val_accuracy":  0.0,
    "val_f1":        0.0,
    "val_precision": 0.0,
    "val_recall":    0.0,
}


def benign_exposure_trainable(
    trial: Dict[str, Any],
    split_data: Dict[str, Dataset],
    max_seq_length: int,
    benign_per_malicious_val: int,
    seed: int,
    template_cfg: Optional[DictConfig] = None,
    training_mode: str = "ft",
    models_registry: Optional[Dict[str, ModelSpec]] = None,
    peft_cfg: Optional[DictConfig] = None,
    perf_cfg: Optional[DictConfig] = None,
    eval_batch_size: Optional[int] = None,
    email_cfg: Optional[DictConfig] = None,
    email_pools=None,
) -> None:
    """
    Ray Tune trainable for benign exposure training.
    
    Args:
        trial: Ray Tune trial dict with hyperparameters
        split_data: Pre-split data dict (16 files)
        max_seq_length: Max sequence length for tokenization
        benign_per_malicious_val: Fixed ratio for validation set
        seed: Random seed
        template_cfg: Optional config for template dilution collator
        training_mode: "ft" for full fine-tuning, "peft" for LoRA
        models_registry: Model specs; required in peft mode for peft_target_modules
        peft_cfg: Optional `peft` config block (qat / extra_modules_to_save)
        perf_cfg: Optional `perf` config block (throughput knobs; see §0a)
        eval_batch_size: Fixed eval batch size, independent of the sampled
            train batch size
    """
    model_id = trial["model_name"]
    hp = _resolve_segment_hp(trial, perf_cfg)
    if _segment_enabled(hp):
        from transformers import set_seed
        set_seed(seed)  # Seed new token-head and adapter initialization as well.
    
    try:
        trial_id = tune.get_context().get_trial_id()
    except Exception:
        trial_id = "local"
    
    # Load tokenizer + model
    try:
        tokenizer, model = load_model(
            model_id, float(hp["classifier_dropout"]),
            attn_implementation=_perf_flag(perf_cfg, "attn_implementation", "eager"),
        )
    except ModelLoadError as exc:
        print(f"\n{_BORDER}\n  ✗ LOAD FAILED  (trial {trial_id})  {model_id}"
              f"\n  {exc}\n{_BORDER}\n")
        _report_to_tune({**_FAILED_REPORT, "error": str(exc)})
        return
    
    # QAT first, then LoRA (see _prepare_model)
    try:
        if _segment_enabled(hp):
            from segment_scoring import make_segment_model
            model = make_segment_model(model, tokenizer, hp)
        model = _prepare_model(
            model, model_id, hp, training_mode, models_registry, peft_cfg
        )
    except Exception as exc:
        print(f"\n{_BORDER}\n  ✗ MODEL PREP FAILED  (trial {trial_id})  {model_id}"
              f"\n  {exc}\n{_BORDER}\n")
        _report_to_tune({**_FAILED_REPORT, "error": str(exc)})
        return
    
    email_train_collator = make_email_collator(
        tokenizer, email_pools, email_cfg, template_cfg, "train", seed, "random"
    )
    email_eval_collator = make_email_collator(
        tokenizer, email_pools, email_cfg, template_cfg, "valid", seed, "fixed"
    )

    # Combine M_core + M_extra for training
    M_train = combine_malicious_pools(
        split_data["M_core_train"], 
        split_data["M_extra_train"]
    )
    
    # Combine B_core + B_extra for training
    B_train = combine_benign_pools(
        split_data["B_core_train"], 
        split_data["B_extra_train"]
    )
    
    # Debug: Print column names to verify data structure
    print(f"  [Trainable] M_train columns: {M_train.column_names}")
    print(f"  [Trainable] B_train columns: {B_train.column_names}")
    
    # Sample training view with benign_to_malicious_ratio
    train_ds = sample_training_view(
        M_train=M_train,
        B_train=B_train,
        benign_to_malicious_ratio=int(hp["benign_to_malicious_ratio"]),
        seed=seed,
        email_collator=email_train_collator,
    )
    
    # Combine M_core + M_extra for validation
    M_val = combine_malicious_pools(
        split_data["M_core_val"], 
        split_data["M_extra_val"]
    )
    
    # Combine B_core + B_extra for validation
    B_val = combine_benign_pools(
        split_data["B_core_val"], 
        split_data["B_extra_val"]
    )
    
    # Debug: Print column names to verify data structure
    print(f"  [Trainable] M_val columns: {M_val.column_names}")
    print(f"  [Trainable] B_val columns: {B_val.column_names}")
    
    # Build validation set with fixed ratio
    val_ds = build_validation_set(
        M_val=M_val,
        B_val=B_val,
        benign_per_malicious=benign_per_malicious_val,
        email_collator=email_eval_collator,
    )
    
    # Debug: Print final dataset columns
    print(f"  [Trainable] train_ds columns: {train_ds.column_names}")
    print(f"  [Trainable] val_ds columns: {val_ds.column_names}")
    
    # Note: Do NOT tokenize here - TemplateDilutionCollator handles tokenization
    # The dataset keeps 'text' and 'label' columns for the collator to use
    
    # Ensure 'text' and 'label' columns exist
    # First check if we have 'labels' that needs to be renamed to 'label'
    if 'labels' in train_ds.column_names and 'label' not in train_ds.column_names:
        train_ds = train_ds.rename_column('labels', 'label')
        print(f"  [Trainable] Renamed 'labels' -> 'label' in train_ds")
    if 'labels' in val_ds.column_names and 'label' not in val_ds.column_names:
        val_ds = val_ds.rename_column('labels', 'label')
        print(f"  [Trainable] Renamed 'labels' -> 'label' in val_ds")
    
    # Check if 'text' column is missing and try to recover it
    if 'text' not in train_ds.column_names:
        available = train_ds.column_names
        print(f"  [Trainable] WARNING: 'text' column missing in train_ds. Available: {available}")
        # If we have 'input_ids' but no 'text', the data was pre-tokenized
        # We need to use the original data source
        if 'input_ids' in available:
            raise ValueError(
                f"Dataset appears to be pre-tokenized (has 'input_ids' but no 'text'). "
                f"TemplateDilutionCollator requires raw 'text' column. "
                f"Available columns: {available}"
            )
    
    if 'text' not in val_ds.column_names:
        available = val_ds.column_names
        print(f"  [Trainable] WARNING: 'text' column missing in val_ds. Available: {available}")
        if 'input_ids' in available:
            raise ValueError(
                f"Dataset appears to be pre-tokenized (has 'input_ids' but no 'text'). "
                f"TemplateDilutionCollator requires raw 'text' column. "
                f"Available columns: {available}"
            )
    
    # Fold gradient accumulation into the per-device batch (same effective batch,
    # same optimizer-step count, ~8x fewer forward/backward launches at bs=2).
    micro_bs, accum = _training_batch(hp, perf_cfg)
    if (micro_bs, accum) != (int(hp["per_device_train_batch_size"]),
                             int(hp["gradient_accumulation_steps"])):
        print(f"  micro-batch {hp['per_device_train_batch_size']}x"
              f"{hp['gradient_accumulation_steps']} -> {micro_bs}x{accum} "
              f"(effective batch unchanged: {micro_bs * accum})")
    eval_bs = int(eval_batch_size or micro_bs * 2)
    
    # Length-sorted eval batches: padding is to the batch maximum, so an unsorted
    # batch is priced by its longest row. Result-neutral (see sort_dataset_by_length).
    if _perf_flag(perf_cfg, "sort_eval_by_length", True):
        val_ds = sort_dataset_by_length(val_ds)
    
    # Build train and eval collators (use TemplateDilutionCollator if enabled)
    if email_train_collator is not None:
        train_collator = email_train_collator
        val_ds = email_eval_collator.materialize(val_ds, "valid")
        if _perf_flag(perf_cfg, "sort_eval_by_length", True):
            val_ds = sort_dataset_by_length(val_ds)
        eval_collator = _evaluation_collator(tokenizer, email_cfg)
    elif template_cfg is not None and template_cfg.enabled:
        print(f"  Using TemplateDilutionCollator (probability={template_cfg.probability})")
        # Load the instruction templates the payload is wrapped into.
        templates = load_templates(
            template_cfg.template_path,
            template_key=template_cfg.template_key,
            placeholder=template_cfg.placeholder,
        )
        print(f"  Loaded {len(templates)} templates from {template_cfg.template_path}")
        # Train collator: random mode (wrap a random ~probability fraction each call)
        train_collator = TemplateDilutionCollator(
            tokenizer=tokenizer,
            templates=templates,
            text_key=template_cfg.text_key,
            label_key=template_cfg.label_key,
            id_key=template_cfg.id_key,
            placeholder=template_cfg.placeholder,
            dilution_probability=float(template_cfg.probability),
            mode="random",
            seed=seed,
            max_length=int(template_cfg.max_length),
        )
        # Eval collator: fixed mode (deterministic wrap decision per sample ID)
        eval_collator = TemplateDilutionCollator(
            tokenizer=tokenizer,
            templates=templates,
            text_key=template_cfg.text_key,
            label_key=template_cfg.label_key,
            id_key=template_cfg.id_key,
            placeholder=template_cfg.placeholder,
            dilution_probability=float(template_cfg.probability),
            mode="fixed",
            seed=seed,
            max_length=int(template_cfg.max_length),
        )
    else:
        train_collator = DataCollatorWithPadding(tokenizer)
        eval_collator = DataCollatorWithPadding(tokenizer)
    
    # Train
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_args = TrainingArguments(
                output_dir=tmpdir,
                num_train_epochs=int(hp["num_train_epochs"]),
                per_device_train_batch_size=micro_bs,
                per_device_eval_batch_size=eval_bs,
                gradient_accumulation_steps=accum,
                learning_rate=float(hp["learning_rate"]),
                weight_decay=float(hp["weight_decay"]),
                warmup_ratio=float(hp["warmup_ratio"]),
                lr_scheduler_type="linear",
                max_grad_norm=1.0,
                eval_strategy="epoch",
                save_strategy="no",
                logging_strategy="epoch",
                load_best_model_at_end=False,
                report_to="none",
                disable_tqdm=True,
                fp16=False,
                bf16=_perf_flag(perf_cfg, "bf16", False) and torch.cuda.is_available(),
                tf32=_perf_flag(perf_cfg, "tf32", False) or None,
                seed=seed,
                remove_unused_columns=False,  # Keep 'text' column for TemplateDilutionCollator
            )
            
            trainer_class, metrics_function = _training_components(hp)
            trainer_class(
                model=model,
                args=train_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                processing_class=tokenizer,
                train_data_collator=train_collator,
                eval_data_collator=eval_collator,
                compute_metrics=metrics_function,
                callbacks=[ReportToRay(save_checkpoints=False, output_dir=tmpdir)],
                # ASL Loss hyperparameters
                gamma_pos=float(hp.get("gamma_pos", 0.0)),
                gamma_neg=float(hp.get("gamma_neg", 0.0)),
                asl_clip=float(hp.get("asl_clip", 0.0)),
                check_finite_every=int(_perf_flag(perf_cfg, "check_finite_every", 50)),
            ).train()
    
    except torch.cuda.OutOfMemoryError:
        print(f"\n{_BORDER}\n  ✗ CUDA OOM  (trial {trial_id})  {model_id}"
              f"  bs={hp['per_device_train_batch_size']}\n{_BORDER}\n")
        _report_to_tune({**_FAILED_REPORT, "error": "cuda_oom"})
    except Exception as exc:
        print(f"\n{_BORDER}\n  ✗ TRAIN ERROR  (trial {trial_id})  {model_id}"
              f"\n  {exc}\n{_BORDER}\n")
        _report_to_tune({**_FAILED_REPORT, "error": str(exc)})


# =============================================================================
# §4  HPO Orchestration
# =============================================================================

def build_benign_exposure_space(
    training_cfg,
    active_models: List[str],
    seed: int,
    search_alg: str = "optuna",
) -> Dict:
    """
    Build Ray Tune search space for benign exposure training.
    
    Only supports optuna search algorithm.
    """
    
    # Only optuna is currently implemented
    if search_alg == "optuna":
        model_space = tune.choice(active_models)
    else:
        raise NotImplementedError(
            f"search_alg='{search_alg}' is not implemented. "
            f"Only 'optuna' is currently supported."
        )
    
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
                f"Unknown search space type '{param_cfg['type']}' for {param_name}. "
                f"Supported: {list(_TUNE_BUILDERS.keys())}"
            )
        space[param_name] = builder(param_cfg)
    
    return space


def run_benign_exposure_hpo(
    cfg: DictConfig,
    active_models: List[str],
    split_data: Dict[str, Dataset],
    models_registry: Optional[Dict[str, ModelSpec]] = None,
    email_pools=None,
) -> Tuple[Dict, Dict]:
    """
    Run Ray Tune HPO for benign exposure training.
    
    Supports resuming from interrupted runs by checking for existing experiment
    at the storage path and using Tuner.restore() if found.
    
    Returns:
        (winner_dict, best_hp_dict)
    """
    metric = cfg.hpo.metric
    mode = _metric_mode(metric)
    output_dir = cfg.ray.output_dir
    seed = cfg.data.seed
    training_mode = cfg.training.get("mode", "ft")
    peft_cfg = cfg.get("peft", None)
    perf_cfg = cfg.get("perf", None)
    # Bound to the search space's largest per-device batch, not to whatever this
    # trial samples, so every trial evaluates at the same speed.
    eval_batch_size = resolve_eval_batch_size(perf_cfg, cfg.training.get("search_space", None))
    
    # Build search space
    space = build_benign_exposure_space(
        cfg.training, active_models, seed, cfg.hpo.search_alg
    )
    
    # Fail before spending the search budget. The trainable catches every
    # exception and reports a failed trial to Ray, so a missing LoRA key would
    # otherwise burn all num_samples trials at val_f1=0 and then pick a "winner"
    # out of the failures.
    if training_mode == "peft":
        missing = [k for k in _LORA_KEYS if k not in space]
        if missing:
            raise ValueError(
                f"training.mode='peft' but training.search_space is missing "
                f"{missing}. LoRA hyperparameters have no defaults; add them to "
                f"the search space (see configs/training/"
                f"peft_benign_exposure_mmbert2_template.yaml)."
            )
        if not models_registry:
            raise ValueError(
                "training.mode='peft' requires a model registry "
                "(it supplies peft_target_modules)."
            )
    
    # Only optuna is currently implemented
    if cfg.hpo.search_alg != "optuna":
        raise NotImplementedError(
            f"search_alg='{cfg.hpo.search_alg}' is not implemented. "
            f"Only 'optuna' is currently supported. "
            f"TODO: Implement random search."
        )
    
    total_trials = len(active_models) * cfg.hpo.num_samples
    if cfg.training.get("segment_scoring", False) and len(active_models) != 1:
        raise ValueError("The 20-trial segment search expects exactly one ModernBERT model")
    num_samples = total_trials  # Optuna controls everything
    
    print(f"\n{'='*64}")
    print(f"  BENIGN EXPOSURE HPO")
    print(f"{'='*64}")
    print(f"  Models: {len(active_models)}")
    print(f"  Configs per model: {cfg.hpo.num_samples}")
    print(f"  Total trials: {total_trials}")
    print(f"  Metric: {metric} (mode={mode})")
    print(f"  Search algorithm: {cfg.hpo.search_alg}")
    print(f"  Training mode: {training_mode}")
    print(f"  Eval batch size: {eval_batch_size} (fixed across trials)")
    print(f"  Max micro-batch: {_perf_flag(perf_cfg, 'max_micro_batch_size', 32)}")
    print(f"{'='*64}\n")
    
    # Only pass data needed for HPO (train/val splits)
    # This reduces memory usage and network transfer
    hpo_split_data = {
        "M_core_train": split_data["M_core_train"],
        "M_extra_train": split_data["M_extra_train"],
        "M_core_val": split_data["M_core_val"],
        "M_extra_val": split_data["M_extra_val"],
        "B_core_train": split_data["B_core_train"],
        "B_extra_train": split_data["B_extra_train"],
        "B_core_val": split_data["B_core_val"],
        "B_extra_val": split_data["B_extra_val"],
    }

    # Get dilution config
    template_cfg = cfg.get("template", None)
    
    # Build trainable with parameters
    train_fn = tune.with_parameters(
        benign_exposure_trainable,
        split_data=hpo_split_data,
        max_seq_length=cfg.data.max_seq_length,
        benign_per_malicious_val=cfg.validation.benign_per_malicious,
        seed=seed,
        template_cfg=template_cfg,
        training_mode=training_mode,
        models_registry=models_registry,
        peft_cfg=peft_cfg,
        perf_cfg=perf_cfg,
        eval_batch_size=eval_batch_size,
        email_cfg=cfg.get("email", None),
        email_pools=({name: email_pools[name] for name in ("train", "valid")} if email_pools is not None else None),
    )
    
    train_fn = tune.with_resources(
        train_fn,
        {"cpu": cfg.hpo.cpus_per_trial,
         "gpu": cfg.hpo.gpus_per_trial,
         "accelerator_type:H200": cfg.hpo.num_nvidia_gpu_per_trial},
    )
    
    # Setup scheduler
    scheduler = ASHAScheduler(
        max_t=cfg.hpo.max_epochs,
        grace_period=int(cfg.hpo.get("grace_period", 1)),
        reduction_factor=2,
    )
    
    # Setup search algorithm
    search_alg_obj = None
    if cfg.hpo.search_alg == "optuna":
        from ray.tune.search.optuna import OptunaSearch
        points = None
        if cfg.training.get("segment_scoring", False):
            from segment_training import initial_segment_trials
            points = initial_segment_trials(cfg.training.search_space, active_models[0], seed)
            points = points[:int(cfg.hpo.num_samples)]
        search_alg_obj = OptunaSearch(metric=metric, mode=mode, points_to_evaluate=points, seed=seed)
    
    # Check for existing experiment to resume
    storage_path = str(Path(output_dir).resolve())
    experiment_name = "benign_exposure_hpo"
    experiment_path = Path(storage_path) / experiment_name
    if cfg.training.get("segment_scoring", False):
        search_record = Path(storage_path) / "segment_search_space.json"
        specification = OmegaConf.to_container(cfg.training.search_space, resolve=True)
        if experiment_path.exists() and not search_record.exists():
            raise ValueError("Cannot resume an unverified segment experiment; use a new ray.output_dir")
        if search_record.exists() and json.loads(search_record.read_text()) != specification:
            raise ValueError("Segment search space changed; use a new ray.output_dir")
        search_record.parent.mkdir(parents=True, exist_ok=True)
        search_record.write_text(json.dumps(specification, indent=2) + "\n")
    
    if experiment_path.exists():
        print(f"  📁 Found existing experiment at {experiment_path}")
        print(f"  🔄 Resuming from previous run...\n")
        tuner = tune.Tuner.restore(
            str(experiment_path),
            trainable=train_fn,
        )
    else:
        print(f"  🆕 Starting new experiment at {experiment_path}\n")
        tuner = tune.Tuner(
            train_fn,
            run_config=tune.RunConfig(
                name=experiment_name,
                storage_path=storage_path,
            ),
            tune_config=tune.TuneConfig(
                metric=metric,
                mode=mode,
                scheduler=scheduler,
                search_alg=search_alg_obj,
                num_samples=num_samples,
                max_concurrent_trials=cfg.hpo.max_concurrent_trials,
            ),
            param_space=space,
        )
    
    # Run HPO
    results = tuner.fit()
    
    # Get best result
    best_result = results.get_best_result(metric=metric, mode=mode)
    if cfg.training.get("segment_scoring", False) and best_result.metrics.get("error"):
        raise RuntimeError(f"Selected segment trial failed: {best_result.metrics['error']}")
    best_hp = best_result.config
    
    # Print results
    print(f"\n{'='*64}")
    print(f"  HPO RESULTS")
    print(f"{'='*64}")
    print(f"  Best model: {best_hp['model_name']}")
    print(f"  Best {metric}: {best_result.metrics.get(metric, 'N/A'):.4f}")
    print(f"  Best hyperparameters:")
    for k, v in best_hp.items():
        if k not in ['model_name', 'seed']:
            print(f"    {k}: {v}")
    print(f"{'='*64}\n")
    
    # Save checkpoint
    _save_checkpoint(output_dir, "hpo_complete", {
        "training_mode": training_mode,
        "scoring_mode": "token_max_subarray_v1" if cfg.training.get("segment_scoring", False) else "sentence",
        "best_hp": best_hp,
        "best_metrics": {k: v for k, v in best_result.metrics.items()
                         if isinstance(v, (int, float))},
    })
    
    winner = {
        "winner_model": best_hp["model_name"],
        "best_hp": best_hp,
        "best_metrics": best_result.metrics,
    }
    
    return winner, best_hp


def _save_checkpoint(output_dir: str, stage: str, data: dict) -> Path:
    """Save intermediate results to {output_dir}/checkpoints/{stage}.json."""
    cp_dir = Path(output_dir) / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    path = cp_dir / f"{stage}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"  💾 Checkpoint saved: {path}")
    return path


# =============================================================================
# §5  Calibration & Evaluation Functions
# =============================================================================

def calibrate_threshold(
    trainer,
    M_cal: Dataset,
    B_cal: Dataset,
    tokenizer,
    max_seq_length: int,
    seed: int,
    temp_ratio: int,
    threshold_ratio: int,
    beta: float = 0.5,
    target_precision: float = 0.95,
    min_recall: float = 0.9,
    email_collator=None,
) -> Tuple[float, PlattCalibrator, dict]:
    """
    Two-step calibration:
    1. Platt calibration on the temp_ratio:1 view, reweighted to threshold_ratio:1
    2. Threshold optimization on the threshold_ratio:1 view (maximize F-beta
       subject to target_precision and min_recall)

    The threshold is a calibrated probability, compared against
    ``scaler.scale(logits)``. That is only safe while the probabilities keep
    their resolution: a float32 rounds to exactly 1.0 past a calibrated margin
    of ~17.3, and every affected row then shares one score. The Platt intercept
    is what holds them clear of it; ``check_score_resolution`` verifies rather
    than assumes.

    Args:
        trainer: Trained HuggingFace Trainer
        M_cal: Combined malicious calibration samples
        B_cal: Combined benign calibration samples
        tokenizer: Tokenizer for tokenization
        max_seq_length: Max sequence length
        seed: Random seed
        temp_ratio: Benign:malicious ratio of the calibration view (default 10)
        threshold_ratio: Benign:malicious ratio for threshold optimization (default 500)
        beta: F-beta parameter (default 0.5)

    Returns:
        threshold: Optimal threshold
        scaler: Fitted Platt calibrator
        cal_metrics: Calibration metrics from threshold_ratio:1 view
    """
    print(f"\n{'='*64}")
    print(f"  CALIBRATION")
    print(f"{'='*64}")

    # Step 1: Build temp_ratio:1 view and fit the calibrator on it
    print(f"\n  Step 1: Calibration ({temp_ratio}:1 view, reweighted to "
          f"{threshold_ratio}:1)")
    temp_cal_ds = build_calibration_view(
        M_cal=M_cal,
        B_cal=B_cal,
        benign_per_malicious=temp_ratio,
        seed=seed,
        email_collator=email_collator,
    )
    temp_cal_ds = (
        email_collator.materialize(temp_cal_ds, f"calib_ratio_{temp_ratio}")
        if email_collator is not None else tokenize_dataset(temp_cal_ds, tokenizer, max_seq_length)
    )
    # Forward-only pass with order-independent metrics, so sorting is free accuracy-wise
    # and collapses the padding waste (see sort_dataset_by_length).
    temp_cal_ds = sort_dataset_by_length(temp_cal_ds)

    # Extract predictions and fit the calibrator
    temp_logits, temp_labels = _extract_predictions(trainer, temp_cal_ds, "temp_cal")
    # Two-parameter Platt rather than a single temperature. Moving from
    # temp_ratio:1 to threshold_ratio:1 is a pure prior shift, i.e. an
    # INTERCEPT, and a temperature can only rescale -- it cannot represent that
    # move, so it lands on a value that fits neither view and leaves the
    # probabilities saturated at 1.0 where the threshold needs to go.
    scaler = PlattCalibrator()
    a, b = scaler.fit(
        temp_logits,
        temp_labels,
        sample_weight=deployment_sample_weight(temp_labels, temp_ratio, threshold_ratio),
    )
    print(f"  Platt calibrator: a = {a:.4f}, b = {b:+.4f} "
          f"(equivalent temperature {scaler.temperature:.4f})")

    # Step 2: Build threshold_ratio:1 view for threshold optimization.
    #
    # With temp_ratio == threshold_ratio, build_calibration_view returns the very
    # same rows (same pool, same ratio, same seed), so reuse the logits instead
    # of paying for a second pass over the whole benign pool.
    print(f"\n  Step 2: Threshold optimization ({threshold_ratio}:1 view)")
    if temp_ratio == threshold_ratio:
        print(f"  Reusing the Step 1 predictions: identical view")
        thresh_logits, thresh_labels = temp_logits, temp_labels
    else:
        thresh_cal_ds = build_calibration_view(
            M_cal=M_cal,
            B_cal=B_cal,
            benign_per_malicious=threshold_ratio,
            seed=seed,
            email_collator=email_collator,
        )
        thresh_cal_ds = (
            email_collator.materialize(thresh_cal_ds, f"calib_ratio_{threshold_ratio}")
            if email_collator is not None else tokenize_dataset(thresh_cal_ds, tokenizer, max_seq_length)
        )
        thresh_cal_ds = sort_dataset_by_length(thresh_cal_ds)
        thresh_logits, thresh_labels = _extract_predictions(
            trainer, thresh_cal_ds, "thresh_cal")

    # Extract predictions and optimize threshold
    thresh_probs = np.float32(scaler.scale(thresh_logits))
    check_score_resolution(thresh_probs, name="threshold calibration probabilities")
    constraints = ThresholdConstraints(
        target_precision=target_precision,
        min_recall=min_recall,
        beta=beta,
    )
    threshold, cal_metrics = optimize_threshold(thresh_probs, thresh_labels, constraints)

    cal_f1 = fbeta(cal_metrics["precision"], cal_metrics["recall"], beta=1.0)
    cal_fbeta = fbeta(cal_metrics["precision"], cal_metrics["recall"], beta=beta)
    print(f"  Optimal threshold = {threshold:.4f}")
    print(f"  Calibration F1 = {cal_f1:.4f}, F{beta} = {cal_fbeta:.4f}")
    print(f"  Calibration precision = {cal_metrics['precision']:.4f}, recall = {cal_metrics['recall']:.4f}")

    print(f"{'='*64}\n")

    return threshold, scaler, cal_metrics


def evaluate_on_test_set(
    trainer,
    M_test: Dataset,
    B_test: Dataset,
    tokenizer,
    max_seq_length: int,
    threshold: float,
    scaler: PlattCalibrator,
    seed: int,
    test_ratio: int,
    email_collator=None,
) -> dict:
    """
    Evaluate model on test set with calibrated threshold.

    Args:
        trainer: Trained HuggingFace Trainer
        M_test: Combined malicious test samples
        B_test: Combined benign test samples
        tokenizer: Tokenizer for tokenization
        max_seq_length: Max sequence length
        threshold: Calibrated threshold
        scaler: Fitted Platt calibrator
        seed: Random seed
        test_ratio: Benign:malicious ratio for test set (default 500)

    Returns:
        Dictionary with precision, recall, f1, fpr, fnr, confusion matrix, etc.
    """
    print(f"\n{'='*64}")
    print(f"  TEST SET EVALUATION")
    print(f"{'='*64}")

    # Build test set
    test_ds = build_test_set(
        M_test=M_test,
        B_test=B_test,
        benign_per_malicious=test_ratio,
        seed=seed,
        email_collator=email_collator,
    )
    test_ds = (
        email_collator.materialize(test_ds, f"test_ratio_{test_ratio}")
        if email_collator is not None else tokenize_dataset(test_ds, tokenizer, max_seq_length)
    )
    test_ds = sort_dataset_by_length(test_ds)

    # Extract predictions
    test_logits, test_labels = _extract_predictions(trainer, test_ds, "test")

    # Apply temperature scaling
    test_probs = np.float32(np.asarray(scaler.scale(test_logits), dtype=float).ravel())
    check_score_resolution(test_probs, name="test probabilities")
    test_labels = np.asarray(test_labels, dtype=int).ravel()

    # Apply threshold
    test_preds = (test_probs >= threshold).astype(int)

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()

    result = {
        "threshold": round(float(threshold), 6),
        "calibrator": scaler.to_dict(),
        "temperature": round(float(scaler.temperature), 4),
        "precision": round(float(precision_score(test_labels, test_preds, average="binary", zero_division=0)), 4),
        "recall": round(float(recall_score(test_labels, test_preds, average="binary", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(test_labels, test_preds)), 4),
        "f1": round(float(f1_score(test_labels, test_preds, average="binary", zero_division=0)), 4),
        "fpr": round(_fpr(test_labels, test_preds), 4),
        "fnr": round(_fnr(test_labels, test_preds), 4),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }

    print(f"  Test Results:")
    print(f"    Precision: {result['precision']:.4f}")
    print(f"    Recall:    {result['recall']:.4f}")
    print(f"    F1:        {result['f1']:.4f}")
    print(f"    FPR:       {result['fpr']:.4f}")
    print(f"    FNR:       {result['fnr']:.4f}")
    print(f"    Confusion Matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f"{'='*64}\n")

    return result


def save_calibration_artifacts(
    model_save_dir: Path,
    scaler: PlattCalibrator,
    threshold: float,
    cal_metrics: dict,
    test_metrics: dict,
) -> None:
    """
    Save the calibrator and threshold alongside the model.

    ``threshold`` is a calibrated probability, compared against
    ``sigmoid(a * (logit_1 - logit_0) + b)``. ``temperature`` is kept for
    readers that predate the ``calibrator`` block; it is the equivalent ``1/a``
    and drops the intercept, so it reproduces the ranking but not the
    probabilities.

    Saves:
    - {model_dir}/calibration.json (calibrator, threshold)
    - {model_dir}/calibration_metrics.json
    - {model_dir}/test_metrics.json
    """
    # Save calibration parameters
    calibration = {
        "calibrator": scaler.to_dict(),
        "threshold": round(float(threshold), 6),
        # Legacy field: equivalent temperature, without the intercept.
        "temperature": round(float(scaler.temperature), 4),
    }
    from segment_scoring import is_segment_checkpoint, SCORING_MODE
    if is_segment_checkpoint(model_save_dir):
        config = json.loads((model_save_dir / "config.json").read_text())
        calibration.update(scoring_mode=SCORING_MODE, segment_tau=config["segment_tau"])
    if np.float32(threshold) >= 1.0:
        print(f"  ⚠ The chosen threshold is 1.0, i.e. it sits inside a saturated "
              f"tie. Refit the calibrator at the deployment prevalence before "
              f"shipping this.")
    calibration_path = model_save_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2))
    print(f"  💾 Calibration saved: {calibration_path}")

    # Save calibration metrics
    cal_metrics_path = model_save_dir / "calibration_metrics.json"
    cal_metrics_path.write_text(json.dumps(cal_metrics, indent=2, default=str))
    print(f"  💾 Calibration metrics saved: {cal_metrics_path}")

    # Save test metrics
    test_metrics_path = model_save_dir / "test_metrics.json"
    test_metrics_path.write_text(json.dumps(test_metrics, indent=2, default=str))
    print(f"  💾 Test metrics saved: {test_metrics_path}")


# =============================================================================
# §6  Calibration & Evaluation Pipeline
# =============================================================================

def calibrate_and_evaluate(
    model_path: str,
    split_data: Dict[str, Dataset],
    max_seq_length: int,
    seed: int,
    calibration_cfg: DictConfig,
    test_cfg: DictConfig,
    training_mode: str = "ft",
    perf_cfg: Optional[DictConfig] = None,
    eval_batch_size: Optional[int] = None,
    email_cfg: Optional[DictConfig] = None,
    email_pools=None,
    template_cfg: Optional[DictConfig] = None,
) -> Tuple[float, PlattCalibrator, dict, dict]:
    """
    Load saved model, calibrate threshold, and evaluate on test set.
    
    Args:
        model_path: Path to saved model
        split_data: Pre-split data dict (16 files)
        max_seq_length: Max sequence length
        seed: Random seed
        calibration_cfg: Calibration config (beta, temp_ratio, threshold_ratio)
        test_cfg: Test config (benign_per_malicious)
        training_mode: "ft" or "peft"; in peft mode the LoRA adapter saved next
            to the base weights is attached before scoring
        perf_cfg: Optional `perf` config block (throughput knobs; see §0a)
        eval_batch_size: Batch size for the ~1.5M calibration + test rows
    
    Returns:
        Tuple of (threshold, scaler, cal_metrics, test_metrics)
    """
    print(f"\n{'='*64}")
    print(f"  LOADING MODEL FOR CALIBRATION & EVALUATION")
    print(f"{'='*64}")
    print(f"  Model path: {model_path}")
    
    # Clear GPU memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Load tokenizer and model from disk. Calibration runs in fp32: the
    # training-time fake-quant is not reapplied here.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_trained_model(model_path, training_mode)
    
    # Create trainer for inference. This sweeps ~747k calibration + ~747k test
    # rows, so the batch size matters as much here as it does inside a trial.
    cal_eval_bs = int(eval_batch_size or 64)
    print(f"  Eval batch size: {cal_eval_bs}")
    train_args = TrainingArguments(
        output_dir=model_path,
        per_device_eval_batch_size=cal_eval_bs,
        report_to="none",
        fp16=False,
        bf16=_perf_flag(perf_cfg, "bf16", False) and torch.cuda.is_available(),
        tf32=_perf_flag(perf_cfg, "tf32", False) or None,
    )
    
    trainer = Trainer(
        model=model,
        args=train_args,
        processing_class=tokenizer,
        data_collator=_evaluation_collator(tokenizer, email_cfg),
    )
    
    print(f"  Model loaded successfully")
    print(f"{'='*64}\n")
    
    # Combine M_core + M_extra for calibration
    M_cal = combine_malicious_pools(
        split_data["M_core_cal"],
        split_data["M_extra_cal"]
    )
    
    # Combine B_core + B_extra for calibration
    B_cal = combine_benign_pools(
        split_data["B_core_cal"],
        split_data["B_extra_cal"]
    )
    
    # Combine M_core + M_extra for test
    M_test = combine_malicious_pools(
        split_data["M_core_test"],
        split_data["M_extra_test"]
    )
    
    # Combine B_core + B_extra for test
    B_test = combine_benign_pools(
        split_data["B_core_test"],
        split_data["B_extra_test"]
    )
    
    # Calibrate threshold
    threshold, scaler, cal_metrics = calibrate_threshold(
        trainer=trainer,
        M_cal=M_cal,
        B_cal=B_cal,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        seed=seed,
        temp_ratio=calibration_cfg.temp_ratio,
        threshold_ratio=calibration_cfg.threshold_ratio,
        beta=calibration_cfg.beta,
        target_precision=calibration_cfg.get("target_precision", 0.95),
        min_recall=calibration_cfg.get("min_recall", 0.9),
        email_collator=make_email_collator(
            tokenizer, email_pools, email_cfg, template_cfg, "calib", seed, "fixed"
        ),
    )
    
    # Evaluate on test set
    test_metrics = evaluate_on_test_set(
        trainer=trainer,
        M_test=M_test,
        B_test=B_test,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        threshold=threshold,
        scaler=scaler,
        seed=seed,
        test_ratio=test_cfg.benign_per_malicious,
        email_collator=make_email_collator(
            tokenizer, email_pools, email_cfg, template_cfg, "test", seed, "fixed"
        ),
    )
    
    # Clean up GPU memory
    del model, trainer, tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return threshold, scaler, cal_metrics, test_metrics


# =============================================================================
# §7  Final Model Training
# =============================================================================

def train_final_model(
    best_hp: Dict,
    split_data: Dict[str, Dataset],
    max_seq_length: int,
    benign_per_malicious_val: int,
    seed: int,
    output_dir: str,
    template_cfg: Optional[DictConfig] = None,
    training_mode: str = "ft",
    models_registry: Optional[Dict[str, ModelSpec]] = None,
    peft_cfg: Optional[DictConfig] = None,
    perf_cfg: Optional[DictConfig] = None,
    eval_batch_size: Optional[int] = None,
    email_cfg: Optional[DictConfig] = None,
    email_pools=None,
) -> str:
    """
    Train final model with best hyperparameters.
    
    Args:
        best_hp: Best hyperparameters from HPO
        split_data: Pre-split data dict (16 files)
        max_seq_length: Max sequence length for tokenization
        benign_per_malicious_val: Fixed ratio for validation set
        seed: Random seed
        output_dir: Output directory for model checkpoint
        template_cfg: Optional config for template dilution collator
        training_mode: "ft" for full fine-tuning, "peft" for LoRA
        models_registry: Model specs; required in peft mode
        peft_cfg: Optional `peft` config block (qat / save_merged /
            extra_modules_to_save)
        perf_cfg: Optional `perf` config block (throughput knobs; see §0a)
        eval_batch_size: Fixed eval batch size, independent of the train batch
    
    Returns:
        Path to saved model
    """
    model_id = best_hp["model_name"]
    is_peft = training_mode == "peft"
    best_hp = _resolve_segment_hp(best_hp, perf_cfg)
    if _segment_enabled(best_hp):
        from transformers import set_seed
        set_seed(seed)
    
    print(f"\n{'='*64}")
    print(f"  TRAINING FINAL MODEL")
    print(f"{'='*64}")
    print(f"  Model: {model_id}")
    print(f"  Training mode: {training_mode}")
    print(f"{'='*64}\n")
    
    # Load tokenizer + model
    tokenizer, model = load_model(
        model_id, float(best_hp["classifier_dropout"]),
        attn_implementation=_perf_flag(perf_cfg, "attn_implementation", "eager"),
    )
    
    # QAT first, then LoRA (see _prepare_model)
    if _segment_enabled(best_hp):
        from segment_scoring import make_segment_model
        model = make_segment_model(model, tokenizer, best_hp)
    model = _prepare_model(
        model, model_id, best_hp, training_mode, models_registry, peft_cfg
    )
    email_train_collator = make_email_collator(
        tokenizer, email_pools, email_cfg, template_cfg, "train", seed, "random"
    )
    email_eval_collator = make_email_collator(
        tokenizer, email_pools, email_cfg, template_cfg, "valid", seed, "fixed"
    )
    
    # Combine M_core + M_extra for training
    M_train = combine_malicious_pools(
        split_data["M_core_train"],
        split_data["M_extra_train"]
    )
    
    # Combine B_core + B_extra for training
    B_train = combine_benign_pools(
        split_data["B_core_train"],
        split_data["B_extra_train"]
    )
    
    # Sample training view
    train_ds = sample_training_view(
        M_train=M_train,
        B_train=B_train,
        benign_to_malicious_ratio=int(best_hp["benign_to_malicious_ratio"]),
        seed=seed,
        email_collator=email_train_collator,
    )
    
    # Combine M_core + M_extra for validation
    M_val = combine_malicious_pools(
        split_data["M_core_val"],
        split_data["M_extra_val"]
    )
    
    # Combine B_core + B_extra for validation
    B_val = combine_benign_pools(
        split_data["B_core_val"],
        split_data["B_extra_val"]
    )
    
    # Build validation set
    val_ds = build_validation_set(
        M_val=M_val,
        B_val=B_val,
        benign_per_malicious=benign_per_malicious_val,
        email_collator=email_eval_collator,
    )
    
    # Note: Do NOT tokenize here - TemplateDilutionCollator handles tokenization
    # The dataset keeps 'text' and 'label' columns for the collator to use
    
    # Debug: Print dataset columns to verify structure
    print(f"  train_ds columns: {train_ds.column_names}")
    print(f"  val_ds columns: {val_ds.column_names}")
    
    # Ensure 'text' and 'label' columns exist (rename 'labels' back to 'label' if needed)
    if 'labels' in train_ds.column_names and 'label' not in train_ds.column_names:
        train_ds = train_ds.rename_column('labels', 'label')
        print(f"  Renamed 'labels' -> 'label' in train_ds")
    if 'labels' in val_ds.column_names and 'label' not in val_ds.column_names:
        val_ds = val_ds.rename_column('labels', 'label')
        print(f"  Renamed 'labels' -> 'label' in val_ds")
    
    # Same repack as the HPO path, so the final model trains under the identical
    # numerics the search selected on.
    micro_bs, accum = _training_batch(best_hp, perf_cfg)
    if (micro_bs, accum) != (int(best_hp["per_device_train_batch_size"]),
                             int(best_hp["gradient_accumulation_steps"])):
        print(f"  micro-batch {best_hp['per_device_train_batch_size']}x"
              f"{best_hp['gradient_accumulation_steps']} -> {micro_bs}x{accum} "
              f"(effective batch unchanged: {micro_bs * accum})")
    eval_bs = int(eval_batch_size or micro_bs * 2)
    
    if _perf_flag(perf_cfg, "sort_eval_by_length", True):
        val_ds = sort_dataset_by_length(val_ds)
    
    # Build train and eval collators (use TemplateDilutionCollator if enabled)
    if email_train_collator is not None:
        train_collator = email_train_collator
        val_ds = email_eval_collator.materialize(val_ds, "valid")
        if _perf_flag(perf_cfg, "sort_eval_by_length", True):
            val_ds = sort_dataset_by_length(val_ds)
        eval_collator = _evaluation_collator(tokenizer, email_cfg)
    elif template_cfg is not None and template_cfg.enabled:
        print(f"  Using TemplateDilutionCollator (probability={template_cfg.probability})")
        # Load the instruction templates the payload is wrapped into.
        templates = load_templates(
            template_cfg.template_path,
            template_key=template_cfg.template_key,
            placeholder=template_cfg.placeholder,
        )
        print(f"  Loaded {len(templates)} templates from {template_cfg.template_path}")
        # Train collator: random mode (wrap a random ~probability fraction each call)
        train_collator = TemplateDilutionCollator(
            tokenizer=tokenizer,
            templates=templates,
            text_key=template_cfg.text_key,
            label_key=template_cfg.label_key,
            id_key=template_cfg.id_key,
            placeholder=template_cfg.placeholder,
            dilution_probability=float(template_cfg.probability),
            mode="random",
            seed=seed,
            max_length=int(template_cfg.max_length),
        )
        # Eval collator: fixed mode (deterministic wrap decision per sample ID)
        eval_collator = TemplateDilutionCollator(
            tokenizer=tokenizer,
            templates=templates,
            text_key=template_cfg.text_key,
            label_key=template_cfg.label_key,
            id_key=template_cfg.id_key,
            placeholder=template_cfg.placeholder,
            dilution_probability=float(template_cfg.probability),
            mode="fixed",
            seed=seed,
            max_length=int(template_cfg.max_length),
        )
    else:
        train_collator = DataCollatorWithPadding(tokenizer)
        eval_collator = DataCollatorWithPadding(tokenizer)
    
    # Train
    model_save_dir = Path(output_dir) / "best_model" / model_id.replace("/", "_")
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        train_args = TrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=int(best_hp["num_train_epochs"]),
            per_device_train_batch_size=micro_bs,
            per_device_eval_batch_size=eval_bs,
            gradient_accumulation_steps=accum,
            learning_rate=float(best_hp["learning_rate"]),
            weight_decay=float(best_hp["weight_decay"]),
            warmup_ratio=float(best_hp["warmup_ratio"]),
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            eval_strategy="no",
            save_strategy="no",
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to="none",
            fp16=False,
            bf16=_perf_flag(perf_cfg, "bf16", False) and torch.cuda.is_available(),
            tf32=_perf_flag(perf_cfg, "tf32", False) or None,
            seed=seed,
            remove_unused_columns=False,  # Keep 'text' column for TemplateDilutionCollator
        )
        
        trainer_class, metrics_function = _training_components(best_hp)
        trainer = trainer_class(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            train_data_collator=train_collator,
            eval_data_collator=eval_collator,
            compute_metrics=metrics_function,
            # ASL Loss hyperparameters
            gamma_pos=float(best_hp.get("gamma_pos", 0.0)),
            gamma_neg=float(best_hp.get("gamma_neg", 4.0)),
            asl_clip=float(best_hp.get("asl_clip", 0.05)),
            check_finite_every=int(_perf_flag(perf_cfg, "check_finite_every", 50)),
        )
        
        trainer.train()
        
        # Save model.
        #
        # In peft mode the adapter is NEVER merged into the backbone: merging
        # would fold the fp32 LoRA delta into weights that are subsequently
        # quantized. So the root directory gets the base only (frozen backbone +
        # trained classification head, via unload()) and the adapter is written
        # beside it. Order matters: unload()/merge_and_unload() mutate the
        # PeftModel in place, hence save the adapter first and deepcopy after.
        if is_peft:
            peft_model = trainer.accelerator.unwrap_model(trainer.model)
            peft_model.save_pretrained(str(model_save_dir / "adapter"))
            copy.deepcopy(peft_model).unload().save_pretrained(
                str(model_save_dir), safe_serialization=True)
            print(f"  Saved base weights (no LoRA delta) + adapter/")
            if bool((peft_cfg or {}).get("save_merged", False)):
                # Debug/reference copy only. This is NOT the artifact the
                # calibration below is computed on.
                copy.deepcopy(peft_model).merge_and_unload().save_pretrained(
                    str(model_save_dir / "merged_fp32"), safe_serialization=True)
                print(f"  Saved debug copy to merged_fp32/")
        else:
            trainer.save_model(str(model_save_dir))
        tokenizer.save_pretrained(str(model_save_dir))
        
        # Save training arguments manually (save_to_json was removed in newer transformers)
        with open(model_save_dir / "training_args.json", "w") as f:
            f.write(train_args.to_json_string())
        
        print(f"\n  ✅ Model saved to: {model_save_dir}")
    
    # Save training summary
    summary = {
        "model_id": model_id,
        "model_path": str(model_save_dir),
        "training_mode": training_mode,
        "qat": _qat_enabled(peft_cfg),
        # Explicit marker for downstream loaders: the root directory holds the
        # base weights only, so scoring without adapter/ is silently wrong.
        "requires_adapter": is_peft,
        "scoring_mode": "token_max_subarray_v1" if _segment_enabled(best_hp) else "sentence",
        "lora": ({
            **{k: best_hp[k] for k in _LORA_KEYS},
            "target_modules": list(models_registry[model_id].peft_target_modules),
            "extra_modules_to_save": list(
                (peft_cfg or {}).get("extra_modules_to_save", []) or []),
        } if is_peft else None),
        "artifacts": ({
            "base": str(model_save_dir),
            "adapter": str(model_save_dir / "adapter"),
        } if is_peft else {"model": str(model_save_dir)}),
        "calibrated_on": ("fp32 base + fp32 adapter" if is_peft else "fp32"),
        "hyperparameters": {k: v for k, v in best_hp.items() if k != "model_name"},
        "seed": seed,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    
    summary_path = model_save_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    
    return str(model_save_dir)


# =============================================================================
# §8  Main
# =============================================================================

def load_best_hyperparameters(
    checkpoint_path: str,
    expected_mode: str = "ft",
    expected_segment: bool = False,
) -> Tuple[Dict, Dict]:
    """Load best_hp from a completed HPO checkpoint for final-only training.

    `expected_mode` is enforced, not merely reported. A PEFT run may only reuse a
    checkpoint that a PEFT run produced: LoRA hyperparameters have no defaults,
    and an FT-tuned learning rate (~1e-5) applied to the ~0.5% of parameters LoRA
    trains would produce a model that trains, saves and calibrates cleanly while
    being quietly useless. Legacy checkpoints carry no `training_mode` field and
    are treated as "ft".
    """
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path

    if not path.exists():
        raise FileNotFoundError(
            f"HPO checkpoint not found: {path}"
        )

    checkpoint = json.loads(path.read_text())
    if (checkpoint.get("scoring_mode") == "token_max_subarray_v1") != expected_segment:
        raise ValueError("HPO checkpoint scoring mode does not match this training run; run a new HPO")
    best_hp = checkpoint.get("best_hp")
    if not isinstance(best_hp, dict):
        raise ValueError(
            f"HPO checkpoint does not contain a best_hp object: {path}"
        )

    checkpoint_mode = checkpoint.get("training_mode", "ft")
    if checkpoint_mode != expected_mode:
        raise ValueError(
            f"training.mode='{expected_mode}' but the HPO checkpoint was "
            f"produced by training_mode='{checkpoint_mode}': {path}\n"
            f"  Hyperparameters are not transferable between the two modes. "
            f"Either run hpo.skip=false to tune '{expected_mode}' from scratch, "
            f"or point hpo.checkpoint_path at a completed '{expected_mode}' run."
        )

    required_keys = {
        "model_name",
        "classifier_dropout",
        "learning_rate",
        "weight_decay",
        "per_device_train_batch_size",
        "num_train_epochs",
        "gradient_accumulation_steps",
        "warmup_ratio",
        "benign_to_malicious_ratio",
    }
    if expected_mode == "peft":
        required_keys |= set(_LORA_KEYS)
    if expected_segment:
        from segment_scoring import SEGMENT_HP
        required_keys -= {"classifier_dropout", "per_device_train_batch_size", "gradient_accumulation_steps"}
        required_keys |= set(SEGMENT_HP) | {"effective_batch_size"}
    missing_keys = sorted(required_keys - best_hp.keys())
    if missing_keys:
        raise ValueError(
            f"HPO checkpoint is missing required hyperparameters "
            f"{missing_keys}: {path}\n"
            f"  These have no defaults. Run hpo.skip=false to tune them."
        )

    winner = {
        "winner_model": best_hp["model_name"],
        "best_hp": best_hp,
        "best_metrics": checkpoint.get("best_metrics", {}),
    }
    print(f"  Loaded tuned hyperparameters from {path}")
    print(f"  Training mode: {checkpoint_mode}")
    print(f"  Model: {best_hp['model_name']}")
    for k, v in sorted(best_hp.items()):
        if k not in ("model_name", "seed"):
            print(f"    {k}: {v}")
    return winner, best_hp


def main(config_override: Optional[DictConfig] = None) -> None:
    """Main entry point for benign exposure training (16-file split structure)."""
    
    # Get path to configs directory
    config_dir = Path(__file__).parent / "configs"
    
    # Training config: the FT email config by default, or a YAML passed as the first
    # positional argument (e.g. the PEFT template). Tested by suffix rather than
    # by a prefix allowlist so that new top-level keys such as `peft.qat=true`
    # are never mistaken for a filename.
    argv = list(sys.argv[1:]) if config_override is None else []
    config_file = "ft_benign_exposure_mmbert2_email.yaml"
    if argv and argv[0].endswith((".yaml", ".yml")):
        config_file = Path(argv.pop(0)).name
    
    cfg = (OmegaConf.load(config_dir / "training" / config_file) if config_override is None
           else OmegaConf.create(OmegaConf.to_container(config_override, resolve=True)))
    if config_override is not None:
        config_file = "interim HPO snapshot"
    # Email experiment configs inherit the existing FT/PEFT settings explicitly.
    if "base_config" in cfg:
        base = OmegaConf.load(config_dir / "training" / str(cfg.pop("base_config")))
        segment_space = cfg.training.search_space if cfg.get("training", {}).get("segment_scoring", False) else None
        cfg = OmegaConf.merge(base, cfg)
        if segment_space is not None:
            cfg.training.search_space = segment_space
    models_cfg = OmegaConf.load(config_dir / "models" / "mmbert.yaml")
    
    # Allow CLI overrides (e.g., hpo.num_samples=3)
    if argv:
        overrides = OmegaConf.from_dotlist(argv)
        cfg = OmegaConf.merge(cfg, overrides)
    
    # Resolve variable interpolations (e.g., ${data.seed} in output_dir)
    OmegaConf.resolve(cfg)
    
    training_mode = cfg.training.get("mode", "ft")
    if cfg.training.get("segment_scoring", False):
        if not cfg.email.enabled or not cfg.email.get("region_supervision", False):
            raise ValueError("Segment training requires email augmentation and region masks")
        if "token_evidence_head" not in cfg.peft.extra_modules_to_save:
            raise ValueError("Segment PEFT must train and save token_evidence_head")
        from segment_scoring import SEGMENT_HP
        required = {*SEGMENT_HP, "effective_batch_size"}
        missing = required - set(cfg.training.search_space)
        if missing:
            raise ValueError(f"Missing segment search parameters: {sorted(missing)}")
    if training_mode not in _TRAINING_MODES:
        raise ValueError(
            f"training.mode='{training_mode}' is not supported; "
            f"expected one of {list(_TRAINING_MODES)}."
        )
    peft_cfg = cfg.get("peft", None)
    perf_cfg = cfg.get("perf", None)
    eval_batch_size = resolve_eval_batch_size(perf_cfg, cfg.training.get("search_space", None))
    if _perf_flag(perf_cfg, "tf32", False):
        # Nothing else in this repo enables TF32 (data.py sets it but is never
        # imported), so fp32 matmuls have been taking the slow IEEE path.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print(f"\n{'='*64}")
    print(f"  BENIGN EXPOSURE TRAINING (16-file split)")
    print(f"{'='*64}")
    print(f"  Config file: {config_file}")
    print(f"  Training mode: {training_mode}")
    print(f"  QAT: {'on' if _qat_enabled(peft_cfg) else 'off'}")
    print(f"  Attention: {_perf_flag(perf_cfg, 'attn_implementation', 'eager')}"
          f"  |  tf32: {_perf_flag(perf_cfg, 'tf32', False)}"
          f"  |  bf16: {_perf_flag(perf_cfg, 'bf16', False)}")
    print(f"  Max micro-batch: {_perf_flag(perf_cfg, 'max_micro_batch_size', 32)}"
          f"  |  eval batch: {eval_batch_size}")
    print(f"  Config: {OmegaConf.to_yaml(cfg)}")
    print(f"{'='*64}\n")
    
    # Get seed from config (used for data loading, shuffling, and TrainingArguments)
    seed = cfg.data.seed
    
    # Load pre-split data (16 files)
    split_data = load_split_data(cfg.data.split_dir, seed)
    email_cfg = cfg.get("email", None)
    email_pools = None
    if email_cfg is not None and email_cfg.get("enabled", False):
        if int(cfg.template.max_length) != int(cfg.data.max_seq_length):
            raise ValueError("Email augmentation requires template.max_length == data.max_seq_length")
        email_pools, email_manifest = load_email_pools(email_cfg, seed)
        output_path = Path(cfg.ray.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        run_manifest = {
            "email_pools": email_manifest,
            "email_config": OmegaConf.to_container(email_cfg, resolve=True),
            "template_config": OmegaConf.to_container(cfg.template, resolve=True),
            "templates": load_templates(cfg.template.template_path, cfg.template.template_key, cfg.template.placeholder),
        }
        manifest_path = output_path / "email_augmentation_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != run_manifest:
            raise ValueError("Email augmentation settings changed; use a new ray.output_dir")
        if not manifest_path.exists() and (output_path / "benign_exposure_hpo").exists():
            raise ValueError("Cannot resume a legacy HPO directory with email augmentation; use a new ray.output_dir")
        manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
        OmegaConf.save(cfg, output_path / "email_training_config.yaml")

    # The registry is built unconditionally: PEFT needs spec.peft_target_modules
    # on the hpo.skip path too. preflight_check stays in the HPO branch so the
    # skip path remains network-free.
    MODELS = build_model_registry_from_yaml(models_cfg)
    print(f"  Loaded {len(MODELS)} models from config")

    # Either load previously-tuned hyperparameters (final-only training) or run HPO.
    if bool(cfg.hpo.get("skip", False)):
        winner, best_hp = load_best_hyperparameters(
            cfg.hpo.checkpoint_path, expected_mode=training_mode,
            expected_segment=bool(cfg.training.get("segment_scoring", False)),
        )
    else:
        active = preflight_check(list(MODELS.keys()))
        if not active:
            raise RuntimeError("No models passed preflight check.")

        winner, best_hp = run_benign_exposure_hpo(
            cfg, active, split_data, models_registry=MODELS, email_pools=email_pools
        )

    if training_mode == "peft" and best_hp["model_name"] not in MODELS:
        raise ValueError(
            f"PEFT mode requires '{best_hp['model_name']}' in the model registry "
            f"(it supplies peft_target_modules). "
            f"Available: {sorted(MODELS)}"
        )
    
    # Get dilution config
    template_cfg = cfg.get("template", None)
    
    # Train final model
    model_path = train_final_model(
        best_hp=best_hp,
        split_data=split_data,
        max_seq_length=cfg.data.max_seq_length,
        benign_per_malicious_val=cfg.validation.benign_per_malicious,
        seed=seed,
        output_dir=cfg.ray.output_dir,
        template_cfg=template_cfg,
        email_cfg=email_cfg,
        email_pools=email_pools,
        training_mode=training_mode,
        models_registry=MODELS,
        peft_cfg=peft_cfg,
        perf_cfg=perf_cfg,
        eval_batch_size=eval_batch_size,
    )
    
    # Calibrate and evaluate
    threshold, scaler, cal_metrics, test_metrics = calibrate_and_evaluate(
        model_path=model_path,
        split_data=split_data,
        max_seq_length=cfg.data.max_seq_length,
        seed=seed,
        calibration_cfg=cfg.calibration,
        test_cfg=cfg.test,
        email_cfg=email_cfg,
        email_pools=email_pools,
        template_cfg=template_cfg,
        training_mode=training_mode,
        perf_cfg=perf_cfg,
        eval_batch_size=eval_batch_size,
    )
    
    # Save calibration artifacts
    model_save_dir = Path(model_path)
    save_calibration_artifacts(
        model_save_dir=model_save_dir,
        scaler=scaler,
        threshold=threshold,
        cal_metrics=cal_metrics,
        test_metrics=test_metrics,
    )
    
    # Save final summary
    final_summary = {
        "winner_model": winner["winner_model"],
        "model_path": model_path,
        "calibration": {
            "temperature": scaler.temperature,
            "threshold": threshold,
        },
        "test_metrics": test_metrics,
    }
    summary_path = model_save_dir / "final_summary.json"
    summary_path.write_text(json.dumps(final_summary, indent=2, default=str))
    
    print(f"\n{'='*64}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*64}")
    print(f"  Best model: {winner['winner_model']}")
    print(f"  Model saved to: {model_path}")
    print(f"  Temperature: {scaler.temperature:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall: {test_metrics['recall']:.4f}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
