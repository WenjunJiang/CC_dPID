"""
plot_pr_curve.py

Plot PR curve for the trained model with labeled special points.

This script:
1. Loads the trained model from benign_exposure_results_seed42
2. Runs inference on test data and saves results as HuggingFace Dataset
3. Computes and plots PR curve with special points:
   - T_p{X}: Minimal threshold for precision >= X (for each target precision)
   - T_r{X}: Maximum threshold for recall >= X (for each target recall)

Usage:
    python plot_pr_curve_mmbert2.py --training-mode ft
    python plot_pr_curve_mmbert2.py --training-mode peft
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from datasets import load_from_disk, Dataset, concatenate_datasets
from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

import matplotlib.pyplot as plt

# Import reusable functions from existing modules
from train_hard_benign_mining import (
    compute_pr_curve,
    find_threshold_for_precision,
    find_threshold_for_recall,
)
from train_benign_exposure_mmbert2 import (
    build_test_set,
    tokenize_dataset,
    combine_benign_pools,
    combine_malicious_pools,
)
from raytune_benchmark import (
    _extract_predictions, _fpr, check_score_resolution, PlattCalibrator,
)
from sklearn.metrics import confusion_matrix


# =============================================================================
# Configuration
# =============================================================================

SEED = int(os.environ.get("PR_SEED", 42))

# Overridable via environment so a re-split or a retrain into a different output
# dir can be evaluated without editing this file. An environment override beats
# apply_training_mode(), which otherwise derives OUTPUT_DIR and PR_CURVE_PATH
# from the chosen mode.
FT_MODEL_PATH = os.environ.get(
    "PR_MODEL_PATH", f"benign_exposure_results_mmbert2_seed{SEED}/best_model/jhu-clsp_mmBERT-small"
)
PEFT_MODEL_PATH = os.environ.get(
    "PR_PEFT_MODEL_PATH",
    f"benign_exposure_peft_mmbert2_template_seed{SEED}/best_model/jhu-clsp_mmBERT-small"
)
# The live value; apply_training_mode() picks one of the two above.
MODEL_PATH = FT_MODEL_PATH
DATA_SPLIT_DIR = os.environ.get("PR_DATA_SPLIT_DIR", f"data_split3/{SEED}")
OUTPUT_DIR = os.environ.get("PR_OUTPUT_DIR", f"inference_results_seed{SEED}")
PR_CURVE_PATH = os.environ.get("PR_CURVE_PATH", f"pr_curve_seed{SEED}.png")

# "ft" | "peft", set by apply_training_mode() from the required --training-mode.
# A PEFT run saves the base weights in the model directory and the LoRA delta in
# adapter/ beside it, so scoring such a directory as "ft" silently drops the
# delta instead of failing.
TRAINING_MODE = "ft"

MAX_SEQ_LENGTH = 512
TEST_RATIO = 1000  # 1000:1 benign:malicious ratio

# Target precision and recall values for special points on PR curve
# Can be empty lists if no special points are needed
TARGET_PRECISIONS = [0.85, 0.90, 0.95]  # Minimal threshold for precision >= target
TARGET_RECALLS = [0.85, 0.90, 0.95]     # Maximum threshold for recall >= target


# =============================================================================
# Training mode
#
# Which artifact is being scored is stated on the command line and has no
# default. It has to be, because the two model directories are structurally
# identical apart from adapter/: a PEFT run writes the base weights (frozen
# backbone + trained classification head) to the root and the LoRA delta to
# adapter/, so loading a PEFT directory as "ft" succeeds and scores without the
# delta -- wrong numbers, no exception. A default would make that the outcome of
# forgetting a flag.
# =============================================================================

def parse_training_mode(doc: Optional[str] = None) -> str:
    """Parse the required --training-mode. Shared by the *_dilute entry point."""
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument(
        "--training-mode", choices=("ft", "peft"), required=True,
        help="Which artifact to score. Required and without a default: a PEFT "
             "model directory holds the base weights only and loads as 'ft' "
             "without erroring, so the mode cannot be left implicit. "
             f"ft -> {FT_MODEL_PATH}; peft -> {PEFT_MODEL_PATH}",
    )
    return parser.parse_args().training_mode


def apply_training_mode(mode: str, suffix: str = "") -> None:
    """Point the module at the chosen artifact and at its own result paths.

    The two models have different calibrators and different scores, and main()
    reuses OUTPUT_DIR instead of re-running inference whenever that directory
    already exists, so sharing one output directory would silently replot the
    other model. ``suffix`` is the entry point's own tag ("_dilute" for the
    diluted mirror); the mode appends "_peft" to it.

    An explicit environment override always wins, so the documented escape hatch
    for evaluating a re-split or a differently-named retrain still works.
    """
    global TRAINING_MODE, MODEL_PATH, OUTPUT_DIR, PR_CURVE_PATH

    TRAINING_MODE = mode
    if "PR_MODEL_PATH" not in os.environ:
        MODEL_PATH = PEFT_MODEL_PATH if mode == "peft" else FT_MODEL_PATH

    tag = f"{suffix}_peft" if mode == "peft" else suffix
    if "PR_OUTPUT_DIR" not in os.environ:
        OUTPUT_DIR = f"inference_results_seed{SEED}{tag}"
    if "PR_CURVE_PATH" not in os.environ:
        PR_CURVE_PATH = f"pr_curve_seed{SEED}{tag}.png"


# =============================================================================
# Provenance of saved inference results
#
# main() reuses OUTPUT_DIR instead of re-running inference whenever it exists.
# The saved dataset records scores but not which model produced them, so
# pointing PR_MODEL_PATH at a new checkpoint while PR_OUTPUT_DIR still names an
# old run replots the old model's scores under the new model's name. The
# manifest below makes the reuse conditional on the same checkpoint files,
# calibration, data and test construction.
# =============================================================================

MANIFEST_NAME = "inference_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inference_identity(model_path: str, training_mode: str, split_dir: str) -> dict:
    """Everything that determines the saved scores, with checkpoint file hashes."""
    root = Path(model_path).resolve()
    files = [root / "config.json", root / "calibration.json", *root.glob("*.safetensors"),
             *root.glob("*.bin"), *(root / "adapter").glob("*")]
    return {
        "model_path": str(root),
        "training_mode": training_mode,
        "data_split_dir": str(Path(split_dir).resolve()),
        "test_ratio": TEST_RATIO,
        "seed": SEED,
        "max_seq_length": MAX_SEQ_LENGTH,
        "files": {str(f.relative_to(root)): _sha256(f) for f in sorted(files) if f.is_file()},
    }


def check_reusable(output_path: Path, expected: dict) -> None:
    """Refuse saved results that another model, calibration or test set produced."""
    manifest = output_path / MANIFEST_NAME
    remedy = (f"Delete {output_path} to re-run inference, or set PR_OUTPUT_DIR "
              f"to a new directory for this model.")
    if not manifest.exists():
        raise RuntimeError(
            f"{output_path} holds inference results with no record of the model that "
            f"produced them, so they may belong to a different checkpoint. {remedy}")
    saved = json.loads(manifest.read_text())
    differing = sorted(key for key in set(saved) | set(expected) if saved.get(key) != expected.get(key))
    if differing:
        lines = []
        for key in differing:
            if key == "files":
                old, new = saved.get("files", {}), expected.get("files", {})
                changed = sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))
                lines.append(f"    files changed: {changed}")
            else:
                lines.append(f"    {key}: saved {saved.get(key)!r}, now {expected.get(key)!r}")
        raise RuntimeError(
            f"{output_path} was produced by a different run:\n" + "\n".join(lines) + f"\n  {remedy}")


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_test_data(split_dir: str) -> Dict[str, Dataset]:
    """Load test data splits."""
    base_path = Path(split_dir)
    
    splits = {
        "M_core_test": load_from_disk(str(base_path / "M_core_test")),
        "M_extra_test": load_from_disk(str(base_path / "M_extra_test")),
        "B_core_test": load_from_disk(str(base_path / "B_core_test")),
        "B_extra_test": load_from_disk(str(base_path / "B_extra_test")),
    }
    
    print(f"Loaded test data:")
    for name, ds in splits.items():
        print(f"  {name}: {len(ds)} samples")
    
    return splits


def load_temperature(model_path: str) -> float:
    """Load temperature from calibration.json."""
    calibration_path = Path(model_path) / "calibration.json"
    if not calibration_path.exists():
        print(f"  ⚠ Warning: calibration.json not found at {calibration_path}, using temperature=1.0")
        return 1.0

    with open(calibration_path) as f:
        calibration = json.load(f)

    temperature = calibration.get("temperature", 1.0)
    print(f"  Loaded temperature: {temperature:.4f}")
    return temperature


def load_calibrator(model_path: str) -> PlattCalibrator:
    """Load the score calibrator from calibration.json.

    Understands both artifact shapes: the two-parameter Platt block written by
    the current training code, and the legacy scalar ``temperature``, which is
    the special case ``a = 1/T, b = 0``.

    The calibrator is a monotone map of the logit margin, so it cannot change
    precision, recall, AP or AUC. What it does decide is whether the resulting
    float32 probability stays usable: a Platt intercept fitted at the deployment
    prevalence subtracts roughly log(ratio) from every score, which keeps the
    probabilities clear of 1.0, while a bare temperature cannot express an
    offset and leaves the top of the ranking saturated.
    """
    calibration_path = Path(model_path) / "calibration.json"
    if not calibration_path.exists():
        from segment_scoring import is_segment_checkpoint
        if is_segment_checkpoint(model_path):
            raise FileNotFoundError(f"Segment model requires its own calibration: {calibration_path}")
        print(f"  ⚠ Warning: calibration.json not found at {calibration_path}, "
              f"using an identity calibrator")
        return PlattCalibrator(a=1.0, b=0.0)

    with open(calibration_path) as f:
        calibration = json.load(f)

    from segment_scoring import validate_segment_calibration
    validate_segment_calibration(model_path, calibration)
    block = calibration.get("calibrator")
    if isinstance(block, dict) and block.get("type") == "platt":
        cal = PlattCalibrator.from_dict(block)
        print(f"  Loaded Platt calibrator: a={cal.a:.4f}, b={cal.b:+.4f} "
              f"(equivalent temperature {cal.temperature:.4f})")
        return cal

    temperature = float(calibration.get("temperature", 1.0))
    print(f"  Loaded legacy temperature {temperature:.4f} -> Platt a={1/temperature:.4f}, b=0")
    return PlattCalibrator(a=1.0 / temperature, b=0.0)


# =============================================================================
# Data Cleaning
# =============================================================================

def clean_dataset(dataset: Dataset, text_column: str = "text") -> Tuple[Dataset, int]:
    """
    Clean dataset by removing samples with invalid text values.
    
    Args:
        dataset: Input dataset
        text_column: Name of the text column to validate
    
    Returns:
        Tuple of (cleaned_dataset, num_removed)
    """
    def is_valid_text(example):
        text = example.get(text_column)
        # Check for None, non-string types, or empty strings
        if text is None:
            return False
        if not isinstance(text, str):
            return False
        if text.strip() == "":
            return False
        return True
    
    # Filter the dataset
    cleaned_dataset = dataset.filter(is_valid_text, desc="Cleaning invalid text samples")
    
    num_removed = len(dataset) - len(cleaned_dataset)
    
    return cleaned_dataset, num_removed


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    model_path: str,
    test_data: Dict[str, Dataset],
    calibrator: PlattCalibrator,
    training_mode: str,
    max_seq_length: int,
    output_dir: str,
) -> Dataset:
    """
    Run inference and save results as HuggingFace Dataset.

    Returns dataset with columns: text, label, logit, prob, source.
    """
    print(f"\n{'='*64}")
    print(f"  RUNNING INFERENCE")
    print(f"{'='*64}")
    print(f"  Model: {model_path}")
    print(f"  Training mode: {training_mode}")
    print(f"  Calibrator: a={calibrator.a:.4f}, b={calibrator.b:+.4f} "
          f"(equivalent temperature {calibrator.temperature:.4f})")

    # Load model and tokenizer. The tokenizer always comes from the model
    # directory: a PEFT adapter/ holds no tokenizer files.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    from segment_scoring import load_scoring_model
    model = load_scoring_model(model_path, training_mode)
    
    # Create trainer for inference
    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=32,
        report_to="none",
        fp16=False,
    )
    
    trainer = Trainer(
        model=model,
        args=train_args,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    
    # Combine M_core_test + M_extra_test into M_test
    M_test = combine_malicious_pools(
        test_data["M_core_test"],
        test_data["M_extra_test"],
    )
    
    # Combine B_core_test + B_extra_test into B_test
    B_test = combine_benign_pools(
        test_data["B_core_test"],
        test_data["B_extra_test"],
    )
    
    # Build test set
    test_ds = build_test_set(
        M_test=M_test,
        B_test=B_test,
        benign_per_malicious=TEST_RATIO,
        seed=SEED,
    )
    
    print(f"  Test set: {len(test_ds)} samples before cleaning")
    
    # Clean dataset - remove samples with invalid text (None, empty strings)
    test_ds_cleaned, num_removed = clean_dataset(test_ds, "text")
    
    if num_removed > 0:
        print(f"  ⚠ Removed {num_removed} samples with invalid text values")
    
    print(f"  Test set: {len(test_ds_cleaned)} samples after cleaning")
    
    # Store original data before tokenization (from cleaned dataset)
    original_texts = test_ds_cleaned["text"]
    original_labels = test_ds_cleaned["label"] if "label" in test_ds_cleaned.column_names else test_ds_cleaned["labels"]
    original_sources = test_ds_cleaned["source"] if "source" in test_ds_cleaned.column_names else ["unknown"] * len(test_ds_cleaned)
    
    # Tokenize the cleaned dataset
    test_ds_tokenized = tokenize_dataset(test_ds_cleaned, tokenizer, max_seq_length)
    
    # Run prediction
    print(f"  Running prediction on {len(test_ds_tokenized)} samples...")
    logits, labels = _extract_predictions(trainer, test_ds_tokenized, "test")

    # text/label/source come from test_ds_cleaned, logits from test_ds_tokenized,
    # and Dataset.from_dict below pairs them by position. Verify the pairing
    # instead of assuming tokenize_dataset kept every row in order: the labels
    # the model was scored with must equal the labels written next to the text.
    if len(test_ds_tokenized) != len(original_texts):
        raise RuntimeError(f"tokenize_dataset changed the row count: "
                           f"{len(original_texts)} -> {len(test_ds_tokenized)}")
    if labels is not None and not np.array_equal(
            np.asarray(labels).reshape(-1).astype(int), np.asarray(original_labels).astype(int)):
        raise RuntimeError("Scores and texts are misaligned: the labels Trainer scored "
                           "differ from the labels saved beside each text")
    if "text" in test_ds_tokenized.column_names and list(test_ds_tokenized["text"]) != list(original_texts):
        raise RuntimeError("Scores and texts are misaligned: tokenize_dataset reordered or "
                           "altered the text column")
    
    # Get logits for positive class (class 1)
    logits_pos = logits[:, 1]

    # Calibrated probability. The Platt intercept subtracts roughly
    # log(deployment ratio) from every score, which keeps the calibrated
    # log-odds clear of the point where a float32 probability rounds to exactly
    # 1.0 (~17.3). check_score_resolution() verifies that rather than assuming
    # it: past that point the top of the ranking collapses into a single tie and
    # no threshold can separate it.
    probs = calibrator.scale(logits)

    print(f"  Logits shape: {logits.shape}")
    print(f"  Probs range: [{probs.min():.6f}, {probs.max():.6f}]")
    check_score_resolution(np.float32(probs), name="inference probabilities")

    # Create results dataset
    # Note: Pass numpy arrays directly to Dataset.from_dict() for efficiency.
    # Avoiding .tolist() provides 10-100x speedup by skipping Python list conversion.
    results = Dataset.from_dict({
        "text": original_texts,
        "label": original_labels,
        "logit": logits_pos,
        "prob": probs,
        "source": original_sources,
    })
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results.save_to_disk(str(output_path))
    (output_path / MANIFEST_NAME).write_text(
        json.dumps(inference_identity(model_path, training_mode, DATA_SPLIT_DIR), indent=2) + "\n")
    print(f"  Results saved to: {output_path}")
    print(f"  Total samples: {len(results)}")
    
    # Clean up
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"{'='*64}\n")
    
    return results


# =============================================================================
# Plotting
# =============================================================================

def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_fpr_fnr(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Tuple[float, float]:
    """
    Compute FPR and FNR at a given threshold.
    
    Args:
        labels: True labels (0 or 1)
        probs: Predicted probabilities for positive class
        threshold: Classification threshold
    
    Returns:
        Tuple of (fpr, fnr)
    """
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return fpr, fnr


def plot_pr_curve(
    pr_curve: Dict[str, np.ndarray],
    target_precisions: List[float],
    t_p_indices: List[int],
    target_recalls: List[float],
    t_r_indices: List[int],
    output_path: str,
    probs: np.ndarray = None,
    labels: np.ndarray = None,
    n_plot_points: int = 1000,
    sampling_method: str = "uniform_recall",
):
    """
    Plot PR curve with special points marked for target precision and recall values.
    
    Args:
        pr_curve: Dict with precision, recall, thresholds
        target_precisions: List of target precision values (e.g., [0.85, 0.90, 0.95])
        t_p_indices: List of indices for each target precision threshold
        target_recalls: List of target recall values (e.g., [0.85, 0.90, 0.95])
        t_r_indices: List of indices for each target recall threshold
        output_path: Path to save the plot
        probs: Predicted probabilities for positive class (required for FPR computation)
        labels: True labels (required for FPR computation)
        n_plot_points: Number of points to use for plotting (downsampling)
        sampling_method: Sampling method for plotting:
            - "uniform_index": Uniform sampling by index (original method).
              May have low resolution in areas where precision is sparse.
            - "uniform_recall": Uniform sampling in recall space with interpolation.
              Ensures even coverage across the recall axis, providing better
              resolution in sparse precision regions.
    """
    precision = pr_curve["precision"]
    recall = pr_curve["recall"]
    thresholds = pr_curve["thresholds"]
    
    print(f"\n{'='*64}")
    print(f"  PLOTTING PR CURVE")
    print(f"{'='*64}")
    print(f"  Total PR points: {len(precision)}")
    print(f"  Plot points: {n_plot_points}")
    print(f"  Sampling method: {sampling_method}")
    print(f"  Target precisions: {target_precisions}")
    print(f"  Target recalls: {target_recalls}")
    
    # Downsample for plotting based on sampling method
    if len(precision) > n_plot_points:
        if sampling_method == "uniform_index":
            # Original method: uniform sampling by index
            # Simple but may have low resolution in sparse precision regions
            indices = np.linspace(0, len(precision) - 1, n_plot_points, dtype=int)
            precision_plot = precision[indices]
            recall_plot = recall[indices]
        elif sampling_method == "uniform_recall":
            # Improved method: uniform sampling in recall space using nearest neighbor
            # Generate evenly spaced target recall values across the recall range
            # Then find the nearest actual recall value and use the corresponding
            # precision value. This ensures even coverage across the recall axis
            # while using actual data points (no interpolation).
            target_recalls_plot = np.linspace(recall.min(), recall.max(), n_plot_points)
            # Find the index of the nearest actual recall value for each target
            indices = np.array([np.argmin(np.abs(recall - r)) for r in target_recalls_plot])
            # Remove duplicate indices to avoid plotting the same point multiple times
            indices = np.unique(indices)
            precision_plot = precision[indices]
            recall_plot = recall[indices]
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}. "
                           f"Choose from: 'uniform_index', 'uniform_recall'")
    else:
        precision_plot = precision
        recall_plot = recall
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot PR curve
    ax.plot(recall_plot, precision_plot, 'b-', linewidth=2, label='PR Curve', alpha=0.8)
    
    # Color maps for precision and recall points
    precision_colors = plt.cm.Reds(np.linspace(0.4, 0.9, max(len(target_precisions), 1)))
    recall_colors = plt.cm.Greens(np.linspace(0.4, 0.9, max(len(target_recalls), 1)))
    
    # Plot target precision points (minimal threshold for precision >= target)
    for i, (target_p, t_p_idx) in enumerate(zip(target_precisions, t_p_indices)):
        t_p = thresholds[t_p_idx]
        p_p = precision[t_p_idx]
        r_p = recall[t_p_idx]
        f1_p = compute_f1(p_p, r_p)
        
        # Compute FPR and FNR if probs and labels are provided
        if probs is not None and labels is not None:
            fpr_p, fnr_p = compute_fpr_fnr(labels, probs, t_p)
            label_str = f'T_p{target_p:.2f}: P={p_p:.4f}, R={r_p:.4f}, T={t_p:.4f}, F1={f1_p:.4f}, FPR={fpr_p:.4f}, FNR={fnr_p:.4f}'
        else:
            label_str = f'T_p{target_p:.2f}: P={p_p:.4f}, R={r_p:.4f}, T={t_p:.4f}, F1={f1_p:.4f}'
        
        # Plot point with label for legend
        ax.scatter([r_p], [p_p], color=precision_colors[i], s=150, zorder=5, 
                   marker='o', edgecolors='black', linewidths=2,
                   label=label_str)
    
    # Plot target recall points (maximum threshold for recall >= target)
    for i, (target_r, t_r_idx) in enumerate(zip(target_recalls, t_r_indices)):
        t_r = thresholds[t_r_idx]
        p_r = precision[t_r_idx]
        r_r = recall[t_r_idx]
        f1_r = compute_f1(p_r, r_r)
        
        # Compute FPR and FNR if probs and labels are provided
        if probs is not None and labels is not None:
            fpr_r, fnr_r = compute_fpr_fnr(labels, probs, t_r)
            label_str = f'T_r{target_r:.2f}: P={p_r:.4f}, R={r_r:.4f}, T={t_r:.4f}, F1={f1_r:.4f}, FPR={fpr_r:.4f}, FNR={fnr_r:.4f}'
        else:
            label_str = f'T_r{target_r:.2f}: P={p_r:.4f}, R={r_r:.4f}, T={t_r:.4f}, F1={f1_r:.4f}'
        
        # Plot point with label for legend
        ax.scatter([r_r], [p_r], color=recall_colors[i], s=150, zorder=5, 
                   marker='s', edgecolors='black', linewidths=2,
                   label=label_str)
    
    # Add reference lines for target precisions
    for target_p in target_precisions:
        ax.axhline(y=target_p, color='orange', linestyle='--', alpha=0.3)
    
    # Add reference lines for target recalls
    for target_r in target_recalls:
        ax.axvline(x=target_r, color='green', linestyle='--', alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve for Prompt Injection Detection', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  PR curve saved to: {output_path}")
    
    # Print summary
    print(f"\n  Special Points Summary:")
    if target_precisions:
        print(f"  Target Precision Points:")
        for i, (target_p, t_p_idx) in enumerate(zip(target_precisions, t_p_indices)):
            t_p = thresholds[t_p_idx]
            p_p = precision[t_p_idx]
            r_p = recall[t_p_idx]
            f1_p = compute_f1(p_p, r_p)
            
            # Compute FPR and FNR if probs and labels are provided
            if probs is not None and labels is not None:
                fpr_p, fnr_p = compute_fpr_fnr(labels, probs, t_p)
                print(f"    T_p{target_p:.2f} (minimal threshold for P≥{target_p:.2f}):")
                print(f"      Threshold: {t_p:.4f}")
                print(f"      Precision: {p_p:.4f}")
                print(f"      Recall:    {r_p:.4f}")
                print(f"      F1:        {f1_p:.4f}")
                print(f"      FPR:       {fpr_p:.4f}")
                print(f"      FNR:       {fnr_p:.4f}")
            else:
                print(f"    T_p{target_p:.2f} (minimal threshold for P≥{target_p:.2f}):")
                print(f"      Threshold: {t_p:.4f}")
                print(f"      Precision: {p_p:.4f}")
                print(f"      Recall:    {r_p:.4f}")
                print(f"      F1:        {f1_p:.4f}")
    else:
        print(f"  No target precision points.")
    
    if target_recalls:
        print(f"  Target Recall Points:")
        for i, (target_r, t_r_idx) in enumerate(zip(target_recalls, t_r_indices)):
            t_r = thresholds[t_r_idx]
            p_r = precision[t_r_idx]
            r_r = recall[t_r_idx]
            f1_r = compute_f1(p_r, r_r)
            
            # Compute FPR and FNR if probs and labels are provided
            if probs is not None and labels is not None:
                fpr_r, fnr_r = compute_fpr_fnr(labels, probs, t_r)
                print(f"    T_r{target_r:.2f} (maximum threshold for R≥{target_r:.2f}):")
                print(f"      Threshold: {t_r:.4f}")
                print(f"      Precision: {p_r:.4f}")
                print(f"      Recall:    {r_r:.4f}")
                print(f"      F1:        {f1_r:.4f}")
                print(f"      FPR:       {fpr_r:.4f}")
                print(f"      FNR:       {fnr_r:.4f}")
            else:
                print(f"    T_r{target_r:.2f} (maximum threshold for R≥{target_r:.2f}):")
                print(f"      Threshold: {t_r:.4f}")
                print(f"      Precision: {p_r:.4f}")
                print(f"      Recall:    {r_r:.4f}")
                print(f"      F1:        {f1_r:.4f}")
    else:
        print(f"  No target recall points.")
    
    print(f"{'='*64}\n")
    
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    print(f"\n{'='*64}")
    print(f"  PR CURVE PLOTTING")
    print(f"{'='*64}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Training mode: {TRAINING_MODE}")
    print(f"  Data: {DATA_SPLIT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*64}\n")
    
    # Check if inference results already exist
    output_path = Path(OUTPUT_DIR)
    
    calibrator = load_calibrator(MODEL_PATH)

    if output_path.exists():
        print(f"  Loading existing inference results from: {output_path}")
        check_reusable(output_path, inference_identity(MODEL_PATH, TRAINING_MODE, DATA_SPLIT_DIR))
        results = load_from_disk(str(output_path))
        print(f"  Loaded {len(results)} samples")
    else:
        # Load test data
        test_data = load_test_data(DATA_SPLIT_DIR)

        # Run inference
        results = run_inference(
            model_path=MODEL_PATH,
            test_data=test_data,
            calibrator=calibrator,
            training_mode=TRAINING_MODE,
            max_seq_length=MAX_SEQ_LENGTH,
            output_dir=OUTPUT_DIR,
        )

    labels = np.array(results["label"])
    probs = np.array(results["prob"], dtype=np.float64)

    print(f"\n  Computing PR curve...")
    print(f"  Total samples: {len(labels)}")
    print(f"  Positive samples: {labels.sum()}")
    print(f"  Negative samples: {len(labels) - labels.sum()}")

    stats = check_score_resolution(probs, name=str(output_path))
    if stats["n_saturated"]:
        capped = labels[probs >= 1.0].mean()
        print(f"    These results were produced with a calibrator that leaves "
              f"scores in the saturating range, so this curve is capped at "
              f"{capped:.4f} precision and has no operating point below recall "
              f"{labels[probs >= 1.0].sum() / labels.sum():.4f}.")
        print(f"    Delete {output_path} and re-run to regenerate with the "
              f"current calibrator.")

    # Compute PR curve
    pr_curve = compute_pr_curve(probs, labels)
    thresholds = pr_curve["thresholds"]
    
    # Find thresholds for each target precision
    t_p_indices = []
    for target_p in TARGET_PRECISIONS:
        t_p = find_threshold_for_precision(pr_curve, target_p)
        t_p_idx = np.argmin(np.abs(thresholds - t_p))
        t_p_indices.append(t_p_idx)
    
    # Find thresholds for each target recall
    t_r_indices = []
    for target_r in TARGET_RECALLS:
        t_r = find_threshold_for_recall(pr_curve, target_r)
        t_r_idx = np.argmin(np.abs(thresholds - t_r))
        t_r_indices.append(t_r_idx)
    
    # Plot PR curve
    plot_pr_curve(
        pr_curve=pr_curve,
        target_precisions=TARGET_PRECISIONS,
        t_p_indices=t_p_indices,
        target_recalls=TARGET_RECALLS,
        t_r_indices=t_r_indices,
        output_path=PR_CURVE_PATH,
        probs=probs,
        labels=labels,
    )
    
    print(f"\n{'='*64}")
    print(f"  COMPLETE")
    print(f"{'='*64}")
    print(f"  Inference results: {OUTPUT_DIR}")
    print(f"  PR curve plot: {PR_CURVE_PATH}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    apply_training_mode(parse_training_mode(__doc__))
    main()
