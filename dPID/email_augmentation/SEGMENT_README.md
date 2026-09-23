# Token-only email model

For INT8 ONNX/CPU export and prediction, see the separate
[segment quantization commands](../quant_engine/SEGMENT_README.md).

This is a new model, not a continuation of the pooled sentence classifier.
The original template training script and the legacy region implementation are
unchanged. The new entry point enables a separate scoring path in the shared
email training loop; existing email configurations retain their old behavior.

## Train

Run from the repository root in the existing training environment:

```bash
python train_benign_exposure_mmbert2_dilute_email_segment.py
```

The default configuration is
`configs/training/peft_benign_exposure_mmbert2_email_segment.yaml`.
It runs **20 total Optuna trials**, then trains a fresh final model with the
winning hyperparameters, calibrates on the calibration split, and evaluates on
the test split. ASHA may stop unsuccessful trials early, after a two-epoch grace
period. The first five trials cover every categorical candidate at least once;
they are not a full grid. Selection remains validation F1, never test metrics.

Output:

```text
benign_exposure_peft_mmbert2_email_segment_seed42/
  checkpoints/hpo_complete.json
  segment_search_space.json
  email_augmentation/
  best_model/jhu-clsp_mmBERT-small/
    config.json
    model.safetensors
    adapter/
    calibration.json
    calibration_metrics.json
    test_metrics.json
```

To retrain the final model using this new run's completed HPO:

```bash
python train_benign_exposure_mmbert2_dilute_email_segment.py hpo.skip=true
```

This retrains and replaces the final model; it is not calibration-only. It rejects
old sentence-model HPO checkpoints. Use a different `ray.output_dir` for a new
independent experiment, especially after changing search or augmentation settings.
An existing compatible Ray experiment resumes its search state; completed trials
are not an additional budget of 20.

## Train an interim final model while HPO continues

Run from the same repository root and environment as the source HPO. First
inspect the selection without training, downloads, or output writes:

```bash
python train_benign_exposure_mmbert2_dilute_email_segment.py \
  --final-from-current --dry-run
```

Then train with the currently best completed trial's parameters:

```bash
CUDA_VISIBLE_DEVICES=1 python train_benign_exposure_mmbert2_dilute_email_segment.py \
  --final-from-current
```

Replace `1` with an available GPU. This is an independent process outside Ray's
resource scheduling; sharing an occupied GPU can cause OOM or slow down HPO.
No HPO pause, restore, or restart is performed.

The default source is `benign_exposure_peft_mmbert2_email_segment_seed42`.
For a different run or a named output directory:

```bash
python train_benign_exposure_mmbert2_dilute_email_segment.py \
  --final-from-current \
  --hpo-dir benign_exposure_peft_mmbert2_email_segment_seed42 \
  --output-dir benign_exposure_peft_mmbert2_email_segment_seed42_preview1
```

- `--hpo-dir` is the run root, not its `benign_exposure_hpo/` child. Its saved
  `email_training_config.yaml` supplies the original data and training settings.
- Selection reads the newest readable Ray JSON state snapshot. Only trials with
  status `TERMINATED`, no recorded failures, and `epoch >= num_train_epochs`
  qualify. Running trials and ASHA-pruned trials below their epoch budget are
  excluded, even when their partial scores are higher.
- Rank by the **last reported `val_f1`**, not an earlier peak. Exact ties use
  trial ID. If no trial qualifies, exit without creating an output directory.
  Ray state flushing may lag behind logs; retry after its next flush. An
  incomplete JSON write falls back to the previous readable snapshot.
- Default output is a unique sibling directory named
  `<source>_interim_<UTC timestamp>_<suffix>`. Existing outputs and paths
  overlapping the source run are rejected. Augmentation artifacts are isolated;
  the external email download cache can be reused.
- `checkpoints/hpo_current_best.json` records selected parameters, metrics, trial
  ID, source state hash, selection time, and excluded-trial counts. No source
  experiment files are changed, and no `hpo_complete.json` is fabricated.
- The normal final-only pipeline retrains from the pretrained encoder with new
  LoRA/head weights, then calibrates and evaluates. It does not resume the trial's
  weights. Point `PR_MODEL_PATH` to the interim directory's
  `best_model/jhu-clsp_mmBERT-small` for manual prediction and PR evaluation.

Repeated interim evaluations should not turn the held-out test set into a tuning
set; use validation for development decisions.

## Score and supervision

`segment_scoring.py` implements the only primary scoring function:

```text
z = token_head(encoder(input))
sequence_logit = max over nonempty contiguous intervals I: sum(z[t] - tau for t in I)
logits = [0, sequence_logit]
loss = ASL(sequence_logit, label) + region_loss_weight * region_loss
region_loss[sample] = 0.5 * positive_region_BCE + 0.5 * negative_region_BCE
```

- Tau is fixed within each trial, selected by HPO, and saved in `config.json`.
  It is not a trainable parameter. There is no `L_loc`, dense token BCE, pooled
  sentence head, or sequence top-k.
- Decoding uses an O(L) vectorized prefix-minimum equivalent of Kadane. Intervals
  must be nonempty, so all-negative inputs retain negative sequence logits.
  Padding and special tokens are excluded and are barriers inside the sequence.
  Gradients flow through the selected token sum; spans never select the interval.
- Training, calibration, manual prediction, and PR inference all call this same
  model forward/scoring function. Region masks are only needed during training.
- Positive-region BCE uses the mean of the highest `malicious_top_k` logits in
  the inserted malicious payload. Negative-region BCE uses the mean of the
  highest `benign_top_k` logits outside that payload, or across all valid tokens
  of a benign example. Each K is clamped to its region size. A missing region
  contributes zero, without redistributing its half-weight. Average across samples.
- The source dataset has payload-level labels, not finer attack spans: the
  inserted payload boundary defines the positive region. Common words inside it
  are **not** forced individually to be malicious. Masks use tokenizer character
  offsets and composition boundaries, not substring searches.
- ASL uses the existing positive/negative focusing and negative probability
  shift, evaluated in log space without hard-clamping the aggregate logit.
  The new Trainer explicitly normalizes gradient accumulation for mean losses.
- Initialize from the pretrained mmBERT encoder with fresh LoRA and token-head
  weights. Keep existing QAT-before-LoRA preparation and FP32 deployment loading.
  The trainable token head is saved in the PEFT adapter as well as the unloaded
  base; it is not deleted after training. Calibration uses the saved deployment
  model, without training-time fake quantization.

The six-form sampling proportions, email splits, insertion positions, and
payload/template preservation stay as described in [README.md](README.md).
Region-enabled fixed views use a new cache identity and store three masks.
Validation logs include recall, FPR, and selected-interval length for short
(under 128 valid tokens), medium (128–255), and long (256–512) inputs.

## HPO ranges

| Parameter | Search range |
| --- | --- |
| `segment_tau` | 0, 0.5, 1, 2, 4 |
| `region_loss_weight` | 0.05, 0.1, 0.2, 0.5, 1 |
| `malicious_top_k` | 1, 3, 5 |
| `benign_top_k` | 3, 8, 16, 32 |
| `learning_rate` | Log-uniform, 1e-5 to 3e-4 |
| `token_head_dropout` | 0, 0.1, 0.2 |
| `effective_batch_size` | 16, 32, 64 |
| `num_train_epochs` | 3, 5, 8 |

Fixed: LoRA r=16, alpha=64, dropout=0; ASL gamma_pos=1, gamma_neg=2,
clip=0.01; weight decay=0.01; warmup=0.1; benign:malicious training ratio=30.
Effective batch is per trial's single GPU: micro-batch is capped at 32, with
gradient accumulation for 64. Lower `perf.max_micro_batch_size` if needed;
the cap must divide the selected effective batch.

## Predict and evaluate

`predict_prompt.py` automatically prints saved tau, selected token/character ranges
(`[start, end)`, zero-based), window text, and selected versus valid token counts.
`--json` includes these under `segment_window`. This is the segment used for the
final score, not a crop fed to the encoder: the encoder still sees the full input
up to 512 tokens. A window is also selected for SAFE inputs; it is not a labeled
attack span. No retraining or recalibration is required to display it.

```bash
export PR_MODEL_PATH="benign_exposure_peft_mmbert2_email_segment_seed42/best_model/jhu-clsp_mmBERT-small"
python predict_prompt.py --training-mode peft --file test_prompt1.txt --single

PR_DATA_SPLIT_DIR=data_split3/42 \
PR_OUTPUT_DIR=inference_segment_original_seed42 \
PR_CURVE_PATH=pr_segment_original_seed42.png \
python plot_pr_curve_mmbert2.py --training-mode peft

PR_DATA_SPLIT_DIR=data_split3_dilute/42 \
PR_OUTPUT_DIR=inference_segment_dilute_seed42 \
PR_CURVE_PATH=pr_segment_dilute_seed42.png \
python plot_pr_curve_mmbert2.py --training-mode peft
```

Use the already generated held-out email dilution dataset; changing the scorer
does not require regenerating it. Both evaluations use the new model's same
`calibration.json`; do not tune thresholds on either test set. Old
`calibration_dilute.json` is ignored for segment checkpoints. Loaders reject
missing calibration or mismatched scorer/tau metadata. Metadata does not detect
manually copied calibration from another model with the same tau; always keep
calibration with the model that produced it.

Use new inference output directories after retraining: the PR script reuses
existing cached predictions. Its existing length limit is still 512 tokens;
this change addresses scoring dilution within that window, not context overflow.
The aggregation does not guarantee length-independent false positives; inspect
long-input recall and FPR on the newly trained model.

## Compare downloaded raw and diluted predictions

Use the dedicated entry point, not the old calibration-comparison scripts:

```bash
python plot_pr_curve_segment_raw_dilute.py
```

Defaults are `inference_segment_original_seed42_v2` and
`inference_segment_dilute_seed42_v2` next to the script. Output is
`pr_segment_raw_vs_dilute.png`. Override paths with `--raw-dir`, `--dilute-dir`,
and `--out`; `--ratio 1000` sets the common benign:malicious projection.
It reads saved `label`/`prob` and the existing model's `calibration.json`, without
loading model weights or adapters. The saved probability threshold is marked on
both curves, together with projected precision/recall/F1/accuracy and FPR/FNR.
Precision=0.90 and recall=0.90 reference lines are included. It never fits a new
calibrator or searches for a threshold on the test sets. Metrics use exact
`score >= threshold` decisions, not the nearest sampled curve threshold.
The default calibration path is the final segment checkpoint under
`benign_exposure_peft_mmbert2_email_segment_seed42/best_model/jhu-clsp_mmBERT-small/`.
For another checkpoint (including an interim model), pass
`--calibration-file /path/to/that/checkpoint/calibration.json`.
Both input runs must use that same checkpoint/calibration; the saved columns
alone cannot verify provenance. `calibration_dilute.json` is not used.

If probabilities are exactly 0 or 1, the script warns about lost ranking
resolution. For these segment-model exports, `--score-column logit` plots the
saved sequence margins directly, without probability inversion or recalibration.
Only the saved threshold is converted to equivalent margin units using
`T_margin = (log(T_prob / (1 - T_prob)) - b) / a`. The plot labels the original
probability threshold and reports the equivalent margin threshold separately.
Do not assume the `logit` column is a margin for unrelated sentence-model exports.
The plot and metrics report label the diluted input as "Augmented". Plot titles,
axis labels, tick numbers and legends use larger fonts; no footer note is drawn.
Input paths, filenames and the existing `--dilute-dir` option remain unchanged.
Input datasets and calibration are unchanged. Rerunning replaces the output PNG
and its companion `.metrics.json`, containing both projected and empirical metrics
and TP/FP/FN/TN counts. Recall, FPR and FNR are unchanged by prevalence projection.

## Offline tests

```bash
python -m unittest discover -s tests -p 'test_segment_scoring.py' -v
python -m unittest discover -s tests -p 'test_interim_segment_training.py' -v
python -m unittest discover -s tests -p 'test_plot_segment_raw_dilute.py' -v
```

Tests use tiny local models without downloading weights: exhaustive interval
checks, gradients, region weights/masks, accumulation equivalence, QAT/PEFT
save/load, the HPO training callback, final training, calibration, and agreement
between manual and PR predictions. They do not establish full-model accuracy
or GPU throughput.
