# Email-background augmentation for mmBERT

For the new token-only maximum-subarray model and its 20-trial HPO, see
[SEGMENT_README.md](SEGMENT_README.md). The instructions below describe the
legacy email model and the shared data pipeline.

The standalone entry point is `train_benign_exposure_mmbert2_dilute_email.py`.
The original `train_benign_exposure_mmbert2_dilute_template.py` is unchanged.
The email experiment supports six input forms. Emails are benign backgrounds. When a payload is present, its
label becomes the input label. Without a payload, the label is always benign.
Payload, template + payload, and the two email + payload forms combined receive
equal shares of the payload-bearing budget (1:1:1). Email-only forms occupy
10% of the benign budget, rather than adding samples to that budget.

## Input forms and sampling

| Form | Within malicious | Within benign |
| --- | ---: | ---: |
| Payload | 33 1/3% | 30% |
| Template + payload | 33 1/3% | 30% |
| Email with inserted payload | 16 2/3% | 15% |
| Template + email with inserted payload | 16 2/3% | 15% |
| Email only | 0% | 5% |
| Template + email only | 0% | 5% |

These are sampling probabilities, so finite datasets and batches will not have
exactly these percentages. Class counts remain those of the existing sampling
pipeline: retain eligible malicious payloads and sample benign slots according
to `benign_to_malicious_ratio`. A benign slot assigned an email-only form does
not use its original payload. No malicious slot can lose its payload.

For any form requiring an email, sample one record uniformly with replacement
from the corresponding email split. Both labels use the same pool and rules.
Each input contains at most one inserted payload and one email background.
The existing `evaluation/bigscience_p3.jsonl` template pool is reused; no new
email-specific template pool is introduced. A template replaces `{PAYLOAD}`
with either the payload, the composed email, or the email-only body. In email
mode, the six form weights supersede `template.probability` and
`template.enabled`; templates are an explicit part of the selected form.

## Email source and splits

Source: [LLM-PBE/enron-email](https://huggingface.co/datasets/LLM-PBE/enron-email).
Read the `text` column from the upstream `train` split. No separate subject
field is used. Forwarded headers, quoted text, signatures, and addresses inside
the body remain part of the background.

The download code is in [`load_email_pools()`](../email_augmentation.py), which
is called by `main()` in the standalone email training script. On a cache miss,
it calls Hugging Face `datasets.load_dataset()` with `LLM-PBE/enron-email`,
`split="train"`, and the configured revision. Downloaded source files are cached
under `email_data_cache/downloads/` by default; the four processed email pools
and their manifest are stored under `email_data_cache/<settings-hash>/`.

Assign source row IDs before removing empty or non-string bodies. Shuffle valid
records with `data.seed` and divide them into:

| Email split | Fraction | Payload source |
| --- | ---: | --- |
| train | 70% | Existing train splits |
| valid | 10% | Existing val splits |
| calib | 10% | Existing cal splits |
| test | 10% | Existing test splits |

Round the first three counts down and give the remainder to test. No
deduplication, thread reconstruction, length stratification, or usage balancing
is performed. Source row IDs are disjoint across splits, but duplicate text or
quoted thread content can still cross split boundaries. Public availability
does not guarantee content-level independence.

The first load downloads and stores a snapshot beneath `email.cache_dir`.
Subsequent runs with the same source settings, revision, seed, and split ratios
reuse it. The manifest records the upstream dataset fingerprint, removed empty
record count, split counts, and split fingerprints. Set `email.revision` to a
commit SHA when an immutable upstream revision is required. With `main`, the
local snapshot is fixed but a new cache on another machine may fetch a newer
upstream revision; preserve the cache and manifest for exact reproduction.

## Insertion positions and truncation

For either payload-bearing email form, choose independently:

- Start of the body: probability 1/3.
- End of the body: probability 1/3.
- A uniformly sampled internal whitespace boundary: probability 1/3.

The internal choice never cuts a whitespace-delimited English word. If no
internal boundary exists, choose start or end with equal probability. Surround
the payload with blank lines. The sampling manifest records both the requested
and effective position. All position and formatting rules are label-independent.

`data.max_seq_length` and `template.max_length` must match (512 by default).
The token limit includes model special tokens. The implementation:

1. Excludes empty payloads, payloads exceeding the limit by themselves, and
   payloads that cannot fit inside any template. Filtering happens before class
   counts and benign sampling are computed. Exclusion counts are saved under
   `payload_filter/`. All four forms remain available to retained payloads.
2. Selects a template uniformly from those that fit the complete payload,
   whenever a template is required. Template prefix and suffix are preserved.
3. Keeps whole email words around the insertion point, using the suffix of the
   left context and the prefix of the right context. Email-only forms keep a
   prefix of the body. Whitespace within retained context is preserved.
4. Searches for a fitting background length and tokenizes the actual assembled
   string again. It never truncates the payload or template tokens. If the
   complete payload/template consumes the budget, a payload-bearing email form
   can retain zero background words; this is recorded in the sampling metadata.

The word-budget search prioritizes a bounded, valid result, not mathematically
maximal token utilization. Token counts can vary with subword boundaries. An
email-only form that cannot retain even one complete word raises an explicit
error instead of silently becoming an empty/background-free example.

## Training, validation, calibration, and test

Training keeps the selected payload view fixed and samples a new form, email,
position, and template whenever the collator reads a slot. Python randomness
is seeded by Trainer and its DataLoader workers; do not reseed each batch.
Changing worker count or batch order can change the training augmentation
sequence, while the fixed evaluation mapping remains unaffected.

Validation, calibration, and test derive randomness from the seed, split name,
slot ID, label, and original payload text. Thus sorting, batching, and repeated
evaluation do not change a slot's combination. Fixed views are materialized
once into tokenized Arrow datasets plus aligned sampling metadata:

```text
<ray.output_dir>/
  email_training_config.yaml
  email_augmentation_manifest.json
  email_augmentation/
    payload_filter/<fingerprint>.json
    fixed_views/<view>/<fingerprint>/
      encoded/     # input_ids, attention_mask, labels
      sampling/    # form, slot/payload/email IDs, template ID, positions, word count
```

The metadata rows align with encoded rows before the trainer's length sorting.
Cache identities include the tokenizer, templates, email pool, seed, weights,
payload view, and algorithm version. HPO and final training share the validation
view when these inputs match. Ray receives only the train and valid email pools.

Validation retains the configured 1:50 malicious:benign ratio; calibration and
test retain 1:500 by default. Actual ratios are logged if insufficient eligible
benign payloads remain. Unlike the legacy template-only workflow, email-mode
calibration and test use the same six-form builder as training. Equal calibration
and threshold ratios reuse the identical prediction pass. Calibration and test
never sample backgrounds from train or valid.

## Standalone held-out test dataset

[`materialize_email_diluted_test_splits.py`](../materialize_email_diluted_test_splits.py)
exports a text-based mirror of the four original `*_test` datasets. It is a
new entry point; neither of the original materialize scripts is modified.
Unlike the training script's internal six-form test view, this standalone test
excludes raw payload and uses the 18 `paraphrase` templates from
`evaluation/heldout_templates.jsonl`. It rejects wording overlap with the
configured training template pool after whitespace/case normalization.

The exporter removes the raw-payload weight and renormalizes each class's
remaining weights. With the current training configuration:

| Form | Within malicious | Within benign |
| --- | ---: | ---: |
| Template + payload | 50% | 42.8571% |
| Email with inserted payload | 25% | 21.4286% |
| Template + email with inserted payload | 25% | 21.4286% |
| Email only | 0% | 7.1429% |
| Template + email only | 0% | 7.1429% |

There is no `--mode` option. Every run uses this five-form mixture; custom
training weights, if supplied, are conditioned on excluding raw payload in the
same way. Sampling is deterministic per source record, and independent of
processing batch size and row order. Only the cached **email test pool** is
used; the existing 70/10/10/10 email partition is reused without deduplication.

Preview the first three source rows from each test split:

```bash
python materialize_email_diluted_test_splits.py --dry-run 3
```

Create the full mirror:

```bash
python materialize_email_diluted_test_splits.py \
  --config configs/training/peft_benign_exposure_mmbert2_email.yaml \
  --output-dir data_split3_dilute
```

The default config is the PEFT email config, but this operation never loads
model weights, trains, or calibrates a model. Both FT and PEFT models may be
evaluated on the same exported dataset. `--tokenizer` accepts a model ID or a
local tokenizer directory; it defaults to `jhu-clsp/mmBERT-small`. `--batch-size`
controls processing only, and dot-list overrides such as `data.seed=123` are
accepted. Relative input/config/output paths are resolved from the repository
root. A dry run does not create the output mirror, but may download and cache
Enron and the tokenizer on first use.

```text
data_split3_dilute/
  README.md
  42/
    M_core_test/
    M_extra_test/
    B_core_test/
    B_extra_test/
    manifest.json
    metadata/<source_split>/
    exclusions/<source_split>/
```

The four datasets preserve the original column order and Arrow schema. Only
`text` is replaced. Each retained source row produces one output, with the same
`id`, `split`, `label`, and other original fields. Metadata is stored separately
and aligns row-for-row with the corresponding output dataset. It records the
source row index, slot/payload/email IDs, template index, sampled form, requested
and effective insertion positions, retained email word count, output token
count, and whether a payload is actually present. Template indices refer to the
ordered held-out template list saved in `manifest.json`.

Eligibility uses the complete payload and **held-out** templates. Empty or
over-budget payloads and payloads that fit no held-out template are excluded
with their original row indices, IDs, labels, and reasons. In addition, an
email-bearing form that cannot retain one complete email word is excluded,
rather than exported as a background-free example. This extra export check
does not change training augmentation. There is no retry or relabeling that
could silently change a rejected sample's form. The manifest records input,
output, and exclusion counts, achieved form/position counts, source and email
fingerprints, exact templates and their file hashes, tokenizer identity, seed,
token budget, weights, and the resolved configuration.

The exporter does not impose a new benign:malicious ratio. It retains source
class counts except for the documented exclusions. Both the class ratio and
form frequencies can therefore shift when exclusions occur; use the manifest
counts when reporting evaluation results. The existing inference pipeline can
load the four datasets directly with `load_from_disk()` and tokenize `text`.
This held-out mixture differs from the distribution used for the training
script's internal calibration; the exporter does not fit or change thresholds.

The default output root is `<data.split_dir>_dilute`, so the current config
writes to `data_split3_dilute/42/`. Existing results for the selected seed are
replaced by default, without an overwrite flag. Other seed directories remain
unchanged. The previous seed directory and root README, when present, are kept
under `data_split3_dilute/.backups/<seed>-<unique-id>/`; the backup path is printed.

Intermediate Arrow files are written to a temporary workspace outside the
original source datasets. Replacement starts only when all four splits have
been saved and their schemas/counts reloaded successfully. A generation failure
leaves existing results intact; a publication failure attempts to restore the
previous seed directory. A short publication lock prevents concurrent swaps.
The original input directory cannot be used as the output directory. A dry run
never replaces data. Use `--output-dir` when a separate output root is desired.

Previously cached inference scores do not update when this dataset is replaced.
Use a fresh inference output directory to evaluate the new samples; this
generator does not delete or modify existing inference results.

On shared/NFS storage, the exporter releases Arrow dataset views before cleaning
temporary files and retries transient cleanup failures. If cleanup still fails,
it prints a warning with the temporary path; successfully published data remains
valid. Generation/publication errors still fail the run and are not hidden by
cleanup errors.

Run the offline exporter tests, including CLI dry-run/export checks with a
small fixture email pool and a local tokenizer:

```bash
python -m unittest discover -s tests -p 'test_materialize_email_diluted_test_splits.py' -v
```

## Running the experiment

Use the project's existing training environment and run from the repository
root. The standalone email script defaults to
`configs/training/ft_benign_exposure_mmbert2_email.yaml`. The original training
script and its template-only configs remain unchanged. Email configurations
inherit the existing FT/PEFT settings via `base_config`
and use separate output directories to avoid restoring old HPO experiments.

Full fine-tuning using the existing FT hyperparameters:

```bash
python train_benign_exposure_mmbert2_dilute_email.py \
  configs/training/ft_benign_exposure_mmbert2_email.yaml
```

Full fine-tuning with a new hyperparameter search:

```bash
python train_benign_exposure_mmbert2_dilute_email.py \
  configs/training/ft_benign_exposure_mmbert2_email.yaml hpo.skip=false
```

LoRA/QAT with a new hyperparameter search:

```bash
python train_benign_exposure_mmbert2_dilute_email.py \
  configs/training/peft_benign_exposure_mmbert2_email.yaml
```

For a controlled LoRA comparison using an existing PEFT HPO checkpoint:

```bash
python train_benign_exposure_mmbert2_dilute_email.py \
  configs/training/peft_benign_exposure_mmbert2_email.yaml \
  hpo.skip=true \
  hpo.checkpoint_path=benign_exposure_peft_mmbert2_template_seed42/checkpoints/hpo_complete.json
```

This reuses hyperparameters, not trained weights. The final model is trained and
its probabilities and threshold are recalibrated. FT and PEFT checkpoints remain
subject to the existing training-mode compatibility checks. Use a new
`ray.output_dir` after changing email/template settings; incompatible manifests
are rejected rather than silently resuming another experiment.

For the original experiment, run the unchanged
`train_benign_exposure_mmbert2_dilute_template.py` with an original template
config. The standalone email script also accepts `email.enabled=false` to use
its template-only fallback and raw calibration/test behavior.

## Evaluation after dataset generation

Run from the repository root after training and exporting the dataset.

1. **Calibration:** a completed email training run already saves
   `calibration.json`, `calibration_metrics.json`, and internal `test_metrics.json`.
   Exporting a new test dataset does not require recalibration. The existing
   standalone recalibration scripts do not implement the email mixture; do not
   use them unchanged. Never fit calibration or select deployment thresholds on
   either test dataset.
2. **Batch inference and PR:** run both commands below. The internal training
   test uses the six-form mixture and training templates, so it does not replace
   these original and held-out diluted evaluations.

```bash
EMAIL_MODEL_DIR="benign_exposure_peft_mmbert2_email_seed42/best_model/jhu-clsp_mmBERT-small"

PR_MODEL_PATH="$EMAIL_MODEL_DIR" \
PR_DATA_SPLIT_DIR=data_split3/42 \
PR_OUTPUT_DIR=inference_results_email_original_v1 \
PR_CURVE_PATH=pr_curve_email_original_v1.png \
python plot_pr_curve_mmbert2.py --training-mode peft

PR_MODEL_PATH="$EMAIL_MODEL_DIR" \
PR_DATA_SPLIT_DIR=data_split3_dilute/42 \
PR_OUTPUT_DIR=inference_results_email_diluted_v1 \
PR_CURVE_PATH=pr_curve_email_diluted_v1.png \
python plot_pr_curve_mmbert2.py --training-mode peft
```

Each command saves predictions and plots PR; no separate batch `predict` pass
is needed. Existing result directories are reused without rerunning inference.
Use fresh names (for example, `v2`) after changing the dataset, model, or calibrator.

3. **Fixed-threshold comparison:** use the same model, `calibration.json`, and
   its threshold for both datasets. Report precision, recall, F1, FPR/FNR, and
   confusion counts alongside PR/AP. This fixed-threshold summary still needs
   separate computation from the saved scores: the PR script's marked thresholds
   are selected on each test curve, not taken from calibration. Its test ratio
   is 1000 benign per malicious, versus the training config's internal 500:1;
   precision is not directly comparable across those ratios.
4. **Optional manual prediction:** score a complete multiline email as one input:

```bash
export PR_MODEL_PATH="benign_exposure_peft_mmbert2_email_seed42/best_model/jhu-clsp_mmBERT-small"
python predict_prompt.py --training-mode peft --file sample_email.txt --single
```

Change `PR_MODEL_PATH` to switch models; `unset PR_MODEL_PATH` restores the
original default. Selection priority is `--model-path` > nonempty `PR_MODEL_PATH`
> the FT/PEFT default. Keep `--training-mode peft` for either PEFT model.
Use your own `sample_email.txt`. The prediction script prefers
`calibration_dilute.json` when present, whereas the PR script reads
`calibration.json`; ensure both use the intended calibration before comparing
outputs. A new email training run writes `calibration.json`.

## Verification

Run the offline regression suite (no model weights or Enron download required):

```bash
python -m unittest discover -s tests -p 'test_email_augmentation.py' -v
```

The suite checks the six forms and label budgets, approximate form and position
frequencies, split isolation without deduplication, deterministic evaluation,
dynamic training, whole-word insertion, special-token-inclusive truncation,
template suffix preservation, over-budget payload filtering, fixed-view cache
round-tripping, and integration with the existing sampling/calibration helpers.
