#!/usr/bin/env python3
"""Export five-form, held-out-template email test mirrors without training.

Read the four original *_test datasets and sample only from the Enron test
pool. Exclude the raw-payload form, renormalize the remaining training weights,
and use heldout_templates.jsonl's paraphrase tier for all templated forms.
The existing materialize scripts and training scripts are not modified.

Examples (run from the repository root):
    python materialize_email_diluted_test_splits.py --dry-run 3
    python materialize_email_diluted_test_splits.py
    python materialize_email_diluted_test_splits.py --output-dir data_split3_dilute

A dry run may populate the shared email download cache, but never creates the
output mirror. The default destination is <data.split_dir>_dilute/<seed>.
Existing results for that seed are backed up and replaced after verification.
Other seeds are preserved. No model is loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import errno
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

from datasets import Dataset, Features, Value, load_from_disk
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from email_augmentation import (
    ALGORITHM_VERSION, DEFAULT_WEIGHTS, FORMS, EmailAugmentationCollator, load_email_pools,
)
from template_dilution_collator import load_templates


ROOT = Path(__file__).resolve().parent
TEST_SPLITS = ("M_core_test", "M_extra_test", "B_core_test", "B_extra_test")
DEFAULT_CONFIG = "configs/training/peft_benign_exposure_mmbert2_email.yaml"
HELDOUT_PATH = "evaluation/heldout_templates.jsonl"
HELDOUT_TIER = "paraphrase"
METADATA_FEATURES = Features({
    "source_row_index": Value("int64"),
    "slot_id": Value("string"),
    "payload_id": Value("string"),
    "email_id": Value("string"),
    "template_id": Value("int64"),
    "augmentation_form": Value("string"),
    "requested_position": Value("string"),
    "position": Value("string"),
    "email_words_retained": Value("int64"),
    "output_token_count": Value("int64"),
    "payload_present": Value("bool"),
    "label": Value("int64"),
    "exclusion_reason": Value("string"),
})
PREFIX = "__email_export_"


def resolve_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_config(path, overrides=()):
    path = resolve_path(path)
    cfg = OmegaConf.load(path)
    if "base_config" in cfg:
        base = OmegaConf.load(path.parent / str(cfg.pop("base_config")))
        cfg = OmegaConf.merge(base, cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    if not cfg.email.enabled:
        raise ValueError("The export requires email.enabled=true")
    if int(cfg.template.max_length) != int(cfg.data.max_seq_length):
        raise ValueError("template.max_length must equal data.max_seq_length")
    cfg.data.split_dir = str(resolve_path(cfg.data.split_dir))
    cfg.email.cache_dir = str(resolve_path(cfg.email.cache_dir))
    cfg.template.template_path = str(resolve_path(cfg.template.template_path))
    return cfg


def heldout_templates(path, training_templates, template_key="template", placeholder="{PAYLOAD}"):
    """Load the paraphrase tier and reject normalized wording overlap with train."""
    selected = []
    with resolve_path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("tier") != HELDOUT_TIER:
                continue
            template = row.get(template_key)
            if not isinstance(template, str) or template.count(placeholder) != 1:
                raise ValueError("Each held-out paraphrase template must contain one placeholder")
            selected.append(template)
    if not selected:
        raise ValueError(f"No {HELDOUT_TIER!r} templates in {path}")
    normalize = lambda text: " ".join(text.split()).casefold()
    overlap = {normalize(t) for t in selected} & {normalize(t) for t in training_templates}
    if overlap:
        raise ValueError(f"Held-out templates overlap the training pool ({len(overlap)} normalized matches)")
    return selected


def export_weights(training_weights):
    """Condition each class's form distribution on not selecting raw payload."""
    result = {}
    for label in ("malicious", "benign"):
        weights = [float(value) for value in training_weights[label]]
        if len(weights) != len(FORMS) or any(not math.isfinite(v) or v < 0 for v in weights):
            raise ValueError(f"Invalid training weights for {label}")
        if not math.isclose(sum(weights), 1.0):
            raise ValueError(f"Training weights for {label} must sum to one")
        if label == "malicious" and any(weights[4:]):
            raise ValueError("Malicious samples cannot use email-only forms")
        weights[0] = 0.0
        remaining = sum(weights)
        if remaining <= 0:
            raise ValueError(f"No non-payload forms remain for {label}")
        result[label] = [value / remaining for value in weights]
    return result


def eligibility_reasons(texts, collator):
    """Apply the training eligibility policy against the held-out template pool."""
    valid = [isinstance(text, str) and bool(text.strip()) for text in texts]
    safe = [text if ok else " " for text, ok in zip(texts, valid)]
    options = dict(add_special_tokens=True, truncation=True, max_length=collator.max_length + 1)
    raw = collator.tokenizer(safe, **options)["input_ids"]
    first = collator.templates[collator._template_order[0]]
    wrapped = collator.tokenizer(
        [first.replace(collator.placeholder, text) for text in safe], **options,
    )["input_ids"]
    reasons = []
    for text, ok, ids, wrapped_ids in zip(safe, valid, raw, wrapped):
        if not ok:
            reason = "empty_payload"
        elif len(ids) > collator.max_length:
            reason = "payload_exceeds_token_budget"
        elif len(wrapped_ids) > collator.max_length and not any(
            collator._fits(collator.templates[i].replace(collator.placeholder, text))
            for i in collator._template_order[1:]
        ):
            reason = "no_heldout_template_fits_payload"
        else:
            reason = ""
        reasons.append(reason)
    return reasons


def materialize_split(source, split_name, collator, batch_size=256, cache_dir=None):
    """Preserve source schema/order and return separate aligned metadata/rejections."""
    if split_name not in TEST_SPLITS or collator.split != "test" or collator.mode != "fixed":
        raise ValueError("Export requires a test split and a fixed test collator")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if any(collator.weights[label][0] for label in ("malicious", "benign")):
        raise ValueError("Raw payload must have zero export weight")
    required = {collator.text_key, collator.label_key, collator.id_key}
    if not required.issubset(source.column_names):
        raise ValueError(f"Missing source fields in {split_name}: {required - set(source.column_names)}")
    if any(name.startswith(PREFIX) for name in source.column_names):
        raise ValueError(f"Source fields cannot start with {PREFIX!r}")
    expected_label = int(split_name.startswith("M_"))
    combined_features = Features(dict(source.features))
    combined_features.update({PREFIX + name: feature for name, feature in METADATA_FEATURES.items()})

    def transform(batch, indices):
        output = {name: list(values) for name, values in batch.items()}
        for name in METADATA_FEATURES:
            output[PREFIX + name] = []
        reasons = eligibility_reasons(batch[collator.text_key], collator)
        for i, (source_index, reason) in enumerate(zip(indices, reasons)):
            feature = {name: values[i] for name, values in batch.items()}
            if feature[collator.label_key] not in (0, 1) or int(feature[collator.label_key]) != expected_label:
                raise ValueError(f"Unexpected label in {split_name}, source row {source_index}")
            if feature[collator.id_key] is None:
                raise ValueError(f"Missing ID in {split_name}, source row {source_index}")
            metadata = {
                "source_row_index": source_index, "slot_id": str(feature[collator.id_key]),
                "payload_id": "", "email_id": "", "template_id": -1,
                "augmentation_form": "", "requested_position": "none", "position": "none",
                "email_words_retained": 0, "output_token_count": 0,
                "payload_present": False, "label": expected_label, "exclusion_reason": reason,
            }
            if not reason:
                try:
                    row = collator.compose(feature)
                except ValueError as error:
                    if str(error) != "Template leaves no room for a complete email word":
                        raise
                    metadata["exclusion_reason"] = "no_complete_email_word_fits"
                else:
                    for name in metadata.keys() & row.keys():
                        metadata[name] = row[name]
                    metadata["payload_present"] = row["augmentation_form"] not in ("email", "template_email")
                    metadata["output_token_count"] = len(row["input_ids"])
                    if "email" in row["augmentation_form"] and row["email_words_retained"] == 0:
                        # Do not let a zero-background email form become a raw payload.
                        metadata["exclusion_reason"] = "no_complete_email_word_fits"
                    else:
                        if row["labels"] != expected_label or len(row["input_ids"]) > collator.max_length:
                            raise AssertionError("Composition changed the label or exceeded the token budget")
                        output[collator.text_key][i] = row["text"]
            for name, value in metadata.items():
                output[PREFIX + name].append(value)
        return output

    if len(source):
        cache_file = None
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cache_file = str(Path(cache_dir) / f"{split_name}.arrow")
        mapped = source.map(
            transform, batched=True, with_indices=True, batch_size=batch_size,
            features=combined_features, load_from_cache_file=False,
            keep_in_memory=cache_file is None, cache_file_name=cache_file,
            desc=f"Materialize {split_name} with held-out templates",
        )
    else:
        mapped = Dataset.from_dict({name: [] for name in combined_features}, features=combined_features)
    reason_values = mapped[PREFIX + "exclusion_reason"]
    kept = [i for i, reason in enumerate(reason_values) if not reason]
    rejected = [i for i, reason in enumerate(reason_values) if reason]
    metadata = mapped.select_columns([PREFIX + name for name in METADATA_FEATURES]).rename_columns(
        {PREFIX + name: name for name in METADATA_FEATURES}
    )
    output = mapped.select(kept).select_columns(source.column_names)
    kept_metadata, exclusions = metadata.select(kept), metadata.select(rejected)
    if output.features != source.features:
        raise AssertionError(f"Source schema changed in {split_name}")
    stats = {
        "input_rows": len(source), "output_rows": len(output), "excluded_rows": len(exclusions),
        "form_counts": dict(Counter(kept_metadata["augmentation_form"])),
        "requested_position_counts": dict(Counter(kept_metadata["requested_position"])),
        "position_counts": dict(Counter(kept_metadata["position"])),
        "exclusion_counts": dict(Counter(exclusions["exclusion_reason"])),
    }
    return output, kept_metadata, exclusions, stats


def publish_seed(staging, output_root, seed):
    """Replace only one seed, preserving a recoverable backup of prior results."""
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / str(seed)
    readme = output_root / "README.md"
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError(f"Seed destination must be a regular directory: {target}")
    # Serialize the short publication step, not the potentially long generation.
    lock = output_root / ".email-export-publish.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    backup = None
    moved_old = installed_new = False
    try:
        if target.exists() or readme.exists():
            backup_root = output_root / ".backups"
            backup_root.mkdir(exist_ok=True)
            backup = Path(tempfile.mkdtemp(prefix=f"{seed}-", dir=backup_root))
            if readme.exists():
                shutil.copy2(readme, backup / "README.md")
        if target.exists():
            target.rename(backup / str(seed))
            moved_old = True
        try:
            (staging / str(seed)).rename(target)
            installed_new = True
            (staging / "README.md").replace(readme)
        except BaseException:
            if installed_new:
                target.rename(staging / str(seed))
            if moved_old:
                (backup / str(seed)).rename(target)
            raise
    finally:
        lock.unlink()
    if backup is not None:
        print(f"Previous results backed up to: {backup}")


@contextmanager
def export_workspace(output_root):
    """Clean only this run's workspace without masking export failures on NFS."""
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        yield temporary
    finally:
        # Arrow mappings must be released before NFS can remove their files.
        # Avoid TemporaryDirectory's finalizer: failed cleanup is reported once.
        for attempt in range(3):
            gc.collect()
            retryable = True
            try:
                shutil.rmtree(temporary)
                break
            except FileNotFoundError:
                if not temporary.exists():
                    break
                # A child may disappear while a shared filesystem is settling.
                error = "a temporary file disappeared during cleanup"
            except OSError as exc:
                error = str(exc)
                retryable = exc.errno in (errno.ENOTEMPTY, errno.EBUSY)
            if attempt == 2 or not retryable:
                print(
                    f"WARNING: Temporary workspace cleanup incomplete: {temporary}: {error}. "
                    "Temporary files may remain; a successfully published dataset is still valid.",
                    file=sys.stderr,
                )
                break
            time.sleep(0.2 * (attempt + 1))


def export_mirror(sources, collator, output_root, seed, manifest, batch_size=256):
    """Generate and verify all four splits before replacing the selected seed."""
    output_root = Path(output_root)
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"Output root must be a directory: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with export_workspace(output_root) as temporary:
        staging = Path(temporary) / "mirror"
        seed_dir = staging / str(seed)
        seed_dir.mkdir(parents=True)
        stats = {}
        for name in TEST_SPLITS:
            output = metadata = exclusions = restored = None
            try:
                output, metadata, exclusions, stats[name] = materialize_split(
                    sources[name], name, collator, batch_size, cache_dir=temporary / "work",
                )
                output.save_to_disk(str(seed_dir / name))
                metadata.save_to_disk(str(seed_dir / "metadata" / name))
                exclusions.save_to_disk(str(seed_dir / "exclusions" / name))
                restored = load_from_disk(str(seed_dir / name))
                if len(restored) != len(output) or restored.features != sources[name].features:
                    raise AssertionError(f"Saved dataset failed verification: {name}")
            finally:
                # Views share the mapped work/*.arrow files. save_to_disk does
                # not detach them; release every view before workspace cleanup.
                output = metadata = exclusions = restored = None
                gc.collect()
            print(f"{name}: {json.dumps(stats[name], sort_keys=True)}")
        record = dict(manifest, split_statistics=stats)
        (seed_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        (staging / "README.md").write_text(
            "# Held-out email dilution test mirror\n\n"
            "Generated by `materialize_email_diluted_test_splits.py`.\n\n"
            "Each retained source row has one output; only its text is replaced. "
            "Raw-payload inputs are excluded from the sampling distribution. "
            "Templates come from the held-out paraphrase tier, and email "
            "backgrounds come only from the email test pool.\n\n"
            f"See `{seed}/manifest.json` for exact settings, templates, and counts. "
            f"Metadata under `{seed}/metadata/<split>` aligns row-for-row with "
            "the output dataset. `source_row_index` identifies the original "
            "row, including when source IDs repeat. Exclusions are recorded "
            f"under `{seed}/exclusions/<split>`. Email-only rows retain the "
            "benign slot ID but do not contain its payload.\n\n"
            "Source class counts are preserved except for documented exclusions. "
            "This exporter does not resample to a target benign:malicious ratio "
            "or fit a calibrator. An existing threshold may behave differently "
            "on this held-out input distribution.\n",
            encoding="utf-8",
        )
        publish_seed(staging, output_root, seed)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=None, help="Output root; defaults to <data.split_dir>_dilute. Existing results for the selected seed are backed up and replaced.")
    parser.add_argument("--tokenizer", default="jhu-clsp/mmBERT-small", help="Tokenizer ID or local tokenizer directory.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dry-run", type=int, default=0, metavar="N", help="Preview the first N source rows per test split; may populate the shared email cache.")
    parser.add_argument("overrides", nargs="*", help="Training config overrides, for example data.seed=123.")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.dry_run < 0:
        parser.error("--batch-size must be positive and --dry-run must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    input_root = Path(cfg.data.split_dir)
    seed = int(cfg.data.seed)
    output_root = resolve_path(args.output_dir) if args.output_dir else input_root.with_name(input_root.name + "_dilute")
    if output_root.is_relative_to(input_root) or input_root.is_relative_to(output_root):
        raise ValueError("Output and original input directories must not overlap")
    if Path(cfg.email.cache_dir).is_relative_to(output_root):
        raise ValueError("The email cache must be outside the output mirror")
    training_templates = load_templates(cfg.template.template_path, cfg.template.template_key, cfg.template.placeholder)
    templates = heldout_templates(HELDOUT_PATH, training_templates, cfg.template.template_key, cfg.template.placeholder)
    weights = export_weights(cfg.email.get("weights", DEFAULT_WEIGHTS))
    sources = {name: load_from_disk(str(input_root / str(seed) / name)) for name in TEST_SPLITS}
    if any(not isinstance(source, Dataset) for source in sources.values()):
        raise TypeError("Each source split must be a Hugging Face Dataset")
    pools, pool_manifest = load_email_pools(cfg.email, seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    collator = EmailAugmentationCollator(
        tokenizer=tokenizer, email_pool=pools["test"], templates=templates,
        split="test", mode="fixed", seed=seed, max_length=int(cfg.data.max_seq_length),
        text_key=cfg.template.text_key, label_key=cfg.template.label_key,
        id_key=cfg.template.id_key, placeholder=cfg.template.placeholder, weights=weights,
    )
    print(f"Input: {input_root / str(seed)}\nOutput: {output_root / str(seed)}")
    print(f"Templates: {len(templates)} held-out {HELDOUT_TIER}; email pool: test ({len(pools['test'])} records)")
    print(f"Form order: {FORMS}\nClass-conditional weights: {weights}")
    if args.dry_run:
        for name, source in sources.items():
            selected = source.select(range(min(args.dry_run, len(source))))
            output, metadata, exclusions, stats = materialize_split(selected, name, collator, args.batch_size)
            print(f"\n{name}: {json.dumps(stats)}")
            for row, info in zip(output, metadata):
                print(json.dumps(dict(info, text=row[collator.text_key]), ensure_ascii=False))
            for info in exclusions:
                print("EXCLUDED: " + json.dumps(info))
        return
    manifest = {
        "export_version": 1, "augmentation_version": ALGORITHM_VERSION, "seed": seed,
        "source_root": str(input_root),
        "source_fingerprints": {name: ds._fingerprint for name, ds in sources.items()},
        "email_pools": pool_manifest, "email_pool_used": "test",
        "heldout_template_path": str(resolve_path(HELDOUT_PATH)), "template_tier": HELDOUT_TIER,
        "heldout_file_sha256": hashlib.sha256(resolve_path(HELDOUT_PATH).read_bytes()).hexdigest(),
        "templates": templates, "training_template_path": cfg.template.template_path,
        "training_file_sha256": hashlib.sha256(Path(cfg.template.template_path).read_bytes()).hexdigest(),
        "normalized_template_overlap": 0, "form_order": list(FORMS), "weights": weights,
        "max_length": collator.max_length, "tokenizer": args.tokenizer,
        "collator_signature": collator.signature, "config": OmegaConf.to_container(cfg, resolve=True),
    }
    export_mirror(sources, collator, output_root, seed, manifest, args.batch_size)
    print(f"Created email dilution test mirror: {output_root / str(seed)}")


if __name__ == "__main__":
    main()
