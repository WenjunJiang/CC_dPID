"""Read a running Ray experiment snapshot and train an isolated final model.

No Ray restore, fit, pickle loading, or writes to the source experiment occur.
Only TERMINATED trials reaching their configured epoch budget are eligible.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid

from omegaconf import OmegaConf


DEFAULT_HPO_DIR = "benign_exposure_peft_mmbert2_email_segment_seed42"
SCORING_MODE = "token_max_subarray_v1"


def read_experiment_snapshot(source):
    """Read the newest complete JSON snapshot, tolerating an in-progress write.

    Metadata and statuses come from the same snapshot, not independently read
    trial logs. A just-finished trial may wait for Ray's next state flush.
    Serialized storage objects are deliberately never unpickled.
    """
    experiment = source / "benign_exposure_hpo"
    files = sorted(experiment.glob("experiment_state-*.json"),
                   key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
    for path in files:
        try:
            raw = path.read_bytes()
            state = json.loads(raw)
            records = []
            for entry in state["trial_data"]:
                trial, metadata = entry
                trial = json.loads(trial) if isinstance(trial, str) else trial
                metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
                if not isinstance(trial, dict) or not isinstance(metadata, dict):
                    raise ValueError("Invalid trial record")
                records.append((trial, metadata))
            return records, path, hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise ValueError(f"No readable Ray experiment snapshot in {experiment}; wait for Ray to flush its state")


def select_current_best(source):
    """Return source config and a reusable parameter snapshot without writing."""
    source = Path(source).resolve()
    config_path = source / "email_training_config.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    if (cfg.training.get("mode") != "peft" or not cfg.training.get("segment_scoring", False)
            or not cfg.email.get("region_supervision", False)):
        raise ValueError("The source must be a PEFT segment-scoring email run")
    if cfg.hpo.metric != "val_f1":
        raise ValueError("Interim selection currently requires hpo.metric=val_f1")
    records, state_path, digest = read_experiment_snapshot(source)
    required = set(cfg.training.search_space) | {"model_name", "seed", "segment_tau",
        "region_loss_weight", "malicious_top_k", "benign_top_k", "effective_batch_size",
        "token_head_dropout", "num_train_epochs", "lora_r", "lora_alpha", "lora_dropout"}
    candidates, skipped = [], Counter()
    for trial, metadata in records:
        if trial.get("status") != "TERMINATED":
            skipped[trial.get("status", "unknown_status")] += 1
            continue
        result = metadata.get("last_result", {})
        if not isinstance(result, dict) or not isinstance(trial.get("config"), dict):
            skipped["invalid_result_or_config"] += 1
            continue
        if (metadata.get("error_filename") or metadata.get("pickled_error_filename")
                or metadata.get("num_failures", 0) or result.get("error")):
            skipped["failed"] += 1
            continue
        hp = trial.get("config", {})
        try:
            if not required <= hp.keys() or int(hp["seed"]) != int(cfg.data.seed):
                raise ValueError("Missing parameters or mismatched seed")
            epochs, epoch, score = float(hp["num_train_epochs"]), float(result["epoch"]), float(result["val_f1"])
            if not all(math.isfinite(v) for v in (epochs, epoch, score)) or epochs <= 0 or not 0 <= score <= 1:
                raise ValueError("Invalid metric or epoch")
            if epoch + 1e-6 < epochs:
                skipped["early_stopped"] += 1
                continue
            if result.get("config", hp) != hp or result.get("trial_id", trial["trial_id"]) != trial["trial_id"]:
                raise ValueError("Result and trial identity disagree")
        except (KeyError, ValueError, TypeError, OverflowError):
            skipped["invalid_result_or_config"] += 1
            continue
        candidates.append((score, str(trial["trial_id"]), hp, result))
    if not candidates:
        raise ValueError(f"No fully completed valid trials yet (excluded: {dict(skipped)}). "
                         "Wait for a full epoch budget and Ray's next state flush, then retry.")
    # Last reported F1, not the historical peak. Trial ID breaks exact ties.
    score, trial_id, hp, result = sorted(candidates, key=lambda row: (-row[0], row[1]))[0]
    metrics = {key: value for key, value in result.items()
               if isinstance(value, (int, float)) and math.isfinite(value)}
    snapshot = dict(training_mode="peft", scoring_mode=SCORING_MODE, interim=True,
        best_hp=hp, best_metrics=metrics, source_trial_id=trial_id,
        source_hpo_dir=str(source), source_state_file=str(state_path),
        source_state_sha256=digest, selected_at=datetime.now(timezone.utc).isoformat(),
        selection="highest last val_f1 among TERMINATED trials reaching num_train_epochs",
        eligible_trials=len(candidates), excluded_trials=dict(skipped))
    return cfg, snapshot


def prepare_interim_run(source, output, cfg, snapshot):
    """Create a new, non-overlapping output directory and freeze the selection."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Interim output must be separate from, and not contain, the source HPO directory")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite interim output: {output}")
    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cache = Path(str(run_cfg.email.cache_dir)).resolve()
    if cache == source or source in cache.parents:
        raise ValueError("The source email cache is inside the HPO directory; use an external shared cache")
    run_cfg.ray.output_dir = str(output)
    run_cfg.email.artifacts_dir = str(output / "email_augmentation")
    run_cfg.hpo.skip = True
    run_cfg.hpo.checkpoint_path = str(output / "checkpoints" / "hpo_current_best.json")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(run_cfg.hpo.checkpoint_path)
    checkpoint.parent.mkdir()
    checkpoint.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    OmegaConf.save(cfg, output / "source_training_config.yaml")
    OmegaConf.save(run_cfg, output / "email_training_config.yaml")
    return run_cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo-dir", default=DEFAULT_HPO_DIR,
                        help="Source run root containing email_training_config.yaml and benign_exposure_hpo/")
    parser.add_argument("--output-dir", help="New output directory; existing paths are rejected")
    parser.add_argument("--dry-run", action="store_true", help="Show current winner without writing or training")
    args = parser.parse_args(argv)
    try:
        source = Path(args.hpo_dir).expanduser().resolve()
        cfg, snapshot = select_current_best(source)
        print(json.dumps(snapshot, indent=2, allow_nan=False))
        if args.dry_run:
            return
        output = (Path(args.output_dir).expanduser() if args.output_dir else source.with_name(
            source.name + "_interim_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]))
        run_cfg = prepare_interim_run(source, output, cfg, snapshot)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Interim output: {run_cfg.ray.output_dir}")
    print("The source HPO remains untouched. Use a free GPU to avoid competing with its trials.")
    from train_benign_exposure_mmbert2_dilute_email import main as train_main
    train_main(config_override=run_cfg)


if __name__ == "__main__":
    main()
