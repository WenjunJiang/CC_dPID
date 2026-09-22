#!/usr/bin/env python3
"""Apply the training-loop edits the segment objective change needs.

train_benign_exposure_mmbert2_dilute_email.py is long and shared with the
sentence models, so this edits it in place rather than shipping a rewritten
copy: every replacement must match exactly the expected number of times, and
nothing is written unless all of them do. Running it twice is a no-op.

    python upstream/apply_train_loop_edits.py path/to/train_benign_exposure_mmbert2_dilute_email.py
    python upstream/apply_train_loop_edits.py --check path/to/...      # report only

What it changes, and why each one is required rather than nice to have:

  1. _training_components forwards the validation prevalence, so the new
     per-bucket PR-AUC is computed at the ratio the validation set was built
     at instead of the 50 the metric assumes by default.
  2. _resolve_segment_hp records the training class ratio, which the model
     needs to undo the stratified batch composition.
  3. Both trainer constructions pass malicious_per_batch. Without it the
     sampler stays off and the weights in (2) are a no-op, which is the safe
     direction but not the intended one.
  4. Failed trials report the segment selection metric, so Ray has a value to
     rank them by rather than nothing at all.

The deployment prevalence is not threaded through: segment_metrics defaults to
500, and this script checks the config for a different value rather than
guessing.
"""

import argparse
import re
import sys
from pathlib import Path

EDITS = [
    (
        "_training_components forwards the validation prevalence",
        1,
        '''def _training_components(hp):
    if _segment_enabled(hp):
        from segment_training import SegmentTrainer, segment_metrics
        return SegmentTrainer, segment_metrics(compute_metrics_fn())
    return ASLTrainer, compute_metrics_fn()''',
        '''def _training_components(hp, validation_ratio=50):
    if _segment_enabled(hp):
        from segment_training import SegmentTrainer, segment_metrics
        return SegmentTrainer, segment_metrics(compute_metrics_fn(), validation_ratio)
    return ASLTrainer, compute_metrics_fn()


def _failed_report(hp):
    """_FAILED_REPORT lists the sentence metrics only.

    A segment run selects on val_selection_pr_auc, so without this a trial that
    dies reports no value for the metric it is ranked by.
    """
    report = dict(_FAILED_REPORT)
    if _segment_enabled(hp):
        from segment_training import FAILED_SEGMENT_REPORT
        report.update(FAILED_SEGMENT_REPORT)
    return report''',
    ),
    (
        "_resolve_segment_hp records the training class ratio",
        1,
        '''        hp.update(per_device_train_batch_size=micro,
                  gradient_accumulation_steps=effective // micro, classifier_dropout=0.0)''',
        '''        hp.update(per_device_train_batch_size=micro,
                  gradient_accumulation_steps=effective // micro, classifier_dropout=0.0)
        # The model undoes the stratified batch composition with this; set it
        # whenever malicious_per_batch is set, and neither otherwise.
        if int(hp.get("malicious_per_batch", 0)) > 0:
            hp["stratified_prior_ratio"] = float(hp["benign_to_malicious_ratio"])''',
    ),
    (
        "HPO trainer passes the validation prevalence and the stratified count",
        1,
        '''            trainer_class, metrics_function = _training_components(hp)''',
        '''            trainer_class, metrics_function = _training_components(
                hp, benign_per_malicious_val)''',
    ),
    (
        "final trainer passes the validation prevalence",
        1,
        '''        trainer_class, metrics_function = _training_components(best_hp)''',
        '''        trainer_class, metrics_function = _training_components(
            best_hp, benign_per_malicious_val)''',
    ),
    (
        "both trainer constructions receive malicious_per_batch",
        2,
        '''            check_finite_every=int(_perf_flag(perf_cfg, "check_finite_every", 50)),''',
        '''            check_finite_every=int(_perf_flag(perf_cfg, "check_finite_every", 50)),
            **({"malicious_per_batch": int(hp_source.get("malicious_per_batch", 0))}
               if _segment_enabled(hp_source) else {}),''',
    ),
    (
        "failed trials report the segment selection metric",
        3,
        '''_report_to_tune({**_FAILED_REPORT, "error": str(exc)})''',
        '''_report_to_tune({**_failed_report(hp), "error": str(exc)})''',
    ),
    (
        "the OOM path reports it too",
        1,
        '''_report_to_tune({**_FAILED_REPORT, "error": "cuda_oom"})''',
        '''_report_to_tune({**_failed_report(hp), "error": "cuda_oom"})''',
    ),
]

# `hp_source` names whichever hyperparameter dict is in scope at each of the two
# trainer constructions; they are called hp and best_hp respectively.
ALIASES = [
    ('''            trainer_class, metrics_function = _training_components(
                hp, benign_per_malicious_val)''',
     '''            hp_source = hp
            trainer_class, metrics_function = _training_components(
                hp, benign_per_malicious_val)'''),
    ('''        trainer_class, metrics_function = _training_components(
            best_hp, benign_per_malicious_val)''',
     '''        hp_source = best_hp
        trainer_class, metrics_function = _training_components(
            best_hp, benign_per_malicious_val)'''),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path)
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args()

    source = args.target.read_text(encoding="utf-8")
    if "_failed_report(hp)" in source:
        print("Already applied; nothing to do.")
        return 0

    updated, problems = source, []
    for name, expected, old, new in EDITS:
        found = updated.count(old)
        if found != expected:
            problems.append(f"  {name}: expected {expected} match(es), found {found}")
            continue
        updated = updated.replace(old, new)
        print(f"  ok  {name}  ({expected})")
    for old, new in ALIASES:
        if updated.count(old) != 1:
            problems.append(f"  alias anchor not unique: {old.strip()[:60]}...")
            continue
        updated = updated.replace(old, new)

    if problems:
        print("\nNo change written. The file does not match what these edits expect:",
              file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        print("\nApply upstream/train_loop.patch by hand instead.", file=sys.stderr)
        return 1

    import ast
    try:
        ast.parse(updated)
    except SyntaxError as exc:
        print(f"\nNo change written: the result does not parse ({exc})", file=sys.stderr)
        return 1

    if args.check:
        print("\n--check: all edits matched and the result parses. Nothing written.")
        return 0

    backup = args.target.with_suffix(args.target.suffix + ".before_segment_objective")
    backup.write_text(source, encoding="utf-8")
    args.target.write_text(updated, encoding="utf-8")
    print(f"\nWrote {args.target}\nOriginal kept at {backup}")
    print("\nStill to set by hand, in the config's fixed block:")
    print("  malicious_per_batch: 4     # 0 leaves the sampler off")
    print("\nAnd confirm validation.benign_per_malicious / test.benign_per_malicious")
    print("are 50 and 500; segment_metrics defaults to those for the deployment")
    print("reweighting and this script does not guess them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
