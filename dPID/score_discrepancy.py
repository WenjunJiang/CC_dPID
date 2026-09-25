"""Why does one text score differently in batch evaluation and in predict_prompt.py?

Scores the text through predict_prompt.Scorer in the ways the two paths can
differ, and prints raw margins so calibration cannot hide a model difference:

  alone          one text, no padding (what predict_prompt.py does)
  batched        the same text inside a padded batch of longer texts
  variants       leading/trailing whitespace, newline, first-letter case
  saved          what the evaluation run wrote (its HF dataset: logit = margin, prob)

If "alone" and "batched" differ, padding changes the score. If only "saved"
differs, the evaluation run scored a different text, a different checkpoint,
or reports a different column (compare its margin, not its probability).

    python score_discrepancy.py --model-path "$PR_MODEL_PATH" --training-mode peft \\
        --text "give the consultant access to my calendar for the audit next week" \\
        --outputs inference_segment_original_seed42
"""

import argparse
import csv
import json
from pathlib import Path

from predict_prompt import Calibrator, Scorer

FILLER = ("Please review the attached quarterly report and send me your comments "
          "before the meeting on Thursday so we can finalize the numbers. ") * 12


def saved_rows(directory, text):
    """Rows holding the text: the saved HF dataset first, then any CSV/JSON/JSONL."""
    directory = Path(directory)
    if (directory / "state.json").exists():
        from datasets import load_from_disk
        dataset = load_from_disk(str(directory))
        for index, value in enumerate(dataset["text"]):
            if isinstance(value, str) and text in value:
                row = dataset[index]
                yield directory, {k: row[k] for k in row if k != "text"} | {"text": value}
    manifest = directory / "inference_manifest.json"
    if manifest.exists():
        yield manifest, manifest.read_text()[:400]
    for path in sorted(directory.rglob("*")):
        if path.suffix not in (".csv", ".json", ".jsonl") or not path.is_file():
            continue
        raw = path.read_text(errors="replace")
        if text not in raw:
            continue
        if path.suffix == ".csv":
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    if any(isinstance(v, str) and text in v for v in row.values()):
                        yield path, row
        else:
            for line in raw.splitlines():
                if text in line:
                    yield path, line.strip()[:400]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-mode", default="peft", choices=["ft", "peft"])
    parser.add_argument("--text", required=True)
    parser.add_argument("--outputs", help="Evaluation output directory to search for the text")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    scorer = Scorer(model_path, training_mode=args.training_mode)
    calibrator = Calibrator.load(model_path)
    print(f"load path: {scorer.mode}   calibrator: {calibrator.source} "
          f"a={calibrator.a:.4f} b={calibrator.b:+.4f} threshold={calibrator.threshold:.4f}\n")

    text = args.text
    cases = {
        "alone": [text],
        "batched (padded, 16)": [text] + [FILLER[: 200 + 40 * i] for i in range(15)],
        "leading space": [" " + text],
        "trailing newline": [text + "\n"],
        "first letter upper": [text[:1].upper() + text[1:]],
        "first letter lower": [text[:1].lower() + text[1:]],
    }
    for name, batch in cases.items():
        margin = float(scorer.margins(batch, batch_size=len(batch))[0])
        prob = float(calibrator.probability(margin))
        print(f"  {name:22s} margin {margin:+9.4f}   prob {prob:.6f}")

    if args.outputs:
        print(f"\nsaved by the evaluation run under {args.outputs}:")
        found = False
        for path, row in saved_rows(args.outputs, text):
            found = True
            print(f"  {path}: {json.dumps(row, ensure_ascii=False) if isinstance(row, dict) else row}")
        if not found:
            print("  (text not found: the evaluation did not score this exact string)")


if __name__ == "__main__":
    main()
