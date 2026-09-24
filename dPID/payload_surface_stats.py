"""How strongly does a payload's surface form predict its label?

Reads the 16-file split directly and reports, per class, the share of payloads
by first-letter case, final punctuation, leading/trailing whitespace, and
internal line breaks. Any feature whose share differs a lot
between M_* and B_* is a shortcut the model can learn instead of the content.

    python payload_surface_stats.py data_split3 42
    python payload_surface_stats.py data_split3 42 --text-key text
"""

import argparse
from pathlib import Path

from datasets import load_from_disk

GROUPS = ("core", "extra")
PARTS = ("train", "val", "cal", "test")


def surface(text):
    letter = next((c for c in text if c.isalpha()), "")
    stripped = text.rstrip()
    return {
        "first_letter_lower": letter.islower(),
        "first_letter_upper": letter.isupper(),
        "no_final_punctuation": bool(stripped) and stripped[-1] not in ".!?。！？\"')]",
        "leading_whitespace": text[:1].isspace(),
        "trailing_whitespace": text[-1:].isspace(),
        "contains_newline": "\n" in text.strip(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split_dir")
    parser.add_argument("seed")
    parser.add_argument("--text-key", default="text")
    args = parser.parse_args()
    base = Path(args.split_dir) / args.seed
    for part in PARTS:
        shares = {}
        for cls in ("M", "B"):
            texts = [t for group in GROUPS
                     for t in load_from_disk(str(base / f"{cls}_{group}_{part}"))[args.text_key]
                     if isinstance(t, str) and t.strip()]
            flags = [surface(t) for t in texts]
            shares[cls] = (len(texts), {k: sum(f[k] for f in flags) / len(flags) for k in flags[0]})
        print(f"\n[{part}]  malicious n={shares['M'][0]}  benign n={shares['B'][0]}")
        for key in shares["M"][1]:
            m, b = shares["M"][1][key], shares["B"][1][key]
            print(f"  {key:22s} malicious {m:6.1%}   benign {b:6.1%}   gap {m - b:+6.1%}")


if __name__ == "__main__":
    main()
