"""Train the token-only email model with 20 HPO trials by default.

Run from the repository root:
    python train_benign_exposure_mmbert2_dilute_email_segment.py
    python train_benign_exposure_mmbert2_dilute_email_segment.py hpo.num_samples=5
    python train_benign_exposure_mmbert2_dilute_email_segment.py --final-from-current --dry-run
    python train_benign_exposure_mmbert2_dilute_email_segment.py --final-from-current

The shared training loop keeps the legacy email and template entry points intact.
"""

import sys


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--final-from-current":
        from interim_segment_training import main
        main(sys.argv[2:])
    else:
        from train_benign_exposure_mmbert2_dilute_email import main
        if len(sys.argv) == 1 or not sys.argv[1].endswith((".yaml", ".yml")):
            sys.argv.insert(1, "configs/training/peft_benign_exposure_mmbert2_email_segment.yaml")
        main()
