"""Does a model trained only on paragraph-separated payloads miss inline ones?

Two arms train the same model on the same data, both with the first-letter
redraw and edge stripping on, and differ only in `insertion_separators`:
"paragraph" keeps the fixed "\\n\\n", "mixed" draws "\\n\\n", "\\n" or " ".

The probe composes every test sample twice with identical draws (email,
position, template, case) except the separator: once as its own paragraph and
once written into the running text with a single space. The threshold is fixed
at 1% FPR on the paragraph benign scores, as calibration on paragraph data
would fix it. Only email+payload forms differ between the two views, so the
rates are reported on those.

    python tests/separator_shift.py --arm paragraph --seed 0 --out p0.json
    python tests/separator_shift.py --arm mixed --seed 0 --out m0.json
"""

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_stack  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import concatenate_datasets  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from transformers import TrainingArguments  # noqa: E402

import train_benign_exposure_mmbert2_dilute_email as loop  # noqa: E402
from ab_segment_objective import ARMS, SHARED, WEIGHTS  # noqa: E402
from email_augmentation import _digest  # noqa: E402
from first_letter_shortcut import margins  # noqa: E402
from raytune_benchmark import ModelSpec  # noqa: E402
from segment_scoring import make_segment_model  # noqa: E402

SEPARATORS = {"paragraph": None, "mixed": ["\n\n", "\n", " "]}
DATA = dict(malicious_density=0.4, benign_density=0.05)
SURFACE = dict(first_letter_upper_probability=0.5, strip_payload_whitespace=True)


def probe_collator(tokenizer, separator):
    collator = tiny_stack.make_collator(tokenizer, "test", "fixed", 42, WEIGHTS,
                                        insertion_separators=[separator], **SURFACE)
    collator._rng = lambda f: random.Random(int(_digest(
        [collator.seed, collator.split, str(f["id"]), int(f["label"])]), 16))
    return collator


def train(arm, seed, workdir, epochs):
    hp = loop._resolve_segment_hp({**SHARED, **ARMS["new"], "seed": seed, "num_train_epochs": epochs},
                                  OmegaConf.create({"max_micro_batch_size": 16}))
    tokenizer = tiny_stack.make_tokenizer(cased=True, newlines=True)
    model = make_segment_model(tiny_stack.make_base_model(tokenizer, seed), tokenizer, hp)
    model = loop._prepare_model(
        model, "tiny", hp, "peft",
        {"tiny": ModelSpec(family="mmbert", peft_target_modules=["Wqkv", "Wi", "Wo"])},
        OmegaConf.create({"qat": True, "extra_modules_to_save": ["token_evidence_head"]}))
    collator = tiny_stack.make_collator(tokenizer, "train", "random", 42, WEIGHTS,
                                        insertion_separators=SEPARATORS[arm], **SURFACE)
    malicious, benign = tiny_stack.cased_payload_split(40, 1300, seed=100 + seed, **DATA)
    train_ds = loop.sample_training_view(malicious, benign, 30, seed=42, email_collator=collator)
    trainer_class, metrics = loop._training_components(hp, 50)
    args = TrainingArguments(
        output_dir=workdir, num_train_epochs=hp["num_train_epochs"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=64, gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"],
        warmup_ratio=hp["warmup_ratio"], lr_scheduler_type="linear", max_grad_norm=1.0,
        eval_strategy="no", save_strategy="no", logging_strategy="epoch", report_to="none",
        disable_tqdm=True, seed=seed, use_cpu=True, remove_unused_columns=False)
    trainer = trainer_class(
        model=model, args=args, train_dataset=train_ds, processing_class=tokenizer,
        train_data_collator=collator,
        eval_data_collator=loop._evaluation_collator(tokenizer, {"region_supervision": True}),
        compute_metrics=metrics, gamma_pos=hp["gamma_pos"], gamma_neg=hp["gamma_neg"],
        asl_clip=hp["asl_clip"], malicious_per_batch=hp["malicious_per_batch"])
    trainer.train()
    return tokenizer, model, trainer


def evaluate(model, trainer, tokenizer, seed):
    malicious, benign = tiny_stack.cased_payload_split(100, 3000, seed=200 + seed, **DATA)
    raw = concatenate_datasets([malicious, benign])
    views = {}
    for name, separator in (("paragraph", "\n\n"), ("inline", " ")):
        collator = probe_collator(tokenizer, separator)
        forms = [collator.compose(f)["augmentation_form"] for f in raw]
        views[name] = margins(model, trainer, collator.materialize(raw, name))
    (para, labels), (inline, inline_labels) = views["paragraph"], views["inline"]
    assert (labels == inline_labels).all()
    inserted = np.array([f in ("email_payload", "template_email_payload") for f in forms])
    threshold = float(np.quantile(para[labels == 0], 0.99))
    mal, ben = inserted & (labels == 1), inserted & (labels == 0)
    weights = np.where(labels[inserted] == 1, 1.0, 500 / 50)

    def rate(values, mask):
        return float((values[mask] > threshold).mean())

    return dict(
        threshold_at_1pct_fpr=threshold,
        n_inserted_malicious=int(mal.sum()), n_inserted_benign=int(ben.sum()),
        recall_paragraph=rate(para, mal), recall_inline=rate(inline, mal),
        fpr_paragraph=rate(para, ben), fpr_inline=rate(inline, ben),
        pr_auc_paragraph=float(average_precision_score(labels[inserted], para[inserted], sample_weight=weights)),
        pr_auc_inline=float(average_precision_score(labels[inserted], inline[inserted], sample_weight=weights)),
        malicious_margin_shift=float(np.mean(inline[mal] - para[mal])),
        benign_margin_shift=float(np.mean(inline[ben] - para[ben])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=sorted(SEPARATORS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as workdir:
        tokenizer, model, trainer = train(args.arm, args.seed, workdir, args.epochs)
        result = dict(arm=args.arm, seed=args.seed, epochs=args.epochs,
                      **evaluate(model, trainer, tokenizer, args.seed))
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
