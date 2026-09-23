"""Does redrawing the payload's first-letter case remove the case shortcut?

Training payloads carry the bias seen in the real data: injections almost
always start lower-case ("ignore ..."), benign payloads almost always
capitalized. Two arms train the same model on the same data and differ only in
`first_letter_upper_probability` on the training collator (None = verbatim,
0.5 = the fix).

The probe scores every test sample twice, as written and with its payload's
first letter flipped, on the same carrier and template, so the only
difference between the pair is that one letter. A model that reads the
content scores both alike; one that learned the shortcut drops injections
that start "Ignore" and flags benign text that starts lower-case.

    python tests/first_letter_shortcut.py --arm verbatim --seed 0 --out v0.json
    python tests/first_letter_shortcut.py --arm redrawn --seed 0 --out r0.json
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
from datasets import Dataset, concatenate_datasets  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from transformers import TrainingArguments  # noqa: E402

import train_benign_exposure_mmbert2_dilute_email as loop  # noqa: E402
from ab_segment_objective import ARMS, SHARED, WEIGHTS, score  # noqa: E402
from email_augmentation import _digest, set_first_letter_case  # noqa: E402
from raytune_benchmark import ModelSpec  # noqa: E402
from segment_scoring import make_segment_model  # noqa: E402

PROBABILITY = {"verbatim": None, "redrawn": 0.5}


def flipped(dataset):
    """Same rows with each payload's first letter in the opposite case."""
    def flip(text):
        first = next((c for c in text if c.isalpha()), "")
        return set_first_letter_case(text, first.islower())
    return Dataset.from_list([{**row, "text": flip(row["text"])} for row in dataset])


def paired_collator(tokenizer):
    """Fixed-mode collator whose draws ignore the payload text.

    The stock fixed mode hashes the text into its seed, so a flipped payload
    would land in a different email at a different position. Hashing only the
    id keeps carrier, template and position identical across the pair.
    """
    collator = tiny_stack.make_collator(tokenizer, "test", "fixed", 42, WEIGHTS)
    collator._rng = lambda f: random.Random(int(_digest(
        [collator.seed, collator.split, str(f["id"]), int(f["label"])]), 16))
    return collator


def train(arm, seed, workdir, epochs):
    hp = loop._resolve_segment_hp({**SHARED, **ARMS["new"], "seed": seed, "num_train_epochs": epochs},
                                  OmegaConf.create({"max_micro_batch_size": 16}))
    tokenizer = tiny_stack.make_tokenizer(cased=True)
    model = make_segment_model(tiny_stack.make_base_model(tokenizer, seed), tokenizer, hp)
    model = loop._prepare_model(
        model, "tiny", hp, "peft",
        {"tiny": ModelSpec(family="mmbert", peft_target_modules=["Wqkv", "Wi", "Wo"])},
        OmegaConf.create({"qat": True, "extra_modules_to_save": ["token_evidence_head"]}))
    collator = tiny_stack.make_collator(tokenizer, "train", "random", 42, WEIGHTS,
                                        first_letter_upper_probability=PROBABILITY[arm])
    malicious, benign = tiny_stack.cased_payload_split(40, 1300, seed=100 + seed)
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


@torch.no_grad()
def margins(model, trainer, dataset):
    model.eval()
    out, labels = [], []
    for batch in trainer.get_eval_dataloader(dataset):
        margin, _, _ = score(model, batch)
        out.append(margin)
        labels.append(batch["labels"])
    return torch.cat(out).numpy(), torch.cat(labels).numpy()


def evaluate(model, trainer, tokenizer, seed):
    malicious, benign = tiny_stack.cased_payload_split(100, 3000, seed=200 + seed)
    raw = concatenate_datasets([malicious, benign])
    collator = paired_collator(tokenizer)
    clean, labels = margins(model, trainer, collator.materialize(raw, "clean"))
    flip, flip_labels = margins(model, trainer, collator.materialize(flipped(raw), "flipped"))
    assert (labels == flip_labels).all()
    threshold = float(np.quantile(clean[labels == 0], 0.99))
    weights = np.where(labels == 1, 1.0, 500 / 50)
    mal, ben = labels == 1, labels == 0

    def rate(values, mask):
        return float((values[mask] > threshold).mean())

    return dict(
        threshold_at_1pct_fpr=threshold,
        pr_auc_as_written=float(average_precision_score(labels, clean, sample_weight=weights)),
        pr_auc_flipped=float(average_precision_score(labels, flip, sample_weight=weights)),
        recall_as_written=rate(clean, mal), recall_flipped=rate(flip, mal),
        fpr_as_written=rate(clean, ben), fpr_flipped=rate(flip, ben),
        malicious_decision_flips=float(((clean > threshold) != (flip > threshold))[mal].mean()),
        benign_decision_flips=float(((clean > threshold) != (flip > threshold))[ben].mean()),
        malicious_margin_shift=float(np.mean(flip[mal] - clean[mal])),
        benign_margin_shift=float(np.mean(flip[ben] - clean[ben])),
        n_malicious=int(mal.sum()), n_benign=int(ben.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=sorted(PROBABILITY), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
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
