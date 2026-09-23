"""Train the old and the new segment objective on identical data; compare.

Both arms run the production stack in tiny_stack.py and differ only in which
copy of the code is imported (DPID_CODE_DIR) and in the hyperparameters that
exist in one objective but not the other. Everything they share is fixed to
the previous HPO winner: tau=2, region_loss_weight=0.1, benign_top_k=32,
gamma_pos=1.

The question is the failure seen in deployment, where changing `ignore` to
`Ignore` flipped a detection: does the score rest on one or two tokens? So
besides PR-AUC this measures the decoded window on injections and recall
after greedily replacing the strongest evidence token inside the window with
filler, one token at a time, against a threshold fixed at 1% FPR on the clean
benign scores.

    DPID_CODE_DIR=/path/to/old/dPID python tests/ab_segment_objective.py --arm old --seed 0 --out old0.json
    python tests/ab_segment_objective.py --arm new --seed 0 --out new0.json
"""

import argparse
import inspect
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_stack  # noqa: E402  (sets sys.path to DPID_CODE_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import concatenate_datasets  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from transformers import TrainingArguments  # noqa: E402

import train_benign_exposure_mmbert2_dilute_email as loop  # noqa: E402
from raytune_benchmark import ModelSpec  # noqa: E402
from segment_scoring import make_segment_model  # noqa: E402

SHARED = dict(
    model_name="tiny", segment_tau=2.0, region_loss_weight=0.1, benign_top_k=32,
    token_head_dropout=0.0, gamma_pos=1.0, gamma_neg=2.0, asl_clip=0.01,
    lora_r=16, lora_alpha=64, lora_dropout=0.0, learning_rate=3e-3, weight_decay=0.01,
    warmup_ratio=0.1, benign_to_malicious_ratio=30, effective_batch_size=32,
    num_train_epochs=8)
ARMS = {
    "old": dict(malicious_top_k=3),
    "new": dict(positive_coverage=0.5, leak_loss_weight=0.1, malicious_per_batch=4),
}
WEIGHTS = {"malicious": [0.2, 0.2, 0.3, 0.3, 0.0, 0.0],
           "benign": [0.2, 0.2, 0.25, 0.25, 0.05, 0.05]}
FILLER_TOKEN = "w0"
MAX_ABLATIONS = 3


def build(arm, seed, workdir, epochs):
    hp = loop._resolve_segment_hp({**SHARED, **ARMS[arm], "seed": seed, "num_train_epochs": epochs},
                                  OmegaConf.create({"max_micro_batch_size": 16}))
    tokenizer = tiny_stack.make_tokenizer()
    model = make_segment_model(tiny_stack.make_base_model(tokenizer, seed), tokenizer, hp)
    model = loop._prepare_model(
        model, "tiny", hp, "peft",
        {"tiny": ModelSpec(family="mmbert", peft_target_modules=["Wqkv", "Wi", "Wo"])},
        OmegaConf.create({"qat": True, "extra_modules_to_save": ["token_evidence_head"]}))

    # Data depends on the seed only, never on the arm.
    train_collator = tiny_stack.make_collator(tokenizer, "train", "random", 42, WEIGHTS)
    test_collator = tiny_stack.make_collator(tokenizer, "test", "fixed", 42, WEIGHTS)
    malicious, benign = tiny_stack.payload_split(40, 1300, seed=100 + seed)
    train_ds = loop.sample_training_view(malicious, benign, 30, seed=42, email_collator=train_collator)
    test_malicious, test_benign = tiny_stack.payload_split(80, 4000, seed=200 + seed)
    test_ds = test_collator.materialize(concatenate_datasets([test_malicious, test_benign]), "test")

    trainer_class, metrics = (loop._training_components(hp, 50)
                              if len(inspect.signature(loop._training_components).parameters) > 1
                              else loop._training_components(hp))
    extra = {}
    if "malicious_per_batch" in inspect.signature(trainer_class.__init__).parameters:
        extra["malicious_per_batch"] = int(hp.get("malicious_per_batch", 0))
    args = TrainingArguments(
        output_dir=workdir, num_train_epochs=hp["num_train_epochs"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=64, gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"],
        warmup_ratio=hp["warmup_ratio"], lr_scheduler_type="linear", max_grad_norm=1.0,
        eval_strategy="no", save_strategy="no", logging_strategy="epoch", report_to="none",
        disable_tqdm=True, seed=seed, use_cpu=True, remove_unused_columns=False)
    trainer = trainer_class(
        model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
        processing_class=tokenizer, train_data_collator=train_collator,
        eval_data_collator=loop._evaluation_collator(tokenizer, {"region_supervision": True}),
        compute_metrics=metrics, gamma_pos=hp["gamma_pos"], gamma_neg=hp["gamma_neg"],
        asl_clip=hp["asl_clip"], **extra)
    return hp, tokenizer, model, trainer, test_ds


@torch.no_grad()
def score(model, batch):
    """Sequence margin, decoded window, and the token logits it was decoded from."""
    captured = {}
    head = model.get_base_model().token_evidence_head
    hook = head.register_forward_hook(lambda module, args, output: captured.update(z=output))
    try:
        out = model(**batch, return_segment_window=True)
    finally:
        hook.remove()
    return out.logits[:, 1] - out.logits[:, 0], out.segment_window, captured["z"].float()


def ablate(model, batch, window, token_logits, filler):
    """Margins after replacing the strongest in-window token with filler, k = 1..MAX."""
    ids, valid, margins = batch["input_ids"].clone(), batch["valid_token_mask"], []
    positions = torch.arange(ids.shape[1])[None, :]
    for _ in range(MAX_ABLATIONS):
        inside = ((positions >= window[:, :1]) & (positions < window[:, 1:])
                  & valid & (ids != filler))
        target = token_logits.masked_fill(~inside, -torch.inf).argmax(dim=1)
        hit = inside.any(dim=1)
        ids[hit, target[hit]] = filler
        margin, window, token_logits = score(model, {**batch, "input_ids": ids})
        margins.append(margin)
    return margins


@torch.no_grad()
def evaluate(model, trainer, tokenizer):
    model.eval()
    filler = tokenizer.convert_tokens_to_ids(FILLER_TOKEN)
    rows = []
    for batch in trainer.get_eval_dataloader():
        margin, window, token_logits = score(model, batch)
        labels, valid, span = batch["labels"], batch["valid_token_mask"], batch["malicious_mask"]
        # Only injections are ablated; the threshold comes from clean benign scores.
        malicious = labels == 1
        ablated = torch.full((len(labels), MAX_ABLATIONS), float("nan"))
        if malicious.any():
            sub = {k: v[malicious] for k, v in batch.items()}
            ablated[malicious] = torch.stack(
                ablate(model, sub, window[malicious], token_logits[malicious], filler), dim=1)
        for i in range(len(labels)):
            start, end = window[i].tolist()
            in_window = torch.zeros_like(span[i])
            in_window[start:end] = True
            overlap = int((in_window & span[i]).sum())
            union = int((in_window | span[i]).sum())
            rows.append(dict(
                label=int(labels[i]), valid=int(valid[i].sum()),
                margins=[float(margin[i])] + ablated[i].tolist(), window=end - start,
                iou=overlap / union if labels[i] else None,
                overflow=(end - start - overlap) / (end - start) if labels[i] else None))
    return rows


def summarize(rows):
    labels = np.array([r["label"] for r in rows])
    margins = np.array([r["margins"] for r in rows])
    lengths = np.array([r["valid"] for r in rows])
    weights = np.where(labels == 1, 1.0, 500 / 50)
    threshold = float(np.quantile(margins[labels == 0, 0], 0.99))
    malicious = [r for r in rows if r["label"] == 1]
    long_ = lengths >= 256
    summary = dict(
        pr_auc=float(average_precision_score(labels, margins[:, 0], sample_weight=weights)),
        long_pr_auc=float(average_precision_score(labels[long_], margins[long_, 0],
                                                  sample_weight=weights[long_])),
        threshold_at_1pct_fpr=threshold,
        recall_after_k_ablations=[float((margins[labels == 1, k] > threshold).mean())
                                  for k in range(MAX_ABLATIONS + 1)],
        long_recall_after_k_ablations=[
            float((margins[(labels == 1) & long_, k] > threshold).mean())
            for k in range(MAX_ABLATIONS + 1)],
        median_malicious_margin_minus_threshold=float(np.median(margins[labels == 1, 0]) - threshold),
        malicious_window=float(np.mean([r["window"] for r in malicious])),
        malicious_window_iou=float(np.mean([r["iou"] for r in malicious])),
        malicious_window_overflow=float(np.mean([r["overflow"] for r in malicious])),
        benign_window=float(np.mean([r["window"] for r in rows if r["label"] == 0])),
        n_malicious=int((labels == 1).sum()), n_long_malicious=int(((labels == 1) & long_).sum()),
        n_benign=int((labels == 0).sum()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=SHARED["num_train_epochs"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as workdir:
        hp, tokenizer, model, trainer, _ = build(args.arm, args.seed, workdir, args.epochs)
        trainer.train()
        rows = evaluate(model, trainer, tokenizer)
    result = dict(arm=args.arm, seed=args.seed, code=str(tiny_stack.ROOT),
                  hp={k: v for k, v in hp.items() if isinstance(v, (int, float, str))},
                  **summarize(rows))
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
