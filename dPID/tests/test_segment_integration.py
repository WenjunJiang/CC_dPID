"""End-to-end check of the segment objective through the production stack.

test_segment_objective.py pins the loss in isolation. This file checks that it
survives the parts it has to pass through in a real run: the training loop's
hyperparameter resolution, QAT then LoRA, PEFT's modules_to_save wrapping,
transformers.Trainer's sampler and gradient accumulation, the email collator's
region masks, and the metrics Ray ranks on. See tiny_stack.py for what is
substituted and why.

Run from the dPID directory:
    python -m unittest tests.test_segment_integration
"""

import tempfile
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import TrainingArguments

from tests import tiny_stack
import train_benign_exposure_mmbert2_dilute_email as loop
from raytune_benchmark import ModelSpec
from segment_scoring import make_segment_model
from segment_training import StratifiedOrder

WEIGHTS = {"malicious": [0.2, 0.2, 0.3, 0.3, 0.0, 0.0],
           "benign": [0.2, 0.2, 0.25, 0.25, 0.05, 0.05]}
SEARCHED = dict(
    model_name="tiny", segment_tau=1.0, region_loss_weight=0.5, positive_coverage=0.5,
    benign_top_k=8, token_head_dropout=0.0, leak_loss_weight=0.1, gamma_pos=0.5,
    gamma_neg=2.0, asl_clip=0.01, lora_r=8, lora_alpha=16, lora_dropout=0.0,
    learning_rate=3e-3, weight_decay=0.01, warmup_ratio=0.1, benign_to_malicious_ratio=30,
    malicious_per_batch=4, effective_batch_size=16, num_train_epochs=2)
PERF = OmegaConf.create({"max_micro_batch_size": 8})
PEFT = OmegaConf.create({"qat": True, "extra_modules_to_save": ["token_evidence_head"]})


class Recorder:
    """Pass-through collator that remembers what each micro-batch contained."""

    def __init__(self, inner):
        self.inner, self.batches = inner, []

    def __call__(self, features):
        batch = self.inner(features)
        self.batches.append(([f["id"] for f in features], batch["labels"].tolist()))
        return batch


class SegmentStackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hp = loop._resolve_segment_hp(dict(SEARCHED), PERF)
        tokenizer = tiny_stack.make_tokenizer()
        model = make_segment_model(tiny_stack.make_base_model(tokenizer), tokenizer, cls.hp)
        cls.model = loop._prepare_model(model, "tiny", cls.hp, "peft",
                                        {"tiny": ModelSpec(family="mmbert",
                                                           peft_target_modules=["Wqkv", "Wi", "Wo"])},
                                        PEFT)
        cls.before = {n: p.detach().clone() for n, p in cls.model.named_parameters()}

        cls.tmp = tempfile.TemporaryDirectory()
        train_collator = tiny_stack.make_collator(tokenizer, "train", "random", 42, WEIGHTS)
        eval_collator = tiny_stack.make_collator(tokenizer, "valid", "fixed", 42, WEIGHTS)
        malicious, benign = tiny_stack.payload_split(10, 400, seed=1)
        train_ds = loop.sample_training_view(malicious, benign, 30, seed=42,
                                             email_collator=train_collator)
        val_malicious, val_benign = tiny_stack.payload_split(8, 400, seed=2)
        from datasets import concatenate_datasets
        val_ds = eval_collator.materialize(concatenate_datasets([val_malicious, val_benign]), "valid")

        cls.recorder = Recorder(train_collator)
        trainer_class, metrics = loop._training_components(cls.hp, 50)
        args = TrainingArguments(
            output_dir=cls.tmp.name, num_train_epochs=cls.hp["num_train_epochs"],
            per_device_train_batch_size=cls.hp["per_device_train_batch_size"],
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=cls.hp["gradient_accumulation_steps"],
            learning_rate=cls.hp["learning_rate"], weight_decay=cls.hp["weight_decay"],
            warmup_ratio=cls.hp["warmup_ratio"], lr_scheduler_type="linear", max_grad_norm=1.0,
            eval_strategy="epoch", save_strategy="no", logging_strategy="epoch",
            report_to="none", disable_tqdm=True, seed=42, use_cpu=True,
            remove_unused_columns=False)
        cls.trainer = trainer_class(
            model=cls.model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
            processing_class=tokenizer, train_data_collator=cls.recorder,
            eval_data_collator=loop._evaluation_collator(tokenizer, {"region_supervision": True}),
            compute_metrics=metrics, gamma_pos=cls.hp["gamma_pos"], gamma_neg=cls.hp["gamma_neg"],
            asl_clip=cls.hp["asl_clip"], malicious_per_batch=cls.hp["malicious_per_batch"])
        cls.train_labels = np.asarray(train_ds["label"])
        cls.output = cls.trainer.train()
        cls.metrics = cls.trainer.evaluate()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_training_loop_sets_the_prior_the_model_reweights_with(self):
        self.assertEqual(self.hp["stratified_prior_ratio"], 30.0)
        self.assertEqual(self.hp["per_device_train_batch_size"], 8)
        self.assertEqual(self.hp["gradient_accumulation_steps"], 2)
        config = self.model.get_base_model().config
        self.assertEqual(config.stratified_prior_ratio, 30.0)
        self.assertEqual(config.positive_coverage, 0.5)

    def test_trainer_uses_the_stratified_sampler(self):
        self.assertIsInstance(self.trainer._get_train_sampler(), StratifiedOrder)

    def test_every_micro_batch_holds_the_requested_positives(self):
        # Accelerate re-wraps the DataLoader; this is the check that it kept
        # the order rather than re-shuffling it.
        positives = [sum(labels) for _, labels in self.recorder.batches]
        self.assertTrue(positives)
        # Without stratification a 30:1 micro-batch of 8 is empty 77% of the time.
        self.assertEqual(set(positives), {SEARCHED["malicious_per_batch"]})

    def test_each_epoch_reorders(self):
        steps = len(self.recorder.batches) // SEARCHED["num_train_epochs"]
        first = [i for ids, _ in self.recorder.batches[:steps] for i in ids]
        second = [i for ids, _ in self.recorder.batches[steps:2 * steps] for i in ids]
        self.assertEqual(len(first), len(second))
        self.assertNotEqual(first, second)

    def test_only_adapter_and_token_head_train(self):
        changed = {n for n, p in self.model.named_parameters()
                   if not torch.equal(p.detach(), self.before[n])}
        self.assertTrue(any("lora_" in n for n in changed), changed)
        self.assertTrue(any("token_evidence_head" in n for n in changed), changed)
        frozen = {n for n in changed if "lora_" not in n and "token_evidence_head" not in n}
        self.assertEqual(frozen, set())

    def test_loss_is_finite(self):
        self.assertTrue(np.isfinite(self.output.training_loss))
        self.assertTrue(np.isfinite(self.metrics["eval_loss"]))

    def test_ray_sees_the_selection_metric(self):
        # Ray ranks on val_*, which the loop's reporter maps from eval_*.
        for key in ("eval_selection_pr_auc", "eval_malicious_segment_length",
                    "eval_long_pr_auc", "eval_long_malicious_segment_length"):
            self.assertIn(key, self.metrics)
        self.assertTrue(0.0 <= self.metrics["eval_selection_pr_auc"] <= 1.0)
        bucket = [v for k, v in self.metrics.items()
                  if k.endswith("_pr_auc") and k != "eval_selection_pr_auc"]
        self.assertAlmostEqual(self.metrics["eval_selection_pr_auc"], min(bucket))

    def test_failed_trials_report_the_selection_metric(self):
        report = loop._failed_report(self.hp)
        self.assertEqual(report["val_selection_pr_auc"], 0.0)
        self.assertIn("val_f1", report)
        self.assertNotIn("val_selection_pr_auc", loop._failed_report({"learning_rate": 1e-4}))

    def test_evaluation_is_not_reweighted(self):
        # Prior weights are a training device; eval loss must be the plain mean
        # so it stays comparable with runs that did not stratify.
        self.model.eval()
        batch = next(iter(self.trainer.get_eval_dataloader()))
        with torch.no_grad():
            out = self.model(**batch)
            config = self.model.get_base_model().config
            ratio, config.stratified_prior_ratio = config.stratified_prior_ratio, None
            plain = self.model(**batch)
            config.stratified_prior_ratio = ratio
        self.assertTrue(torch.allclose(out.loss, plain.loss))


if __name__ == "__main__":
    unittest.main()
