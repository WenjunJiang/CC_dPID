from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from region_supervision import compute_region_supervision_loss
from typing import Any, Optional
from transformers.modeling_outputs import SequenceClassifierOutput


class AsymmetricLossBinary(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip: float = 0.01,
        eps: float = 1e-5,
        prob_clip: float = 1e-4,
        check_finite_every: int = 50,
    ):
        super().__init__()

        self.check_finite_every = int(check_finite_every)
        self._step_count = 0
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.prob_clip = prob_clip

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.float().view(-1)

        # Convert two-class logits into one malicious-class logit.
        binary_logits = logits[:, 1] - logits[:, 0]
        
        # Clip binary logits to prevent extreme values that cause NaN/Inf
        binary_logits = binary_logits.clamp(min=-50, max=50)

        log_p = F.logsigmoid(binary_logits)
        log_1_minus_p = F.logsigmoid(-binary_logits)

        p = log_p.exp().clamp(
            min=self.prob_clip,
            max=1.0 - self.prob_clip,
        )

        pos_focus = (1.0 - p).pow(self.gamma_pos)
        neg_focus = p.pow(self.gamma_neg)

        pos_loss = -labels * log_p * pos_focus

        if self.clip is not None and self.clip > 0:
            neg_p_shifted = (1.0 - p + self.clip).clamp(
                min=self.eps,
                max=1.0 - self.prob_clip
            )

            neg_loss = (
                -(1.0 - labels)
                * torch.log(neg_p_shifted.clamp(min=self.eps))
                * neg_focus
            )
        else:
            neg_loss = (
                -(1.0 - labels)
                * log_1_minus_p.clamp(min=-50, max=0)
                * neg_focus
            )

        loss = (pos_loss + neg_loss).mean()

        # Guard against NaN/Inf propagation.
        #
        # `if not <0-dim cuda tensor>` forces a .item(), i.e. a cudaStreamSynchronize
        # on every micro-step, which stops the CPU from running ahead of the kernel
        # queue - costly in a launch-bound workload and fatal to CUDA-graph capture.
        # So check periodically rather than every step; a non-finite loss poisons the
        # weights and stays detectable on the next check.
        self._step_count += 1
        if self.check_finite_every > 0 and self._step_count % self.check_finite_every == 0:
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "ASL produced a non-finite loss."
                )

        return loss


class ASLTrainer(Trainer):
    def __init__(
        self,
        *args,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        asl_clip: float = 0.01,
        check_finite_every: int = 50,
        train_data_collator=None,
        eval_data_collator=None,
        **kwargs,
    ):
        # Keep backward compatibility with Trainer's data_collator argument.
        default_collator = kwargs.pop("data_collator", None)

        self.train_data_collator = (
            train_data_collator
            if train_data_collator is not None
            else default_collator
        )

        self.eval_data_collator = (
            eval_data_collator
            if eval_data_collator is not None
            else default_collator
        )

        # Trainer still requires one default collator during initialization.
        kwargs["data_collator"] = self.train_data_collator

        super().__init__(*args, **kwargs)

        self.loss_fn = AsymmetricLossBinary(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=asl_clip,
            check_finite_every=check_finite_every,
        )

    def get_train_dataloader(self):
        # Use dynamic/random dilution for training.
        original_collator = self.data_collator
        self.data_collator = self.train_data_collator

        try:
            return super().get_train_dataloader()
        finally:
            self.data_collator = original_collator

    def get_eval_dataloader(self, eval_dataset=None):
        # Use deterministic/fixed dilution for validation.
        original_collator = self.data_collator
        self.data_collator = self.eval_data_collator

        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_collator

    def get_test_dataloader(self, test_dataset):
        # Use the evaluation collator for prediction and testing.
        original_collator = self.data_collator
        self.data_collator = self.eval_data_collator

        try:
            return super().get_test_dataloader(test_dataset)
        finally:
            self.data_collator = original_collator

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # Compute the loss in fp32 even under bf16 autocast: ASL clamps to
        # [-50, 50] and clips probabilities at 1e-4, and bf16's ~3 significant
        # decimal digits would saturate those bounds far too early.
        loss = self.loss_fn(
            logits=outputs.logits.float(),
            labels=labels,
        )

        return (loss, outputs) if return_outputs else loss



class ASLRegionTrainer(ASLTrainer):
    def __init__(
        self,
        *args,
        region_loss_weight: float = 0.2,
        malicious_top_k: int = 3,
        benign_top_k: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if region_loss_weight < 0:
            raise ValueError(
                "region_loss_weight must be non-negative."
            )
        if malicious_top_k <= 0:
            raise ValueError(
                "malicious_top_k must be positive."
            )
        if benign_top_k <= 0:
            raise ValueError(
                "benign_top_k must be positive."
            )

        self.region_loss_weight = float(region_loss_weight)
        self.malicious_top_k = int(malicious_top_k)
        self.benign_top_k = int(benign_top_k)

    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        inputs = dict(inputs)

        labels = inputs.pop("labels")

        # Remove auxiliary masks before model.forward().
        malicious_mask = inputs.pop("malicious_mask", None)
        benign_mask = inputs.pop("benign_mask", None)

        use_region_loss = (
            model.training
            and self.region_loss_weight > 0
        )

        outputs = model(
            **inputs,
            output_hidden_states=use_region_loss,
            return_dict=True,
        )

        sequence_loss = self.loss_fn(
            logits=outputs.logits,
            labels=labels,
        )

        if not torch.isfinite(sequence_loss):
            raise FloatingPointError(
                f"Non-finite sequence loss: "
                f"{sequence_loss.detach().item()}"
            )

        if use_region_loss:
            if malicious_mask is None or benign_mask is None:
                raise KeyError(
                    "Region-supervised training requires "
                    "malicious_mask and benign_mask."
                )

            if outputs.hidden_states is None:
                raise RuntimeError(
                    "Hidden states were not returned during "
                    "region-supervised training."
                )

            malicious_mask = malicious_mask.bool()
            benign_mask = benign_mask.bool()

            final_hidden_states = outputs.hidden_states[-1]

            unwrapped_model = self.accelerator.unwrap_model(
                model
            )

            if not hasattr(
                unwrapped_model,
                "token_evidence_head",
            ):
                raise AttributeError(
                    "Model does not have token_evidence_head."
                )

            token_logits = (
                unwrapped_model.token_evidence_head(
                    final_hidden_states
                )
            )

            if token_logits.shape != malicious_mask.shape:
                raise RuntimeError(
                    "token_logits and malicious_mask have "
                    "different shapes: "
                    f"{tuple(token_logits.shape)} vs "
                    f"{tuple(malicious_mask.shape)}"
                )

            if token_logits.shape != benign_mask.shape:
                raise RuntimeError(
                    "token_logits and benign_mask have "
                    "different shapes: "
                    f"{tuple(token_logits.shape)} vs "
                    f"{tuple(benign_mask.shape)}"
                )

            region_loss, _ = (
                compute_region_supervision_loss(
                    token_logits=token_logits,
                    malicious_mask=malicious_mask,
                    benign_mask=benign_mask,
                    malicious_top_k=self.malicious_top_k,
                    benign_top_k=self.benign_top_k,
                )
            )

            if not torch.isfinite(region_loss):
                raise FloatingPointError(
                    f"Non-finite region loss: "
                    f"{region_loss.detach().item()}"
                )

            loss = (
                sequence_loss
                + self.region_loss_weight * region_loss
            )
        else:
            # Validation evaluates the deployed sequence classifier.
            loss = sequence_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite total loss: "
                f"{loss.detach().item()}"
            )

        if return_outputs:
            # Do not expose hidden_states to Trainer evaluation.
            metric_outputs = SequenceClassifierOutput(
                logits=outputs.logits,
            )
            return loss, metric_outputs

        return loss

