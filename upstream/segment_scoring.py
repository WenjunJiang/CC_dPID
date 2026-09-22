"""Shared token-only scoring, losses, padding, and checkpoint loading for mmBERT.

The nonempty maximum-subarray score is the same in training and inference.
Span masks are used only by the auxiliary region loss, never by decoding.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertModel, ModernBertPreTrainedModel

SCORING_MODE = "token_max_subarray_v1"
MASK_KEYS = ("valid_token_mask", "malicious_mask", "benign_mask")
SEGMENT_HP = ("segment_tau", "region_loss_weight", "positive_coverage", "benign_top_k",
              "token_head_dropout", "leak_loss_weight")

# `positive_coverage` replaces the former constant `malicious_top_k`. The scoring
# function is unchanged, so checkpoints trained under either objective decode
# identically and SCORING_MODE stays put; only training needs the new key, and
# an old HPO checkpoint is rejected because it cannot supply it.


def maximum_subarray(logits, valid_mask, tau):
    """Return nonempty maximum sums and [start, end) boundaries in O(B * L).

    Prefix minima are the vectorized equivalent of Kadane. Selection is discrete;
    recomputing the selected sum from the original tensor preserves its exact
    subgradient, without building a sequential per-token autograd graph.
    Invalid tokens are barriers, including invalid positions inside the input.
    """
    valid = valid_mask.bool()
    if logits.ndim != 2 or valid.shape != logits.shape:
        raise ValueError("Expected matching [batch, length] logits and validity mask")
    if not valid.any(dim=1).all():
        raise ValueError("Maximum-subarray scoring requires at least one valid token per sample")
    if not math.isfinite(float(tau)):
        raise ValueError("segment_tau must be finite")
    values = logits.float() - float(tau)
    if not torch.isfinite(values).all():
        raise ValueError("Segment scoring received non-finite token logits")
    with torch.no_grad():
        # A barrier larger than the total absolute mass cannot belong to the
        # optimum. FP64 selection avoids cancellation around padded barriers.
        # MPS has no FP64 kernels; only its small decoding tensor moves to CPU.
        # Separate transfer from casting: a combined to(cpu, float64) can ask
        # MPS to perform the unsupported FP64 cast before transferring.
        detached = values.detach().cpu().double() if values.device.type == "mps" else values.double()
        selection_valid = valid.to(detached.device)
        penalty = detached.abs().sum(dim=1, keepdim=True) + 1.0
        search = torch.where(selection_valid, detached, -penalty)
        prefix = torch.cat((search.new_zeros((len(search), 1)), search.cumsum(dim=1)), dim=1)
        minima, indices = prefix[:, :-1].cummin(dim=1)
        best_ending = (prefix[:, 1:] - minima).masked_fill(~selection_valid, -torch.inf)
        end = best_ending.argmax(dim=1)
        start = indices.gather(1, end[:, None]).squeeze(1)
        start, end = start.to(values.device), end.to(values.device)
        positions = torch.arange(values.shape[1], device=values.device)[None, :]
        selected = (positions >= start[:, None]) & (positions <= end[:, None]) & valid
    score = values.masked_fill(~selected, 0).sum(dim=1)
    return score, start, end + 1


def stable_sequence_asl(margin, labels, gamma_pos=1.0, gamma_neg=2.0, clip=0.01):
    """ASL without a hard logit clamp or a saturated sigmoid-to-log round trip."""
    z = margin.float()
    y = labels.float().reshape(-1)
    log_p, log_not_p = F.logsigmoid(z), F.logsigmoid(-z)
    positive = -log_p * torch.exp(gamma_pos * log_not_p)
    if clip > 0:
        # Preserve the existing shifted-negative ASL objective, but evaluate it
        # in log space. Hard-negative gradients are small, not hard-clamped away.
        shifted = torch.logaddexp(log_not_p, z.new_tensor(math.log(clip))).clamp(max=0)
    else:
        shifted = log_not_p
    negative = -shifted * torch.exp(gamma_neg * log_p)
    return (y * positive + (1 - y) * negative).mean()


def _pooled_top(token_logits, mask, count):
    """Mean of the highest `count[i]` in-region logits, zero where the region is empty.

    `count` varies per sample, so the top-k width is the batch maximum and ranks
    at or beyond a row's own count are zeroed. An empty region yields count 0,
    which masks every rank away before the sum and therefore never lets the
    -inf placeholders reach the result.
    """
    width = max(1, int(count.max().item()))
    selected = token_logits.float().masked_fill(~mask, -torch.inf).topk(
        min(width, token_logits.shape[1]), dim=1,
    ).values
    ranks = torch.arange(selected.shape[1], device=selected.device)[None, :]
    return selected.masked_fill(ranks >= count[:, None], 0).sum(dim=1) / count.clamp(min=1)


def region_losses(token_logits, malicious_mask, benign_mask, positive_coverage, benign_k):
    """Per-sample region losses plus which samples own each region.

    Returned unreduced so the caller can normalize each half by the number of
    samples that HAVE that region. Averaging the combined term over the batch
    instead makes the two halves unequal the moment the classes are imbalanced:
    at 30:1 only one sample in 31 owns a positive region, so its aggregate
    weight collapses by that factor while the negative half keeps full weight.

    The positive width scales with the payload rather than being a constant:
    a fixed k is fully satisfied by concentrating the whole score on k tokens,
    which is exactly the degenerate two-token window this model converges to
    otherwise. `positive_coverage` is the fraction of the payload that must read
    as evidence, so widening the selected segment is what lowers the loss.
    """
    positive, negative = malicious_mask.bool(), benign_mask.bool()
    if (positive & negative).any():
        raise ValueError("Malicious and benign regions must not overlap")
    if not 0.0 < float(positive_coverage) <= 1.0:
        raise ValueError("positive_coverage must lie in (0, 1]")
    if int(benign_k) <= 0:
        raise ValueError("Region top-k must be positive")

    positive_size = positive.sum(dim=1)
    positive_count = torch.minimum(
        (positive_size.float() * float(positive_coverage)).ceil().long(), positive_size)
    positive_loss = F.softplus(-_pooled_top(token_logits, positive, positive_count))

    negative_size = negative.sum(dim=1)
    negative_count = negative_size.clamp(max=int(benign_k))
    negative_loss = F.softplus(_pooled_top(token_logits, negative, negative_count))

    return positive_loss, positive_size > 0, negative_loss, negative_size > 0


def _region_half(losses, owned):
    """Mean over the samples that own the region, zero when none of them do."""
    owned = owned.to(losses.dtype)
    return (losses * owned).sum() / owned.sum().clamp(min=1)


def leak_loss_per_sample(token_logits, window, malicious_mask, tau):
    """Positive evidence the decoder picked up outside the annotated payload.

    One-sided on purpose: it penalizes a window that reaches past the payload and
    says nothing about one that stops short. The dataset carries payload
    boundaries, not finer attack spans, so requiring the window to equal the span
    would assume a precision the labels do not have; forbidding leakage does not.
    """
    outside = window & ~malicious_mask.bool()
    return F.relu((token_logits.float() - float(tau)).masked_fill(~outside, 0).sum(dim=1))


@dataclass
class SegmentOutput(SequenceClassifierOutput):
    # Small per-sample diagnostics, not all token logits/hidden states.
    segment_diagnostics: torch.Tensor | None = None
    # Opt-in so Trainer prediction tuples and existing callers remain unchanged.
    segment_window: torch.Tensor | None = None


class SegmentTokenHead(nn.Module):
    """Avoid PEFT's automatic SEQ_CLS wrapping of nested 'classifier' layers."""

    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        return self.projection(self.dropout(hidden_states)).squeeze(-1)


class SegmentForSequenceClassification(ModernBertPreTrainedModel):
    """ModernBERT encoder + trainable token head; no pooled sentence head."""

    def __init__(self, config, encoder=None):
        super().__init__(config)
        if getattr(config, "scoring_mode", None) != SCORING_MODE:
            raise ValueError("Missing or incompatible segment-scoring configuration")
        if (not math.isfinite(float(config.segment_tau))
                or not math.isfinite(float(config.region_loss_weight))
                or float(config.region_loss_weight) < 0):
            raise ValueError("Invalid segment tau or region loss weight")
        if int(config.benign_top_k) <= 0:
            raise ValueError("Region top-k values must be positive")
        # Checkpoints written before the coverage objective carry neither key and
        # still decode correctly, so validate them only where they are present.
        if hasattr(config, "positive_coverage") and not 0.0 < float(config.positive_coverage) <= 1.0:
            raise ValueError("positive_coverage must lie in (0, 1]")
        if hasattr(config, "leak_loss_weight") and float(config.leak_loss_weight) < 0:
            raise ValueError("leak_loss_weight must not be negative")
        self.num_labels = 2
        self.model = encoder if encoder is not None else ModernBertModel(config)
        self.token_evidence_head = SegmentTokenHead(config.hidden_size, config.token_head_dropout)
        if encoder is None:
            self.post_init()
        else:
            self.token_evidence_head.apply(self._init_weights)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                inputs_embeds=None, labels=None, valid_token_mask=None,
                malicious_mask=None, benign_mask=None, return_dict=None,
                return_segment_window=False, **kwargs):
        if input_ids is None:
            raise ValueError("Segment scoring requires input_ids to exclude special tokens")
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("output_hidden_states", None)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, inputs_embeds=inputs_embeds,
                             return_dict=True, **kwargs)
        token_logits = self.token_evidence_head(outputs.last_hidden_state).float()
        valid = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        for special_id in self.config.segment_special_token_ids:
            valid = valid & (input_ids != special_id)
        if valid_token_mask is not None and not torch.equal(valid, valid_token_mask.bool()):
            raise ValueError("Collator and model disagree on valid tokens")
        scores, start, end = maximum_subarray(token_logits, valid, self.config.segment_tau)
        logits = torch.stack((torch.zeros_like(scores), scores), dim=-1)
        loss = None
        if labels is not None:
            loss = stable_sequence_asl(scores, labels, self.config.gamma_pos,
                                       self.config.gamma_neg, self.config.asl_clip)
            if self.training and self.config.region_loss_weight > 0:
                if malicious_mask is None or benign_mask is None:
                    raise ValueError("Segment training requires both region masks")
                positive, negative = malicious_mask.bool(), benign_mask.bool()
                if not torch.equal(positive | negative, valid):
                    raise ValueError("Region masks must cover exactly the valid tokens")
                if not torch.equal(positive.any(dim=1), labels.bool()):
                    raise ValueError("Region masks and sentence labels disagree")
                positive_loss, owns_positive, negative_loss, owns_negative = region_losses(
                    token_logits, positive, negative,
                    self.config.positive_coverage, self.config.benign_top_k,
                )
                # Each half is a mean over its own owners, so the positive term
                # keeps full weight at any class ratio.
                loss = loss + self.config.region_loss_weight * (
                    0.5 * _region_half(positive_loss, owns_positive)
                    + 0.5 * _region_half(negative_loss, owns_negative)
                )
                leak_weight = float(getattr(self.config, "leak_loss_weight", 0.0))
                if leak_weight > 0:
                    positions = torch.arange(valid.shape[1], device=valid.device)[None, :]
                    window = (positions >= start[:, None]) & (positions < end[:, None]) & valid
                    loss = loss + leak_weight * _region_half(
                        leak_loss_per_sample(token_logits, window, positive,
                                             self.config.segment_tau),
                        owns_positive,
                    )
        return SegmentOutput(loss=loss, logits=logits,
                             segment_diagnostics=torch.stack((valid.sum(dim=1), end - start), dim=1),
                             segment_window=torch.stack((start, end), dim=1) if return_segment_window else None)


def make_segment_model(base_model, tokenizer, hp):
    """Reuse pretrained encoder weights, initialize only the new token head."""
    if base_model.config.model_type != "modernbert":
        raise ValueError("Segment scoring currently supports ModernBERT only")
    config = base_model.config
    config.scoring_mode = SCORING_MODE
    config.segment_special_token_ids = list(tokenizer.all_special_ids)
    config.num_labels = 2
    for name in SEGMENT_HP:
        if name not in hp:
            raise ValueError(f"Missing segment hyperparameter: {name}")
        setattr(config, name, hp[name])
    for name, default in (("gamma_pos", 1.0), ("gamma_neg", 2.0), ("asl_clip", 0.01)):
        setattr(config, name, float(hp.get(name, default)))
    return SegmentForSequenceClassification(config, encoder=base_model.model)


class RegionPaddingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        result = self.padding([{k: v for k, v in row.items() if k not in MASK_KEYS} for row in features])
        length = result["input_ids"].shape[1]
        for key in MASK_KEYS:
            if key not in features[0]:
                continue
            masks = []
            for row in features:
                mask = list(row[key])
                extra = [0] * (length - len(mask))
                masks.append(extra + mask if self.tokenizer.padding_side == "left" else mask + extra)
            result[key] = torch.tensor(masks, dtype=torch.bool)
        return result


def is_segment_checkpoint(path):
    config = Path(path) / "config.json"
    return config.exists() and json.loads(config.read_text()).get("scoring_mode") == SCORING_MODE


def validate_segment_calibration(path, calibration):
    """Reject legacy calibration or calibration fitted at a different tau."""
    if is_segment_checkpoint(path):
        config = json.loads((Path(path) / "config.json").read_text())
        if (calibration.get("scoring_mode") != SCORING_MODE
                or calibration.get("segment_tau") != config["segment_tau"]):
            raise ValueError("Calibration does not match this segment scorer; recalibrate the new model")


def load_scoring_base(path, **kwargs):
    """Shared by training calibration, PR inference, and manual prediction."""
    config_path = Path(path) / "config.json"
    if config_path.exists():
        mode = json.loads(config_path.read_text()).get("scoring_mode")
        if mode not in (None, "sentence", SCORING_MODE):
            raise ValueError(f"Unsupported checkpoint scoring mode: {mode}")
    if not is_segment_checkpoint(path):
        return AutoModelForSequenceClassification.from_pretrained(path, **kwargs)
    model, info = SegmentForSequenceClassification.from_pretrained(path, output_loading_info=True, **kwargs)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise ValueError(f"Incomplete segment checkpoint: {info}")
    return model


def load_scoring_model(path, training_mode="ft", **kwargs):
    model = load_scoring_base(path, **kwargs)
    if training_mode == "peft":
        from peft import PeftModel
        adapter = Path(path) / "adapter"
        if not adapter.is_dir():
            raise FileNotFoundError(f"PEFT checkpoint requires {adapter}")
        if is_segment_checkpoint(path):
            data = json.loads((adapter / "adapter_config.json").read_text())
            if "token_evidence_head" not in (data.get("modules_to_save") or []):
                raise ValueError("Segment adapter must include token_evidence_head in modules_to_save")
        model = PeftModel.from_pretrained(model, str(adapter))
    elif training_mode != "ft":
        raise ValueError(f"Unknown training mode: {training_mode}")
    elif is_segment_checkpoint(path) and (Path(path) / "adapter").exists():
        raise ValueError("This segment checkpoint requires --training-mode peft")
    return model.eval()
