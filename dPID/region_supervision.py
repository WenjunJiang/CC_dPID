from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEvidenceHead(nn.Module):
    """
    Produce one malicious-evidence logit for every token.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:
                Shape [batch_size, sequence_length, hidden_size].

        Returns:
            Token logits with shape [batch_size, sequence_length].
        """
        hidden_states = self.dropout(hidden_states)
        return self.classifier(hidden_states).squeeze(-1)


def attach_token_evidence_head(
    model: nn.Module,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Register the token head as a trainable model submodule.
    """
    if hasattr(model, "token_evidence_head"):
        raise ValueError(
            "The model already has a token_evidence_head."
        )

    hidden_size = getattr(model.config, "hidden_size", None)

    if hidden_size is None:
        hidden_size = getattr(model.config, "d_model", None)

    if hidden_size is None:
        raise ValueError(
            "Cannot determine hidden size from model.config."
        )

    model.token_evidence_head = TokenEvidenceHead(
        hidden_size=int(hidden_size),
        dropout=float(dropout),
    )

    return model


def masked_topk_mean(
    token_logits: torch.Tensor,
    region_mask: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pool the highest-k token logits inside each non-empty region.

    If a region has fewer than k tokens, all available region tokens
    are used.

    Args:
        token_logits:
            Shape [batch_size, sequence_length].

        region_mask:
            Boolean tensor with shape [batch_size, sequence_length].

        k:
            Maximum number of region tokens used for pooling.

    Returns:
        pooled_scores:
            Shape [number_of_samples_with_nonempty_region].

        has_region:
            Boolean tensor with shape [batch_size].
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    region_mask = region_mask.bool()
    region_lengths = region_mask.sum(dim=-1)
    has_region = region_lengths > 0

    if not has_region.any():
        return token_logits.new_empty((0,)), has_region

    selected_logits = token_logits[has_region]
    selected_mask = region_mask[has_region]
    selected_lengths = region_lengths[has_region]

    masked_logits = selected_logits.masked_fill(
        ~selected_mask,
        torch.finfo(selected_logits.dtype).min,
    )

    max_k = min(int(k), selected_logits.shape[1])

    topk_values = torch.topk(
        masked_logits,
        k=max_k,
        dim=-1,
    ).values

    effective_k = selected_lengths.clamp(max=max_k)

    positions = torch.arange(
        max_k,
        device=token_logits.device,
    ).unsqueeze(0)

    valid_topk = positions < effective_k.unsqueeze(1)

    # Invalid positions come from regions shorter than max_k.
    topk_values = topk_values.masked_fill(
        ~valid_topk,
        0.0,
    )

    pooled_scores = (
        topk_values.sum(dim=-1)
        / effective_k.to(token_logits.dtype)
    )

    return pooled_scores, has_region


def compute_region_supervision_loss(
    token_logits: torch.Tensor,
    malicious_mask: torch.Tensor,
    benign_mask: torch.Tensor,
    malicious_top_k: int,
    benign_top_k: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Region-level supervision:

        malicious_mask -> top-k mean -> positive BCE
        benign_mask    -> top-k mean -> negative BCE

    This does not apply individual labels to every token.
    """
    losses: list[torch.Tensor] = []
    details: dict[str, torch.Tensor] = {}

    malicious_scores, _ = masked_topk_mean(
        token_logits=token_logits,
        region_mask=malicious_mask,
        k=malicious_top_k,
    )

    if malicious_scores.numel() > 0:
        malicious_region_loss = (
            F.binary_cross_entropy_with_logits(
                malicious_scores,
                torch.ones_like(malicious_scores),
            )
        )

        losses.append(malicious_region_loss)
        details["malicious_region_loss"] = (
            malicious_region_loss.detach()
        )

    benign_scores, _ = masked_topk_mean(
        token_logits=token_logits,
        region_mask=benign_mask,
        k=benign_top_k,
    )

    if benign_scores.numel() > 0:
        benign_region_loss = (
            F.binary_cross_entropy_with_logits(
                benign_scores,
                torch.zeros_like(benign_scores),
            )
        )

        losses.append(benign_region_loss)
        details["benign_region_loss"] = (
            benign_region_loss.detach()
        )

    if losses:
        # Equal weighting between malicious-region and benign-region
        # objectives when both are present.
        region_loss = torch.stack(losses).mean()
    else:
        region_loss = token_logits.sum() * 0.0

    details["region_loss"] = region_loss.detach()

    return region_loss, details