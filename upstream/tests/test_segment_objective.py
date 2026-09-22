"""The objective must prefer wide evidence over a concentrated peak.

The previous objective did not. With a constant `malicious_top_k`, a payload
represented by two very high tokens scored a lower loss than the same payload
represented by evidence spread across all of it, because only the top k entered
the region term and the sequence term had already saturated. The model converged
to a two-token window and a 6% margin, and a single substituted token removed it.

These are ordering tests, not accuracy tests: they pin the property that made
the failure possible, so it cannot come back silently.
"""

import unittest

import torch
from torch.nn import functional as F

from segment_scoring import (leak_loss_per_sample, region_losses, stable_sequence_asl,
                             _region_half)

TAU, REGION_WEIGHT, PAYLOAD, CARRIER = 2.0, 0.1, 35, 200


def total_loss(span_logits, coverage, benign_k=32, gamma_pos=1.0):
    """One malicious sample: payload logits, then a quiet carrier."""
    logits = torch.tensor([list(span_logits) + [-3.0] * CARRIER])
    malicious = torch.zeros_like(logits, dtype=torch.bool)
    malicious[0, :len(span_logits)] = True
    benign = ~malicious
    best = max(sum(x - TAU for x in span_logits[i:j])
               for i in range(len(span_logits)) for j in range(i + 1, len(span_logits) + 1))
    sequence = stable_sequence_asl(torch.tensor([best]), torch.tensor([1.0]), gamma_pos)
    positive, owns_positive, negative, owns_negative = region_losses(
        logits, malicious, benign, coverage, benign_k)
    region = 0.5 * _region_half(positive, owns_positive) + 0.5 * _region_half(negative, owns_negative)
    return float(sequence + REGION_WEIGHT * region)


class CoveragePrefersWidth(unittest.TestCase):
    CONCENTRATED = [5.2] * 2 + [-1.0] * (PAYLOAD - 2)
    PARTIAL = [3.5] * 8 + [-1.0] * (PAYLOAD - 8)
    COVERED = [2.5] * PAYLOAD

    def test_wide_evidence_wins(self):
        wide = total_loss(self.COVERED, coverage=0.5)
        partial = total_loss(self.PARTIAL, coverage=0.5)
        narrow = total_loss(self.CONCENTRATED, coverage=0.5)
        self.assertLess(wide, partial, "covering the payload must beat covering part of it")
        self.assertLess(partial, narrow, "eight tokens must beat two")

    def test_constant_k_is_what_inverted_the_order(self):
        """Regression witness: coverage 3/PAYLOAD reproduces the old k=3 behaviour."""
        as_constant_k = 3 / PAYLOAD
        self.assertGreater(total_loss(self.COVERED, coverage=as_constant_k),
                           total_loss(self.CONCENTRATED, coverage=as_constant_k))

    def test_width_requirement_follows_payload_length(self):
        """Covering half a payload costs the same whatever the payload's length.

        Asserted on the region term alone. The sequence term is a sum, so it
        does move with the payload length, and mixing the two in would test the
        saturation curve rather than the property meant here.
        """
        def positive_half(n):
            logits = torch.tensor([[2.5] * n + [-3.0] * CARRIER])
            malicious = torch.zeros_like(logits, dtype=torch.bool)
            malicious[0, :n] = True
            positive, owns, _, _ = region_losses(logits, malicious, ~malicious, 0.5, 32)
            return float(_region_half(positive, owns))

        costs = {n: positive_half(n) for n in (8, 35, 120)}
        self.assertLess(max(costs.values()) - min(costs.values()), 1e-6, costs)


class RegionHalvesAreBalanced(unittest.TestCase):
    """Each half must keep full weight at any class ratio.

    Averaging the combined per-sample term over the batch does not: at 30:1 only
    one sample in 31 owns a positive region, so the positive half arrives ~31x
    weaker than the negative one while the per-sample formula still reads 0.5/0.5.
    """

    def _batch(self, n_malicious, n_benign, length=40, payload=20):
        logits = torch.full((n_malicious + n_benign, length), -1.0)
        logits[:n_malicious, :payload] = 1.0
        malicious = torch.zeros_like(logits, dtype=torch.bool)
        malicious[:n_malicious, :payload] = True
        return logits, malicious, ~malicious

    def test_positive_half_is_ratio_invariant(self):
        halves = []
        for n_benign in (1, 30, 300):
            logits, malicious, benign = self._batch(1, n_benign)
            positive, owns, _, _ = region_losses(logits, malicious, benign, 0.5, 32)
            halves.append(float(_region_half(positive, owns)))
        self.assertLess(max(halves) - min(halves), 1e-6,
                        f"positive half drifted with the class ratio: {halves}")

    def test_missing_region_contributes_zero_without_redistributing(self):
        logits = torch.tensor([[5.0] * 10])                       # bare payload
        malicious = torch.ones_like(logits, dtype=torch.bool)
        _, owns_positive, negative, owns_negative = region_losses(
            logits, malicious, torch.zeros_like(malicious), 0.5, 32)
        self.assertTrue(bool(owns_positive.item()))
        self.assertFalse(bool(owns_negative.item()))
        self.assertEqual(float(_region_half(negative, owns_negative)), 0.0)

    def test_empty_region_stays_finite(self):
        logits = torch.tensor([[0.0] * 6])
        empty = torch.zeros_like(logits, dtype=torch.bool)
        positive, _, negative, _ = region_losses(logits, empty, ~empty, 0.5, 32)
        self.assertTrue(torch.isfinite(positive).all() and torch.isfinite(negative).all())


class LeakPenaltyIsOneSided(unittest.TestCase):
    """It forbids reaching past the payload; it never demands stopping short.

    The labels carry payload boundaries, not finer attack spans, so a two-sided
    term would assume a precision the annotation does not have.
    """

    def setUp(self):
        self.logits = torch.tensor([[1.0] * 5 + [4.0] * 10 + [1.0] * 5])
        self.malicious = torch.zeros_like(self.logits, dtype=torch.bool)
        self.malicious[0, 5:15] = True

    def _window(self, start, end):
        mask = torch.zeros_like(self.logits, dtype=torch.bool)
        mask[0, start:end] = True
        return mask

    def test_window_inside_the_payload_is_free(self):
        self.assertEqual(float(leak_loss_per_sample(
            self.logits, self._window(7, 12), self.malicious, TAU)), 0.0)

    def test_quiet_overrun_is_free(self):
        self.assertEqual(float(leak_loss_per_sample(
            self.logits, self._window(3, 18), self.malicious, TAU)), 0.0)

    def test_evidence_outside_the_payload_is_charged(self):
        logits = self.logits.clone()
        logits[0, 15:18] = 5.0
        self.assertGreater(float(leak_loss_per_sample(
            logits, self._window(3, 18), self.malicious, TAU)), 0.0)

    def test_gradient_pushes_the_leaked_tokens_down(self):
        logits = self.logits.clone()
        logits[0, 15:18] = 5.0
        logits.requires_grad_(True)
        leak_loss_per_sample(logits, self._window(3, 18), self.malicious, TAU).sum().backward()
        self.assertTrue((logits.grad[0, 15:18] > 0).all(), "leaked tokens must be pushed down")
        self.assertTrue((logits.grad[0, 5:15] == 0).all(), "payload tokens must be untouched")


class SequenceTermSaturates(unittest.TestCase):
    """Documents why the region term has to carry the width signal.

    Past a margin of about six the sequence term has no opinion left, and the
    previous run's decision boundary sat at 6.055.
    """

    def _gradient(self, margin, gamma_pos):
        z = torch.tensor([float(margin)], requires_grad=True)
        stable_sequence_asl(z, torch.tensor([1.0]), gamma_pos).backward()
        return abs(float(z.grad))

    def test_gradient_is_spent_at_the_observed_boundary(self):
        self.assertLess(self._gradient(6.055, 1.0) / self._gradient(1.0, 1.0), 1e-3)

    def test_lower_focusing_extends_the_useful_range(self):
        self.assertGreater(self._gradient(6.055, 0.0), 50 * self._gradient(6.055, 1.0))


if __name__ == "__main__":
    unittest.main()
