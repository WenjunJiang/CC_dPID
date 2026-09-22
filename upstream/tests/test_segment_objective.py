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

import numpy as np
import torch
from torch.nn import functional as F
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


class StratifiedBatchesPreserveTheObjective(unittest.TestCase):
    """Positives in every micro-batch, at the same expected loss.

    The micro-batch is capped by memory rather than by the sampled effective
    batch, so at 30:1 roughly a third of optimizer steps saw no malicious sample
    at all and about 71% saw no long-context one. Reordering the same indices
    fixes that for free; the importance weights keep it from also changing the
    problem being solved.
    """

    RATIO, BATCH, PER_BATCH = 30, 32, 4

    def setUp(self):
        from segment_training import StratifiedOrder
        self.labels = np.array([1] * 300 + [0] * 9000)
        self.sampler = StratifiedOrder(self.labels, self.BATCH, self.PER_BATCH, seed=42)

    def _batches(self):
        order = list(iter(self.sampler))
        return [order[i:i + self.BATCH]
                for i in range(0, len(order) - self.BATCH + 1, self.BATCH)]

    def test_every_batch_holds_the_requested_positives(self):
        counts = {sum(self.labels[i] == 1 for i in batch) for batch in self._batches()}
        self.assertEqual(counts, {self.PER_BATCH})

    def test_negatives_are_not_repeated_within_an_epoch(self):
        drawn = [i for batch in self._batches() for i in batch if self.labels[i] == 0]
        self.assertEqual(len(drawn), len(set(drawn)))

    def test_both_classes_are_rescaled_by_the_same_factor(self):
        """The epoch grows ~10%; what must not change is the ratio between classes.

        Exposure times weight is the total gradient each class contributes per
        epoch. Stratified batches spend 28 negative slots where an unstratified
        batch of 32 spends about 31, so an epoch covers more steps -- equally for
        both classes, which is why the objective is untouched.
        """
        batches = self._batches()
        share = 1.0 / (1.0 + self.RATIO)
        weight_positive = share / (self.PER_BATCH / self.BATCH)
        weight_negative = (1 - share) / (1 - self.PER_BATCH / self.BATCH)
        positive = len(batches) * self.PER_BATCH / (self.labels == 1).sum() * weight_positive
        negative = len(batches) * (self.BATCH - self.PER_BATCH) / (self.labels == 0).sum() * weight_negative
        self.assertAlmostEqual(positive, negative, places=6)

    def test_epochs_differ(self):
        first = list(iter(self.sampler))
        self.sampler.set_epoch(1)
        self.assertNotEqual(first, list(iter(self.sampler)))


class PriorWeightsRestoreTheUnstratifiedLoss(unittest.TestCase):
    RATIO = 30

    def _loss(self, margins, labels, prior_ratio=None):
        return float(stable_sequence_asl(
            torch.tensor(margins), torch.tensor(labels), prior_ratio=prior_ratio))

    def test_a_batch_already_at_the_prior_is_left_alone(self):
        labels = [1.0] + [0.0] * self.RATIO
        margins = [3.0] + [-3.0] * self.RATIO
        self.assertAlmostEqual(self._loss(margins, labels),
                               self._loss(margins, labels, self.RATIO), places=6)

    def test_a_stratified_batch_matches_the_population_loss(self):
        """Same per-class losses, different batch composition, same weighted mean."""
        def weighted(n_positive, n_negative):
            labels = [1.0] * n_positive + [0.0] * n_negative
            margins = [3.0] * n_positive + [-3.0] * n_negative
            return self._loss(margins, labels, self.RATIO)

        self.assertAlmostEqual(weighted(4, 28), weighted(1, 30), places=6)
        self.assertAlmostEqual(weighted(4, 28), weighted(16, 16), places=6)

    def test_an_absent_class_cannot_contribute(self):
        from segment_scoring import prior_weights
        weights = prior_weights(torch.zeros(8), self.RATIO)
        self.assertTrue(torch.isfinite(weights).all())


class MatchesTheLegacyRegionNormalization(unittest.TestCase):
    """The per-owner mean is not a new idea; region_supervision.py already did it.

    That file selects `token_logits[has_region]` before its BCE, so each region's
    mean runs over the samples that own it. The segment rewrite replaced that
    with a per-sample combined term averaged over the whole batch, which is where
    the positive half lost a factor of the class ratio. This pins the two back
    together so the regression cannot repeat.
    """

    def _legacy_half(self, token_logits, mask, k, positive):
        """region_supervision.masked_topk_mean + its BCE, transcribed."""
        mask = mask.bool()
        lengths = mask.sum(dim=-1)
        owns = lengths > 0
        if not owns.any():
            return None
        logits, sub_mask, sub_lengths = token_logits[owns], mask[owns], lengths[owns]
        width = min(int(k), logits.shape[1])
        top = logits.masked_fill(~sub_mask, torch.finfo(logits.dtype).min).topk(width, dim=-1).values
        effective = sub_lengths.clamp(max=width)
        keep = torch.arange(width)[None, :] < effective[:, None]
        pooled = top.masked_fill(~keep, 0.0).sum(dim=-1) / effective.to(logits.dtype)
        target = torch.ones_like(pooled) if positive else torch.zeros_like(pooled)
        return float(F.binary_cross_entropy_with_logits(pooled, target))

    def test_each_half_matches_the_legacy_computation(self):
        torch.manual_seed(0)
        logits = torch.randn(31, 40)
        malicious = torch.zeros_like(logits, dtype=torch.bool)
        malicious[0, 5:25] = True                       # one malicious row in 31
        benign = ~malicious
        # coverage chosen so ceil(coverage * 20) == 3, the legacy constant k
        positive, owns_positive, negative, owns_negative = region_losses(
            logits, malicious, benign, 3 / 20, 8)
        self.assertAlmostEqual(float(_region_half(positive, owns_positive)),
                               self._legacy_half(logits, malicious, 3, True), places=5)
        self.assertAlmostEqual(float(_region_half(negative, owns_negative)),
                               self._legacy_half(logits, benign, 8, False), places=5)

    def test_a_missing_half_is_dropped_rather_than_redistributed(self):
        """Where the two deliberately differ, and why.

        The legacy code stacks only the terms that exist and means them, so a
        region absent from the whole batch hands its weight to the other one.
        The segment design states the opposite -- a missing region contributes
        zero without redistributing its half -- and that is what is kept here.
        """
        logits = torch.randn(4, 12)
        malicious = torch.ones_like(logits, dtype=torch.bool)   # bare payloads
        positive, owns_positive, negative, owns_negative = region_losses(
            logits, malicious, torch.zeros_like(malicious), 0.5, 8)
        combined = (0.5 * _region_half(positive, owns_positive)
                    + 0.5 * _region_half(negative, owns_negative))
        self.assertAlmostEqual(float(combined),
                               0.5 * self._legacy_half(logits, malicious, 6, True), places=5)
