"""The first-letter case draw must be label-independent and must not move spans.

Run from the dPID directory:
    python -m unittest tests.test_first_letter_case
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

from tests import tiny_stack
from email_augmentation import set_first_letter_case

WEIGHTS = {"malicious": [0.2, 0.2, 0.3, 0.3, 0.0, 0.0],
           "benign": [0.2, 0.2, 0.25, 0.25, 0.05, 0.05]}


class SetFirstLetterCase(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(set_first_letter_case("ignore all", True), "Ignore all")
        self.assertEqual(set_first_letter_case("Ignore all", False), "ignore all")
        self.assertEqual(set_first_letter_case('"ignore', True), '"Ignore')
        self.assertEqual(set_first_letter_case("1. ignore", True), "1. Ignore")
        self.assertEqual(set_first_letter_case("123", True), "123")
        self.assertEqual(set_first_letter_case("", True), "")
        self.assertEqual(set_first_letter_case("éclair", True), "Éclair")

    def test_length_never_changes(self):
        # "ß".upper() is "SS"; changing length would shift every span offset.
        self.assertEqual(set_first_letter_case("ßtraße", True), "ßtraße")


class CollatorDraw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = tiny_stack.make_tokenizer(cased=True)
        malicious, benign = tiny_stack.cased_payload_split(600, 600, seed=3)
        cls.features = list(malicious) + list(benign)

    def composed(self, probability, mode="fixed"):
        collator = tiny_stack.make_collator(self.tokenizer, "train", mode, 7, WEIGHTS,
                                            first_letter_upper_probability=probability)
        return collator, [collator.compose(f) for f in self.features]

    def test_upper_share_is_the_same_for_both_labels(self):
        _, rows = self.composed(0.5)
        share = {}
        for label in (0, 1):
            # Email-only benign forms carry no payload and draw nothing.
            cases = [r["payload_first_letter"] for r in rows
                     if r["labels"] == label and r["payload_first_letter"]]
            share[label] = cases.count("upper") / len(cases)
        # The source data is 5% vs 90% upper; after the draw both sit at p.
        self.assertAlmostEqual(share[0], 0.5, delta=0.06)
        self.assertAlmostEqual(share[1], 0.5, delta=0.06)

    def test_malicious_mask_covers_the_recased_payload(self):
        _, rows = self.composed(0.5)
        by_id = {f["id"]: f["text"] for f in self.features}
        checked = 0
        for row in rows:
            if row["labels"] != 1:
                continue
            tokens = self.tokenizer.convert_ids_to_tokens(row["input_ids"])
            span = " ".join(t for t, m in zip(tokens, row["malicious_mask"]) if m)
            original = by_id[row["payload_id"]]
            self.assertEqual(span, set_first_letter_case(original, row["payload_first_letter"] == "upper"))
            checked += 1
        self.assertGreater(checked, 500)

    def test_fixed_mode_is_reproducible(self):
        _, first = self.composed(0.5)
        _, second = self.composed(0.5)
        self.assertEqual([r["text"] for r in first], [r["text"] for r in second])

    def test_probability_changes_the_cache_identity(self):
        off, _ = self.composed(None)
        half, _ = self.composed(0.5)
        self.assertNotEqual(off.signature, half.signature)

    def test_off_matches_the_code_before_this_change(self):
        # The pre-change module, if a copy is available (see DPID_OLD_CODE).
        old = os.environ.get("DPID_OLD_CODE")
        if not old:
            self.skipTest("set DPID_OLD_CODE to the pre-change dPID directory")
        spec = importlib.util.spec_from_file_location(
            "email_augmentation_before", Path(old) / "email_augmentation.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses resolve their module by name
        spec.loader.exec_module(module)
        collator, rows = self.composed(None)
        before = module.EmailAugmentationCollator(
            tokenizer=self.tokenizer, email_pool=collator.email_pool, templates=collator.templates,
            split="train", mode="fixed", seed=7, max_length=512, weights=WEIGHTS,
            region_supervision=True)
        self.assertEqual(collator.signature, before.signature)
        self.assertEqual(rows, [before.compose(f) for f in self.features])


if __name__ == "__main__":
    unittest.main()
