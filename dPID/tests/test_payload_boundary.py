"""Separator draw and whitespace stripping: label-independent, spans exact.

Run from the dPID directory:
    python -m unittest tests.test_payload_boundary
"""

import unittest

from datasets import Dataset

from tests import tiny_stack
from email_augmentation import set_first_letter_case

WEIGHTS = {"malicious": [0.2, 0.2, 0.3, 0.3, 0.0, 0.0],
           "benign": [0.2, 0.2, 0.25, 0.25, 0.05, 0.05]}
SEPARATORS = ["\n\n", "\n", " "]


class BoundaryDraw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = tiny_stack.make_tokenizer(cased=True, newlines=True)
        malicious, benign = tiny_stack.cased_payload_split(600, 600, seed=5)
        # Give one class ragged edges, as a second data source might.
        malicious = Dataset.from_list([{**r, "text": "  " + r["text"] + " \n"} for r in malicious])
        cls.features = list(malicious) + list(benign)
        cls.by_id = {f["id"]: f["text"] for f in cls.features}
        cls.collator = tiny_stack.make_collator(
            cls.tokenizer, "train", "fixed", 11, WEIGHTS, first_letter_upper_probability=0.5,
            strip_payload_whitespace=True, insertion_separators=SEPARATORS)
        cls.rows = [cls.collator.compose(f) for f in cls.features]

    def inserted(self):
        return [r for r in self.rows if r["separator"]]

    def test_separator_share_is_the_same_for_both_labels(self):
        # Pooled over three collator seeds: one seed gives about 300 insertions
        # per class, where a single share can sit 0.09 from 1/3 by chance.
        rows = list(self.rows)
        for seed in (12, 13):
            collator = tiny_stack.make_collator(
                self.tokenizer, "train", "fixed", seed, WEIGHTS, first_letter_upper_probability=0.5,
                strip_payload_whitespace=True, insertion_separators=SEPARATORS)
            rows += [collator.compose(f) for f in self.features]
        for label in (0, 1):
            seps = [r["separator"] for r in rows if r["separator"] and r["labels"] == label]
            self.assertGreater(len(seps), 800)
            for sep in SEPARATORS:
                self.assertAlmostEqual(seps.count(sep) / len(seps), 1 / 3, delta=0.06, msg=(label, sep))

    def test_only_email_payload_forms_draw_a_separator(self):
        for row in self.rows:
            has = row["augmentation_form"] in ("email_payload", "template_email_payload")
            self.assertEqual(bool(row["separator"]), has, row["augmentation_form"])

    def test_payload_sits_between_the_drawn_separators(self):
        for row in self.inserted():
            payload = self.expected_payload(row)
            text, sep = row["text"], row["separator"]
            at = text.index(payload)
            if row["position"] != "start":
                self.assertTrue(text[:at].endswith(sep), (row["position"], repr(text[at - 3:at])))
                self.assertFalse(text[:at - len(sep)].endswith((" ", "\n")))
            if row["position"] != "end":
                after = text[at + len(payload):]
                if after:
                    self.assertTrue(after.startswith(sep))

    def test_edges_are_stripped_for_both_labels(self):
        for row in self.rows:
            if row["payload_id"]:
                payload = self.expected_payload(row)
                self.assertEqual(payload, payload.strip())

    def test_malicious_mask_is_exactly_the_payload(self):
        checked = 0
        for row in self.rows:
            if row["labels"] != 1:
                continue
            tokens = self.tokenizer.convert_ids_to_tokens(row["input_ids"])
            span = [t for t, m in zip(tokens, row["malicious_mask"]) if m]
            self.assertEqual(span, self.expected_payload(row).split(), row["separator"] or "none")
            checked += 1
        self.assertGreater(checked, 500)

    def expected_payload(self, row):
        return set_first_letter_case(self.by_id[row["payload_id"]].strip(),
                                     row["payload_first_letter"] == "upper")

    def test_options_change_the_cache_identity(self):
        plain = tiny_stack.make_collator(self.tokenizer, "train", "fixed", 11, WEIGHTS,
                                         first_letter_upper_probability=0.5)
        self.assertNotEqual(plain.signature, self.collator.signature)

    def test_rejects_non_whitespace_separators(self):
        with self.assertRaises(ValueError):
            tiny_stack.make_collator(self.tokenizer, "train", "fixed", 11, WEIGHTS,
                                     insertion_separators=["\n", " -- "])


if __name__ == "__main__":
    unittest.main()
