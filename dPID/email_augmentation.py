"""Six-form email augmentation with isolated pools and payload-safe truncation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import DataCollatorWithPadding

from template_dilution_collator import load_templates


FORMS = (
    "payload", "template_payload", "email_payload", "template_email_payload",
    "email", "template_email",
)
DEFAULT_WEIGHTS = {
    # Equal shares for payload, template + payload, and both email + payload forms combined.
    "malicious": [1 / 3, 1 / 3, 1 / 6, 1 / 6, 0.0, 0.0],
    "benign": [0.30, 0.30, 0.15, 0.15, 0.05, 0.05],
}
SPLITS = ("train", "valid", "calib", "test")
ALGORITHM_VERSION = 1


_FIRST_LETTER = re.compile(r"[^\W\d_]")


def set_first_letter_case(text: str, upper: bool) -> str:
    """Upper- or lower-case the first letter of `text` without changing its length.

    Letters whose case mapping changes length (German sharp s) are left alone so
    that character offsets, and therefore span masks, stay valid.
    """
    match = _FIRST_LETTER.search(text)
    if match is None:
        return text
    char = match.group()
    replaced = char.upper() if upper else char.lower()
    if len(replaced) != 1:
        return text
    return text[:match.start()] + replaced + text[match.end():]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def split_email_dataset(dataset: Dataset, seed: int, ratios: dict) -> DatasetDict:
    """Split records, without deduplication or inferred thread grouping."""
    if set(ratios) != set(SPLITS):
        raise ValueError(f"Email split ratios must contain exactly {SPLITS}")
    values = [float(ratios[name]) for name in SPLITS]
    if any(not math.isfinite(v) or v <= 0 for v in values) or not math.isclose(sum(values), 1.0):
        raise ValueError("Email split ratios must be positive and sum to one")
    shuffled = dataset.shuffle(seed=seed)
    result, start = {}, 0
    for i, name in enumerate(SPLITS):
        end = len(dataset) if i == len(SPLITS) - 1 else start + int(len(dataset) * values[i])
        if end <= start:
            raise ValueError(f"Email pool is too small for a non-empty {name} split")
        result[name] = shuffled.select(range(start, end))
        start = end
    return DatasetDict(result)


def load_email_pools(cfg: Any, seed: int) -> tuple[DatasetDict, dict]:
    """Download once, persist record-level splits, and reuse the saved snapshot."""
    settings = {
        "dataset_name": str(cfg.dataset_name),
        "revision": str(cfg.get("revision", "main")),
        "source_split": str(cfg.get("source_split", "train")),
        "text_key": str(cfg.get("text_key", "text")),
        "seed": int(seed),
        "ratios": {name: float(cfg.split_ratios[name]) for name in SPLITS},
        "algorithm_version": ALGORITHM_VERSION,
    }
    cache_root = Path(cfg.cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / _digest(settings)[:20]
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if manifest["settings"] != settings:
            raise ValueError(f"Email cache settings do not match: {destination}")
        pools = load_from_disk(str(destination / "pools"))
        for name in SPLITS:
            if len(pools[name]) != manifest["counts"][name]:
                raise ValueError(f"Email cache count mismatch in {name}")
        return pools, manifest

    raw = load_dataset(
        settings["dataset_name"], split=settings["source_split"],
        revision=settings["revision"], cache_dir=str(cache_root / "downloads"),
    )
    source_fingerprint = raw._fingerprint
    key = settings["text_key"]
    if key not in raw.column_names:
        raise ValueError(f"Email dataset has no {key!r} column: {raw.column_names}")

    def normalize(batch, indices):
        return {
            "text": batch[key],
            "id": [f"enron:{settings['source_split']}:{i}" for i in indices],
        }

    normalized = raw.map(
        normalize, batched=True, with_indices=True, remove_columns=raw.column_names,
        desc="Normalize email records",
    )
    normalized = normalized.filter(
        lambda row: isinstance(row["text"], str) and bool(row["text"].strip()),
        desc="Remove empty email records (no deduplication)",
    )
    pools = split_email_dataset(normalized, seed, settings["ratios"])
    manifest = {
        "settings": settings,
        "source_fingerprint": source_fingerprint,
        "source_records": len(raw),
        "empty_records_removed": len(raw) - len(normalized),
        "counts": {name: len(pools[name]) for name in SPLITS},
        "fingerprints": {name: pools[name]._fingerprint for name in SPLITS},
    }
    with tempfile.TemporaryDirectory(prefix="email-split-", dir=cache_root) as temporary:
        staging = Path(temporary) / "snapshot"
        pools.save_to_disk(str(staging / "pools"))
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(destination)
    # Reload memory-mapped data so Ray workers share compact Arrow references.
    return load_from_disk(str(destination / "pools")), manifest


@dataclass
class EmailAugmentationCollator:
    tokenizer: Any
    email_pool: Dataset
    templates: list[str]
    split: str
    mode: str = "random"
    seed: int = 42
    max_length: int = 512
    text_key: str = "text"
    label_key: str = "label"
    id_key: str = "id"
    placeholder: str = "{PAYLOAD}"
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    artifacts_dir: str | None = None
    region_supervision: bool = False
    # Probability that a payload is inserted with an upper-case first letter,
    # drawn independently of the label. None keeps payloads verbatim.
    first_letter_upper_probability: float | None = None

    def __post_init__(self):
        if self.mode not in ("random", "fixed") or self.split not in SPLITS:
            raise ValueError("Invalid email augmentation mode or split")
        if not len(self.email_pool):
            raise ValueError("Email augmentation requires a non-empty email pool")
        if not self.templates or any(t.count(self.placeholder) != 1 for t in self.templates):
            raise ValueError("Each email augmentation template must contain exactly one placeholder")
        for label in ("benign", "malicious"):
            values = [float(v) for v in self.weights[label]]
            if len(values) != len(FORMS) or any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError(f"Invalid {label} form weights")
            if not math.isclose(sum(values), 1.0):
                raise ValueError(f"{label} form weights must sum to one")
        if any(self.weights["malicious"][4:]):
            raise ValueError("Email-only forms cannot have a malicious label")
        if self.max_length <= self.tokenizer.num_special_tokens_to_add(pair=False):
            raise ValueError("max_length leaves no room for content")
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("Email augmentation requires a fast tokenizer for reproducible cache identities")
        self.padding_collator = DataCollatorWithPadding(self.tokenizer)
        if self.region_supervision:
            from segment_scoring import RegionPaddingCollator
            self.padding_collator = RegionPaddingCollator(self.tokenizer)
        self._template_order = sorted(
            range(len(self.templates)),
            key=lambda i: len(self._encode(self.templates[i].replace(self.placeholder, ""))),
        )
        self.signature = _digest({
            "version": ALGORITHM_VERSION, "split": self.split, "seed": self.seed,
            "max_length": self.max_length, "weights": self.weights,
            "templates": self.templates, "placeholder": self.placeholder,
            "keys": [self.text_key, self.label_key, self.id_key],
            "email_fingerprint": self.email_pool._fingerprint,
            "tokenizer": self.tokenizer.backend_tokenizer.to_str(),
            "special_tokens": self.tokenizer.special_tokens_map,
        })
        if self.region_supervision:
            # Do not reuse legacy fixed views that contain no region masks.
            self.signature = _digest([self.signature, "segment_region_masks_v1"])
        if self.first_letter_upper_probability is not None:
            p = float(self.first_letter_upper_probability)
            if not 0.0 <= p <= 1.0:
                raise ValueError("first_letter_upper_probability must lie in [0, 1]")
            self.first_letter_upper_probability = p
            self.signature = _digest([self.signature, "first_letter_case_v1", p])

    def _encode(self, text: str) -> list[int]:
        # Encode the actual final string, including the tokenizer's special tokens.
        return self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]

    def _fits(self, text: str) -> bool:
        return len(self.tokenizer(
            text, add_special_tokens=True, truncation=True,
            max_length=self.max_length + 1,
        )["input_ids"]) <= self.max_length

    def filter_payloads(self, dataset: Dataset) -> Dataset:
        """Exclude unusable payloads before class-budget sampling, without relabeling."""
        def eligible(batch):
            if self.first_letter_upper_probability is None:
                return eligible_as_written(batch[self.text_key])
            # Either case may be drawn at composition time, so both must fit.
            texts = batch[self.text_key]
            return [a and b for a, b in zip(
                eligible_as_written([set_first_letter_case(t, True) if isinstance(t, str) else t
                                     for t in texts]),
                eligible_as_written([set_first_letter_case(t, False) if isinstance(t, str) else t
                                     for t in texts]))]

        def eligible_as_written(texts):
            valid = [isinstance(text, str) and bool(text.strip()) for text in texts]
            safe = [text if ok else " " for text, ok in zip(texts, valid)]
            raw = self.tokenizer(
                safe, add_special_tokens=True, truncation=True, max_length=self.max_length + 1,
            )["input_ids"]
            first = self.templates[self._template_order[0]]
            wrapped = self.tokenizer(
                [first.replace(self.placeholder, text) for text in safe],
                add_special_tokens=True, truncation=True, max_length=self.max_length + 1,
            )["input_ids"]
            return [
                ok and len(ids) <= self.max_length and (
                    len(wrapped_ids) <= self.max_length or any(
                        self._fits(self.templates[i].replace(self.placeholder, text))
                        for i in self._template_order[1:]
                    )
                )
                for text, ok, ids, wrapped_ids in zip(safe, valid, raw, wrapped)
            ]

        fingerprint = _digest(["eligible", dataset._fingerprint, self.signature])
        filtered = dataset.filter(
            eligible, batched=True, batch_size=256, new_fingerprint=fingerprint,
            desc="Exclude empty or over-budget payloads",
        )
        report = {"input": len(dataset), "retained": len(filtered), "excluded": len(dataset) - len(filtered)}
        print(f"  Email payload eligibility ({self.split}): {report}")
        if self.artifacts_dir:
            root = Path(self.artifacts_dir) / "payload_filter"
            root.mkdir(parents=True, exist_ok=True)
            (root / f"{fingerprint}.json").write_text(json.dumps(report, indent=2) + "\n")
        if not len(filtered):
            raise ValueError(f"No eligible payloads remain in {self.split}")
        return filtered

    def _rng(self, feature: dict):
        if self.mode == "random":
            # Trainer seeds Python and DataLoader workers. Do not reset per batch.
            return random
        identity = [
            self.seed, self.split, str(feature[self.id_key]),
            int(feature[self.label_key]), feature[self.text_key],
        ]
        return random.Random(int(_digest(identity), 16))

    @staticmethod
    def _insert_parts(body: str, position: str, rng) -> tuple[str, str, str]:
        if position == "random":
            boundaries = [m.end() for m in re.finditer(r"\s+", body) if m.start() > 0 and m.end() < len(body)]
            if boundaries:
                index = rng.choice(boundaries)
                return body[:index], body[index:], position
            position = rng.choice(("start", "end"))
        if position == "start":
            return "", body, position
        return body, "", position

    def compose(self, feature: dict) -> dict:
        """Return the final text and its reproducible sampling metadata."""
        label = int(feature[self.label_key])
        if label not in (0, 1):
            raise ValueError(f"Expected binary label, got {label}")
        rng = self._rng(feature)
        form = rng.choices(FORMS, weights=self.weights["malicious" if label else "benign"], k=1)[0]
        has_payload = form not in ("email", "template_email")
        has_email = "email" in form
        has_template = form.startswith("template_")
        payload = feature[self.text_key] if has_payload else ""
        payload_case = ""
        if has_payload and self.first_letter_upper_probability is not None and isinstance(payload, str):
            # Payload sources differ in how they capitalize, and the difference
            # correlates with the label: an injection that starts "ignore ..."
            # inserted as its own paragraph looks unlike a benign sentence. The
            # same draw for both labels makes the first letter carry no label.
            upper = rng.random() < self.first_letter_upper_probability
            payload = set_first_letter_case(payload, upper)
            payload_case = "upper" if upper else "lower"
        if has_payload and (not isinstance(payload, str) or not payload.strip() or not self._fits(payload)):
            raise ValueError("Payload is empty or exceeds max_length; call filter_payloads before sampling")

        template_id, template = -1, self.placeholder
        if has_template:
            # Uniformly try templates without replacement; retain the complete payload.
            candidates = list(range(len(self.templates)))
            rng.shuffle(candidates)
            for candidate in candidates:
                if self._fits(self.templates[candidate].replace(self.placeholder, payload)):
                    template_id, template = candidate, self.templates[candidate]
                    break
            else:
                raise ValueError("No template fits the complete payload within max_length")

        left = right = ""
        email_id, requested_position, position = "", "none", "none"
        if has_email:
            email = self.email_pool[rng.randrange(len(self.email_pool))]
            email_id, body = str(email["id"]), email["text"]
            if not isinstance(body, str) or not body.strip():
                raise ValueError("Email pool contains an empty body")
            if has_payload:
                requested_position = rng.choice(("start", "end", "random"))
                left, right, position = self._insert_parts(body, requested_position, rng)
            else:
                right = body

        # Keep only complete words near the insertion point. A bounded candidate
        # avoids tokenizing multi-megabyte forwarded threads on every micro-step.
        left_words = list(re.finditer(r"\S+", left))
        right_words = list(re.finditer(r"\S+", right))
        max_left = min(len(left_words), self.max_length)
        max_right = min(len(right_words), self.max_length)

        def render_parts(word_budget: int):
            n_left = min(max_left, word_budget // 2)
            n_right = min(max_right, word_budget - n_left)
            n_left = min(max_left, word_budget - n_right)
            before = left[left_words[-n_left].start():].rstrip() if n_left else ""
            after = right[:right_words[n_right - 1].end()].lstrip() if n_right else ""
            content = "\n\n".join(part for part in (before, payload, after) if part)
            prefix, suffix = template.split(self.placeholder)
            start = len(prefix) + (len(before) + 2 if before and payload else 0)
            return prefix + content + suffix, start, start + len(payload)

        def render(word_budget: int) -> str:
            return render_parts(word_budget)[0]

        # The zero-background version must fit. Search only changes email words;
        # it never slices payload text or the template prefix/suffix.
        low, high = 0, max_left + max_right
        text = render(0)
        if not self._fits(text):
            raise ValueError("The complete payload and template exceed max_length")
        full = render(high)
        if self._fits(full):
            low, text = high, full
        while low < high:
            middle = (low + high + 1) // 2
            candidate = render(middle)
            if self._fits(candidate):
                low, text = middle, candidate
            else:
                high = middle - 1
        ids = self._encode(text)
        if len(ids) > self.max_length:
            raise AssertionError("Email truncation exceeded max_length")
        if not has_payload and not low:
            raise ValueError("Template leaves no room for a complete email word")
        result = {
            "text": text, "input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": label, "augmentation_form": form, "email_id": email_id,
            "payload_id": str(feature[self.id_key]) if has_payload else "",
            "slot_id": str(feature[self.id_key]), "template_id": template_id,
            "requested_position": requested_position, "position": position,
            "email_words_retained": low,
        }
        if self.first_letter_upper_probability is not None:
            result["payload_first_letter"] = payload_case
        if self.region_supervision:
            final_text, start, end = render_parts(low)
            if final_text != text:
                raise AssertionError("Span boundaries must refer to the final rendered text")
            tokens = self.tokenizer(text, add_special_tokens=True, truncation=False,
                                    return_offsets_mapping=True)
            if tokens["input_ids"] != ids:
                raise AssertionError("Offset tokenization changed model inputs")
            special_ids = set(self.tokenizer.all_special_ids)
            valid = [int(token_id not in special_ids) for token_id in ids]
            positive = [int(bool(label) and bool(ok) and b > a and a < end and b > start)
                        for ok, (a, b) in zip(valid, tokens["offset_mapping"])]
            if not any(valid) or (label and not any(positive)):
                raise ValueError("Sample has no valid content or no tokenizable malicious span")
            result.update(valid_token_mask=valid, malicious_mask=positive,
                          benign_mask=[int(ok and not pos) for ok, pos in zip(valid, positive)])
        return result

    def __call__(self, features: list[dict]) -> dict:
        encoded = [self.compose(feature) for feature in features]
        keys = ("input_ids", "attention_mask", "labels")
        if self.region_supervision:
            keys += ("valid_token_mask", "malicious_mask", "benign_mask")
        return self.padding_collator([
            {key: row[key] for key in keys}
            for row in encoded
        ])

    def materialize(self, dataset: Dataset, name: str) -> Dataset:
        """Persist fixed, unpadded model inputs and a separate sampling manifest."""
        if self.mode != "fixed":
            raise ValueError("Only fixed augmentation may be materialized")
        fingerprint = _digest([self.signature, dataset._fingerprint, name])
        target = Path(self.artifacts_dir) / "fixed_views" / name / fingerprint if self.artifacts_dir else None
        if target is not None and target.exists():
            return load_from_disk(str(target / "encoded"))

        def encode_batch(batch):
            rows = [self.compose(dict(zip(batch, values))) for values in zip(*batch.values())]
            # Keep tokens and metadata, not another full-text copy of a large view.
            return {key: [row[key] for row in rows] for key in rows[0] if key != "text"}

        encoded = dataset.map(
            encode_batch, batched=True, batch_size=128, remove_columns=dataset.column_names,
            new_fingerprint=fingerprint, desc=f"Build fixed email view: {name}",
        )
        model_keys = ["input_ids", "attention_mask", "labels"]
        if self.region_supervision:
            model_keys += ["valid_token_mask", "malicious_mask", "benign_mask"]
        metadata = encoded.remove_columns(model_keys)
        encoded = encoded.select_columns(model_keys)
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="email-view-", dir=target.parent) as temporary:
                staging = Path(temporary) / "snapshot"
                encoded.save_to_disk(str(staging / "encoded"))
                metadata.save_to_disk(str(staging / "sampling"))
                staging.rename(target)
            return load_from_disk(str(target / "encoded"))
        return encoded


def make_email_collator(tokenizer, pools, email_cfg, template_cfg, split, seed, mode):
    if email_cfg is None or not email_cfg.get("enabled", False):
        return None
    if template_cfg is None:
        raise ValueError("Email augmentation requires template configuration")
    return EmailAugmentationCollator(
        tokenizer=tokenizer, email_pool=pools[split], split=split, seed=seed, mode=mode,
        templates=load_templates(template_cfg.template_path, template_cfg.template_key, template_cfg.placeholder),
        max_length=int(template_cfg.max_length), text_key=template_cfg.text_key,
        label_key=template_cfg.label_key, id_key=template_cfg.id_key,
        placeholder=template_cfg.placeholder,
        weights={key: list(email_cfg.get("weights", DEFAULT_WEIGHTS)[key]) for key in DEFAULT_WEIGHTS},
        artifacts_dir=str(email_cfg.artifacts_dir),
        region_supervision=bool(email_cfg.get("region_supervision", False)),
        first_letter_upper_probability=email_cfg.get("first_letter_upper_probability", None),
    )
