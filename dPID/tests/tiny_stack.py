"""A tiny but real instance of the segment training stack, for tests on CPU.

Everything that decides behaviour is the production code: the email collator
builds the six forms and the region masks from tokenizer offsets,
make_segment_model and _prepare_model attach the token head, QAT and LoRA, and
SegmentTrainer drives transformers.Trainer. Only three things are substituted,
because none is reachable offline: the encoder is a randomly initialized
two-layer ModernBERT, the tokenizer is a whitespace word-level fast tokenizer,
and payloads and emails are synthetic.

`template_dilution_collator` is not part of the shipped tree; it is stubbed
only when missing, and the email path under test never calls it.
"""

import os
import random
import sys
import types
from pathlib import Path

# DPID_CODE_DIR points the harness at another copy of the code, so the same
# harness can train the pre-change objective for comparison.
ROOT = Path(os.environ.get("DPID_CODE_DIR") or Path(__file__).resolve().parents[1]).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "template_dilution_collator" not in sys.modules:
    try:
        import template_dilution_collator  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("template_dilution_collator")
        stub.TemplateDilutionCollator = object
        stub.load_templates = lambda *args, **kwargs: []
        sys.modules["template_dilution_collator"] = stub

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402
from transformers import (ModernBertConfig, ModernBertForSequenceClassification,  # noqa: E402
                          PreTrainedTokenizerFast)

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
FILLER = [f"w{i}" for i in range(300)]
# Attack vocabulary. Each payload uses several of these, so a detector that
# reads the whole payload survives losing any one token; one that keys on a
# single token does not.
ATTACK = ["ignore", "Ignore", "IGNORE", "disregard", "previous", "prior", "instructions",
          "rules", "reveal", "system", "prompt", "secret", "override", "bypass", "now",
          "output", "password", "admin", "mode", "developer"]
TEMPLATE_WORDS = ["Summarize", "Classify", "sentiment", "the", "following", "text", ":", "end"]
TEMPLATES = ["Summarize the following text : {PAYLOAD} end",
             "Classify the sentiment : {PAYLOAD}",
             "{PAYLOAD}"]


def make_tokenizer(cased=False):
    """`cased` adds a capitalized form of every word, as a cased subword vocabulary has."""
    words = SPECIALS + FILLER + ATTACK + TEMPLATE_WORDS
    if cased:
        words = list(dict.fromkeys(words + [w.capitalize() for w in FILLER + ATTACK]))
    vocab = {token: i for i, token in enumerate(words)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def make_base_model(tokenizer, seed=0):
    torch.manual_seed(seed)
    config = ModernBertConfig(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=512, local_attention=512,
        pad_token_id=tokenizer.pad_token_id, cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id, bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id, num_labels=2, attn_implementation="eager")
    return ModernBertForSequenceClassification(config)


def malicious_payload(rng, length):
    """An injection: attack words throughout, lightly interleaved with filler."""
    words = [rng.choice(ATTACK) if rng.random() < 0.8 else rng.choice(FILLER) for _ in range(length)]
    words[0] = rng.choice(["ignore", "Ignore", "disregard"])
    return " ".join(words)


def benign_payload(rng, length):
    """Benign text that still contains the odd attack word, as real text does."""
    return " ".join(rng.choice(ATTACK) if rng.random() < 0.05 else rng.choice(FILLER)
                    for _ in range(length))


def payload_split(n_malicious, n_benign, seed, lengths=(8, 40)):
    rng = random.Random(seed)
    rows = [dict(text=malicious_payload(rng, rng.randint(*lengths)), label=1, id=f"m{seed}_{i}")
            for i in range(n_malicious)]
    rows += [dict(text=benign_payload(rng, rng.randint(*lengths)), label=0, id=f"b{seed}_{i}")
             for i in range(n_benign)]
    malicious = Dataset.from_list([r for r in rows if r["label"] == 1])
    benign = Dataset.from_list([r for r in rows if r["label"] == 0])
    return malicious, benign


def email_pool(n, seed):
    rng = random.Random(seed)
    return Dataset.from_list([
        dict(id=f"e{seed}_{i}", text=" ".join(rng.choice(FILLER) for _ in range(rng.randint(150, 450))))
        for i in range(n)])


def make_collator(tokenizer, split, mode, seed, weights, artifacts_dir=None,
                  first_letter_upper_probability=None):
    from email_augmentation import EmailAugmentationCollator
    extra = ({} if first_letter_upper_probability is None
             else {"first_letter_upper_probability": first_letter_upper_probability})
    return EmailAugmentationCollator(
        tokenizer=tokenizer, email_pool=email_pool(40, seed), templates=TEMPLATES, split=split,
        mode=mode, seed=seed, max_length=512, weights=weights, artifacts_dir=artifacts_dir,
        region_supervision=True, **extra)


# Payloads whose first-letter case depends on the label, as when injections
# come from sources that start them "ignore ..." and benign text is ordinary
# capitalized prose. Content is kept weak on purpose so that the case of the
# first word is the cheapest cue available, which is when a model takes it.
OPENERS = ["ignore", "disregard", "override", "bypass", "reveal"]


def _cased(word, upper):
    return word.capitalize() if upper else word.lower()


def cased_payload_split(n_malicious, n_benign, seed, malicious_upper=0.05, benign_upper=0.9,
                        lengths=(8, 40)):
    rng = random.Random(seed)

    def body(length, density):
        return [rng.choice(ATTACK).lower() if rng.random() < density else rng.choice(FILLER)
                for _ in range(length - 1)]

    malicious = [dict(text=" ".join([_cased(rng.choice(OPENERS), rng.random() < malicious_upper)]
                                    + body(rng.randint(*lengths), 0.15)),
                      label=1, id=f"m{seed}_{i}") for i in range(n_malicious)]
    benign = []
    for i in range(n_benign):
        # Benign text sometimes opens with the same verbs ("Ignore the noise ...").
        opener = rng.choice(OPENERS) if rng.random() < 0.1 else rng.choice(FILLER)
        benign.append(dict(text=" ".join([_cased(opener, rng.random() < benign_upper)]
                                         + body(rng.randint(*lengths), 0.05)),
                           label=0, id=f"b{seed}_{i}"))
    return Dataset.from_list(malicious), Dataset.from_list(benign)
