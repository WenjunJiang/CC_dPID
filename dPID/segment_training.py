"""Small Trainer/HPO integration for the token-only maximum-subarray model."""

import numpy as np
from transformers import EvalPrediction

from asl_loss import ASLTrainer


class SegmentTrainer(ASLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Our losses are micro-batch means, not sums normalized by the number
        # of items across the accumulated batch. Ask Trainer to divide once.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # The model computes both objectives from its deployed segment score.
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def segment_metrics(base_metrics):
    def compute(prediction):
        logits, diagnostics = prediction.predictions
        labels = prediction.label_ids
        result = base_metrics(EvalPrediction(predictions=logits, label_ids=labels))
        guessed = logits.argmax(axis=-1)
        for lower, upper, name in ((0, 128, "short"), (128, 256, "medium"), (256, 513, "long")):
            selected = (diagnostics[:, 0] >= lower) & (diagnostics[:, 0] < upper)
            positives, negatives = selected & (labels == 1), selected & (labels == 0)
            result[f"{name}_positives"] = int(positives.sum())
            result[f"{name}_negatives"] = int(negatives.sum())
            if positives.any():
                result[f"{name}_recall"] = float((guessed[positives] == 1).mean())
            if negatives.any():
                result[f"{name}_fpr"] = float((guessed[negatives] == 1).mean())
            if selected.any():
                result[f"{name}_segment_length"] = float(diagnostics[selected, 1].mean())
        return result
    return compute


def initial_segment_trials(search_space, model_name, seed=42):
    """Cover every categorical value in the first five trials, not a full grid."""
    rng = np.random.default_rng(seed)
    categorical = {key: list(cfg["values"]) for key, cfg in search_space.items() if cfg["type"] == "choice"}
    for values in categorical.values():
        rng.shuffle(values)
    points = []
    for i in range(max(map(len, categorical.values()))):
        # Ray supplies constant config fields separately. Including its fixed
        # seed here makes Optuna reject the warm-start point's dimensionality.
        point = {"model_name": model_name}
        for key, cfg in search_space.items():
            if key in categorical:
                point[key] = categorical[key][i % len(categorical[key])]
            elif cfg["type"] == "loguniform":
                point[key] = float(np.exp(rng.uniform(np.log(cfg["low"]), np.log(cfg["high"]))))
            else:
                raise ValueError(f"Unsupported initial segment search distribution: {key}")
        points.append(point)
    return points
