"""Small Trainer/HPO integration for the token-only maximum-subarray model."""

import numpy as np
from sklearn.metrics import average_precision_score
from transformers import EvalPrediction

from asl_loss import ASLTrainer

# Validation runs at one benign:malicious ratio and deployment at another, so a
# metric read straight off the validation set answers a question nobody asked.
# Reweighting the negatives by the ratio of the two prevalences turns it back
# into the deployment question without rebuilding the set.
LENGTH_BUCKETS = ((0, 128, "short"), (128, 256, "medium"), (256, 513, "long"))


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


def _deployment_weights(labels, validation_ratio, deployment_ratio):
    """Negatives stand in for `deployment_ratio / validation_ratio` of themselves.

    Computed here rather than through the calibration helper because this is the
    whole of it, and a metric that silently reweights the wrong way would look
    like a better model rather than like a bug.
    """
    scale = float(deployment_ratio) / float(validation_ratio)
    return np.where(np.asarray(labels) == 1, 1.0, scale)


def segment_metrics(base_metrics, validation_ratio=50, deployment_ratio=500):
    """Report per-length behaviour and a selection score that cannot saturate.

    `val_f1` at the validation prevalence separates the top trials by under a
    thousandth, which is seed noise, and it is blind to the two properties this
    model actually has to hold: recall on long inputs, and a selected segment
    wide enough to survive losing a token. The returned `selection_pr_auc` is the
    WORST length bucket's PR-AUC at deployment prevalence, so a trial cannot buy
    the metric by giving up the regime it was built for.
    """
    def compute(prediction):
        logits, diagnostics = prediction.predictions
        labels = np.asarray(prediction.label_ids)
        result = base_metrics(EvalPrediction(predictions=logits, label_ids=labels))
        guessed = logits.argmax(axis=-1)
        margin = logits[:, 1] - logits[:, 0]
        bucket_scores = []
        for lower, upper, name in LENGTH_BUCKETS:
            selected = (diagnostics[:, 0] >= lower) & (diagnostics[:, 0] < upper)
            positives, negatives = selected & (labels == 1), selected & (labels == 0)
            result[f"{name}_positives"] = int(positives.sum())
            result[f"{name}_negatives"] = int(negatives.sum())
            if positives.any():
                result[f"{name}_recall"] = float((guessed[positives] == 1).mean())
            if negatives.any():
                result[f"{name}_fpr"] = float((guessed[negatives] == 1).mean())
            if selected.any():
                # Width of the decoded segment. A value stuck at one or two
                # tokens means the score rests on a single token and one
                # substitution removes it, whatever the headline metric says.
                result[f"{name}_segment_length"] = float(diagnostics[selected, 1].mean())
            if positives.any() and negatives.any():
                score = float(average_precision_score(
                    labels[selected], margin[selected],
                    sample_weight=_deployment_weights(
                        labels[selected], validation_ratio, deployment_ratio),
                ))
                result[f"{name}_pr_auc"] = score
                bucket_scores.append(score)
        if bucket_scores:
            result["selection_pr_auc"] = float(min(bucket_scores))
            result["mean_segment_length"] = float(diagnostics[:, 1].mean())
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
