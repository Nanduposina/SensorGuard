"""Chronological split and lightweight binary evaluation helpers."""

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from .detection import normalize_input


def chronological_split(frame: pd.DataFrame, train_fraction=0.7, validation_fraction=0.15):
    """Split by time without shuffling or allowing future rows into training."""
    if (
        not 0 < train_fraction < 1
        or not 0 <= validation_fraction < 1
        or train_fraction + validation_fraction >= 1
    ):
        raise ValueError("fractions must be nonnegative and sum to less than one")
    data = normalize_input(frame).sort_values("timestamp", kind="stable")
    times = data.timestamp.dropna().drop_duplicates().sort_values().to_numpy()
    if len(times) < 3:
        raise ValueError("chronological splitting requires at least three distinct timestamps")
    a = max(1, min(len(times) - 2, int(len(times) * train_fraction)))
    b = max(a + 1, min(len(times) - 1, int(len(times) * (train_fraction + validation_fraction))))
    t1, t2 = times[a], times[b]
    return (
        data[data.timestamp < t1].copy(),
        data[(data.timestamp >= t1) & (data.timestamp < t2)].copy(),
        data[data.timestamp >= t2].copy(),
    )


def evaluate_predictions(results: pd.DataFrame, labels: pd.Series):
    """Return basic metrics for explicit ground-truth labels; no threshold fitting."""
    pred = (
        results.groupby(results.index)["is_anomaly"].any().reindex(labels.index, fill_value=False)
    )
    return binary_metrics(labels.to_numpy(bool), pred.to_numpy(bool))


def binary_metrics(labels, predictions):
    """Calculate binary metrics and confusion counts with explicit undefined FPR."""
    import numpy as np

    y = np.asarray(labels, dtype=bool)
    pred = np.asarray(predictions, dtype=bool)
    if y.ndim != 1 or pred.ndim != 1 or len(y) != len(pred):
        raise ValueError("labels and predictions must be equally sized one-dimensional arrays")
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum())
    return {
        "evaluated_pairs": len(y),
        "positive_pairs": int(y.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def partially_labeled_metrics(labels, predictions):
    """Score known labels while leaving unlabeled observations unknown.

    Recall is defined over known positive labels. Precision, F1 and FPR are only
    returned when every evaluated observation has an explicit label; otherwise,
    predictions on unlabeled observations are reported separately.
    """
    import numpy as np

    truth = pd.array(labels, dtype="boolean")
    pred = np.asarray(predictions, dtype=bool)
    if truth.ndim != 1 or pred.ndim != 1 or len(truth) != len(pred):
        raise ValueError("labels and predictions must be equally sized one-dimensional arrays")
    known = ~np.asarray(pd.isna(truth))
    positive = np.asarray(truth.fillna(False), dtype=bool)
    tp = int((known & positive & pred).sum())
    fn = int((known & positive & ~pred).sum())
    known_negative = known & ~positive
    fp = int((known_negative & pred).sum())
    tn = int((known_negative & ~pred).sum())
    unlabeled = ~known
    unknown_alerts = int((unlabeled & pred).sum())
    fully_labeled = bool(known.all())
    positive_count = int((known & positive).sum())
    recall = tp / positive_count if positive_count else None
    if fully_labeled:
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else None
    else:
        precision = f1 = fpr = None
        fp = tn = None
    return {
        "evaluated_pairs": len(truth),
        "labeled_pairs": int(known.sum()),
        "unlabeled_pairs": int(unlabeled.sum()),
        "positive_pairs": positive_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "total_alerts": int(pred.sum()),
        "alerts_on_unlabeled_pairs": unknown_alerts,
        "unlabeled_alert_rate": unknown_alerts / int(unlabeled.sum()) if unlabeled.any() else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }
