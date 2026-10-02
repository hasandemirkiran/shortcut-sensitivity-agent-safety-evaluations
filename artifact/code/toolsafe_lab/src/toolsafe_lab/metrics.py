from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def strict_value(value: object) -> int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number not in {0.0, 0.5, 1.0}:
        return None
    return int(number != 0.0)


def strict_pairs(
    predictions: Iterable[object], labels: Iterable[object]
) -> tuple[list[int], list[int], int]:
    y_pred: list[int] = []
    y_true: list[int] = []
    total = 0
    for prediction, label in zip(predictions, labels):
        total += 1
        mapped_prediction = strict_value(prediction)
        mapped_label = strict_value(label)
        if mapped_prediction is None or mapped_label is None:
            continue
        y_pred.append(mapped_prediction)
        y_true.append(mapped_label)
    return y_pred, y_true, total


def binary_metrics(
    labels: Iterable[int],
    predictions: Iterable[int],
    *,
    total_predictions: int | None = None,
) -> dict[str, float | int]:
    y_true = np.asarray(list(labels), dtype=np.int8)
    y_pred = np.asarray(list(predictions), dtype=np.int8)
    if y_true.size == 0 or y_true.size != y_pred.size:
        raise ValueError("Metrics require equally sized, non-empty inputs")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = int(total_predictions if total_predictions is not None else y_true.size)
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    false_positive_rate = float(fp / (fp + tn)) if fp + tn else 0.0
    false_negative_rate = float(fn / (fn + tp)) if fn + tp else 0.0
    return {
        "n": int(y_true.size),
        "coverage": float(y_true.size / total) if total else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

