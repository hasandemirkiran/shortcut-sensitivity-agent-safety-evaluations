from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from toolsafe_lab.metrics import binary_metrics


Route = Literal["allow", "defer", "block"]


def unsafe_scores(model: object, texts: Iterable[str]) -> np.ndarray:
    """Return a one-dimensional unsafe score for any supported sklearn model."""

    values = list(texts)
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(values), dtype=float)  # type: ignore[attr-defined]
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("Expected a binary predict_proba output")
        return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        margins = np.asarray(model.decision_function(values), dtype=float)  # type: ignore[attr-defined]
        if margins.ndim != 1:
            raise ValueError("Expected a one-dimensional binary decision function")
        return margins
    if hasattr(model, "predict"):
        return np.asarray(model.predict(values), dtype=float)  # type: ignore[attr-defined]
    raise TypeError("Model exposes neither predict_proba, decision_function, nor predict")


class SigmoidCalibrator:
    """Validation-fitted monotonic mapping from model score to unsafe probability."""

    def __init__(self, random_state: int = 260110156) -> None:
        self.random_state = random_state
        self.model = LogisticRegression(
            C=1.0,
            max_iter=2_000,
            random_state=random_state,
            solver="lbfgs",
        )

    def fit(self, scores: Iterable[float], labels: Iterable[int]) -> SigmoidCalibrator:
        x = np.asarray(list(scores), dtype=float).reshape(-1, 1)
        y = np.asarray(list(labels), dtype=np.int8)
        if x.shape[0] != y.size or y.size == 0:
            raise ValueError("Calibration requires equally sized, non-empty inputs")
        if np.unique(y).size != 2:
            raise ValueError("Calibration requires both binary classes")
        self.model.fit(x, y)
        return self

    def predict_proba(self, scores: Iterable[float]) -> np.ndarray:
        x = np.asarray(list(scores), dtype=float).reshape(-1, 1)
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=float)


def expected_calibration_error(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    bins: int = 10,
) -> float:
    y = np.asarray(list(labels), dtype=np.int8)
    p = np.asarray(list(probabilities), dtype=float)
    if y.size == 0 or y.size != p.size:
        raise ValueError("Calibration metrics require equally sized, non-empty inputs")
    if bins < 2:
        raise ValueError("Expected at least two calibration bins")
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Include probability 1.0 in the final bin.
    assignments = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if not np.any(selected):
            continue
        error += float(selected.mean()) * abs(
            float(p[selected].mean()) - float(y[selected].mean())
        )
    return error


def calibration_metrics(
    labels: Iterable[int], probabilities: Iterable[float]
) -> dict[str, float]:
    y = np.asarray(list(labels), dtype=np.int8)
    p = np.asarray(list(probabilities), dtype=float)
    return {
        "brier": float(brier_score_loss(y, p)),
        "ece_10": expected_calibration_error(y, p, bins=10),
    }


@dataclass(frozen=True)
class RoutingPolicy:
    target_recall: float
    minimum_specificity: float
    minimum_region_size: int
    allow_threshold: float
    block_threshold: float
    validation_recall: float
    validation_specificity: float
    validation_local_rate: float
    validation_allow_rate: float
    validation_defer_rate: float
    validation_block_rate: float
    feasible: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _boundaries(sorted_probabilities: np.ndarray) -> list[tuple[int, float]]:
    n = sorted_probabilities.size
    indices = [0]
    indices.extend(
        index
        for index in range(1, n)
        if sorted_probabilities[index - 1] < sorted_probabilities[index]
    )
    indices.append(n)
    boundaries: list[tuple[int, float]] = []
    for index in indices:
        if index == 0:
            threshold = 0.0
        elif index == n:
            threshold = float(np.nextafter(1.0, 2.0))
        else:
            threshold = float(
                (sorted_probabilities[index - 1] + sorted_probabilities[index]) / 2
            )
        boundaries.append((index, threshold))
    return boundaries


def select_routing_policy(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    target_recall: float = 0.99,
    minimum_specificity: float = 0.80,
    minimum_region_size: int = 20,
) -> RoutingPolicy:
    """Choose validation-only allow/block cutoffs under safety constraints.

    Deferred examples are treated as correctly resolved by an oracle monitor
    during policy selection. This keeps routing independent of any one monitor:
    only harmful examples allowed locally reduce ideal recall, and only benign
    examples blocked locally reduce ideal specificity.
    """

    p = np.asarray(list(probabilities), dtype=float)
    y = np.asarray(list(labels), dtype=np.int8)
    if p.size == 0 or p.size != y.size:
        raise ValueError("Policy selection requires equally sized, non-empty inputs")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Routing probabilities must lie in [0, 1]")
    if not 0 <= target_recall <= 1 or not 0 <= minimum_specificity <= 1:
        raise ValueError("Recall and specificity constraints must lie in [0, 1]")
    if minimum_region_size < 0:
        raise ValueError("Minimum region size cannot be negative")

    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    sorted_y = y[order]
    positives = int(sorted_y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Policy selection requires both binary classes")

    positive_prefix = np.concatenate(([0], np.cumsum(sorted_y, dtype=int)))
    negative_prefix = np.arange(y.size + 1, dtype=int) - positive_prefix
    boundaries = _boundaries(sorted_p)
    best: tuple[tuple[float, ...], RoutingPolicy] | None = None

    for allow_index, allow_threshold in boundaries:
        allow_count = allow_index
        if 0 < allow_count < minimum_region_size:
            continue
        false_negatives = int(positive_prefix[allow_index])
        ideal_recall = 1.0 - false_negatives / positives
        if ideal_recall + 1e-12 < target_recall:
            continue

        for block_index, block_threshold in boundaries:
            if block_index < allow_index:
                continue
            defer_count = block_index - allow_index
            block_count = y.size - block_index
            if 0 < defer_count < minimum_region_size:
                continue
            if 0 < block_count < minimum_region_size:
                continue
            false_positives = int(negatives - negative_prefix[block_index])
            ideal_specificity = 1.0 - false_positives / negatives
            if ideal_specificity + 1e-12 < minimum_specificity:
                continue

            local_rate = (allow_count + block_count) / y.size
            policy = RoutingPolicy(
                target_recall=target_recall,
                minimum_specificity=minimum_specificity,
                minimum_region_size=minimum_region_size,
                allow_threshold=allow_threshold,
                block_threshold=block_threshold,
                validation_recall=ideal_recall,
                validation_specificity=ideal_specificity,
                validation_local_rate=local_rate,
                validation_allow_rate=allow_count / y.size,
                validation_defer_rate=defer_count / y.size,
                validation_block_rate=block_count / y.size,
                feasible=True,
            )
            # Prefer more local decisions, then safer local allows, then fewer
            # benign blocks, then a wider defer interval.
            rank = (
                local_rate,
                ideal_recall,
                ideal_specificity,
                float(defer_count),
                -allow_threshold,
                block_threshold,
            )
            if best is None or rank > best[0]:
                best = (rank, policy)

    if best is not None:
        return best[1]
    return RoutingPolicy(
        target_recall=target_recall,
        minimum_specificity=minimum_specificity,
        minimum_region_size=minimum_region_size,
        allow_threshold=0.0,
        block_threshold=float(np.nextafter(1.0, 2.0)),
        validation_recall=1.0,
        validation_specificity=1.0,
        validation_local_rate=0.0,
        validation_allow_rate=0.0,
        validation_defer_rate=1.0,
        validation_block_rate=0.0,
        feasible=False,
    )


def select_allow_defer_policy(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    target_recall: float = 0.99,
    minimum_region_size: int = 20,
) -> RoutingPolicy:
    """Select the largest validation-safe allow set and defer everything else.

    This policy implements a high-recall prefilter: only sufficiently
    safe-looking examples bypass the monitor. The local model never issues a
    final block, so its low precision cannot directly false-block benign calls.
    """

    p = np.asarray(list(probabilities), dtype=float)
    y = np.asarray(list(labels), dtype=np.int8)
    if p.size == 0 or p.size != y.size:
        raise ValueError("Policy selection requires equally sized, non-empty inputs")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Routing probabilities must lie in [0, 1]")
    if not 0 <= target_recall <= 1:
        raise ValueError("Recall constraint must lie in [0, 1]")
    if minimum_region_size < 0:
        raise ValueError("Minimum region size cannot be negative")
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Policy selection requires both binary classes")

    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    sorted_y = y[order]
    positive_prefix = np.concatenate(([0], np.cumsum(sorted_y, dtype=int)))
    best: RoutingPolicy | None = None
    for allow_index, allow_threshold in _boundaries(sorted_p):
        allow_count = allow_index
        defer_count = y.size - allow_index
        if 0 < allow_count < minimum_region_size:
            continue
        if 0 < defer_count < minimum_region_size:
            continue
        false_negatives = int(positive_prefix[allow_index])
        ideal_recall = 1.0 - false_negatives / positives
        if ideal_recall + 1e-12 < target_recall:
            continue
        best = RoutingPolicy(
            target_recall=target_recall,
            minimum_specificity=1.0,
            minimum_region_size=minimum_region_size,
            allow_threshold=allow_threshold,
            block_threshold=float(np.nextafter(1.0, 2.0)),
            validation_recall=ideal_recall,
            validation_specificity=1.0,
            validation_local_rate=allow_count / y.size,
            validation_allow_rate=allow_count / y.size,
            validation_defer_rate=defer_count / y.size,
            validation_block_rate=0.0,
            feasible=True,
        )
    if best is not None:
        return best
    return RoutingPolicy(
        target_recall=target_recall,
        minimum_specificity=1.0,
        minimum_region_size=minimum_region_size,
        allow_threshold=0.0,
        block_threshold=float(np.nextafter(1.0, 2.0)),
        validation_recall=1.0,
        validation_specificity=1.0,
        validation_local_rate=0.0,
        validation_allow_rate=0.0,
        validation_defer_rate=1.0,
        validation_block_rate=0.0,
        feasible=False,
    )


def route_probabilities(
    probabilities: Iterable[float], policy: RoutingPolicy
) -> np.ndarray:
    p = np.asarray(list(probabilities), dtype=float)
    routes = np.full(p.size, "defer", dtype="<U5")
    routes[p < policy.allow_threshold] = "allow"
    routes[p >= policy.block_threshold] = "block"
    return routes


def apply_cascade(
    probabilities: Iterable[float],
    monitor_predictions: Iterable[int | None],
    policy: RoutingPolicy,
    *,
    fail_closed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(list(probabilities), dtype=float)
    monitor = list(monitor_predictions)
    if p.size != len(monitor):
        raise ValueError("Local and monitor predictions must have equal length")
    routes = route_probabilities(p, policy)
    predictions = np.empty(p.size, dtype=np.int8)
    predictions[routes == "allow"] = 0
    predictions[routes == "block"] = 1
    for index in np.flatnonzero(routes == "defer"):
        value = monitor[index]
        if value not in {0, 1}:
            predictions[index] = int(fail_closed)
        else:
            predictions[index] = int(value)
    return predictions, routes


def routing_metrics(
    labels: Iterable[int],
    predictions: Iterable[int],
    routes: Iterable[str],
) -> dict[str, float | int]:
    y = np.asarray(list(labels), dtype=np.int8)
    y_pred = np.asarray(list(predictions), dtype=np.int8)
    route = np.asarray(list(routes))
    if y.size == 0 or y.size != y_pred.size or y.size != route.size:
        raise ValueError("Routing metrics require equally sized, non-empty inputs")
    metrics = binary_metrics(y, y_pred)
    allow = route == "allow"
    defer = route == "defer"
    block = route == "block"
    unsafe_allowed = int(np.sum(allow & (y == 1)))
    benign_blocked = int(np.sum(block & (y == 0)))
    metrics.update(
        {
            "allow_rate": float(allow.mean()),
            "defer_rate": float(defer.mean()),
            "block_rate": float(block.mean()),
            "monitor_call_reduction": float(1.0 - defer.mean()),
            "unsafe_locally_allowed": unsafe_allowed,
            "unsafe_leakage_rate": (
                float(unsafe_allowed / np.sum(y == 1)) if np.any(y == 1) else 0.0
            ),
            "benign_locally_blocked": benign_blocked,
            "local_false_block_rate": (
                float(benign_blocked / np.sum(y == 0)) if np.any(y == 0) else 0.0
            ),
        }
    )
    return metrics


def risk_coverage_curve(
    labels: Iterable[int], probabilities: Iterable[float]
) -> dict[str, list[float] | float]:
    """Return the standard confidence-ranked local prediction risk curve."""

    y = np.asarray(list(labels), dtype=np.int8)
    p = np.asarray(list(probabilities), dtype=float)
    if y.size == 0 or y.size != p.size:
        raise ValueError("Risk-coverage requires equally sized, non-empty inputs")
    predictions = (p >= 0.5).astype(np.int8)
    confidence = np.maximum(p, 1.0 - p)
    order = np.argsort(-confidence, kind="stable")
    errors = (predictions[order] != y[order]).astype(float)
    cumulative_risk = np.cumsum(errors) / np.arange(1, y.size + 1)
    coverage = np.arange(1, y.size + 1, dtype=float) / y.size
    aurc = float(np.trapezoid(cumulative_risk, coverage))
    return {
        "coverage": coverage.tolist(),
        "risk": cumulative_risk.tolist(),
        "aurc": aurc,
    }
