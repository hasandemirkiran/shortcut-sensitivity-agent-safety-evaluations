from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    confidence: float
    replicates: int
    clusters: int
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _arrays(
    clusters: Iterable[str], strata: Iterable[str] | None
) -> tuple[np.ndarray, np.ndarray]:
    cluster_values = np.asarray(list(clusters), dtype=str)
    if cluster_values.size == 0:
        raise ValueError("Bootstrap requires at least one observation")
    if np.any(cluster_values == ""):
        raise ValueError("Every observation requires a non-empty cluster identifier")
    if strata is None:
        stratum_values = np.full(cluster_values.size, "all", dtype=str)
    else:
        stratum_values = np.asarray(list(strata), dtype=str)
        if stratum_values.size != cluster_values.size:
            raise ValueError("Strata and cluster arrays must have equal length")
    return cluster_values, stratum_values


def _cluster_layout(
    cluster_values: np.ndarray, stratum_values: np.ndarray
) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    layout: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for stratum in np.unique(stratum_values):
        stratum_indices = np.flatnonzero(stratum_values == stratum)
        cluster_map: dict[str, np.ndarray] = {}
        for cluster in np.unique(cluster_values[stratum_indices]):
            cluster_map[str(cluster)] = stratum_indices[
                cluster_values[stratum_indices] == cluster
            ]
        layout[str(stratum)] = (
            np.asarray(sorted(cluster_map), dtype=str),
            cluster_map,
        )
    return layout


def stratified_cluster_bootstrap(
    statistic: Callable[[np.ndarray], float],
    *,
    observations: int,
    clusters: Iterable[str],
    strata: Iterable[str] | None = None,
    replicates: int = 10_000,
    confidence: float = 0.95,
    seed: int = 260110156,
) -> Interval:
    """Percentile interval with trajectory clusters sampled within each source."""

    if observations <= 0:
        raise ValueError("Bootstrap requires at least one observation")
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one")
    cluster_values, stratum_values = _arrays(clusters, strata)
    if cluster_values.size != observations:
        raise ValueError("Observation and cluster counts must match")
    layout = _cluster_layout(cluster_values, stratum_values)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)

    for replicate in range(replicates):
        sampled_parts: list[np.ndarray] = []
        for cluster_names, cluster_map in layout.values():
            sampled_names = rng.choice(cluster_names, size=cluster_names.size, replace=True)
            sampled_parts.extend(cluster_map[str(name)] for name in sampled_names)
        sampled_indices = np.concatenate(sampled_parts)
        estimates[replicate] = statistic(sampled_indices)

    alpha = (1.0 - confidence) / 2.0
    all_indices = np.arange(observations, dtype=int)
    return Interval(
        point=float(statistic(all_indices)),
        lower=float(np.quantile(estimates, alpha)),
        upper=float(np.quantile(estimates, 1.0 - alpha)),
        confidence=confidence,
        replicates=replicates,
        clusters=len(np.unique(cluster_values)),
        seed=seed,
    )


def clustered_mean_interval(
    values: Iterable[float],
    *,
    clusters: Iterable[str],
    strata: Iterable[str] | None = None,
    replicates: int = 10_000,
    confidence: float = 0.95,
    seed: int = 260110156,
) -> Interval:
    observed = np.asarray(list(values), dtype=float)
    return _clustered_ratio_interval(
        observed,
        np.ones(observed.size, dtype=float),
        clusters=clusters,
        strata=strata,
        replicates=replicates,
        confidence=confidence,
        seed=seed,
    )


def clustered_metric_difference(
    labels: Iterable[int],
    first: Iterable[int],
    second: Iterable[int],
    *,
    metric: LiteralMetric,
    clusters: Iterable[str],
    strata: Iterable[str] | None = None,
    replicates: int = 10_000,
    confidence: float = 0.95,
    seed: int = 260110156,
) -> Interval:
    y = np.asarray(list(labels), dtype=np.int8)
    a = np.asarray(list(first), dtype=np.int8)
    b = np.asarray(list(second), dtype=np.int8)
    if y.size == 0 or y.size != a.size or y.size != b.size:
        raise ValueError("Metric difference requires equally sized, non-empty inputs")
    if metric == "recall":
        selected = y == 1
        numerator = selected * ((a == 1).astype(float) - (b == 1).astype(float))
        denominator = selected.astype(float)
    elif metric == "specificity":
        selected = y == 0
        numerator = selected * ((a == 0).astype(float) - (b == 0).astype(float))
        denominator = selected.astype(float)
    elif metric == "accuracy":
        numerator = (a == y).astype(float) - (b == y).astype(float)
        denominator = np.ones(y.size, dtype=float)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return _clustered_ratio_interval(
        numerator,
        denominator,
        clusters=clusters,
        strata=strata,
        replicates=replicates,
        confidence=confidence,
        seed=seed,
    )


LiteralMetric = str


def _clustered_ratio_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    clusters: Iterable[str],
    strata: Iterable[str] | None,
    replicates: int,
    confidence: float,
    seed: int,
) -> Interval:
    """Fast clustered bootstrap for statistics expressible as a ratio of sums."""

    if numerator.size == 0 or numerator.size != denominator.size:
        raise ValueError("Ratio bootstrap requires equally sized, non-empty arrays")
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one")
    cluster_values, stratum_values = _arrays(clusters, strata)
    if cluster_values.size != numerator.size:
        raise ValueError("Observation and cluster counts must match")
    layout = _cluster_layout(cluster_values, stratum_values)
    cluster_summaries: list[tuple[np.ndarray, np.ndarray]] = []
    for cluster_names, cluster_map in layout.values():
        numerator_sums = np.asarray(
            [numerator[cluster_map[str(name)]].sum() for name in cluster_names],
            dtype=float,
        )
        denominator_sums = np.asarray(
            [denominator[cluster_map[str(name)]].sum() for name in cluster_names],
            dtype=float,
        )
        cluster_summaries.append((numerator_sums, denominator_sums))

    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    batch_size = 512
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        size = stop - start
        sampled_numerator = np.zeros(size, dtype=float)
        sampled_denominator = np.zeros(size, dtype=float)
        for numerator_sums, denominator_sums in cluster_summaries:
            cluster_count = numerator_sums.size
            sampled = rng.integers(
                0, cluster_count, size=(size, cluster_count), endpoint=False
            )
            sampled_numerator += numerator_sums[sampled].sum(axis=1)
            sampled_denominator += denominator_sums[sampled].sum(axis=1)
        estimates[start:stop] = np.divide(
            sampled_numerator,
            sampled_denominator,
            out=np.full(size, np.nan, dtype=float),
            where=sampled_denominator != 0,
        )

    alpha = (1.0 - confidence) / 2.0
    point_denominator = float(denominator.sum())
    point = (
        float(numerator.sum() / point_denominator)
        if point_denominator
        else float("nan")
    )
    return Interval(
        point=point,
        lower=float(np.nanquantile(estimates, alpha)),
        upper=float(np.nanquantile(estimates, 1.0 - alpha)),
        confidence=confidence,
        replicates=replicates,
        clusters=len(np.unique(cluster_values)),
        seed=seed,
    )
