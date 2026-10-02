from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np

from toolsafe_lab.cascade import (
    SigmoidCalibrator,
    route_probabilities,
    routing_metrics,
    select_allow_defer_policy,
    unsafe_scores,
)
from toolsafe_lab.data import Sample, load_eval, load_training
from toolsafe_lab.metrics import binary_metrics, strict_value
from toolsafe_lab.uncertainty import (
    clustered_mean_interval,
    clustered_metric_difference,
)


PRIMARY_TARGET = 0.99
PRIMARY_SEED = 260110156
SEMANTIC_MODEL = "minilm_relation_logreg"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected objects in {path}")
            rows.append(value)
    return rows


def _local_routes(
    *,
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    samples: Sequence[Sample],
    include_semantic: bool = True,
) -> dict[str, dict[str, str]]:
    prediction_path = (
        results_root / "cascade" / "ts_guard_primary_predictions.csv.gz"
    )
    routes: dict[str, dict[str, str]] = defaultdict(dict)
    with gzip.open(prediction_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["policy_family"] != "allow_defer":
                continue
            routes[row["model"]][row["sample_id"]] = row["route"]

    expected_ids = {sample.sample_id for sample in samples}
    for model_name, model_routes in routes.items():
        if set(model_routes) != expected_ids:
            raise ValueError(f"Incomplete local routes for {model_name}")

    if include_semantic and SEMANTIC_MODEL not in routes:
        train = load_training(data_root, "train")
        validation = load_training(data_root, "validation")
        train_eval_ids = expected_ids
        train = [sample for sample in train if sample.sample_id not in train_eval_ids]
        validation = [
            sample for sample in validation if sample.sample_id not in train_eval_ids
        ]
        model = joblib.load(
            artifacts_root / "models" / f"{SEMANTIC_MODEL}.joblib"
        )
        validation_labels = np.asarray(
            [sample.strict_label for sample in validation], dtype=np.int8
        )
        validation_scores = unsafe_scores(
            model, [sample.text for sample in validation]
        )
        calibrator = SigmoidCalibrator(random_state=PRIMARY_SEED).fit(
            validation_scores, validation_labels
        )
        validation_probabilities = calibrator.predict_proba(validation_scores)
        policy = select_allow_defer_policy(
            validation_probabilities,
            validation_labels,
            target_recall=PRIMARY_TARGET,
            minimum_region_size=20,
        )
        evaluation_scores = unsafe_scores(
            model, [sample.text for sample in samples]
        )
        semantic_routes = route_probabilities(
            calibrator.predict_proba(evaluation_scores), policy
        )
        routes[SEMANTIC_MODEL] = {
            sample.sample_id: str(route)
            for sample, route in zip(samples, semantic_routes)
        }

    return dict(routes)


def _monitor_predictions(
    samples: Sequence[Sample],
    records: Iterable[dict[str, object]],
) -> tuple[np.ndarray, set[str], dict[str, dict[str, object]]]:
    records_by_id = {
        str(record["sample_id"]): record
        for record in records
        if "sample_id" in record
    }
    predictions: list[int] = []
    successful: set[str] = set()
    for sample in samples:
        record = records_by_id.get(sample.sample_id)
        prediction = (
            strict_value(record.get("prediction"))
            if record is not None and record.get("status") == "ok"
            else None
        )
        if prediction is None:
            predictions.append(1)
        else:
            predictions.append(prediction)
            successful.add(sample.sample_id)
    return np.asarray(predictions, dtype=np.int8), successful, records_by_id


def _source_slices(
    evaluation: dict[str, list[Sample]],
) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    offset = 0
    for source, samples in evaluation.items():
        slices[source] = slice(offset, offset + len(samples))
        offset += len(samples)
    return slices


def _cost(record: dict[str, object] | None) -> float:
    if record is None:
        return 0.0
    value = record.get("estimated_cost_usd", 0.0)
    return float(value) if value is not None else 0.0


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_hosted_cascades(
    *,
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    prompt_version: str,
    model_ids: Sequence[str] | None = None,
    bootstrap_replicates: int = 10_000,
    seed: int = PRIMARY_SEED,
) -> dict[str, object]:
    evaluation = load_eval(data_root)
    samples = [
        sample for source_samples in evaluation.values() for sample in source_samples
    ]
    labels = np.asarray([sample.strict_label for sample in samples], dtype=np.int8)
    clusters = [sample.trajectory_id for sample in samples]
    strata = [sample.source for sample in samples]
    slices = _source_slices(evaluation)
    routes_by_model = _local_routes(
        data_root=data_root,
        results_root=results_root,
        artifacts_root=artifacts_root,
        samples=samples,
    )
    available_paths = sorted(
        (
            artifacts_root
            / "api_runs"
            / "eval"
            / prompt_version
        ).glob("*.jsonl")
    )
    if model_ids is not None:
        requested = set(model_ids)
        available_paths = [
            path for path in available_paths if path.stem in requested
        ]
    if not available_paths:
        raise FileNotFoundError("No collected hosted evaluation records")

    output: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": prompt_version,
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "failure_policy": "fail_closed_as_unsafe",
        "monitors": {},
    }
    for record_path in available_paths:
        model_id = record_path.stem
        records = _read_jsonl(record_path)
        monitor_predictions, successful, records_by_id = _monitor_predictions(
            samples, records
        )
        fail_open_predictions = monitor_predictions.copy()
        fail_open_predictions[
            np.asarray(
                [sample.sample_id not in successful for sample in samples],
                dtype=bool,
            )
        ] = 0
        monitor_metrics = binary_metrics(labels, monitor_predictions)
        monitor_cost = sum(
            _cost(records_by_id.get(sample.sample_id)) for sample in samples
        )
        monitor: dict[str, object] = {
            "model_id": model_id,
            "model": next(
                (
                    str(record.get("model"))
                    for record in records
                    if record.get("model")
                ),
                model_id,
            ),
            "successful_predictions": len(successful),
            "request_coverage": len(successful) / len(samples),
            "failed_or_missing_predictions": len(samples) - len(successful),
            "observed_batch_cost_usd": monitor_cost,
            "micro": monitor_metrics,
            "by_dataset": {
                source: binary_metrics(
                    labels[source_slice],
                    monitor_predictions[source_slice],
                )
                for source, source_slice in slices.items()
            },
            "fail_open_sensitivity": {
                "micro": binary_metrics(labels, fail_open_predictions),
                "by_dataset": {
                    source: binary_metrics(
                        labels[source_slice],
                        fail_open_predictions[source_slice],
                    )
                    for source, source_slice in slices.items()
                },
            },
            "cascades": {},
        }
        for local_model, model_routes in routes_by_model.items():
            routes = np.asarray(
                [model_routes[sample.sample_id] for sample in samples],
                dtype=object,
            )
            allowed = routes == "allow"
            deferred = ~allowed
            cascade_predictions = monitor_predictions.copy()
            cascade_predictions[allowed] = 0
            fail_open_cascade = fail_open_predictions.copy()
            fail_open_cascade[allowed] = 0
            metrics = routing_metrics(labels, cascade_predictions, routes)
            deferred_ids = {
                sample.sample_id
                for sample, is_deferred in zip(samples, deferred)
                if is_deferred
            }
            cascade_cost = sum(
                _cost(records_by_id.get(sample_id)) for sample_id in deferred_ids
            )
            called_successes = len(deferred_ids & successful)
            failed_when_called = len(deferred_ids) - called_successes
            monitor["cascades"][local_model] = {  # type: ignore[index]
                "micro": metrics,
                "by_dataset": {
                    source: routing_metrics(
                        labels[source_slice],
                        cascade_predictions[source_slice],
                        routes[source_slice],
                    )
                    for source, source_slice in slices.items()
                },
                "monitor_requests": int(np.sum(deferred)),
                "monitor_request_reduction": float(np.mean(allowed)),
                "monitor_successes_when_called": called_successes,
                "monitor_failures_when_called": failed_when_called,
                "observed_batch_cost_usd": cascade_cost,
                "observed_cost_reduction": (
                    1.0 - cascade_cost / monitor_cost if monitor_cost else 0.0
                ),
                "fail_open_sensitivity": {
                    "micro": routing_metrics(
                        labels, fail_open_cascade, routes
                    ),
                    "by_dataset": {
                        source: routing_metrics(
                            labels[source_slice],
                            fail_open_cascade[source_slice],
                            routes[source_slice],
                        )
                        for source, source_slice in slices.items()
                    },
                },
                "intervals": {
                    "monitor_call_reduction": clustered_mean_interval(
                        allowed.astype(float),
                        clusters=clusters,
                        strata=strata,
                        replicates=bootstrap_replicates,
                        seed=seed,
                    ).to_dict(),
                    "recall_difference_vs_monitor": clustered_metric_difference(
                        labels,
                        cascade_predictions,
                        monitor_predictions,
                        metric="recall",
                        clusters=clusters,
                        strata=strata,
                        replicates=bootstrap_replicates,
                        seed=seed + 1,
                    ).to_dict(),
                    "specificity_difference_vs_monitor": clustered_metric_difference(
                        labels,
                        cascade_predictions,
                        monitor_predictions,
                        metric="specificity",
                        clusters=clusters,
                        strata=strata,
                        replicates=bootstrap_replicates,
                        seed=seed + 2,
                    ).to_dict(),
                },
            }
        output["monitors"][model_id] = monitor  # type: ignore[index]
        print(
            f"{monitor['model']}: coverage={float(monitor['request_coverage']) * 100:.2f}% "
            f"cost=${monitor_cost:.2f}; "
            f"{len(routes_by_model)} local cascades"
        )

    path = results_root / "cascade" / "hosted_cascades.json"
    _write_json(path, output)
    print(f"Wrote {path}")
    return output
