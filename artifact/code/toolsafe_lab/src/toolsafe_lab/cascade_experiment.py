from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from toolsafe_lab.cascade import (
    SigmoidCalibrator,
    apply_cascade,
    calibration_metrics,
    risk_coverage_curve,
    route_probabilities,
    routing_metrics,
    select_allow_defer_policy,
    select_routing_policy,
    unsafe_scores,
)
from toolsafe_lab.data import Sample, load_eval, load_training
from toolsafe_lab.metrics import binary_metrics, strict_value
from toolsafe_lab.models import RANDOM_STATE, model_factories
from toolsafe_lab.robustness import TRANSFORMATIONS, transform_samples
from toolsafe_lab.uncertainty import (
    clustered_mean_interval,
    clustered_metric_difference,
)


TARGET_RECALLS = (0.995, 0.99, 0.98, 0.95)
PRIMARY_TARGET = 0.99
REFERENCE_DIRECTORIES = {
    "AgentHarm-Traj": "agentharm",
    "ASB-Traj": "asb",
    "AgentDojo-Traj": "agentdojo",
}
REFERENCE_SUBSET_ORDER = {
    # The released arrays follow the authors' evaluator order, which differs
    # from lexical filename order used by load_eval.
    "AgentHarm-Traj": ("harmful_steps", "benign_steps"),
    "ASB-Traj": ("DPI_attack_success", "OPI_attack_success", "atttack_failure"),
    "AgentDojo-Traj": ("workspace", "travel", "slack", "banking"),
}
MACRO_FIELDS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "false_positive_rate",
    "false_negative_rate",
    "allow_rate",
    "defer_rate",
    "block_rate",
    "monitor_call_reduction",
    "unsafe_leakage_rate",
    "local_false_block_rate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(project_root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _load_ts_guard(
    data_root: Path, evaluation: dict[str, list[Sample]]
) -> dict[str, list[int | None]]:
    predictions: dict[str, list[int | None]] = {}
    for source, directory in REFERENCE_DIRECTORIES.items():
        root = data_root / "reference" / directory
        raw_predictions = json.loads((root / "preds.json").read_text(encoding="utf-8"))
        raw_labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
        if len(raw_predictions) != len(evaluation[source]):
            raise ValueError(f"TS-Guard prediction count mismatch for {source}")
        samples_by_subset = {
            subset: [
                sample for sample in evaluation[source] if sample.subset == subset
            ]
            for subset in REFERENCE_SUBSET_ORDER[source]
        }
        reference_order = [
            sample
            for subset in REFERENCE_SUBSET_ORDER[source]
            for sample in samples_by_subset[subset]
        ]
        if [float(value) for value in raw_labels] != [
            sample.label for sample in reference_order
        ]:
            raise ValueError(f"TS-Guard label alignment mismatch for {source}")
        prediction_by_id = {
            sample.sample_id: strict_value(value)
            for sample, value in zip(reference_order, raw_predictions)
        }
        if len(prediction_by_id) != len(reference_order):
            raise ValueError(f"Duplicate semantic sample IDs prevent alignment for {source}")
        predictions[source] = [
            prediction_by_id[sample.sample_id] for sample in evaluation[source]
        ]
    return predictions


def _macro(dataset_metrics: dict[str, dict[str, float | int]]) -> dict[str, float]:
    return {
        field: float(
            np.mean(
                [
                    float(metrics[field])
                    for metrics in dataset_metrics.values()
                    if field in metrics
                ]
            )
        )
        for field in MACRO_FIELDS
        if all(field in metrics for metrics in dataset_metrics.values())
    }


def _write_predictions(
    path: Path, rows: Iterable[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "model",
        "policy_family",
        "target_recall",
        "sample_id",
        "source",
        "subset",
        "trajectory_id",
        "segment_id",
        "label",
        "local_probability",
        "route",
        "monitor_prediction",
        "cascade_prediction",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_ts_guard_cascades(
    *,
    project_root: Path,
    data_root: Path,
    results_root: Path,
    seed: int = RANDOM_STATE,
    bootstrap_replicates: int = 10_000,
) -> dict[str, object]:
    train = load_training(data_root, "train")
    validation = load_training(data_root, "validation")
    evaluation = load_eval(data_root)
    eval_all = [sample for samples in evaluation.values() for sample in samples]
    eval_ids = {sample.sample_id for sample in eval_all}
    train = [sample for sample in train if sample.sample_id not in eval_ids]
    validation = [sample for sample in validation if sample.sample_id not in eval_ids]

    validation_texts = [sample.text for sample in validation]
    validation_labels = np.asarray(
        [sample.strict_label for sample in validation], dtype=np.int8
    )
    eval_texts = [sample.text for sample in eval_all]
    eval_labels = np.asarray([sample.strict_label for sample in eval_all], dtype=np.int8)
    eval_clusters = [sample.trajectory_id for sample in eval_all]
    eval_strata = [sample.source for sample in eval_all]

    monitor_by_source = _load_ts_guard(data_root, evaluation)
    monitor_predictions = [
        prediction
        for source in evaluation
        for prediction in monitor_by_source[source]
    ]
    monitor_fail_closed = np.asarray(
        [1 if prediction is None else prediction for prediction in monitor_predictions],
        dtype=np.int8,
    )

    source_slices: dict[str, slice] = {}
    offset = 0
    for source, samples in evaluation.items():
        source_slices[source] = slice(offset, offset + len(samples))
        offset += len(samples)

    monitor_metrics = {
        source: binary_metrics(
            eval_labels[source_slice], monitor_fail_closed[source_slice]
        )
        for source, source_slice in source_slices.items()
    }
    output: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": _git_revision(project_root),
        "protocol": {
            "path": "docs/EXPERIMENT_PROTOCOL.md",
            "sha256": _sha256(project_root / "docs" / "EXPERIMENT_PROTOCOL.md"),
        },
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "evaluation": len(eval_all),
        },
        "monitor": {
            "name": "TS-Guard (released artifacts)",
            "micro": binary_metrics(eval_labels, monitor_fail_closed),
            "by_dataset": monitor_metrics,
            "macro": _macro(monitor_metrics),
        },
        "models": {},
    }
    prediction_rows: list[dict[str, object]] = []
    train_texts = [sample.text for sample in train]
    train_labels = [sample.strict_label for sample in train]

    for model_name, factory in model_factories(random_state=seed).items():
        print(f"Training and calibrating {model_name}")
        model = factory()
        model.fit(train_texts, train_labels)
        validation_raw = unsafe_scores(model, validation_texts)
        calibrator = SigmoidCalibrator(random_state=seed).fit(
            validation_raw, validation_labels
        )
        validation_probabilities = calibrator.predict_proba(validation_raw)
        evaluation_raw = unsafe_scores(model, eval_texts)
        evaluation_probabilities = calibrator.predict_proba(evaluation_raw)
        model_result: dict[str, object] = {
            "validation_calibration": calibration_metrics(
                validation_labels, validation_probabilities
            ),
            "evaluation_calibration": calibration_metrics(
                eval_labels, evaluation_probabilities
            ),
            "evaluation_risk_coverage": {
                "aurc": risk_coverage_curve(
                    eval_labels, evaluation_probabilities
                )["aurc"]
            },
            "policies": {},
            "three_way_policies": {},
        }

        for target_recall in TARGET_RECALLS:
            policy = select_allow_defer_policy(
                validation_probabilities,
                validation_labels,
                target_recall=target_recall,
                minimum_region_size=20,
            )
            cascade_predictions, routes = apply_cascade(
                evaluation_probabilities,
                monitor_predictions,
                policy,
                fail_closed=True,
            )
            dataset_metrics = {
                source: routing_metrics(
                    eval_labels[source_slice],
                    cascade_predictions[source_slice],
                    routes[source_slice],
                )
                for source, source_slice in source_slices.items()
            }
            micro = routing_metrics(eval_labels, cascade_predictions, routes)
            policy_result: dict[str, object] = {
                "policy": policy.to_dict(),
                "micro": micro,
                "by_dataset": dataset_metrics,
                "macro": _macro(dataset_metrics),
            }

            if target_recall == PRIMARY_TARGET:
                local_decision = (routes != "defer").astype(float)
                policy_result["intervals"] = {
                    "monitor_call_reduction": clustered_mean_interval(
                        local_decision,
                        clusters=eval_clusters,
                        strata=eval_strata,
                        replicates=bootstrap_replicates,
                        seed=seed,
                    ).to_dict(),
                    "recall_difference_vs_monitor": clustered_metric_difference(
                        eval_labels,
                        cascade_predictions,
                        monitor_fail_closed,
                        metric="recall",
                        clusters=eval_clusters,
                        strata=eval_strata,
                        replicates=bootstrap_replicates,
                        seed=seed + 1,
                    ).to_dict(),
                    "specificity_difference_vs_monitor": clustered_metric_difference(
                        eval_labels,
                        cascade_predictions,
                        monitor_fail_closed,
                        metric="specificity",
                        clusters=eval_clusters,
                        strata=eval_strata,
                        replicates=bootstrap_replicates,
                        seed=seed + 2,
                    ).to_dict(),
                }
                prediction_rows.extend(
                    {
                        "model": model_name,
                        "policy_family": "allow_defer",
                        "target_recall": target_recall,
                        "sample_id": sample.sample_id,
                        "source": sample.source,
                        "subset": sample.subset,
                        "trajectory_id": sample.trajectory_id,
                        "segment_id": sample.segment_id,
                        "label": int(label),
                        "local_probability": float(probability),
                        "route": str(route),
                        "monitor_prediction": (
                            "" if monitor is None else int(monitor)
                        ),
                        "cascade_prediction": int(cascade_prediction),
                    }
                    for sample, label, probability, route, monitor, cascade_prediction in zip(
                        eval_all,
                        eval_labels,
                        evaluation_probabilities,
                        routes,
                        monitor_predictions,
                        cascade_predictions,
                    )
                )

            model_result["policies"][str(target_recall)] = policy_result  # type: ignore[index]
            three_way_policy = select_routing_policy(
                validation_probabilities,
                validation_labels,
                target_recall=target_recall,
                minimum_specificity=0.80,
                minimum_region_size=20,
            )
            three_way_predictions, three_way_routes = apply_cascade(
                evaluation_probabilities,
                monitor_predictions,
                three_way_policy,
                fail_closed=True,
            )
            three_way_dataset_metrics = {
                source: routing_metrics(
                    eval_labels[source_slice],
                    three_way_predictions[source_slice],
                    three_way_routes[source_slice],
                )
                for source, source_slice in source_slices.items()
            }
            model_result["three_way_policies"][str(target_recall)] = {  # type: ignore[index]
                "policy": three_way_policy.to_dict(),
                "micro": routing_metrics(
                    eval_labels, three_way_predictions, three_way_routes
                ),
                "by_dataset": three_way_dataset_metrics,
                "macro": _macro(three_way_dataset_metrics),
                "status": "exploratory_ablation_observed_before_protocol_v1.1",
            }
            print(
                f"  target={target_recall:.3f} "
                f"validation allow={policy.validation_allow_rate * 100:.2f}% "
                f"test calls saved={float(micro['monitor_call_reduction']) * 100:.2f}% "
                f"recall={float(micro['recall']) * 100:.2f}%"
            )

        primary_policy = select_allow_defer_policy(
            validation_probabilities,
            validation_labels,
            target_recall=PRIMARY_TARGET,
            minimum_region_size=20,
        )
        original_routes = route_probabilities(
            evaluation_probabilities, primary_policy
        )
        robustness: dict[str, object] = {}
        for transformation in TRANSFORMATIONS:
            transformed = transform_samples(eval_all, transformation)
            transformed_raw = unsafe_scores(
                model, [sample.text for sample in transformed]
            )
            transformed_probabilities = calibrator.predict_proba(transformed_raw)
            transformed_routes = route_probabilities(
                transformed_probabilities, primary_policy
            )
            allowed = transformed_routes == "allow"
            positives = eval_labels == 1
            negatives = eval_labels == 0
            newly_allowed_harmful = (
                positives & (original_routes != "allow") & allowed
            )
            robustness[transformation] = {
                "allow_rate": float(allowed.mean()),
                "defer_rate": float(np.mean(transformed_routes == "defer")),
                "ideal_gate_recall": float(
                    1.0 - np.sum(allowed & positives) / np.sum(positives)
                ),
                "safe_fraction_among_allowed": (
                    float(np.sum(allowed & negatives) / np.sum(allowed))
                    if np.any(allowed)
                    else 1.0
                ),
                "newly_allowed_harmful": int(np.sum(newly_allowed_harmful)),
                "route_change_rate": float(
                    np.mean(transformed_routes != original_routes)
                ),
                "mean_absolute_probability_shift": float(
                    np.mean(
                        np.abs(
                            transformed_probabilities - evaluation_probabilities
                        )
                    )
                ),
                "local_threshold_metrics": binary_metrics(
                    eval_labels, (transformed_probabilities >= 0.5).astype(np.int8)
                ),
            }
        model_result["robustness"] = robustness
        output["models"][model_name] = model_result  # type: ignore[index]

    cascade_root = results_root / "cascade"
    cascade_root.mkdir(parents=True, exist_ok=True)
    output_path = cascade_root / "ts_guard_summary.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    _write_predictions(
        cascade_root / "ts_guard_primary_predictions.csv.gz",
        prediction_rows,
    )
    print(f"Wrote {output_path}")
    return output


def run_repeated_cascades(
    *,
    data_root: Path,
    results_root: Path,
    seeds: Sequence[int],
    include_semantic: bool = True,
) -> list[dict[str, object]]:
    if not seeds:
        raise ValueError("At least one seed is required")
    train = load_training(data_root, "train")
    validation = load_training(data_root, "validation")
    evaluation = load_eval(data_root)
    eval_all = [sample for samples in evaluation.values() for sample in samples]
    eval_ids = {sample.sample_id for sample in eval_all}
    train = [sample for sample in train if sample.sample_id not in eval_ids]
    validation = [sample for sample in validation if sample.sample_id not in eval_ids]
    train_texts = [sample.text for sample in train]
    train_labels = [sample.strict_label for sample in train]
    validation_texts = [sample.text for sample in validation]
    validation_labels = np.asarray(
        [sample.strict_label for sample in validation], dtype=np.int8
    )
    eval_texts = [sample.text for sample in eval_all]
    eval_labels = np.asarray([sample.strict_label for sample in eval_all], dtype=np.int8)
    monitor_by_source = _load_ts_guard(data_root, evaluation)
    monitor_predictions = [
        prediction
        for source in evaluation
        for prediction in monitor_by_source[source]
    ]
    source_slices: dict[str, slice] = {}
    offset = 0
    for source, samples in evaluation.items():
        source_slices[source] = slice(offset, offset + len(samples))
        offset += len(samples)

    rows: list[dict[str, object]] = []
    for seed in seeds:
        factories = model_factories(random_state=seed)
        if include_semantic:
            from toolsafe_lab.semantic_model import SemanticRelationGuard

            factories["minilm_relation_logreg"] = (
                lambda current_seed=seed: SemanticRelationGuard(
                    random_state=current_seed
                )
            )
        for model_name, factory in factories.items():
            print(f"seed={seed} model={model_name}")
            model = factory()
            model.fit(train_texts, train_labels)
            validation_raw = unsafe_scores(model, validation_texts)
            calibrator = SigmoidCalibrator(random_state=seed).fit(
                validation_raw, validation_labels
            )
            validation_probabilities = calibrator.predict_proba(validation_raw)
            policy = select_allow_defer_policy(
                validation_probabilities,
                validation_labels,
                target_recall=PRIMARY_TARGET,
                minimum_region_size=20,
            )
            eval_raw = unsafe_scores(model, eval_texts)
            eval_probabilities = calibrator.predict_proba(eval_raw)
            predictions, routes = apply_cascade(
                eval_probabilities,
                monitor_predictions,
                policy,
                fail_closed=True,
            )
            calibration = calibration_metrics(
                validation_labels, validation_probabilities
            )
            for dataset, dataset_slice in {
                "micro": slice(0, len(eval_all)),
                **source_slices,
            }.items():
                metrics = routing_metrics(
                    eval_labels[dataset_slice],
                    predictions[dataset_slice],
                    routes[dataset_slice],
                )
                rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "dataset": dataset,
                        "target_recall": PRIMARY_TARGET,
                        "allow_threshold": policy.allow_threshold,
                        "validation_recall": policy.validation_recall,
                        "validation_allow_rate": policy.validation_allow_rate,
                        "validation_brier": calibration["brier"],
                        "validation_ece_10": calibration["ece_10"],
                        **metrics,
                    }
                )

    fields = (
        "seed",
        "model",
        "dataset",
        "target_recall",
        "allow_threshold",
        "validation_recall",
        "validation_allow_rate",
        "validation_brier",
        "validation_ece_10",
        "n",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "mcc",
        "false_positive_rate",
        "false_negative_rate",
        "allow_rate",
        "defer_rate",
        "block_rate",
        "monitor_call_reduction",
        "unsafe_locally_allowed",
        "unsafe_leakage_rate",
        "benign_locally_blocked",
        "local_false_block_rate",
        "tn",
        "fp",
        "fn",
        "tp",
    )
    output_path = results_root / "cascade" / "repeated_seeds.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)
    print(f"Wrote {output_path}")
    return rows
