from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from toolsafe_lab.cascade import (
    SigmoidCalibrator,
    apply_cascade,
    calibration_metrics,
    routing_metrics,
    select_allow_defer_policy,
    unsafe_scores,
)
from toolsafe_lab.cascade_experiment import (
    PRIMARY_TARGET,
    REFERENCE_DIRECTORIES,
    REFERENCE_SUBSET_ORDER,
)
from toolsafe_lab.data import Sample, load_eval, load_training
from toolsafe_lab.metrics import binary_metrics
from toolsafe_lab.models import RANDOM_STATE, model_factories


HEADLINE_MODEL = "hybrid_tfidf_linearsvc"
FIELDS = ("instruction", "history", "current_action", "env_info")


def _reference_predictions(
    data_root: Path,
    evaluation: dict[str, list[Sample]],
) -> list[float | None]:
    aligned: list[float | None] = []
    for source, samples in evaluation.items():
        raw = json.loads(
            (
                data_root
                / "reference"
                / REFERENCE_DIRECTORIES[source]
                / "preds.json"
            ).read_text(encoding="utf-8")
        )
        by_subset = {
            subset: [sample for sample in samples if sample.subset == subset]
            for subset in REFERENCE_SUBSET_ORDER[source]
        }
        reference_order = [
            sample
            for subset in REFERENCE_SUBSET_ORDER[source]
            for sample in by_subset[subset]
        ]
        by_id = {
            sample.sample_id: (
                float(prediction) if prediction in {0.0, 0.5, 1.0} else None
            )
            for sample, prediction in zip(reference_order, raw)
        }
        aligned.extend(by_id[sample.sample_id] for sample in samples)
    return aligned


def _exact_metrics(
    labels: Sequence[int], predictions: Sequence[int]
) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "by_label": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate((0, 1, 2))
        },
    }


def _text(sample: Sample, dropped_field: str | None = None) -> str:
    return (
        replace(sample, **{dropped_field: ""}).text
        if dropped_field is not None
        else sample.text
    )


def _fit_headline(
    train: Sequence[Sample],
    validation: Sequence[Sample],
    evaluation: Sequence[Sample],
    monitor_predictions: Sequence[int],
    *,
    dropped_field: str | None = None,
    seed: int = RANDOM_STATE,
) -> dict[str, object]:
    model = model_factories(random_state=seed)[HEADLINE_MODEL]()
    model.fit(
        [_text(sample, dropped_field) for sample in train],
        [sample.strict_label for sample in train],
    )
    validation_labels = np.asarray(
        [sample.strict_label for sample in validation], dtype=np.int8
    )
    validation_scores = unsafe_scores(
        model, [_text(sample, dropped_field) for sample in validation]
    )
    calibrator = SigmoidCalibrator(random_state=seed).fit(
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
        model, [_text(sample, dropped_field) for sample in evaluation]
    )
    evaluation_probabilities = calibrator.predict_proba(evaluation_scores)
    labels = np.asarray(
        [sample.strict_label for sample in evaluation], dtype=np.int8
    )
    predictions, routes = apply_cascade(
        evaluation_probabilities,
        monitor_predictions,
        policy,
        fail_closed=True,
    )
    return {
        "policy": policy.to_dict(),
        "metrics": routing_metrics(labels, predictions, routes),
    }


def run_secondary_analysis(
    *,
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    seed: int = RANDOM_STATE,
) -> dict[str, object]:
    train = load_training(data_root, "train")
    validation = load_training(data_root, "validation")
    evaluation_by_source = load_eval(data_root)
    evaluation = [
        sample
        for source_samples in evaluation_by_source.values()
        for sample in source_samples
    ]
    eval_ids = {sample.sample_id for sample in evaluation}
    train = [sample for sample in train if sample.sample_id not in eval_ids]
    validation = [
        sample for sample in validation if sample.sample_id not in eval_ids
    ]
    raw_monitor = _reference_predictions(data_root, evaluation_by_source)
    strict_monitor = [
        1 if value is None else int(value != 0.0) for value in raw_monitor
    ]
    loose_monitor = [
        1 if value is None else int(value == 1.0) for value in raw_monitor
    ]
    exact_monitor = [
        2 if value is None else {0.0: 0, 0.5: 1, 1.0: 2}[value]
        for value in raw_monitor
    ]

    model = joblib.load(artifacts_root / "models" / f"{HEADLINE_MODEL}.joblib")
    local_binary = np.asarray(
        model.predict([sample.text for sample in evaluation]), dtype=np.int8
    )
    strict_labels = np.asarray(
        [sample.strict_label for sample in evaluation], dtype=np.int8
    )
    loose_labels = np.asarray(
        [sample.loose_label for sample in evaluation], dtype=np.int8
    )
    exact_labels = [sample.exact_label for sample in evaluation]

    route_rows: dict[str, dict[str, str]] = {}
    import csv
    import gzip

    with gzip.open(
        results_root / "cascade" / "ts_guard_primary_predictions.csv.gz",
        "rt",
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            if row["model"] == HEADLINE_MODEL:
                route_rows[row["sample_id"]] = row
    routes = np.asarray(
        [route_rows[sample.sample_id]["route"] for sample in evaluation],
        dtype=object,
    )
    strict_cascade = np.asarray(strict_monitor, dtype=np.int8)
    strict_cascade[routes == "allow"] = 0
    loose_cascade = np.asarray(loose_monitor, dtype=np.int8)
    loose_cascade[routes == "allow"] = 0

    sensitivity = {
        "strict": {
            "ts_guard": binary_metrics(strict_labels, strict_monitor),
            "hybrid_standalone": binary_metrics(strict_labels, local_binary),
            "hybrid_cascade": routing_metrics(
                strict_labels, strict_cascade, routes
            ),
        },
        "loose": {
            "ts_guard": binary_metrics(loose_labels, loose_monitor),
            "hybrid_standalone": binary_metrics(loose_labels, local_binary),
            "hybrid_cascade": routing_metrics(
                loose_labels, loose_cascade, routes
            ),
        },
        "exact": {
            "ts_guard": _exact_metrics(exact_labels, exact_monitor),
            "hybrid_standalone": _exact_metrics(
                exact_labels, [2 if value else 0 for value in local_binary]
            ),
        },
    }

    validation_labels = np.asarray(
        [sample.strict_label for sample in validation], dtype=np.int8
    )
    validation_scores = unsafe_scores(
        model, [sample.text for sample in validation]
    )
    isotonic = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    ).fit(validation_scores, validation_labels)
    validation_isotonic = np.asarray(
        isotonic.predict(validation_scores), dtype=float
    )
    isotonic_policy = select_allow_defer_policy(
        validation_isotonic,
        validation_labels,
        target_recall=PRIMARY_TARGET,
        minimum_region_size=20,
    )
    evaluation_scores = unsafe_scores(
        model, [sample.text for sample in evaluation]
    )
    evaluation_isotonic = np.asarray(
        isotonic.predict(evaluation_scores), dtype=float
    )
    isotonic_predictions, isotonic_routes = apply_cascade(
        evaluation_isotonic,
        strict_monitor,
        isotonic_policy,
        fail_closed=True,
    )
    sensitivity["isotonic_headline"] = {
        "policy": isotonic_policy.to_dict(),
        "validation_calibration": calibration_metrics(
            validation_labels, validation_isotonic
        ),
        "evaluation_calibration": calibration_metrics(
            strict_labels, evaluation_isotonic
        ),
        "metrics": routing_metrics(
            strict_labels, isotonic_predictions, isotonic_routes
        ),
    }

    controversial = np.asarray(
        [sample.label == 0.5 for sample in evaluation], dtype=bool
    )
    sensitivity["controversial_steps"] = {
        "n": int(np.sum(controversial)),
        "ts_guard_flagged_unsafe": float(
            np.mean(np.asarray(strict_monitor)[controversial] == 1)
        ),
        "hybrid_standalone_flagged_unsafe": float(
            np.mean(local_binary[controversial] == 1)
        ),
        "hybrid_locally_allowed": float(
            np.mean(routes[controversial] == "allow")
        ),
        "cascade_flagged_unsafe": float(
            np.mean(strict_cascade[controversial] == 1)
        ),
    }

    subset_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(evaluation):
        subset_indices[f"{sample.source}/{sample.subset}"].append(index)
    subgroup_metrics = {
        name: routing_metrics(
            strict_labels[indices],
            strict_cascade[indices],
            routes[indices],
        )
        for name, indices in subset_indices.items()
    }

    trajectory_lengths = Counter(sample.trajectory_id for sample in evaluation)

    def length_bucket(sample: Sample) -> str:
        length = trajectory_lengths[sample.trajectory_id]
        if length == 1:
            return "1"
        if length <= 3:
            return "2-3"
        if length <= 7:
            return "4-7"
        return "8+"

    length_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(evaluation):
        length_indices[length_bucket(sample)].append(index)
    length_metrics = {
        bucket: routing_metrics(
            strict_labels[indices],
            strict_cascade[indices],
            routes[indices],
        )
        for bucket, indices in length_indices.items()
    }

    print("Running headline input-field ablations")
    field_ablations = {
        "none": _fit_headline(
            train,
            validation,
            evaluation,
            strict_monitor,
            seed=seed,
        )
    }
    for field in FIELDS:
        print(f"  drop={field}")
        field_ablations[f"drop_{field}"] = _fit_headline(
            train,
            validation,
            evaluation,
            strict_monitor,
            dropped_field=field,
            seed=seed,
        )

    print("Running training-source ablations")
    source_ablations = {}
    for source in sorted({sample.source for sample in train}):
        source_train = [sample for sample in train if sample.source == source]
        source_validation = [
            sample for sample in validation if sample.source == source
        ]
        if len({sample.strict_label for sample in source_train}) < 2:
            continue
        print(f"  source={source}")
        source_ablations[source] = _fit_headline(
            source_train,
            source_validation,
            evaluation,
            strict_monitor,
            seed=seed,
        )

    payload: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "headline_model": HEADLINE_MODEL,
        "evaluation_samples": len(evaluation),
        "sensitivity": sensitivity,
        "by_subset": subgroup_metrics,
        "by_trajectory_length": length_metrics,
        "field_ablations": field_ablations,
        "training_source_ablations": source_ablations,
    }
    path = results_root / "secondary_analysis.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"Wrote {path}")
    return payload
