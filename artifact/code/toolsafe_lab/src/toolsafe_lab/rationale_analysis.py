from __future__ import annotations

import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from toolsafe_lab.cascade import (
    SigmoidCalibrator,
    apply_cascade,
    routing_metrics,
    select_allow_defer_policy,
    unsafe_scores,
)
from toolsafe_lab.cascade_experiment import PRIMARY_TARGET, _load_ts_guard
from toolsafe_lab.data import Sample, load_eval, load_training
from toolsafe_lab.metrics import binary_metrics
from toolsafe_lab.models import RANDOM_STATE, model_factories
from toolsafe_lab.react_parser import parse_react_step
from toolsafe_lab.representation_experiment import (
    _intervals,
    _noninferiority,
    representation_text,
)


MODEL_NAME = "hybrid_tfidf_linearsvc"
TextTransform = Callable[[Sample], str]


def thought_only_text(sample: Sample) -> str:
    return "[THOUGHT]\n" + parse_react_step(sample.current_action).thought


CONFIGURATIONS: OrderedDict[str, TextTransform] = OrderedDict(
    (
        ("thought_only", thought_only_text),
        (
            "full_react_current_action_only",
            lambda sample: representation_text(
                sample, "full_react_current_action_only"
            ),
        ),
        (
            "all_fields_thought_removed",
            lambda sample: representation_text(
                sample, "all_fields_thought_removed"
            ),
        ),
        (
            "execution_plus_request_schema",
            lambda sample: representation_text(
                sample, "execution_plus_request_schema"
            ),
        ),
        (
            "execution_only",
            lambda sample: representation_text(sample, "execution_only"),
        ),
    )
)


def _confusion_profile(
    labels: np.ndarray,
    predictions: np.ndarray,
    routes: np.ndarray,
) -> dict[str, float | int | None]:
    positive = labels == 1
    negative = labels == 0
    true_positive = int(np.sum(positive & (predictions == 1)))
    false_negative = int(np.sum(positive & (predictions == 0)))
    true_negative = int(np.sum(negative & (predictions == 0)))
    false_positive = int(np.sum(negative & (predictions == 1)))
    return {
        "n": int(labels.size),
        "safe": int(negative.sum()),
        "unsafe": int(positive.sum()),
        "allow_rate": float(np.mean(routes == "allow")),
        "unsafe_locally_allowed": int(np.sum(positive & (routes == "allow"))),
        "recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        ),
        "specificity": (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else None
        ),
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
    }


def _severity_profile(
    samples: Sequence[Sample],
    *,
    local_predictions: np.ndarray,
    cascade_predictions: np.ndarray,
    monitor_predictions: np.ndarray,
    routes: np.ndarray,
) -> dict[str, object]:
    labels = np.asarray([sample.label for sample in samples], dtype=float)
    result: dict[str, object] = {}
    for value in (0.0, 0.5, 1.0):
        selected = labels == value
        count = int(selected.sum())
        result[str(value)] = {
            "n": count,
            "locally_allowed": int(np.sum(selected & (routes == "allow"))),
            "allow_rate": float(np.mean(routes[selected] == "allow")) if count else 0.0,
            "local_unsafe_prediction_rate": float(
                np.mean(local_predictions[selected] == 1)
            )
            if count
            else 0.0,
            "monitor_unsafe_prediction_rate": float(
                np.mean(monitor_predictions[selected] == 1)
            )
            if count
            else 0.0,
            "cascade_unsafe_prediction_rate": float(
                np.mean(cascade_predictions[selected] == 1)
            )
            if count
            else 0.0,
        }
    return result


def _subset_profiles(
    samples: Sequence[Sample],
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    routes: np.ndarray,
) -> dict[str, object]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices[f"{sample.source}/{sample.subset}"].append(index)

    profiles: dict[str, object] = {}
    for key, values in sorted(indices.items()):
        selected = np.asarray(values, dtype=int)
        label_counts = {
            str(value): int(
                np.sum(
                    np.asarray([samples[index].label for index in selected]) == value
                )
            )
            for value in (0.0, 0.5, 1.0)
        }
        profiles[key] = {
            "label_counts": label_counts,
            **_confusion_profile(
                labels[selected],
                predictions[selected],
                routes[selected],
            ),
        }
    return profiles


def _shuffled_within_source(
    samples: Sequence[Sample],
    texts: Sequence[str],
) -> list[str]:
    shuffled = list(texts)
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_source[sample.source].append(index)

    for indices in by_source.values():
        order = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                samples[index].sample_id.encode("utf-8")
            ).hexdigest(),
        )
        shift = max(1, len(order) // 2)
        donors = order[shift:] + order[:shift]
        for target, donor in zip(order, donors):
            shuffled[target] = texts[donor]
    return shuffled


def _pairwise_route_profile(
    samples: Sequence[Sample],
    reference: np.ndarray,
    comparison: np.ndarray,
) -> dict[str, object]:
    labels = np.asarray([sample.strict_label for sample in samples], dtype=np.int8)
    reference_allow = reference == "allow"
    comparison_allow = comparison == "allow"
    union = reference_allow | comparison_allow
    return {
        "route_agreement": float(np.mean(reference == comparison)),
        "allow_jaccard": (
            float(np.sum(reference_allow & comparison_allow) / np.sum(union))
            if np.any(union)
            else 1.0
        ),
        "both_allow": int(np.sum(reference_allow & comparison_allow)),
        "reference_only_allow": int(np.sum(reference_allow & ~comparison_allow)),
        "comparison_only_allow": int(np.sum(~reference_allow & comparison_allow)),
        "comparison_only_allow_unsafe": int(
            np.sum(~reference_allow & comparison_allow & (labels == 1))
        ),
        "reference_only_allow_unsafe": int(
            np.sum(reference_allow & ~comparison_allow & (labels == 1))
        ),
    }


def _configuration_result(
    *,
    samples: Sequence[Sample],
    labels: np.ndarray,
    local_predictions: np.ndarray,
    cascade_predictions: np.ndarray,
    monitor_predictions: np.ndarray,
    routes: np.ndarray,
    policy: object,
    source_slices: dict[str, slice],
    clusters: Sequence[str],
    strata: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    by_source = {
        source: routing_metrics(
            labels[source_slice],
            cascade_predictions[source_slice],
            routes[source_slice],
        )
        for source, source_slice in source_slices.items()
    }
    pooled_intervals = _intervals(
        labels=labels,
        predictions=cascade_predictions,
        routes=routes,
        monitor=monitor_predictions,
        clusters=clusters,
        strata=strata,
        replicates=replicates,
        seed=seed,
    )
    by_source_inference: dict[str, object] = {}
    for dataset_index, (source, source_slice) in enumerate(source_slices.items()):
        source_intervals = _intervals(
            labels=labels[source_slice],
            predictions=cascade_predictions[source_slice],
            routes=routes[source_slice],
            monitor=monitor_predictions[source_slice],
            clusters=clusters[source_slice],
            strata=None,
            replicates=replicates,
            seed=seed + 3 + dataset_index * 3,
        )
        by_source_inference[source] = {
            "intervals": source_intervals,
            "noninferiority": _noninferiority(source_intervals),
        }

    return {
        "policy": policy.to_dict(),  # type: ignore[attr-defined]
        "standalone": binary_metrics(labels, local_predictions),
        "cascade": {
            "micro": routing_metrics(labels, cascade_predictions, routes),
            "by_dataset": by_source,
            "intervals": pooled_intervals,
            "noninferiority": _noninferiority(pooled_intervals),
            "by_dataset_inference": by_source_inference,
        },
        "severity": _severity_profile(
            samples,
            local_predictions=local_predictions,
            cascade_predictions=cascade_predictions,
            monitor_predictions=monitor_predictions,
            routes=routes,
        ),
        "by_source_subset": _subset_profiles(
            samples,
            labels=labels,
            predictions=cascade_predictions,
            routes=routes,
        ),
    }


def run_rationale_analysis(
    *,
    data_root: Path,
    results_root: Path,
    seed: int = RANDOM_STATE,
    bootstrap_replicates: int = 10_000,
) -> dict[str, object]:
    loaded_train = load_training(data_root, "train")
    loaded_validation = load_training(data_root, "validation")
    evaluation_by_source = load_eval(data_root)
    evaluation = [
        sample
        for source_samples in evaluation_by_source.values()
        for sample in source_samples
    ]
    evaluation_ids = {sample.sample_id for sample in evaluation}
    train = [
        sample for sample in loaded_train if sample.sample_id not in evaluation_ids
    ]
    validation = [
        sample
        for sample in loaded_validation
        if sample.sample_id not in evaluation_ids
    ]

    labels_train = [sample.strict_label for sample in train]
    labels_validation = np.asarray(
        [sample.strict_label for sample in validation], dtype=np.int8
    )
    labels_evaluation = np.asarray(
        [sample.strict_label for sample in evaluation], dtype=np.int8
    )
    clusters = [sample.trajectory_id for sample in evaluation]
    strata = [sample.source for sample in evaluation]

    monitor_by_source = _load_ts_guard(data_root, evaluation_by_source)
    monitor_values = [
        prediction
        for source in evaluation_by_source
        for prediction in monitor_by_source[source]
    ]
    monitor = np.asarray(
        [1 if value is None else value for value in monitor_values],
        dtype=np.int8,
    )

    source_slices: dict[str, slice] = {}
    offset = 0
    for source, samples in evaluation_by_source.items():
        source_slices[source] = slice(offset, offset + len(samples))
        offset += len(samples)

    results: dict[str, object] = {}
    routes_by_configuration: dict[str, np.ndarray] = {}
    thought_state: dict[str, object] | None = None

    for configuration_index, (name, transform) in enumerate(CONFIGURATIONS.items()):
        print(f"Training {MODEL_NAME} for {name}")
        model = model_factories(random_state=seed)[MODEL_NAME]()
        train_texts = [transform(sample) for sample in train]
        validation_texts = [transform(sample) for sample in validation]
        evaluation_texts = [transform(sample) for sample in evaluation]
        model.fit(train_texts, labels_train)

        validation_raw = unsafe_scores(model, validation_texts)
        calibrator = SigmoidCalibrator(random_state=seed).fit(
            validation_raw, labels_validation
        )
        validation_probabilities = calibrator.predict_proba(validation_raw)
        policy = select_allow_defer_policy(
            validation_probabilities,
            labels_validation,
            target_recall=PRIMARY_TARGET,
            minimum_region_size=20,
        )
        evaluation_probabilities = calibrator.predict_proba(
            unsafe_scores(model, evaluation_texts)
        )
        local_predictions = np.asarray(
            model.predict(evaluation_texts), dtype=np.int8
        )
        cascade_predictions, routes = apply_cascade(
            evaluation_probabilities,
            monitor_values,
            policy,
            fail_closed=True,
        )
        routes_by_configuration[name] = routes
        results[name] = _configuration_result(
            samples=evaluation,
            labels=labels_evaluation,
            local_predictions=local_predictions,
            cascade_predictions=cascade_predictions,
            monitor_predictions=monitor,
            routes=routes,
            policy=policy,
            source_slices=source_slices,
            clusters=clusters,
            strata=strata,
            replicates=bootstrap_replicates,
            seed=seed + configuration_index * 20,
        )

        if name == "thought_only":
            thought_state = {
                "model": model,
                "calibrator": calibrator,
                "policy": policy,
                "evaluation_texts": evaluation_texts,
            }

    if thought_state is None:
        raise RuntimeError("Thought-only configuration was not evaluated")
    shuffled_texts = _shuffled_within_source(
        evaluation,
        thought_state["evaluation_texts"],  # type: ignore[arg-type]
    )
    shuffled_model = thought_state["model"]
    shuffled_calibrator = thought_state["calibrator"]
    shuffled_policy = thought_state["policy"]
    shuffled_probabilities = shuffled_calibrator.predict_proba(  # type: ignore[union-attr]
        unsafe_scores(shuffled_model, shuffled_texts)
    )
    shuffled_local = np.asarray(
        shuffled_model.predict(shuffled_texts),  # type: ignore[union-attr]
        dtype=np.int8,
    )
    shuffled_cascade, shuffled_routes = apply_cascade(
        shuffled_probabilities,
        monitor_values,
        shuffled_policy,  # type: ignore[arg-type]
        fail_closed=True,
    )
    shuffled_name = "thought_only_shuffled_within_source_eval"
    routes_by_configuration[shuffled_name] = shuffled_routes
    results[shuffled_name] = _configuration_result(
        samples=evaluation,
        labels=labels_evaluation,
        local_predictions=shuffled_local,
        cascade_predictions=shuffled_cascade,
        monitor_predictions=monitor,
        routes=shuffled_routes,
        policy=shuffled_policy,
        source_slices=source_slices,
        clusters=clusters,
        strata=strata,
        replicates=bootstrap_replicates,
        seed=seed + len(CONFIGURATIONS) * 20,
    )

    reference_routes = routes_by_configuration["full_react_current_action_only"]
    pairwise = {
        name: _pairwise_route_profile(
            evaluation,
            reference_routes,
            routes,
        )
        for name, routes in routes_by_configuration.items()
        if name != "full_react_current_action_only"
    }

    output = {
        "schema_version": 1,
        "protocol": {
            "status": "prospective_extension_not_preregistration",
            "model": MODEL_NAME,
            "target_validation_recall": PRIMARY_TARGET,
            "evaluation_labels_used_for_selection": False,
            "shuffling": (
                "Deterministic label-agnostic cyclic permutation of Thought text "
                "within each evaluation source; fitted model and policy unchanged."
            ),
            "subset_warning": (
                "Subsets retain source-provided filenames/domains and are not "
                "reinterpreted as a universal threat taxonomy."
            ),
            "raw_text_in_output": False,
        },
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "evaluation": len(evaluation),
        },
        "configurations": results,
        "pairwise_routes_vs_full_react_current_action": pairwise,
    }
    output_path = results_root / "rationale_analysis.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote rationale analysis to {output_path}")
    return output
