from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

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
from toolsafe_lab.system_benchmark import (
    benchmark_model,
    fit_with_metrics,
    host_metadata,
)


TOKEN = re.compile(r"(?u)\b[\w.-]{2,}\b")
MODEL_NAME = "field_namespaced_sparse"
BASELINE_NAME = "concatenated_execution_request_schema"


def _request_text(samples: Sequence[Sample]) -> list[str]:
    return [sample.instruction for sample in samples]


def _action_text(samples: Sequence[Sample]) -> list[str]:
    return [
        parse_react_step(sample.current_action).execution_only_text
        for sample in samples
    ]


def _schema_text(samples: Sequence[Sample]) -> list[str]:
    return [sample.env_info for sample in samples]


def _tokens(value: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN.finditer(value)}


def _walk_arguments(
    value: object,
    *,
    prefix: str,
    depth: int,
    features: Counter[str],
    keys: set[str],
) -> int:
    maximum_depth = depth
    if isinstance(value, dict):
        features["argument_type=object"] += 1
        for key, child in value.items():
            normalized_key = str(key).lower()
            keys.add(normalized_key)
            features[f"argument_key={normalized_key}"] += 1
            features[f"argument_path={prefix}.{normalized_key}"] += 1
            maximum_depth = max(
                maximum_depth,
                _walk_arguments(
                    child,
                    prefix=f"{prefix}.{normalized_key}",
                    depth=depth + 1,
                    features=features,
                    keys=keys,
                ),
            )
    elif isinstance(value, list):
        features["argument_type=array"] += 1
        features[f"array_length={min(len(value), 10)}"] += 1
        for child in value:
            maximum_depth = max(
                maximum_depth,
                _walk_arguments(
                    child,
                    prefix=f"{prefix}.item",
                    depth=depth + 1,
                    features=features,
                    keys=keys,
                ),
            )
    elif value is None:
        features["argument_type=null"] += 1
    elif isinstance(value, bool):
        features["argument_type=boolean"] += 1
    elif isinstance(value, (int, float)):
        features["argument_type=number"] += 1
    else:
        features["argument_type=string"] += 1
    return maximum_depth


def structured_relation_features(sample: Sample) -> dict[str, float]:
    parsed = parse_react_step(sample.current_action)
    features: Counter[str] = Counter()
    features[f"parse_status={parsed.status}"] = 1
    features[f"tool_name={parsed.tool_name.lower()}"] = 1
    features["thought_removed_available"] = int(bool(parsed.thought_removed_text))
    features["argument_json_parseable"] = int(parsed.argument_json_parseable)
    features["argument_characters_log1p"] = float(
        np.log1p(len(parsed.normalized_arguments))
    )

    argument_keys: set[str] = set()
    maximum_depth = 0
    if parsed.argument_json_parseable:
        maximum_depth = _walk_arguments(
            parsed.arguments_json,
            prefix="arg",
            depth=0,
            features=features,
            keys=argument_keys,
        )
    else:
        features["argument_type=unparsed"] = 1
    features["argument_max_depth"] = maximum_depth
    features["argument_key_count_log1p"] = float(np.log1p(len(argument_keys)))

    request_tokens = _tokens(sample.instruction)
    action_tokens = _tokens(
        f"{parsed.tool_name} {parsed.normalized_arguments}"
    )
    schema_tokens = _tokens(sample.env_info)
    request_action_overlap = request_tokens & action_tokens
    action_schema_overlap = action_tokens & schema_tokens
    union_request_action = request_tokens | action_tokens
    union_action_schema = action_tokens | schema_tokens

    features["request_action_jaccard"] = (
        len(request_action_overlap) / len(union_request_action)
        if union_request_action
        else 0.0
    )
    features["action_schema_jaccard"] = (
        len(action_schema_overlap) / len(union_action_schema)
        if union_action_schema
        else 0.0
    )
    features["tool_name_in_schema"] = int(
        bool(parsed.tool_name)
        and parsed.tool_name.lower() in sample.env_info.lower()
    )
    features["argument_keys_in_schema_rate"] = (
        sum(key in schema_tokens for key in argument_keys) / len(argument_keys)
        if argument_keys
        else 0.0
    )
    for token in sorted(request_action_overlap)[:50]:
        features[f"request_action_overlap={token}"] = 1
    for token in sorted(action_schema_overlap)[:50]:
        features[f"action_schema_overlap={token}"] = 1
    return dict(features)


def _structured_features(samples: Sequence[Sample]) -> list[dict[str, float]]:
    return [structured_relation_features(sample) for sample in samples]


def field_namespaced_model(random_state: int = RANDOM_STATE) -> Pipeline:
    features = FeatureUnion(
        (
            (
                "request_word",
                Pipeline(
                    (
                        (
                            "select",
                            FunctionTransformer(_request_text, validate=False),
                        ),
                        (
                            "vectorize",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=2,
                                max_features=40_000,
                                sublinear_tf=True,
                                strip_accents="unicode",
                                dtype=np.float32,
                            ),
                        ),
                    )
                ),
            ),
            (
                "action_word",
                Pipeline(
                    (
                        (
                            "select",
                            FunctionTransformer(_action_text, validate=False),
                        ),
                        (
                            "vectorize",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=2,
                                max_features=50_000,
                                sublinear_tf=True,
                                strip_accents="unicode",
                                dtype=np.float32,
                            ),
                        ),
                    )
                ),
            ),
            (
                "action_char",
                Pipeline(
                    (
                        (
                            "select",
                            FunctionTransformer(_action_text, validate=False),
                        ),
                        (
                            "vectorize",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=70_000,
                                sublinear_tf=True,
                                dtype=np.float32,
                            ),
                        ),
                    )
                ),
            ),
            (
                "schema_word",
                Pipeline(
                    (
                        (
                            "select",
                            FunctionTransformer(_schema_text, validate=False),
                        ),
                        (
                            "vectorize",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=2,
                                max_features=50_000,
                                sublinear_tf=True,
                                strip_accents="unicode",
                                dtype=np.float32,
                            ),
                        ),
                    )
                ),
            ),
            (
                "structured_relations",
                Pipeline(
                    (
                        (
                            "select",
                            FunctionTransformer(_structured_features, validate=False),
                        ),
                        ("vectorize", DictVectorizer(dtype=np.float32)),
                    )
                ),
            ),
        )
    )
    return Pipeline(
        (
            ("features", features),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        )
    )


def _evaluate_model(
    *,
    model: object,
    train_inputs: Sequence[object],
    validation_inputs: Sequence[object],
    evaluation_inputs: Sequence[object],
    train_labels: Sequence[int],
    validation_labels: np.ndarray,
    evaluation_labels: np.ndarray,
    monitor_values: Sequence[int | None],
    monitor: np.ndarray,
    evaluation: Sequence[Sample],
    source_slices: dict[str, slice],
    bootstrap_replicates: int,
    seed: int,
    model_path: Path,
) -> dict[str, object]:
    train_metrics = fit_with_metrics(model, train_inputs, train_labels)  # type: ignore[arg-type]
    standalone = np.asarray(model.predict(evaluation_inputs), dtype=np.int8)  # type: ignore[attr-defined]
    validation_raw = unsafe_scores(model, validation_inputs)  # type: ignore[arg-type]
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
    evaluation_probabilities = calibrator.predict_proba(
        unsafe_scores(model, evaluation_inputs)  # type: ignore[arg-type]
    )
    cascade, routes = apply_cascade(
        evaluation_probabilities,
        monitor_values,
        policy,
        fail_closed=True,
    )
    clusters = [sample.trajectory_id for sample in evaluation]
    strata = [sample.source for sample in evaluation]
    pooled_intervals = _intervals(
        labels=evaluation_labels,
        predictions=cascade,
        routes=routes,
        monitor=monitor,
        clusters=clusters,
        strata=strata,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    by_source = {
        source: routing_metrics(
            evaluation_labels[source_slice],
            cascade[source_slice],
            routes[source_slice],
        )
        for source, source_slice in source_slices.items()
    }
    by_source_inference: dict[str, object] = {}
    for index, (source, source_slice) in enumerate(source_slices.items()):
        source_intervals = _intervals(
            labels=evaluation_labels[source_slice],
            predictions=cascade[source_slice],
            routes=routes[source_slice],
            monitor=monitor[source_slice],
            clusters=clusters[source_slice],
            strata=None,
            replicates=bootstrap_replicates,
            seed=seed + 3 + index * 3,
        )
        by_source_inference[source] = {
            "intervals": source_intervals,
            "noninferiority": _noninferiority(source_intervals),
        }

    systems = benchmark_model(
        model,
        model_path,
        evaluation_inputs,  # type: ignore[arg-type]
        train_metrics=train_metrics,
    )
    return {
        "policy": policy.to_dict(),
        "standalone": binary_metrics(evaluation_labels, standalone),
        "cascade": {
            "micro": routing_metrics(evaluation_labels, cascade, routes),
            "by_dataset": by_source,
            "intervals": pooled_intervals,
            "noninferiority": _noninferiority(pooled_intervals),
            "by_dataset_inference": by_source_inference,
        },
        "systems": systems,
    }


def run_field_sparse_experiment(
    *,
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
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
    train_labels = [sample.strict_label for sample in train]
    validation_labels = np.asarray(
        [sample.strict_label for sample in validation], dtype=np.int8
    )
    evaluation_labels = np.asarray(
        [sample.strict_label for sample in evaluation], dtype=np.int8
    )

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

    baseline_transform = lambda sample: representation_text(  # noqa: E731
        sample, "execution_plus_request_schema"
    )
    baseline_inputs = {
        "train": [baseline_transform(sample) for sample in train],
        "validation": [baseline_transform(sample) for sample in validation],
        "evaluation": [baseline_transform(sample) for sample in evaluation],
    }
    common = {
        "train_labels": train_labels,
        "validation_labels": validation_labels,
        "evaluation_labels": evaluation_labels,
        "monitor_values": monitor_values,
        "monitor": monitor,
        "evaluation": evaluation,
        "source_slices": source_slices,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
    }
    print(f"Training {BASELINE_NAME}")
    baseline = _evaluate_model(
        model=model_factories(random_state=seed)["hybrid_tfidf_linearsvc"](),
        train_inputs=baseline_inputs["train"],
        validation_inputs=baseline_inputs["validation"],
        evaluation_inputs=baseline_inputs["evaluation"],
        model_path=artifacts_root / "models" / "field_sparse" / "concatenated.joblib",
        **common,
    )
    print(f"Training {MODEL_NAME}")
    namespaced = _evaluate_model(
        model=field_namespaced_model(random_state=seed),
        train_inputs=train,
        validation_inputs=validation,
        evaluation_inputs=evaluation,
        model_path=artifacts_root / "models" / "field_sparse" / "namespaced.joblib",
        **common,
    )

    output = {
        "schema_version": 1,
        "protocol": {
            "status": "prospective_extension_not_preregistration",
            "input": "execution + user request + tool schema; no Thought text",
            "target_validation_recall": PRIMARY_TARGET,
            "evaluation_labels_used_for_selection": False,
            "model_selection": (
                "Fixed concatenated hybrid baseline versus one field-namespaced "
                "sparse architecture; no evaluation-driven retuning."
            ),
        },
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "host": host_metadata(),
        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "evaluation": len(evaluation),
        },
        "models": {
            BASELINE_NAME: baseline,
            MODEL_NAME: namespaced,
        },
    }
    output_path = results_root / "field_sparse.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote field-namespaced result to {output_path}")
    return output
