from __future__ import annotations

import hashlib
import json
import csv
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

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
    _load_ts_guard,
)
from toolsafe_lab.data import Sample, load_eval, load_training
from toolsafe_lab.metrics import binary_metrics
from toolsafe_lab.models import RANDOM_STATE, model_factories
from toolsafe_lab.react_parser import parse_react_step
from toolsafe_lab.uncertainty import (
    clustered_mean_interval,
    clustered_metric_difference,
)


HEADLINE_MODEL = "hybrid_tfidf_linearsvc"
Representation = Callable[[Sample], str]


def _marked_fields(
    *,
    instruction: str | None = None,
    history: str | None = None,
    current_action: str | None = None,
    env_info: str | None = None,
) -> str:
    values = (
        ("[USER_REQUEST]", instruction),
        ("[INTERACTION_HISTORY]", history),
        ("[CURRENT_ACTION]", current_action),
        ("[TOOL_DESCRIPTIONS]", env_info),
    )
    parts: list[str] = []
    for marker, value in values:
        if value is not None:
            parts.extend((marker, value))
    return "\n".join(parts)


def _thought_removed(sample: Sample) -> str:
    parsed = parse_react_step(sample.current_action)
    return replace(sample, current_action=parsed.thought_removed_text).text


def _execution_only(sample: Sample) -> str:
    return parse_react_step(sample.current_action).execution_only_text


def _execution_plus_request(sample: Sample) -> str:
    return _marked_fields(
        instruction=sample.instruction,
        current_action=_execution_only(sample),
    )


def _execution_plus_schema(sample: Sample) -> str:
    return _marked_fields(
        current_action=_execution_only(sample),
        env_info=sample.env_info,
    )


def _execution_plus_request_schema(sample: Sample) -> str:
    return _marked_fields(
        instruction=sample.instruction,
        current_action=_execution_only(sample),
        env_info=sample.env_info,
    )


def _structure_tokens(value: object, prefix: str = "arg") -> list[str]:
    if isinstance(value, dict):
        tokens = [f"{prefix}:object"]
        for key in sorted(value):
            child = f"{prefix}.{key}"
            tokens.append(f"{child}:key")
            tokens.extend(_structure_tokens(value[key], child))
        return tokens
    if isinstance(value, list):
        tokens = [f"{prefix}:array", f"{prefix}:length:{len(value)}"]
        for item in value:
            tokens.extend(_structure_tokens(item, f"{prefix}.item"))
        return tokens
    if value is None:
        return [f"{prefix}:null"]
    if isinstance(value, bool):
        return [f"{prefix}:boolean"]
    if isinstance(value, (int, float)):
        return [f"{prefix}:number"]
    return [f"{prefix}:string"]


def _structured_only(sample: Sample) -> str:
    parsed = parse_react_step(sample.current_action)
    tokens = (
        _structure_tokens(parsed.arguments_json)
        if parsed.argument_json_parseable
        else ["arguments:unparsed"]
    )
    return "\n".join(
        (
            "[PARSE_STATUS]",
            parsed.status,
            "[TOOL_NAME]",
            parsed.tool_name,
            "[ARGUMENT_STRUCTURE]",
            " ".join(tokens),
        )
    )


def _no_history_full(sample: Sample) -> str:
    return replace(sample, history="").text


def _full_current_action_only(sample: Sample) -> str:
    return _marked_fields(current_action=sample.current_action)


def _drop_request_full(sample: Sample) -> str:
    return replace(sample, instruction="").text


def _drop_current_action_full(sample: Sample) -> str:
    return replace(sample, current_action="").text


def _drop_schema_full(sample: Sample) -> str:
    return replace(sample, env_info="").text


REPRESENTATIONS: OrderedDict[str, Representation] = OrderedDict(
    (
        ("all_fields_full_react", lambda sample: sample.text),
        ("all_fields_thought_removed", _thought_removed),
        ("execution_only", _execution_only),
        ("execution_plus_request", _execution_plus_request),
        ("execution_plus_schema", _execution_plus_schema),
        ("execution_plus_request_schema", _execution_plus_request_schema),
        ("structured_only", _structured_only),
        ("no_history_full_react", _no_history_full),
        ("full_react_current_action_only", _full_current_action_only),
        ("drop_request_full_react", _drop_request_full),
        ("drop_current_action_full_react", _drop_current_action_full),
        ("drop_schema_full_react", _drop_schema_full),
    )
)


def representation_text(sample: Sample, name: str) -> str:
    try:
        function = REPRESENTATIONS[name]
    except KeyError as error:
        raise ValueError(f"Unknown representation: {name}") from error
    return function(sample)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _macro(rows: dict[str, dict[str, float | int]]) -> dict[str, float]:
    fields = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "mcc",
    )
    return {
        field: float(np.mean([float(row[field]) for row in rows.values()]))
        for field in fields
    }


def _intervals(
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    routes: np.ndarray,
    monitor: np.ndarray,
    clusters: Sequence[str],
    strata: Sequence[str] | None,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    return {
        "monitor_call_reduction": clustered_mean_interval(
            (routes != "defer").astype(float),
            clusters=clusters,
            strata=strata,
            replicates=replicates,
            seed=seed,
        ).to_dict(),
        "recall_difference_vs_monitor": clustered_metric_difference(
            labels,
            predictions,
            monitor,
            metric="recall",
            clusters=clusters,
            strata=strata,
            replicates=replicates,
            seed=seed + 1,
        ).to_dict(),
        "specificity_difference_vs_monitor": clustered_metric_difference(
            labels,
            predictions,
            monitor,
            metric="specificity",
            clusters=clusters,
            strata=strata,
            replicates=replicates,
            seed=seed + 2,
        ).to_dict(),
    }


def _noninferiority(intervals: dict[str, object]) -> dict[str, bool]:
    recall = intervals["recall_difference_vs_monitor"]
    specificity = intervals["specificity_difference_vs_monitor"]
    assert isinstance(recall, dict) and isinstance(specificity, dict)
    return {
        "recall_margin_minus_1pp": float(recall["lower"]) > -0.01,
        "specificity_margin_minus_2pp": float(specificity["lower"]) > -0.02,
        "joint_pass": (
            float(recall["lower"]) > -0.01
            and float(specificity["lower"]) > -0.02
        ),
    }


def _write_summary_csv(path: Path, result: dict[str, object]) -> None:
    fields = (
        "model",
        "representation",
        "standalone_accuracy",
        "standalone_precision",
        "standalone_recall",
        "standalone_specificity",
        "standalone_f1",
        "cascade_monitor_call_reduction",
        "cascade_precision",
        "cascade_recall",
        "cascade_specificity",
        "cascade_f1",
        "pooled_recall_difference_lower",
        "pooled_specificity_difference_lower",
        "pooled_joint_noninferiority",
        "all_domain_joint_noninferiority",
        "agentharm_saved",
        "agentharm_recall",
        "asb_saved",
        "asb_recall",
        "agentdojo_saved",
        "agentdojo_recall",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        models = result["models"]
        assert isinstance(models, dict)
        for model_name, representations in models.items():
            assert isinstance(representations, dict)
            for representation_name, raw_row in representations.items():
                assert isinstance(raw_row, dict)
                standalone = raw_row["standalone"]
                cascade = raw_row["cascade"]
                assert isinstance(standalone, dict) and isinstance(cascade, dict)
                standalone_micro = standalone["micro"]
                cascade_micro = cascade["micro"]
                by_dataset = cascade["by_dataset"]
                assert (
                    isinstance(standalone_micro, dict)
                    and isinstance(cascade_micro, dict)
                    and isinstance(by_dataset, dict)
                )

                intervals = cascade.get("intervals")
                noninferiority = cascade.get("noninferiority")
                by_dataset_inference = cascade.get("by_dataset_inference")
                recall_lower: float | str = ""
                specificity_lower: float | str = ""
                pooled_pass: bool | str = ""
                all_domain_pass: bool | str = ""
                if (
                    isinstance(intervals, dict)
                    and isinstance(noninferiority, dict)
                    and isinstance(by_dataset_inference, dict)
                ):
                    recall_interval = intervals["recall_difference_vs_monitor"]
                    specificity_interval = intervals[
                        "specificity_difference_vs_monitor"
                    ]
                    assert isinstance(recall_interval, dict)
                    assert isinstance(specificity_interval, dict)
                    recall_lower = float(recall_interval["lower"])
                    specificity_lower = float(specificity_interval["lower"])
                    pooled_pass = bool(noninferiority["joint_pass"])
                    all_domain_pass = all(
                        bool(value["noninferiority"]["joint_pass"])  # type: ignore[index]
                        for value in by_dataset_inference.values()
                    )

                writer.writerow(
                    {
                        "model": model_name,
                        "representation": representation_name,
                        "standalone_accuracy": standalone_micro["accuracy"],
                        "standalone_precision": standalone_micro["precision"],
                        "standalone_recall": standalone_micro["recall"],
                        "standalone_specificity": standalone_micro["specificity"],
                        "standalone_f1": standalone_micro["f1"],
                        "cascade_monitor_call_reduction": cascade_micro[
                            "monitor_call_reduction"
                        ],
                        "cascade_precision": cascade_micro["precision"],
                        "cascade_recall": cascade_micro["recall"],
                        "cascade_specificity": cascade_micro["specificity"],
                        "cascade_f1": cascade_micro["f1"],
                        "pooled_recall_difference_lower": recall_lower,
                        "pooled_specificity_difference_lower": specificity_lower,
                        "pooled_joint_noninferiority": pooled_pass,
                        "all_domain_joint_noninferiority": all_domain_pass,
                        "agentharm_saved": by_dataset["AgentHarm-Traj"][
                            "monitor_call_reduction"
                        ],
                        "agentharm_recall": by_dataset["AgentHarm-Traj"]["recall"],
                        "asb_saved": by_dataset["ASB-Traj"][
                            "monitor_call_reduction"
                        ],
                        "asb_recall": by_dataset["ASB-Traj"]["recall"],
                        "agentdojo_saved": by_dataset["AgentDojo-Traj"][
                            "monitor_call_reduction"
                        ],
                        "agentdojo_recall": by_dataset["AgentDojo-Traj"]["recall"],
                    }
                )


def run_representation_ablation(
    *,
    project_root: Path,
    data_root: Path,
    results_root: Path,
    model_names: Sequence[str] | None = None,
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

    factories = model_factories(random_state=seed)
    selected_names = list(model_names) if model_names is not None else list(factories)
    unknown = sorted(set(selected_names) - set(factories))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")

    result: dict[str, object] = {
        "schema_version": 1,
        "protocol": {
            "version": "1.2",
            "path": "docs/EXPERIMENT_PROTOCOL.md",
            "sha256": _sha256(project_root / "docs" / "EXPERIMENT_PROTOCOL.md"),
            "status": "prospective_extension_not_preregistration",
            "prior_full_react_evaluation_results_known": True,
            "execution_only_representations_frozen_before_their_evaluation": True,
            "evaluation_labels_used_for_selection": False,
            "primary_target_validation_recall": PRIMARY_TARGET,
            "minimum_region_size": 20,
            "strict_labels": {"safe": [0.0], "unsafe": [0.5, 1.0]},
        },
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "primary_model_for_intervals": HEADLINE_MODEL,
        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "evaluation": len(evaluation),
        },
        "representations": list(REPRESENTATIONS),
        "models": {},
    }

    for model_index, model_name in enumerate(selected_names):
        model_result: dict[str, object] = {}
        for representation_index, (representation_name, transform) in enumerate(
            REPRESENTATIONS.items()
        ):
            print(f"Training {model_name} on {representation_name}")
            train_texts = [transform(sample) for sample in train]
            validation_texts = [transform(sample) for sample in validation]
            evaluation_texts = [transform(sample) for sample in evaluation]

            model = factories[model_name]()
            model.fit(train_texts, labels_train)
            standalone = np.asarray(model.predict(evaluation_texts), dtype=np.int8)
            validation_raw = unsafe_scores(model, validation_texts)
            calibrator = SigmoidCalibrator(random_state=seed).fit(
                validation_raw, labels_validation
            )
            validation_probabilities = calibrator.predict_proba(validation_raw)
            evaluation_probabilities = calibrator.predict_proba(
                unsafe_scores(model, evaluation_texts)
            )
            policy = select_allow_defer_policy(
                validation_probabilities,
                labels_validation,
                target_recall=PRIMARY_TARGET,
                minimum_region_size=20,
            )
            cascade, routes = apply_cascade(
                evaluation_probabilities,
                monitor_values,
                policy,
                fail_closed=True,
            )

            standalone_by_source = {
                source: binary_metrics(
                    labels_evaluation[source_slice],
                    standalone[source_slice],
                )
                for source, source_slice in source_slices.items()
            }
            cascade_by_source = {
                source: routing_metrics(
                    labels_evaluation[source_slice],
                    cascade[source_slice],
                    routes[source_slice],
                )
                for source, source_slice in source_slices.items()
            }
            row: dict[str, object] = {
                "policy": policy.to_dict(),
                "validation_calibration": calibration_metrics(
                    labels_validation, validation_probabilities
                ),
                "standalone": {
                    "micro": binary_metrics(labels_evaluation, standalone),
                    "macro": _macro(standalone_by_source),
                    "by_dataset": standalone_by_source,
                },
                "cascade": {
                    "micro": routing_metrics(
                        labels_evaluation, cascade, routes
                    ),
                    "macro": _macro(cascade_by_source),
                    "by_dataset": cascade_by_source,
                },
            }

            if model_name == HEADLINE_MODEL:
                interval_seed = seed + representation_index * 20 + model_index * 1_000
                pooled_intervals = _intervals(
                    labels=labels_evaluation,
                    predictions=cascade,
                    routes=routes,
                    monitor=monitor,
                    clusters=clusters,
                    strata=strata,
                    replicates=bootstrap_replicates,
                    seed=interval_seed,
                )
                dataset_intervals: dict[str, object] = {}
                for dataset_index, (source, source_slice) in enumerate(
                    source_slices.items()
                ):
                    source_intervals = _intervals(
                        labels=labels_evaluation[source_slice],
                        predictions=cascade[source_slice],
                        routes=routes[source_slice],
                        monitor=monitor[source_slice],
                        clusters=clusters[source_slice],
                        strata=None,
                        replicates=bootstrap_replicates,
                        seed=interval_seed + 3 + dataset_index * 3,
                    )
                    dataset_intervals[source] = {
                        "intervals": source_intervals,
                        "noninferiority": _noninferiority(source_intervals),
                    }
                row["cascade"]["intervals"] = pooled_intervals  # type: ignore[index]
                row["cascade"]["noninferiority"] = _noninferiority(  # type: ignore[index]
                    pooled_intervals
                )
                row["cascade"]["by_dataset_inference"] = dataset_intervals  # type: ignore[index]

            model_result[representation_name] = row
        result["models"][model_name] = model_result  # type: ignore[index]

    output_path = results_root / "representation_ablation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(results_root / "representation_ablation.csv", result)
    print(f"Wrote representation ablation to {output_path}")
    return result
