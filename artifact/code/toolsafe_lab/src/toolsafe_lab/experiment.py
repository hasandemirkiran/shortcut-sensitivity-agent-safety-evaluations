from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

from toolsafe_lab.data import load_eval, load_training, split_summary
from toolsafe_lab.metrics import binary_metrics, strict_pairs
from toolsafe_lab.models import model_factories
from toolsafe_lab.system_benchmark import benchmark_model, fit_with_metrics, host_metadata


QUALITY_FIELDS = (
    "model",
    "dataset",
    "evaluation_mode",
    "source",
    "n",
    "coverage",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "false_positive_rate",
    "false_negative_rate",
    "tn",
    "fp",
    "fn",
    "tp",
)

SYSTEM_FIELDS = (
    "model",
    "variant",
    "source",
    "hardware",
    "provider",
    "api_model",
    "reasoning_effort",
    "prompt_version",
    "prompt_sha256",
    "run_label",
    "model_size_mib",
    "train_seconds",
    "train_peak_rss_delta_mib",
    "load_ms_p50",
    "single_n",
    "latency_ms_mean",
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_ms_p99",
    "single_peak_rss_delta_mib",
    "batch_n",
    "batch_repeats",
    "batch_throughput_samples_s",
    "batch_peak_rss_delta_mib",
    "requests",
    "successful_requests",
    "failed_requests",
    "request_coverage",
    "request_retries",
    "refused_requests",
    "truncated_requests",
    "http_error_requests",
    "structured_output_error_requests",
    "concurrency",
    "active_wall_seconds",
    "wall_throughput_samples_s",
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "estimated_cost_usd",
    "cost_per_1k_predictions_usd",
    "timing_semantics",
    "client_queue_ms_mean",
    "client_queue_ms_p50",
    "client_queue_ms_p95",
)

PAPER_STRICT = {
    "AgentHarm-Traj": {"accuracy": 0.8481, "f1": 0.9016, "recall": 0.9695},
    "ASB-Traj": {"accuracy": 0.9497, "f1": 0.9476, "recall": 0.9385},
    "AgentDojo-Traj": {"accuracy": 0.9172, "f1": 0.8618, "recall": 0.8949},
}

PAPER_COUNTS = {
    "AgentHarm-Traj": 731,
    "ASB-Traj": 5237,
    "AgentDojo-Traj": 1220,
}

REFERENCE_DIRECTORIES = {
    "AgentHarm-Traj": "agentharm",
    "ASB-Traj": "asb",
    "AgentDojo-Traj": "agentdojo",
}


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def reproduce_ts_guard(data_root: Path, results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {
        "method": "Recomputed from authors' released labels and predictions",
        "evaluation_mode": "strict",
        "datasets": {},
    }
    eval_datasets = load_eval(data_root)

    for dataset, directory in REFERENCE_DIRECTORIES.items():
        reference = data_root / "reference" / directory
        predictions = _load_json(reference / "preds.json")
        labels = _load_json(reference / "labels.json")
        released_metrics = _load_json(reference / "metrics_strict.json")
        if not isinstance(predictions, list) or not isinstance(labels, list):
            raise TypeError(f"Unexpected released prediction schema for {dataset}")
        if not isinstance(released_metrics, dict):
            raise TypeError(f"Unexpected released metric schema for {dataset}")
        if len(predictions) != len(labels):
            raise ValueError(f"Prediction/label length mismatch for {dataset}")
        if len(labels) != len(eval_datasets[dataset]):
            raise ValueError(
                f"Released labels ({len(labels)}) do not cover downloaded "
                f"{dataset} ({len(eval_datasets[dataset])})"
            )

        y_pred, y_true, total = strict_pairs(predictions, labels)
        metrics = binary_metrics(y_true, y_pred, total_predictions=total)
        comparisons = {}
        for key in ("accuracy", "f1", "recall"):
            released_value = float(released_metrics[key])
            recomputed_value = float(metrics[key])
            comparisons[key] = {
                "recomputed": recomputed_value,
                "released_json": released_value,
                "paper_rounded": PAPER_STRICT[dataset][key],
                "matches_released": math.isclose(
                    recomputed_value, released_value, rel_tol=0, abs_tol=1e-12
                ),
                "matches_paper_at_2dp_percent": (
                    round(recomputed_value * 100, 2)
                    == round(PAPER_STRICT[dataset][key] * 100, 2)
                ),
            }
        if not all(item["matches_released"] for item in comparisons.values()):
            raise AssertionError(f"Released TS-Guard metric mismatch for {dataset}")

        row = {
            "model": "TS-Guard (released artifacts)",
            "dataset": dataset,
            "evaluation_mode": "strict",
            "source": "official_cached_predictions",
            **metrics,
        }
        rows.append(row)
        rows.append(
            {
                "model": "TS-Guard (paper Table 3)",
                "dataset": dataset,
                "evaluation_mode": "strict",
                "source": "paper_table_3",
                "n": PAPER_COUNTS[dataset],
                **PAPER_STRICT[dataset],
            }
        )
        details["datasets"][dataset] = {
            "paper_sample_count": PAPER_COUNTS[dataset],
            "released_predictions": total,
            "valid_predictions": metrics["n"],
            "invalid_predictions": total - int(metrics["n"]),
            "released_sample_count_matches_paper": total == PAPER_COUNTS[dataset],
            "comparisons": comparisons,
        }
        print(
            f"{dataset:16} ACC={metrics['accuracy'] * 100:6.2f} "
            f"F1={metrics['f1'] * 100:6.2f} "
            f"Recall={metrics['recall'] * 100:6.2f} "
            f"coverage={metrics['coverage'] * 100:6.2f}"
        )

    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "ts_guard_reproduction.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )
    existing = [
        row
        for row in _read_csv(results_root / "predictive_quality.csv")
        if row.get("source") not in {"official_cached_predictions", "paper_table_3"}
    ]
    _write_csv(
        results_root / "predictive_quality.csv",
        [*rows, *existing],
        QUALITY_FIELDS,
    )

    systems = [
        row
        for row in _read_csv(results_root / "systems_performance.csv")
        if row.get("source") != "paper_table_11"
    ]
    paper_system_row = {
        "model": "TS-Guard",
        "variant": "Qwen2.5-7B-Instruct, BF16",
        "source": "paper_table_11",
        "hardware": "paper environment: 8x NVIDIA H20 96GB",
        # Published checkpoint's four BF16 safetensor shards total 15,231,271,912 bytes.
        "model_size_mib": 15231271912 / (1024**2),
        "latency_ms_mean": 1360.0,
    }
    _write_csv(
        results_root / "systems_performance.csv",
        [paper_system_row, *systems],
        SYSTEM_FIELDS,
    )
    return rows


def run_baselines(
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = load_training(data_root, "train")
    validation = load_training(data_root, "validation")
    evaluation = load_eval(data_root)
    eval_all = [sample for dataset in evaluation.values() for sample in dataset]

    eval_ids = {sample.sample_id for sample in eval_all}
    train_overlap = sorted(eval_ids & {sample.sample_id for sample in train})
    validation_overlap = sorted(eval_ids & {sample.sample_id for sample in validation})
    # Keep the test set untouched and make the local comparison conservative.
    # Upstream's released training artifacts contain two exact ASB-eval copies.
    train = [sample for sample in train if sample.sample_id not in eval_ids]
    validation = [sample for sample in validation if sample.sample_id not in eval_ids]

    diagnostics: dict[str, object] = {
        "host": host_metadata(),
        "train_after_leakage_filter": split_summary(train),
        "validation_after_leakage_filter": split_summary(validation),
        "evaluation": {
            dataset: split_summary(samples) for dataset, samples in evaluation.items()
        },
        "deduplicated_counts": {
            "train": len(train),
            "validation": len(validation),
            "evaluation": len(eval_all),
        },
        "exact_text_overlap": {
            "train_evaluation": len(train_overlap),
            "validation_evaluation": len(validation_overlap),
            "removed_train_ids": train_overlap,
            "removed_validation_ids": validation_overlap,
        },
        "validation_metrics": {},
    }
    quality_rows: list[dict[str, object]] = []
    system_rows: list[dict[str, object]] = []
    train_texts = [sample.text for sample in train]
    train_labels = [sample.strict_label for sample in train]
    validation_texts = [sample.text for sample in validation]
    validation_labels = [sample.strict_label for sample in validation]
    eval_texts = [sample.text for sample in eval_all]
    hardware = (
        f"{platform_name()} {diagnostics['host']['machine']}; "
        f"{diagnostics['host']['ram_gib']} GiB RAM"
    )

    for model_name, factory in model_factories().items():
        print(f"\nTraining {model_name} on {len(train)} samples")
        model = factory()
        training_system = fit_with_metrics(model, train_texts, train_labels)

        validation_predictions = model.predict(validation_texts)
        validation_metrics = binary_metrics(validation_labels, validation_predictions)
        diagnostics["validation_metrics"][model_name] = validation_metrics
        print(
            f"  validation ACC={validation_metrics['accuracy'] * 100:.2f} "
            f"F1={validation_metrics['f1'] * 100:.2f}"
        )

        for dataset, samples in evaluation.items():
            predictions = model.predict([sample.text for sample in samples])
            metrics = binary_metrics(
                [sample.strict_label for sample in samples], predictions
            )
            quality_rows.append(
                {
                    "model": model_name,
                    "dataset": dataset,
                    "evaluation_mode": "strict",
                    "source": "local_measurement",
                    **metrics,
                }
            )
            print(
                f"  {dataset:16} ACC={metrics['accuracy'] * 100:6.2f} "
                f"F1={metrics['f1'] * 100:6.2f} Recall={metrics['recall'] * 100:6.2f}"
            )

        model_path = artifacts_root / "models" / f"{model_name}.joblib"
        system_metrics = benchmark_model(
            model,
            model_path,
            eval_texts,
            train_metrics=training_system,
        )
        system_rows.append(
            {
                "model": model_name,
                "variant": "strict binary classifier",
                "source": "local_measurement",
                "hardware": hardware,
                **system_metrics,
            }
        )
        print(
            f"  latency p50={system_metrics['latency_ms_p50']:.3f} ms, "
            f"p99={system_metrics['latency_ms_p99']:.3f} ms, "
            f"throughput={system_metrics['batch_throughput_samples_s']:.1f} sample/s"
        )

    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "experiment_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )

    reference_quality = [
        row
        for row in _read_csv(results_root / "predictive_quality.csv")
        if row.get("source") != "local_measurement"
    ]
    _write_csv(
        results_root / "predictive_quality.csv",
        [*reference_quality, *quality_rows],
        QUALITY_FIELDS,
    )
    reference_systems = [
        row
        for row in _read_csv(results_root / "systems_performance.csv")
        if row.get("source") != "local_measurement"
    ]
    _write_csv(
        results_root / "systems_performance.csv",
        [*reference_systems, *system_rows],
        SYSTEM_FIELDS,
    )
    return quality_rows, system_rows


def run_semantic_baseline(
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    from toolsafe_lab.semantic_model import SemanticRelationGuard

    train = load_training(data_root, "train")
    validation = load_training(data_root, "validation")
    evaluation = load_eval(data_root)
    eval_all = [sample for dataset in evaluation.values() for sample in dataset]
    eval_ids = {sample.sample_id for sample in eval_all}
    train = [sample for sample in train if sample.sample_id not in eval_ids]
    validation = [sample for sample in validation if sample.sample_id not in eval_ids]
    train_texts = [sample.text for sample in train]
    train_labels = [sample.strict_label for sample in train]

    model_name = "minilm_relation_logreg"
    print(f"Training {model_name} on {len(train)} samples")
    model = SemanticRelationGuard()
    training_system = fit_with_metrics(model, train_texts, train_labels)

    validation_predictions = model.predict([sample.text for sample in validation])
    validation_metrics = binary_metrics(
        [sample.strict_label for sample in validation], validation_predictions
    )
    print(
        f"  validation ACC={validation_metrics['accuracy'] * 100:.2f} "
        f"F1={validation_metrics['f1'] * 100:.2f}"
    )

    quality_rows: list[dict[str, object]] = []
    for dataset, samples in evaluation.items():
        predictions = model.predict([sample.text for sample in samples])
        metrics = binary_metrics([sample.strict_label for sample in samples], predictions)
        quality_rows.append(
            {
                "model": model_name,
                "dataset": dataset,
                "evaluation_mode": "strict",
                "source": "local_measurement",
                **metrics,
            }
        )
        print(
            f"  {dataset:16} ACC={metrics['accuracy'] * 100:6.2f} "
            f"F1={metrics['f1'] * 100:6.2f} Recall={metrics['recall'] * 100:6.2f}"
        )

    model_path = artifacts_root / "models" / f"{model_name}.joblib"
    system_metrics = benchmark_model(
        model,
        model_path,
        [sample.text for sample in eval_all],
        train_metrics=training_system,
        single_samples=128,
        warmup=8,
        batch_repeats=3,
    )
    host = host_metadata()
    system_row = {
        "model": model_name,
        "variant": "all-MiniLM-L6-v2 frozen + logistic regression",
        "source": "local_measurement",
        "hardware": (
            f"{platform_name()} {host['machine']}; {host['ram_gib']} GiB RAM; CPU"
        ),
        **system_metrics,
    }
    print(
        f"  latency p50={system_metrics['latency_ms_p50']:.3f} ms, "
        f"p99={system_metrics['latency_ms_p99']:.3f} ms, "
        f"throughput={system_metrics['batch_throughput_samples_s']:.1f} sample/s"
    )

    existing_quality = [
        row
        for row in _read_csv(results_root / "predictive_quality.csv")
        if row.get("model") != model_name
    ]
    _write_csv(
        results_root / "predictive_quality.csv",
        [*existing_quality, *quality_rows],
        QUALITY_FIELDS,
    )
    existing_systems = [
        row
        for row in _read_csv(results_root / "systems_performance.csv")
        if row.get("model") != model_name
    ]
    _write_csv(
        results_root / "systems_performance.csv",
        [*existing_systems, system_row],
        SYSTEM_FIELDS,
    )

    diagnostics_path = results_root / "experiment_diagnostics.json"
    diagnostics = (
        json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics_path.exists()
        else {}
    )
    diagnostics.setdefault("validation_metrics", {})[model_name] = validation_metrics
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    return quality_rows, system_row


def platform_name() -> str:
    import platform

    return f"{platform.system()} {platform.release()}"
