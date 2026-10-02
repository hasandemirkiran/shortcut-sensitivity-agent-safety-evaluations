from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from toolsafe_lab.cascade import SigmoidCalibrator, calibration_metrics
from toolsafe_lab.data import Sample, load_eval, load_training
from toolsafe_lab.metrics import binary_metrics
from toolsafe_lab.standalone_encoder import (
    ENCODER_SPECS,
    EXECUTION_CONTEXT,
    FIELD_BUDGETS,
    FIELD_MARKERS,
    FULL_REACT_CONTEXT,
    MAX_LENGTH,
    TrainingConfig,
    _benchmark,
    _encode_samples,
    _load_model,
    _PackedDataset,
    _probability_logits,
    _quality_summary,
    _sha256,
    _train_one,
    _unsafe_margins,
    select_high_recall_threshold,
)
from toolsafe_lab.system_benchmark import host_metadata


def _validation_fit(
    model: Any,
    tokenizer: Any,
    validation: Sequence[Sample],
    *,
    input_mode: str,
    device: str,
    config: TrainingConfig,
) -> tuple[SigmoidCalibrator, dict[str, object]]:
    labels = np.asarray([sample.strict_label for sample in validation], dtype=np.int8)
    dataset = _PackedDataset(
        _encode_samples(tokenizer, validation, input_mode=input_mode), labels
    )
    logits, _ = _probability_logits(
        model,
        dataset,
        device=device,
        batch_size=config.batch_size * 2,
    )
    margins = _unsafe_margins(logits)
    calibrator = SigmoidCalibrator(random_state=config.seed).fit(margins, labels)
    probabilities = calibrator.predict_proba(margins)
    threshold = select_high_recall_threshold(
        probabilities,
        labels,
        target_recall=config.target_recall,
    )
    return calibrator, {
        "input_mode": input_mode,
        "calibration": calibration_metrics(labels, probabilities),
        "default_0_5": binary_metrics(
            labels, (probabilities >= 0.5).astype(np.int8)
        ),
        "high_recall": threshold,
        "calibrator_coefficient": float(calibrator.model.coef_[0, 0]),
        "calibrator_intercept": float(calibrator.model.intercept_[0]),
    }


def _evaluate(
    model: Any,
    tokenizer: Any,
    evaluation: Sequence[Sample],
    labels: np.ndarray,
    sources: Sequence[str],
    *,
    input_mode: str,
    device: str,
    batch_size: int,
    calibrator: SigmoidCalibrator,
    high_recall_threshold: float,
) -> dict[str, object]:
    dataset = _PackedDataset(
        _encode_samples(tokenizer, evaluation, input_mode=input_mode), labels
    )
    logits, _ = _probability_logits(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
    )
    probabilities = calibrator.predict_proba(_unsafe_margins(logits))
    return {
        "input_mode": input_mode,
        "default_0_5": _quality_summary(labels, probabilities, sources, 0.5),
        "high_recall": _quality_summary(
            labels, probabilities, sources, high_recall_threshold
        ),
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_thought_encoder_ablation(
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    *,
    model_keys: Sequence[str] = ("minilm_l12", "deberta_v3_xsmall"),
    requested_device: str = "auto",
    epochs: int = 5,
    resume: bool = False,
) -> dict[str, object]:
    try:
        import torch
        import transformers
    except ImportError as error:
        raise ImportError(
            "Install the encoder extra: pip install -e '.[encoders]'"
        ) from error

    unknown = sorted(set(model_keys) - set(ENCODER_SPECS))
    if unknown:
        raise ValueError(f"Unknown encoder models: {', '.join(unknown)}")

    baseline_path = results_root / "standalone_encoders.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            "Run train-standalone-encoders before the symmetric Thought ablation"
        )
    baseline = json.loads(baseline_path.read_text())
    missing_baselines = sorted(set(model_keys) - set(baseline.get("models", {})))
    if missing_baselines:
        raise ValueError(
            "The no-Thought baseline is missing models: " + ", ".join(missing_baselines)
        )

    config = TrainingConfig(epochs=epochs)
    train = load_training(data_root, "train")
    validation = load_training(data_root, "validation")
    evaluation_by_source = load_eval(data_root)
    evaluation_ids = {
        sample.sample_id
        for samples in evaluation_by_source.values()
        for sample in samples
    }
    train = [sample for sample in train if sample.sample_id not in evaluation_ids]
    validation = [
        sample for sample in validation if sample.sample_id not in evaluation_ids
    ]
    print(
        f"Thought encoder protocol: train={len(train)} validation={len(validation)} "
        f"models={','.join(model_keys)}"
    )

    # Preserve the evaluation barrier: train and validation-freeze every new
    # candidate before computing any evaluation metric.
    thought_models: dict[str, dict[str, object]] = {}
    for key in model_keys:
        spec = ENCODER_SPECS[key]
        print(f"Training Thought-inclusive encoder {key} ({spec.model_id})")
        artifact_dir = artifacts_root / "models" / "thought_encoders" / key
        metadata = _train_one(
            spec,
            train,
            validation,
            artifact_dir,
            config=config,
            requested_device=requested_device,
            resume=resume,
            input_mode=FULL_REACT_CONTEXT,
        )
        device = str(metadata["device"])
        model, tokenizer = _load_model(artifact_dir, device)
        _, validation_summary = _validation_fit(
            model,
            tokenizer,
            validation,
            input_mode=FULL_REACT_CONTEXT,
            device=device,
            config=config,
        )
        metadata["validation"] = validation_summary
        thought_models[key] = metadata
        del model
        if device == "mps":
            torch.mps.empty_cache()

    evaluation = [
        sample for samples in evaluation_by_source.values() for sample in samples
    ]
    labels = np.asarray([sample.strict_label for sample in evaluation], dtype=np.int8)
    sources = [sample.source for sample in evaluation]

    for key in model_keys:
        metadata = thought_models[key]
        artifact_dir = artifacts_root / "models" / "thought_encoders" / key
        device = str(metadata["device"])
        model, tokenizer = _load_model(artifact_dir, device)
        calibrator, validation_summary = _validation_fit(
            model,
            tokenizer,
            validation,
            input_mode=FULL_REACT_CONTEXT,
            device=device,
            config=config,
        )
        metadata["validation"] = validation_summary
        threshold = float(validation_summary["high_recall"]["threshold"])  # type: ignore[index]
        metadata["evaluation"] = {
            "matched_full_react": _evaluate(
                model,
                tokenizer,
                evaluation,
                labels,
                sources,
                input_mode=FULL_REACT_CONTEXT,
                device=device,
                batch_size=config.batch_size * 2,
                calibrator=calibrator,
                high_recall_threshold=threshold,
            ),
            "cross_execution_context": _evaluate(
                model,
                tokenizer,
                evaluation,
                labels,
                sources,
                input_mode=EXECUTION_CONTEXT,
                device=device,
                batch_size=config.batch_size * 2,
                calibrator=calibrator,
                high_recall_threshold=threshold,
            ),
        }
        metadata["systems"] = _benchmark(
            artifact_dir,
            evaluation,
            device=device,
            batch_size=config.batch_size * 2,
            input_mode=FULL_REACT_CONTEXT,
        )
        primary = metadata["evaluation"]["matched_full_react"]["default_0_5"][  # type: ignore[index]
            "micro"
        ]
        print(
            f"  {key} Thought matched: accuracy={primary['accuracy']:.4f} "
            f"recall={primary['recall']:.4f} MCC={primary['mcc']:.4f}"
        )
        del model
        if device == "mps":
            torch.mps.empty_cache()

    no_thought_models: dict[str, dict[str, object]] = {}
    for key in model_keys:
        baseline_model = baseline["models"][key]
        artifact_dir = artifacts_root / "models" / "standalone_encoders" / key
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Missing no-Thought checkpoint: {artifact_dir}")
        device = str(baseline_model["device"])
        model, tokenizer = _load_model(artifact_dir, device)
        calibrator, validation_summary = _validation_fit(
            model,
            tokenizer,
            validation,
            input_mode=EXECUTION_CONTEXT,
            device=device,
            config=config,
        )
        frozen_threshold = float(
            baseline_model["validation"]["high_recall"]["threshold"]
        )
        no_thought_models[key] = {
            "device": device,
            "parameter_count": baseline_model["parameter_count"],
            "validation_recomputed": validation_summary,
            "evaluation": {
                "matched_execution_context": {
                    "input_mode": EXECUTION_CONTEXT,
                    "default_0_5": baseline_model["evaluation"]["default_0_5"],
                    "high_recall": baseline_model["evaluation"]["high_recall"],
                    "source": "results/standalone_encoders.json",
                },
                "cross_full_react": _evaluate(
                    model,
                    tokenizer,
                    evaluation,
                    labels,
                    sources,
                    input_mode=FULL_REACT_CONTEXT,
                    device=device,
                    batch_size=config.batch_size * 2,
                    calibrator=calibrator,
                    high_recall_threshold=frozen_threshold,
                ),
            },
            "systems_matched_execution_context": baseline_model["systems"],
        }
        del model
        if device == "mps":
            torch.mps.empty_cache()

    protocol_path = Path(__file__).parents[2] / "docs" / "THOUGHT_ENCODER_ABLATION_PROTOCOL.md"
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "exploratory_prior_evaluation_results_inspected",
        "protocol": {
            "version": "1.0",
            "path": "docs/THOUGHT_ENCODER_ABLATION_PROTOCOL.md",
            "sha256": _sha256(protocol_path),
            "evaluation_metrics_computed_after_all_new_candidates_trained": True,
        },
        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "evaluation": len(evaluation),
        },
        "input": {
            "fields": list(FIELD_MARKERS),
            "max_length": MAX_LENGTH,
            "field_budgets": FIELD_BUDGETS,
            "training_mode": FULL_REACT_CONTEXT,
            "evaluation_modes": [FULL_REACT_CONTEXT, EXECUTION_CONTEXT],
        },
        "thought_trained_models": thought_models,
        "no_thought_reference": {
            "path": "results/standalone_encoders.json",
            "sha256": _sha256(baseline_path),
            "models": no_thought_models,
        },
        "host": {
            **host_metadata(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "requested_device": requested_device,
        },
        "environment": {"pythonhashseed": os.environ.get("PYTHONHASHSEED")},
    }
    output = results_root / "thought_encoder_ablation.json"
    _write_json(output, result)
    print(f"Wrote {output}")
    return result
