from __future__ import annotations

import json
import math
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from toolsafe_lab.cascade import SigmoidCalibrator, calibration_metrics
from toolsafe_lab.data import Sample, load_eval, load_eval_sample_ids, load_training
from toolsafe_lab.metrics import binary_metrics
from toolsafe_lab.standalone_encoder import (
    ENCODER_SPECS,
    FIELD_MARKERS,
    PeakRSS,
    _artifact_size_mib,
    _device,
    _ensure_field_markers,
    _quality_summary,
    _seed_everything,
    _unsafe_margins,
    execution_context_fields,
    host_metadata,
    pack_sample,
    select_high_recall_threshold,
)


RANDOM_STATE = 260110156
MODEL_KEY = "minilm_l12"
RELATION_MAX_LENGTH = 256
TASK_CLASSES = {
    "strict": 2,
    "harmfulness": 3,
    "malicious_user_request": 2,
    "prompt_injection": 2,
}
DEFAULT_TASK_WEIGHTS = {
    "strict": 1.0,
    "harmfulness": 0.5,
    "malicious_user_request": 0.25,
    "prompt_injection": 0.25,
}
VARIANTS = ("packed_multitask", "relation_multitask")


@dataclass(frozen=True)
class FollowupConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_fraction: float = 0.1
    early_stopping_patience: int = 2
    target_recall: float = 0.95
    seed: int = RANDOM_STATE
    relation_max_length: int = RELATION_MAX_LENGTH
    teacher_alpha: float = 0.0


def relation_texts(sample: Sample) -> tuple[str, str, str]:
    fields = execution_context_fields(sample)
    return (
        "\n".join(
            (
                FIELD_MARKERS["request"],
                fields["request"],
                FIELD_MARKERS["action"],
                fields["action"],
            )
        ),
        "\n".join(
            (
                FIELD_MARKERS["history"],
                fields["history"],
                FIELD_MARKERS["action"],
                fields["action"],
            )
        ),
        "\n".join(
            (
                FIELD_MARKERS["action"],
                fields["action"],
                FIELD_MARKERS["schema"],
                fields["schema"],
            )
        ),
    )


def _teacher_targets(
    path: Path | None,
    samples: Sequence[Sample],
) -> dict[str, float]:
    if path is None:
        return {}
    values: dict[str, float] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        probability = float(row["unsafe_probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Teacher probability outside [0, 1] at {path}:{line_number}"
            )
        if sample_id in values:
            raise ValueError(f"Duplicate teacher target for {sample_id}")
        values[sample_id] = probability
    expected = {sample.sample_id for sample in samples}
    missing = expected - set(values)
    extra = set(values) - expected
    if missing or extra:
        raise ValueError(
            "Teacher targets must exactly cover training rows: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    return values


class _FollowupDataset:
    def __init__(
        self,
        tokenizer: Any,
        samples: Sequence[Sample],
        *,
        representation: str,
        relation_max_length: int,
        include_labels: bool,
        teacher_targets: Mapping[str, float] | None = None,
    ):
        import torch

        if not samples:
            raise ValueError("Follow-up encoder dataset cannot be empty")
        self.representation = representation
        if representation == "packed":
            encoded = [pack_sample(tokenizer, sample) for sample in samples]
            self.input_ids = torch.tensor(
                [row["input_ids"] for row in encoded], dtype=torch.long
            )
            self.attention_mask = torch.tensor(
                [row["attention_mask"] for row in encoded], dtype=torch.long
            )
        elif representation == "relation":
            flat_texts = [
                text for sample in samples for text in relation_texts(sample)
            ]
            encoded = tokenizer(
                flat_texts,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=relation_max_length,
                return_tensors="pt",
            )
            self.input_ids = encoded["input_ids"].reshape(
                len(samples), 3, relation_max_length
            )
            self.attention_mask = encoded["attention_mask"].reshape(
                len(samples), 3, relation_max_length
            )
        else:
            raise ValueError(f"Unsupported representation: {representation}")

        self.labels: dict[str, Any] = {}
        if include_labels:
            targets = [sample.auxiliary_labels for sample in samples]
            self.labels = {
                task: torch.tensor([row[task] for row in targets], dtype=torch.long)
                for task in TASK_CLASSES
            }
        self.teacher_probabilities = None
        if teacher_targets:
            self.teacher_probabilities = torch.tensor(
                [teacher_targets[sample.sample_id] for sample in samples],
                dtype=torch.float32,
            )

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
        }
        row.update({f"label_{task}": value[index] for task, value in self.labels.items()})
        if self.teacher_probabilities is not None:
            row["teacher_probability"] = self.teacher_probabilities[index]
        return row


def _build_model(
    *,
    model_source: str | Path,
    revision: str | None,
    representation: str,
) -> Any:
    import torch
    from transformers import AutoModel

    class MultiTaskSafetyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_source, revision=revision)
            hidden_size = int(self.encoder.config.hidden_size)
            self.representation = representation
            self.fusion = (
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_size * 3, hidden_size),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.1),
                )
                if representation == "relation"
                else torch.nn.Identity()
            )
            self.dropout = torch.nn.Dropout(0.1)
            self.heads = torch.nn.ModuleDict(
                {
                    task: torch.nn.Linear(hidden_size, classes)
                    for task, classes in TASK_CLASSES.items()
                }
            )

        def forward(self, input_ids: Any, attention_mask: Any) -> dict[str, Any]:
            if self.representation == "relation":
                batch_size, relation_count, sequence_length = input_ids.shape
                encoded = self.encoder(
                    input_ids=input_ids.reshape(batch_size * relation_count, sequence_length),
                    attention_mask=attention_mask.reshape(
                        batch_size * relation_count, sequence_length
                    ),
                ).last_hidden_state[:, 0]
                pooled = self.fusion(encoded.reshape(batch_size, -1))
            else:
                pooled = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state[:, 0]
            pooled = self.dropout(pooled)
            return {task: head(pooled) for task, head in self.heads.items()}

    return MultiTaskSafetyModel()


def _class_weights(samples: Sequence[Sample], device: str) -> dict[str, Any]:
    import torch

    targets = [sample.auxiliary_labels for sample in samples]
    weights: dict[str, Any] = {}
    for task, classes in TASK_CLASSES.items():
        values = np.asarray([row[task] for row in targets], dtype=np.int64)
        counts = np.bincount(values, minlength=classes)
        if np.any(counts == 0):
            raise ValueError(f"Training has an empty class for auxiliary task {task}")
        weights[task] = torch.tensor(
            [len(values) / (classes * count) for count in counts],
            dtype=torch.float32,
            device=device,
        )
    return weights


def _loss(
    logits: Mapping[str, Any],
    batch: Mapping[str, Any],
    class_weights: Mapping[str, Any],
    *,
    device: str,
    teacher_alpha: float,
) -> Any:
    import torch.nn.functional as functional

    task_losses = {
        task: functional.cross_entropy(
            logits[task],
            batch[f"label_{task}"].to(device),
            weight=class_weights[task].to(dtype=logits[task].dtype),
        )
        for task in TASK_CLASSES
    }
    supervised = sum(
        DEFAULT_TASK_WEIGHTS[task] * task_losses[task] for task in TASK_CLASSES
    )
    if teacher_alpha == 0.0:
        return supervised
    if "teacher_probability" not in batch:
        raise ValueError("Distillation requested without teacher probabilities")
    unsafe_margin = logits["strict"][:, 1] - logits["strict"][:, 0]
    distillation = functional.binary_cross_entropy_with_logits(
        unsafe_margin,
        batch["teacher_probability"].to(device=device, dtype=unsafe_margin.dtype),
    )
    strict_supervised = DEFAULT_TASK_WEIGHTS["strict"] * task_losses["strict"]
    auxiliary = supervised - strict_supervised
    return auxiliary + (1.0 - teacher_alpha) * strict_supervised + teacher_alpha * distillation


def _predict(
    model: Any,
    dataset: _FollowupDataset,
    *,
    device: str,
    batch_size: int,
    class_weights: Mapping[str, Any] | None = None,
    teacher_alpha: float = 0.0,
) -> tuple[dict[str, np.ndarray], float | None]:
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    accumulated = {task: [] for task in TASK_CLASSES}
    losses: list[float] = []
    seen = 0
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            logits = model(**inputs)
            for task in TASK_CLASSES:
                accumulated[task].append(logits[task].detach().cpu().float().numpy())
            if class_weights is not None:
                batch_loss = _loss(
                    logits,
                    batch,
                    class_weights,
                    device=device,
                    teacher_alpha=teacher_alpha,
                )
                batch_count = int(batch["input_ids"].shape[0])
                losses.append(float(batch_loss.detach().cpu()) * batch_count)
                seen += batch_count
    outputs = {
        task: np.concatenate(task_logits, axis=0)
        for task, task_logits in accumulated.items()
    }
    return outputs, (sum(losses) / seen if seen else None)


def _save_artifact(
    artifact_dir: Path,
    model: Any,
    tokenizer: Any,
    *,
    variant: str,
    config: FollowupConfig,
) -> None:
    import torch

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(artifact_dir / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(artifact_dir / "tokenizer")
    torch.save(
        {
            "fusion": model.fusion.state_dict(),
            "heads": model.heads.state_dict(),
        },
        artifact_dir / "task_heads.pt",
    )
    (artifact_dir / "toolsafe_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": variant,
                "representation": variant.split("_", 1)[0],
                "training_config": asdict(config),
                "task_classes": TASK_CLASSES,
                "task_weights": DEFAULT_TASK_WEIGHTS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_artifact(artifact_dir: Path, device: str) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoTokenizer

    metadata = json.loads(
        (artifact_dir / "toolsafe_config.json").read_text(encoding="utf-8")
    )
    representation = str(metadata["representation"])
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir / "tokenizer")
    model = _build_model(
        model_source=artifact_dir / "encoder",
        revision=None,
        representation=representation,
    )
    state = torch.load(artifact_dir / "task_heads.pt", map_location="cpu", weights_only=True)
    model.fusion.load_state_dict(state["fusion"])
    model.heads.load_state_dict(state["heads"])
    model.to(device)
    model.eval()
    return model, tokenizer, representation


def _train_variant(
    variant: str,
    train: Sequence[Sample],
    validation: Sequence[Sample],
    artifact_dir: Path,
    *,
    config: FollowupConfig,
    requested_device: str,
    teacher_targets: Mapping[str, float],
) -> dict[str, object]:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    if variant not in VARIANTS:
        raise ValueError(f"Unknown follow-up MiniLM variant: {variant}")
    if not 0.0 <= config.teacher_alpha < 1.0:
        raise ValueError("teacher_alpha must lie in [0, 1)")
    if bool(teacher_targets) != (config.teacher_alpha > 0.0):
        raise ValueError("Teacher targets and a positive teacher_alpha are required together")

    representation = variant.split("_", 1)[0]
    spec = ENCODER_SPECS[MODEL_KEY]
    _seed_everything(config.seed)
    selected_device = _device(requested_device)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
    _ensure_field_markers(tokenizer)
    model = _build_model(
        model_source=spec.model_id,
        revision=spec.revision,
        representation=representation,
    )
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.float()
    model.to(selected_device)

    train_dataset = _FollowupDataset(
        tokenizer,
        train,
        representation=representation,
        relation_max_length=config.relation_max_length,
        include_labels=True,
        teacher_targets=teacher_targets,
    )
    validation_dataset = _FollowupDataset(
        tokenizer,
        validation,
        representation=representation,
        relation_max_length=config.relation_max_length,
        include_labels=True,
    )
    weights = _class_weights(train, selected_device)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * config.warmup_fraction),
        num_training_steps=total_updates,
    )

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    best_loss = float("inf")
    best_epoch = 0
    patience = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    with PeakRSS() as training_memory:
        for epoch in range(1, config.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            seen = 0
            for step, batch in enumerate(train_loader, 1):
                inputs = {
                    "input_ids": batch["input_ids"].to(selected_device),
                    "attention_mask": batch["attention_mask"].to(selected_device),
                }
                logits = model(**inputs)
                loss = _loss(
                    logits,
                    batch,
                    weights,
                    device=selected_device,
                    teacher_alpha=config.teacher_alpha,
                )
                (loss / config.gradient_accumulation_steps).backward()
                batch_count = int(batch["input_ids"].shape[0])
                running_loss += float(loss.detach().cpu()) * batch_count
                seen += batch_count
                if (
                    step % config.gradient_accumulation_steps == 0
                    or step == len(train_loader)
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            _, validation_loss = _predict(
                model,
                validation_dataset,
                device=selected_device,
                batch_size=config.batch_size * 2,
                class_weights=weights,
            )
            if validation_loss is None:
                raise RuntimeError("Validation loss was not computed")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": running_loss / seen,
                    "validation_loss": validation_loss,
                }
            )
            print(
                f"  {variant} epoch={epoch} train_loss={running_loss / seen:.4f} "
                f"validation_loss={validation_loss:.4f}"
            )
            if validation_loss < best_loss - 1e-4:
                best_loss = validation_loss
                best_epoch = epoch
                patience = 0
                _save_artifact(
                    artifact_dir,
                    model,
                    tokenizer,
                    variant=variant,
                    config=config,
                )
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    break

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    train_seconds = time.perf_counter() - started
    del model
    if selected_device == "mps":
        torch.mps.empty_cache()
    return {
        "variant": variant,
        "representation": representation,
        "encoder_spec": asdict(spec),
        "device": selected_device,
        "parameter_count": parameter_count,
        "training": {
            "config": asdict(config),
            "task_weights": DEFAULT_TASK_WEIGHTS,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "history": history,
            "train_seconds": train_seconds,
            "train_peak_rss_delta_mib": training_memory.delta_mib,
            "teacher_target_count": len(teacher_targets),
        },
    }


def _validation_summary(
    model: Any,
    dataset: _FollowupDataset,
    samples: Sequence[Sample],
    *,
    device: str,
    batch_size: int,
    target_recall: float,
) -> tuple[dict[str, object], SigmoidCalibrator]:
    outputs, _ = _predict(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
    )
    strict_labels = np.asarray([sample.strict_label for sample in samples], dtype=np.int8)
    strict_margins = _unsafe_margins(outputs["strict"])
    calibrator = SigmoidCalibrator(random_state=RANDOM_STATE).fit(
        strict_margins, strict_labels
    )
    probabilities = calibrator.predict_proba(strict_margins)
    threshold = select_high_recall_threshold(
        probabilities,
        strict_labels,
        target_recall=target_recall,
    )
    auxiliary: dict[str, dict[str, float]] = {}
    for task in ("harmfulness", "malicious_user_request", "prompt_injection"):
        labels = np.asarray(
            [sample.auxiliary_labels[task] for sample in samples], dtype=np.int64
        )
        predictions = outputs[task].argmax(axis=1)
        auxiliary[task] = {"accuracy": float(np.mean(labels == predictions))}
    return (
        {
            "calibration": calibration_metrics(strict_labels, probabilities),
            "default_0_5": binary_metrics(
                strict_labels, (probabilities >= 0.5).astype(np.int8)
            ),
            "high_recall": threshold,
            "auxiliary": auxiliary,
            "calibrator_coefficient": float(calibrator.model.coef_[0, 0]),
            "calibrator_intercept": float(calibrator.model.intercept_[0]),
        },
        calibrator,
    )


def _benchmark(
    artifact_dir: Path,
    samples: Sequence[Sample],
    *,
    device: str,
    batch_size: int,
    relation_max_length: int,
) -> dict[str, object]:
    import torch

    load_times: list[float] = []
    for _ in range(3):
        started = time.perf_counter_ns()
        model, tokenizer, representation = _load_artifact(artifact_dir, device)
        load_times.append((time.perf_counter_ns() - started) / 1e6)
        del model
        if device == "mps":
            torch.mps.empty_cache()
    model, tokenizer, representation = _load_artifact(artifact_dir, device)
    sample_count = min(128, len(samples))
    indices = np.linspace(0, len(samples) - 1, sample_count, dtype=int)
    selected = [samples[index] for index in indices]

    single_dataset = _FollowupDataset(
        tokenizer,
        selected,
        representation=representation,
        relation_max_length=relation_max_length,
        include_labels=False,
    )
    for index in range(min(8, len(selected))):
        row = single_dataset[index]
        with torch.inference_mode():
            model(
                input_ids=row["input_ids"].unsqueeze(0).to(device),
                attention_mask=row["attention_mask"].unsqueeze(0).to(device),
            )
    latencies: list[float] = []
    with PeakRSS() as single_memory:
        for index in range(len(selected)):
            row = single_dataset[index]
            started = time.perf_counter_ns()
            with torch.inference_mode():
                model(
                    input_ids=row["input_ids"].unsqueeze(0).to(device),
                    attention_mask=row["attention_mask"].unsqueeze(0).to(device),
                )
            latencies.append((time.perf_counter_ns() - started) / 1e6)

    batch_dataset = _FollowupDataset(
        tokenizer,
        samples,
        representation=representation,
        relation_max_length=relation_max_length,
        include_labels=False,
    )
    batch_times: list[float] = []
    with PeakRSS() as batch_memory:
        for _ in range(3):
            started = time.perf_counter()
            _predict(
                model,
                batch_dataset,
                device=device,
                batch_size=batch_size,
            )
            batch_times.append(time.perf_counter() - started)
    median_batch = statistics.median(batch_times)
    return {
        "artifact_size_mib": _artifact_size_mib(artifact_dir),
        "load_ms_p50": statistics.median(load_times),
        "single_n": len(latencies),
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "latency_ms_p99": float(np.percentile(latencies, 99)),
        "single_peak_rss_delta_mib": single_memory.delta_mib,
        "batch_n": len(samples),
        "batch_repeats": len(batch_times),
        "batch_throughput_samples_s": len(samples) / median_batch,
        "batch_peak_rss_delta_mib": batch_memory.delta_mib,
    }


def run_multitask_encoder_experiment(
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    *,
    variants: Sequence[str] = VARIANTS,
    requested_device: str = "auto",
    epochs: int = 5,
    teacher_targets_path: Path | None = None,
    teacher_alpha: float = 0.0,
) -> dict[str, object]:
    try:
        import torch
        import transformers
    except ImportError as error:
        raise ImportError(
            "Install the encoder extra: pip install -e '.[encoders]'"
        ) from error

    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown follow-up MiniLM variants: {', '.join(unknown)}")
    config = FollowupConfig(epochs=epochs, teacher_alpha=teacher_alpha)
    train = load_training(data_root, "train")
    validation = load_training(data_root, "validation")
    evaluation_ids = load_eval_sample_ids(data_root)
    train_before_filtering = len(train)
    validation_before_filtering = len(validation)
    train = [sample for sample in train if sample.sample_id not in evaluation_ids]
    validation = [
        sample for sample in validation if sample.sample_id not in evaluation_ids
    ]
    teachers = _teacher_targets(teacher_targets_path, train)
    print(
        f"Follow-up MiniLM protocol: train={len(train)} validation={len(validation)} "
        f"variants={','.join(variants)} teacher_targets={len(teachers)}"
    )

    # Train and validation-select every frozen candidate before loading evaluation rows.
    trained: dict[str, dict[str, object]] = {}
    calibrators: dict[str, SigmoidCalibrator] = {}
    for variant in variants:
        artifact_dir = artifacts_root / "models" / "multitask_encoders" / variant
        metadata = _train_variant(
            variant,
            train,
            validation,
            artifact_dir,
            config=config,
            requested_device=requested_device,
            teacher_targets=teachers,
        )
        device = str(metadata["device"])
        model, tokenizer, representation = _load_artifact(artifact_dir, device)
        validation_dataset = _FollowupDataset(
            tokenizer,
            validation,
            representation=representation,
            relation_max_length=config.relation_max_length,
            include_labels=True,
        )
        metadata["validation"], calibrators[variant] = _validation_summary(
            model,
            validation_dataset,
            validation,
            device=device,
            batch_size=config.batch_size * 2,
            target_recall=config.target_recall,
        )
        trained[variant] = metadata
        del model
        if device == "mps":
            torch.mps.empty_cache()

    validation_ranking = sorted(
        variants,
        key=lambda variant: (
            -float(
                trained[variant]["validation"]["high_recall"]["validation_metrics"][  # type: ignore[index]
                    "mcc"
                ]
            ),
            -float(
                trained[variant]["validation"]["high_recall"]["validation_metrics"][  # type: ignore[index]
                    "balanced_accuracy"
                ]
            ),
            variant,
        ),
    )
    selected_variant = validation_ranking[0]

    evaluation_by_source = load_eval(data_root)
    evaluation = [
        sample for source_samples in evaluation_by_source.values() for sample in source_samples
    ]
    labels = np.asarray([sample.strict_label for sample in evaluation], dtype=np.int8)
    sources = [sample.source for sample in evaluation]
    for variant in variants:
        metadata = trained[variant]
        artifact_dir = artifacts_root / "models" / "multitask_encoders" / variant
        device = str(metadata["device"])
        model, tokenizer, representation = _load_artifact(artifact_dir, device)
        evaluation_dataset = _FollowupDataset(
            tokenizer,
            evaluation,
            representation=representation,
            relation_max_length=config.relation_max_length,
            include_labels=False,
        )
        outputs, _ = _predict(
            model,
            evaluation_dataset,
            device=device,
            batch_size=config.batch_size * 2,
        )
        probabilities = calibrators[variant].predict_proba(
            _unsafe_margins(outputs["strict"])
        )
        high_recall_threshold = float(
            metadata["validation"]["high_recall"]["threshold"]  # type: ignore[index]
        )
        metadata["evaluation"] = {
            "default_0_5": _quality_summary(labels, probabilities, sources, 0.5),
            "high_recall": _quality_summary(
                labels, probabilities, sources, high_recall_threshold
            ),
        }
        metadata["systems"] = _benchmark(
            artifact_dir,
            evaluation,
            device=device,
            batch_size=config.batch_size * 2,
            relation_max_length=config.relation_max_length,
        )
        del model
        if device == "mps":
            torch.mps.empty_cache()

    result = {
        "schema_version": 1,
        "protocol": {
            "name": "multitask_relation_minilm_v1",
            "gold_auxiliary_targets": list(TASK_CLASSES),
            "task_weights": DEFAULT_TASK_WEIGHTS,
            "counterfactual_rows_used": 0,
            "evaluation_used_for_selection": False,
            "teacher_targets_path": (
                str(teacher_targets_path) if teacher_targets_path is not None else None
            ),
        },
        "input": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "evaluation_rows": len(evaluation),
            "evaluation_input_overlap_removed_before_fitting": {
                "train": train_before_filtering - len(train),
                "validation": validation_before_filtering - len(validation),
            },
        },
        "validation_ranking": validation_ranking,
        "validation_selected_variant": selected_variant,
        "models": trained,
        "host": host_metadata(),
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    results_root.mkdir(parents=True, exist_ok=True)
    output_path = results_root / "multitask_relation_minilm.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    return result
