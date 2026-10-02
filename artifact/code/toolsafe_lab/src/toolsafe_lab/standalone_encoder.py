from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import log_loss

from toolsafe_lab.cascade import SigmoidCalibrator, calibration_metrics
from toolsafe_lab.data import Sample, load_eval, load_training
from toolsafe_lab.metrics import binary_metrics
from toolsafe_lab.react_parser import parse_react_step
from toolsafe_lab.system_benchmark import PeakRSS, host_metadata


RANDOM_STATE = 260110156
TARGET_RECALL = 0.95
MAX_LENGTH = 512
EXECUTION_CONTEXT = "execution_context"
FULL_REACT_CONTEXT = "full_react_context"
INPUT_MODES = (EXECUTION_CONTEXT, FULL_REACT_CONTEXT)
FIELD_BUDGETS = {
    "request": 96,
    "history": 144,
    "action": 128,
    "schema": 128,
}
FIELD_MARKERS = {
    "request": "[USER_REQUEST]",
    "history": "[INTERACTION_HISTORY]",
    "action": "[CURRENT_ACTION]",
    "schema": "[TOOL_DESCRIPTIONS]",
}


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    model_id: str
    revision: str


ENCODER_SPECS = {
    "minilm_l12": EncoderSpec(
        key="minilm_l12",
        model_id="microsoft/MiniLM-L12-H384-uncased",
        revision="44acabbec0ef496f6dbc93adadea57f376b7c0ec",
    ),
    "deberta_v3_xsmall": EncoderSpec(
        key="deberta_v3_xsmall",
        model_id="microsoft/deberta-v3-xsmall",
        revision="4b419818330868dff6a60ad3e6b1c730f8b8c0c6",
    ),
}


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_fraction: float = 0.1
    early_stopping_patience: int = 2
    max_length: int = MAX_LENGTH
    target_recall: float = TARGET_RECALL
    seed: int = RANDOM_STATE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def execution_context_fields(sample: Sample) -> dict[str, str]:
    parsed = parse_react_step(sample.current_action)
    return {
        "request": sample.instruction,
        "history": sample.history,
        "action": parsed.execution_only_text,
        "schema": sample.env_info,
    }


def full_react_context_fields(sample: Sample) -> dict[str, str]:
    return {
        "request": sample.instruction,
        "history": sample.history,
        "action": sample.current_action,
        "schema": sample.env_info,
    }


def context_fields(sample: Sample, input_mode: str) -> dict[str, str]:
    if input_mode == EXECUTION_CONTEXT:
        return execution_context_fields(sample)
    if input_mode == FULL_REACT_CONTEXT:
        return full_react_context_fields(sample)
    raise ValueError(f"Unknown standalone encoder input mode: {input_mode}")


def _head_tail(values: Sequence[int], budget: int) -> list[int]:
    if budget < 0:
        raise ValueError("Token budget cannot be negative")
    if len(values) <= budget:
        return list(values)
    if budget == 0:
        return []
    head = (budget + 1) // 2
    tail = budget - head
    return [*values[:head], *values[-tail:]] if tail else list(values[:head])


def _head(values: Sequence[int], budget: int) -> list[int]:
    if budget < 0:
        raise ValueError("Token budget cannot be negative")
    return list(values[:budget])


def _ensure_field_markers(tokenizer: Any) -> None:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(FIELD_MARKERS.values())}
    )


def pack_sample(
    tokenizer: Any,
    sample: Sample,
    *,
    max_length: int = MAX_LENGTH,
    field_budgets: dict[str, int] | None = None,
    input_mode: str = EXECUTION_CONTEXT,
) -> dict[str, list[int]]:
    budgets = field_budgets or FIELD_BUDGETS
    if set(budgets) != set(FIELD_MARKERS):
        raise ValueError("Field budgets must cover every standalone input field")

    fields = context_fields(sample, input_mode)
    start_id = tokenizer.cls_token_id
    if start_id is None:
        start_id = tokenizer.bos_token_id
    separator_id = tokenizer.sep_token_id
    if separator_id is None:
        separator_id = tokenizer.eos_token_id
    if start_id is None or separator_id is None:
        raise ValueError("Tokenizer requires start and separator tokens")

    input_ids = [int(start_id)]
    for name in ("request", "history", "action", "schema"):
        marker_id = tokenizer.convert_tokens_to_ids(FIELD_MARKERS[name])
        if marker_id is None or marker_id == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer is missing field marker {FIELD_MARKERS[name]}")
        tokens = tokenizer.encode(fields[name], add_special_tokens=False)
        selected = (
            _head(tokens, budgets[name])
            if name == "request"
            else _head_tail(tokens, budgets[name])
        )
        input_ids.extend((int(marker_id), *selected, int(separator_id)))

    if len(input_ids) > max_length:
        raise ValueError(
            f"Packed sequence has {len(input_ids)} tokens, over limit {max_length}"
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = separator_id
    attention_mask = [1] * len(input_ids)
    padding = max_length - len(input_ids)
    input_ids.extend([int(pad_id)] * padding)
    attention_mask.extend([0] * padding)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def select_high_recall_threshold(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    target_recall: float = TARGET_RECALL,
) -> dict[str, object]:
    p = np.asarray(list(probabilities), dtype=float)
    y = np.asarray(list(labels), dtype=np.int8)
    if p.size == 0 or p.size != y.size:
        raise ValueError("Threshold selection requires equally sized non-empty inputs")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    if not 0 < target_recall <= 1:
        raise ValueError("Target recall must lie in (0, 1]")
    if np.unique(y).size != 2:
        raise ValueError("Threshold selection requires both classes")

    candidates = sorted(
        {
            0.0,
            float(np.nextafter(1.0, 2.0)),
            *(float(value) for value in np.unique(p)),
        }
    )
    best: tuple[tuple[float, ...], float, dict[str, float | int]] | None = None
    for threshold in candidates:
        predictions = (p >= threshold).astype(np.int8)
        metrics = binary_metrics(y, predictions)
        if float(metrics["recall"]) + 1e-12 < target_recall:
            continue
        rank = (
            float(metrics["specificity"]),
            float(metrics["mcc"]),
            float(metrics["balanced_accuracy"]),
            threshold,
        )
        if best is None or rank > best[0]:
            best = (rank, threshold, metrics)
    if best is None:
        raise RuntimeError("No threshold satisfies the recall target")
    return {
        "threshold": best[1],
        "target_recall": target_recall,
        "validation_metrics": best[2],
    }


class _PackedDataset:
    def __init__(self, encoded: Sequence[dict[str, list[int]]], labels: Sequence[int]):
        import torch

        if not encoded or len(encoded) != len(labels):
            raise ValueError("Dataset requires equally sized non-empty inputs and labels")
        self.input_ids = torch.tensor(
            [row["input_ids"] for row in encoded], dtype=torch.long
        )
        self.attention_mask = torch.tensor(
            [row["attention_mask"] for row in encoded], dtype=torch.long
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def _encode_samples(
    tokenizer: Any,
    samples: Sequence[Sample],
    *,
    input_mode: str = EXECUTION_CONTEXT,
) -> list[dict[str, list[int]]]:
    return [
        pack_sample(tokenizer, sample, input_mode=input_mode) for sample in samples
    ]


def _device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _probability_logits(
    model: Any,
    dataset: _PackedDataset,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    outputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch_labels = batch.pop("labels")
            inputs = {name: value.to(device) for name, value in batch.items()}
            logits = model(**inputs).logits.detach().cpu().float().numpy()
            outputs.append(logits)
            labels.append(batch_labels.numpy())
    all_logits = np.concatenate(outputs, axis=0)
    all_labels = np.concatenate(labels, axis=0)
    return all_logits, float(log_loss(all_labels, _softmax(all_logits), labels=[0, 1]))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _unsafe_margins(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Expected two-class logits")
    return logits[:, 1] - logits[:, 0]


def _checkpoint_path(artifact_dir: Path) -> Path:
    return artifact_dir / "training_checkpoint.pt"


def _cpu_copy(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_copy(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(child) for child in value)
    return value


def _optimizer_to_device(optimizer: Any, device: str) -> None:
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _save_training_checkpoint(
    path: Path,
    *,
    spec: EncoderSpec,
    config: TrainingConfig,
    epoch: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    generator: Any,
    best_loss: float,
    best_epoch: int,
    patience: int,
    history: Sequence[dict[str, float | int]],
    elapsed_train_seconds: float,
    train_peak_rss_delta_mib: float,
    device: str,
    input_mode: str = EXECUTION_CONTEXT,
) -> None:
    import torch

    state: dict[str, Any] = {
        "schema_version": 1,
        "spec": asdict(spec),
        "config": asdict(config),
        "input_mode": input_mode,
        "epoch": epoch,
        "model_state_dict": _cpu_copy(model.state_dict()),
        "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
        "scheduler_state_dict": scheduler.state_dict(),
        "generator_state": generator.get_state(),
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "patience": patience,
        "history": list(history),
        "elapsed_train_seconds": elapsed_train_seconds,
        "train_peak_rss_delta_mib": train_peak_rss_delta_mib,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if device == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["device_random_state"] = torch.mps.get_rng_state()
    elif device == "cuda":
        state["device_random_state"] = torch.cuda.get_rng_state()
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _load_training_checkpoint(
    path: Path,
    *,
    spec: EncoderSpec,
    config: TrainingConfig,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    generator: Any,
    device: str,
    input_mode: str = EXECUTION_CONTEXT,
) -> dict[str, Any]:
    import torch

    if not path.exists():
        raise FileNotFoundError(f"No resumable training checkpoint at {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported standalone training-checkpoint schema")
    if state.get("spec") != asdict(spec):
        raise ValueError("Checkpoint encoder specification does not match")
    if state.get("config") != asdict(config):
        raise ValueError("Checkpoint training configuration does not match")
    if state.get("input_mode", EXECUTION_CONTEXT) != input_mode:
        raise ValueError("Checkpoint input representation does not match")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    _optimizer_to_device(optimizer, device)
    scheduler.load_state_dict(state["scheduler_state_dict"])
    generator.set_state(state["generator_state"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    device_state = state.get("device_random_state")
    if device_state is not None:
        if device == "mps" and hasattr(torch.mps, "set_rng_state"):
            torch.mps.set_rng_state(device_state)
        elif device == "cuda":
            torch.cuda.set_rng_state(device_state)
    return state


def _train_one(
    spec: EncoderSpec,
    train: Sequence[Sample],
    validation: Sequence[Sample],
    artifact_dir: Path,
    *,
    config: TrainingConfig,
    requested_device: str,
    resume: bool,
    input_mode: str = EXECUTION_CONTEXT,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as functional
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    _seed_everything(config.seed)
    selected_device = _device(requested_device)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
    _ensure_field_markers(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        num_labels=2,
    )
    model.resize_token_embeddings(len(tokenizer))
    # Some upstream checkpoints advertise half precision even when loaded on
    # Apple MPS. Normalize both frozen candidates to the protocol's shared
    # full-precision training path.
    model.float()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    model.to(selected_device)

    train_labels = [sample.strict_label for sample in train]
    validation_labels = [sample.strict_label for sample in validation]
    train_dataset = _PackedDataset(
        _encode_samples(tokenizer, train, input_mode=input_mode), train_labels
    )
    validation_dataset = _PackedDataset(
        _encode_samples(tokenizer, validation, input_mode=input_mode),
        validation_labels,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    counts = np.bincount(np.asarray(train_labels, dtype=np.int8), minlength=2)
    class_weights = torch.tensor(
        [len(train_labels) / (2 * count) for count in counts],
        dtype=torch.float32,
        device=selected_device,
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
    warmup_steps = round(total_updates * config.warmup_fraction)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    checkpoint_path = _checkpoint_path(artifact_dir)
    if not resume:
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True)
        best_loss = float("inf")
        best_epoch = 0
        patience = 0
        history: list[dict[str, float | int]] = []
        completed_epoch = 0
        prior_train_seconds = 0.0
        prior_peak_rss_delta_mib = 0.0
    else:
        state = _load_training_checkpoint(
            checkpoint_path,
            spec=spec,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            device=selected_device,
            input_mode=input_mode,
        )
        best_loss = float(state["best_loss"])
        best_epoch = int(state["best_epoch"])
        patience = int(state["patience"])
        history = list(state["history"])
        completed_epoch = int(state["epoch"])
        prior_train_seconds = float(state["elapsed_train_seconds"])
        prior_peak_rss_delta_mib = float(state["train_peak_rss_delta_mib"])
        print(f"  resuming {spec.key} after completed epoch {completed_epoch}")

    started = time.perf_counter()
    with PeakRSS() as training_memory:
        for epoch in range(completed_epoch + 1, config.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            seen = 0
            for step, batch in enumerate(train_loader, start=1):
                labels = batch.pop("labels").to(selected_device)
                inputs = {name: value.to(selected_device) for name, value in batch.items()}
                logits = model(**inputs).logits
                loss = functional.cross_entropy(
                    logits,
                    labels,
                    weight=class_weights.to(dtype=logits.dtype),
                )
                (loss / config.gradient_accumulation_steps).backward()
                running_loss += float(loss.detach().cpu()) * int(labels.shape[0])
                seen += int(labels.shape[0])
                if (
                    step % config.gradient_accumulation_steps == 0
                    or step == len(train_loader)
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            _, validation_loss = _probability_logits(
                model,
                validation_dataset,
                device=selected_device,
                batch_size=config.batch_size * 2,
            )
            epoch_result = {
                "epoch": epoch,
                "train_loss": running_loss / seen,
                "validation_loss": validation_loss,
            }
            history.append(epoch_result)
            print(
                f"  {spec.key} epoch={epoch} "
                f"train_loss={epoch_result['train_loss']:.4f} "
                f"validation_loss={validation_loss:.4f}"
            )
            should_stop = False
            if validation_loss < best_loss - 1e-4:
                best_loss = validation_loss
                best_epoch = epoch
                patience = 0
                model.save_pretrained(artifact_dir, safe_serialization=True)
                tokenizer.save_pretrained(artifact_dir)
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    should_stop = True
            _save_training_checkpoint(
                checkpoint_path,
                spec=spec,
                config=config,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                generator=generator,
                best_loss=best_loss,
                best_epoch=best_epoch,
                patience=patience,
                history=history,
                elapsed_train_seconds=(
                    prior_train_seconds + time.perf_counter() - started
                ),
                train_peak_rss_delta_mib=max(
                    prior_peak_rss_delta_mib, training_memory.delta_mib
                ),
                device=selected_device,
                input_mode=input_mode,
            )
            if should_stop:
                break

    train_seconds = prior_train_seconds + time.perf_counter() - started
    del model
    if selected_device == "mps":
        torch.mps.empty_cache()

    return {
        "spec": asdict(spec),
        "device": selected_device,
        "parameter_count": parameter_count,
        "training": {
            "input_mode": input_mode,
            "config": asdict(config),
            "class_counts": {"safe": int(counts[0]), "unsafe": int(counts[1])},
            "class_weights": {
                "safe": float(class_weights[0].detach().cpu()),
                "unsafe": float(class_weights[1].detach().cpu()),
            },
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "history": history,
            "train_seconds": train_seconds,
            "train_peak_rss_delta_mib": max(
                prior_peak_rss_delta_mib, training_memory.delta_mib
            ),
            "resumed": resume,
            "checkpoint_path": str(checkpoint_path.relative_to(artifact_dir)),
        },
    }


def _load_model(artifact_dir: Path, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def _artifact_size_mib(path: Path) -> float:
    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file() and file.name != "training_checkpoint.pt"
    ) / (1024**2)


def _quality_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    datasets: Sequence[str],
    threshold: float,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int8)
    by_dataset: dict[str, dict[str, object]] = {}
    for dataset in sorted(set(datasets)):
        selected = np.asarray([value == dataset for value in datasets], dtype=bool)
        metrics = binary_metrics(labels[selected], predictions[selected])
        by_dataset[dataset] = {
            **metrics,
            "calibration": calibration_metrics(
                labels[selected], probabilities[selected]
            ),
        }
    macro_fields = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "mcc",
    )
    macro = {
        field: float(
            np.mean([float(metrics[field]) for metrics in by_dataset.values()])
        )
        for field in macro_fields
    }
    return {
        "threshold": threshold,
        "micro": {
            **binary_metrics(labels, predictions),
            "calibration": calibration_metrics(labels, probabilities),
        },
        "macro": macro,
        "worst_source": {
            field: min(float(metrics[field]) for metrics in by_dataset.values())
            for field in ("recall", "specificity", "balanced_accuracy", "mcc")
        },
        "by_dataset": by_dataset,
    }


def _benchmark(
    artifact_dir: Path,
    samples: Sequence[Sample],
    *,
    device: str,
    batch_size: int,
    input_mode: str = EXECUTION_CONTEXT,
) -> dict[str, object]:
    import torch

    load_times: list[float] = []
    model = tokenizer = None
    for _ in range(3):
        started = time.perf_counter_ns()
        model, tokenizer = _load_model(artifact_dir, device)
        load_times.append((time.perf_counter_ns() - started) / 1e6)
        del model
        if device == "mps":
            torch.mps.empty_cache()
    model, tokenizer = _load_model(artifact_dir, device)
    sample_count = min(128, len(samples))
    indices = np.linspace(0, len(samples) - 1, sample_count, dtype=int)
    selected = [samples[index] for index in indices]

    def predict_one(sample: Sample) -> None:
        packed = pack_sample(tokenizer, sample, input_mode=input_mode)
        inputs = {
            name: torch.tensor([values], dtype=torch.long, device=device)
            for name, values in packed.items()
        }
        with torch.inference_mode():
            model(**inputs)

    for sample in selected[:8]:
        predict_one(sample)
    latencies: list[float] = []
    with PeakRSS() as single_memory:
        for sample in selected:
            started = time.perf_counter_ns()
            predict_one(sample)
            latencies.append((time.perf_counter_ns() - started) / 1e6)

    encoded = _PackedDataset(
        _encode_samples(tokenizer, samples, input_mode=input_mode),
        [sample.strict_label for sample in samples],
    )
    batch_times: list[float] = []
    with PeakRSS() as batch_memory:
        for _ in range(3):
            started = time.perf_counter()
            _probability_logits(
                model,
                encoded,
                device=device,
                batch_size=batch_size,
            )
            batch_times.append(time.perf_counter() - started)
    median_batch = statistics.median(batch_times)
    mps_allocated = (
        float(torch.mps.current_allocated_memory() / (1024**2))
        if device == "mps"
        else None
    )
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
        "mps_current_allocated_mib": mps_allocated,
    }


def run_standalone_encoder_experiment(
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
        f"Standalone encoder protocol: train={len(train)} "
        f"validation={len(validation)} models={','.join(model_keys)}"
    )

    # Train and freeze every candidate before reading any evaluation labels.
    trained: dict[str, dict[str, object]] = {}
    for key in model_keys:
        spec = ENCODER_SPECS[key]
        print(f"Training standalone encoder {key} ({spec.model_id})")
        artifact_dir = artifacts_root / "models" / "standalone_encoders" / key
        metadata = _train_one(
            spec,
            train,
            validation,
            artifact_dir,
            config=config,
            requested_device=requested_device,
            resume=resume,
            input_mode=EXECUTION_CONTEXT,
        )
        device = str(metadata["device"])
        model, tokenizer = _load_model(artifact_dir, device)
        validation_dataset = _PackedDataset(
            _encode_samples(tokenizer, validation, input_mode=EXECUTION_CONTEXT),
            [sample.strict_label for sample in validation],
        )
        validation_logits, _ = _probability_logits(
            model,
            validation_dataset,
            device=device,
            batch_size=config.batch_size * 2,
        )
        margins = _unsafe_margins(validation_logits)
        calibrator = SigmoidCalibrator(random_state=config.seed).fit(
            margins, [sample.strict_label for sample in validation]
        )
        validation_probabilities = calibrator.predict_proba(margins)
        threshold = select_high_recall_threshold(
            validation_probabilities,
            [sample.strict_label for sample in validation],
            target_recall=config.target_recall,
        )
        metadata["validation"] = {
            "calibration": calibration_metrics(
                [sample.strict_label for sample in validation],
                validation_probabilities,
            ),
            "default_0_5": binary_metrics(
                [sample.strict_label for sample in validation],
                (validation_probabilities >= 0.5).astype(np.int8),
            ),
            "high_recall": threshold,
            "calibrator_coefficient": float(calibrator.model.coef_[0, 0]),
            "calibrator_intercept": float(calibrator.model.intercept_[0]),
        }
        trained[key] = metadata
        del model
        if device == "mps":
            torch.mps.empty_cache()

    evaluation = [
        sample for samples in evaluation_by_source.values() for sample in samples
    ]
    evaluation_labels = np.asarray(
        [sample.strict_label for sample in evaluation], dtype=np.int8
    )
    evaluation_sources = [sample.source for sample in evaluation]
    for key in model_keys:
        metadata = trained[key]
        artifact_dir = artifacts_root / "models" / "standalone_encoders" / key
        device = str(metadata["device"])
        model, tokenizer = _load_model(artifact_dir, device)
        validation_dataset = _PackedDataset(
            _encode_samples(tokenizer, validation, input_mode=EXECUTION_CONTEXT),
            [sample.strict_label for sample in validation],
        )
        validation_logits, _ = _probability_logits(
            model,
            validation_dataset,
            device=device,
            batch_size=config.batch_size * 2,
        )
        calibrator = SigmoidCalibrator(random_state=config.seed).fit(
            _unsafe_margins(validation_logits),
            [sample.strict_label for sample in validation],
        )
        evaluation_dataset = _PackedDataset(
            _encode_samples(tokenizer, evaluation, input_mode=EXECUTION_CONTEXT),
            evaluation_labels,
        )
        evaluation_logits, _ = _probability_logits(
            model,
            evaluation_dataset,
            device=device,
            batch_size=config.batch_size * 2,
        )
        evaluation_probabilities = calibrator.predict_proba(
            _unsafe_margins(evaluation_logits)
        )
        high_recall_threshold = float(
            metadata["validation"]["high_recall"]["threshold"]  # type: ignore[index]
        )
        metadata["evaluation"] = {
            "default_0_5": _quality_summary(
                evaluation_labels,
                evaluation_probabilities,
                evaluation_sources,
                0.5,
            ),
            "high_recall": _quality_summary(
                evaluation_labels,
                evaluation_probabilities,
                evaluation_sources,
                high_recall_threshold,
            ),
        }
        metadata["systems"] = _benchmark(
            artifact_dir,
            evaluation,
            device=device,
            batch_size=config.batch_size * 2,
        )
        print(
            f"  {key} standalone high-recall: "
            f"R={metadata['evaluation']['high_recall']['micro']['recall']:.4f} "  # type: ignore[index]
            f"Spec={metadata['evaluation']['high_recall']['micro']['specificity']:.4f} "  # type: ignore[index]
            f"MCC={metadata['evaluation']['high_recall']['micro']['mcc']:.4f}"  # type: ignore[index]
        )
        del model
        if device == "mps":
            torch.mps.empty_cache()

    validation_ranking = sorted(
        model_keys,
        key=lambda key: (
            float(
                trained[key]["validation"]["high_recall"]["validation_metrics"]["mcc"]  # type: ignore[index]
            ),
            float(
                trained[key]["validation"]["high_recall"]["validation_metrics"][  # type: ignore[index]
                    "balanced_accuracy"
                ]
            ),
            float(
                trained[key]["validation"]["high_recall"]["validation_metrics"][  # type: ignore[index]
                    "specificity"
                ]
            ),
            -float(trained[key]["systems"]["artifact_size_mib"]),  # type: ignore[index]
            -float(trained[key]["systems"]["latency_ms_p95"]),  # type: ignore[index]
        ),
        reverse=True,
    )
    protocol_path = Path(__file__).parents[2] / "docs" / "STANDALONE_LOCAL_GUARD_PROTOCOL.md"
    result = {
        "schema_version": 1,
        "protocol": {
            "version": "1.0",
            "path": "docs/STANDALONE_LOCAL_GUARD_PROTOCOL.md",
            "sha256": _sha256(protocol_path),
            "evaluation_metrics_computed_after_all_candidates_trained": True,
        },
        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "evaluation": len(evaluation),
        },
        "input": {
            "fields": list(FIELD_MARKERS),
            "thought_included": False,
            "max_length": MAX_LENGTH,
            "field_budgets": FIELD_BUDGETS,
        },
        "validation_selected_model": validation_ranking[0],
        "validation_ranking": validation_ranking,
        "models": trained,
        "host": {
            **host_metadata(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "requested_device": requested_device,
        },
        "environment": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }
    results_root.mkdir(parents=True, exist_ok=True)
    output_path = results_root / "standalone_encoders.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return result
