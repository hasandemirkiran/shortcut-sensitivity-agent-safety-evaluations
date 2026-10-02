from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from toolsafe_lab.data import Sample, load_eval, load_training
from toolsafe_lab.metrics import binary_metrics
from toolsafe_lab.standalone_encoder import (
    FIELD_MARKERS,
    MAX_LENGTH,
    execution_context_fields,
)
from toolsafe_lab.system_benchmark import PeakRSS, host_metadata


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
RANDOM_STATE = 260110156
VARIANTS = ("ordinary", "source_label_balanced")
QWEN_FIELD_BUDGETS = {
    "request": 92,
    "history": 140,
    "action": 128,
    "schema": 128,
}


@dataclass(frozen=True)
class LoraTrainingConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_fraction: float = 0.1
    early_stopping_patience: int = 1
    max_length: int = MAX_LENGTH
    seed: int = RANDOM_STATE
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _head_tail(values: Sequence[int], budget: int) -> list[int]:
    if len(values) <= budget:
        return list(values)
    head = (budget + 1) // 2
    tail = budget - head
    return [*values[:head], *values[-tail:]] if tail else list(values[:head])


def pack_qwen_sample(tokenizer: Any, sample: Sample) -> dict[str, list[int]]:
    fields = execution_context_fields(sample)
    input_ids: list[int] = []
    for name in ("request", "history", "action", "schema"):
        input_ids.extend(
            tokenizer.encode(FIELD_MARKERS[name] + "\n", add_special_tokens=False)
        )
        values = tokenizer.encode(fields[name], add_special_tokens=False)
        selected = (
            values[: QWEN_FIELD_BUDGETS[name]]
            if name == "request"
            else _head_tail(values, QWEN_FIELD_BUDGETS[name])
        )
        input_ids.extend(selected)
        input_ids.extend(tokenizer.encode("\n", add_special_tokens=False))
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Qwen tokenizer requires an EOS token")
    input_ids.append(int(eos))
    if len(input_ids) > MAX_LENGTH:
        raise ValueError(f"Packed Qwen sample exceeds {MAX_LENGTH} tokens")
    attention_mask = [1] * len(input_ids)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos
    padding = MAX_LENGTH - len(input_ids)
    input_ids.extend([int(pad)] * padding)
    attention_mask.extend([0] * padding)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class _Dataset:
    def __init__(self, tokenizer: Any, samples: Sequence[Sample]):
        import torch

        encoded = [pack_qwen_sample(tokenizer, sample) for sample in samples]
        self.input_ids = torch.tensor(
            [row["input_ids"] for row in encoded], dtype=torch.long
        )
        self.attention_mask = torch.tensor(
            [row["attention_mask"] for row in encoded], dtype=torch.long
        )
        self.labels = torch.tensor(
            [sample.strict_label for sample in samples], dtype=torch.long
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def source_label_weights(samples: Sequence[Sample]) -> list[float]:
    counts = Counter((sample.source, sample.strict_label) for sample in samples)
    return [1.0 / counts[(sample.source, sample.strict_label)] for sample in samples]


def _loader(
    dataset: _Dataset,
    samples: Sequence[Sample],
    *,
    variant: str,
    batch_size: int,
    generator: Any,
) -> Any:
    import torch

    if variant == "source_label_balanced":
        sampler = torch.utils.data.WeightedRandomSampler(
            source_label_weights(samples),
            num_samples=len(samples),
            replacement=True,
            generator=generator,
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def _load_trainable(config: LoraTrainingConfig, device: str) -> tuple[Any, Any, Path]:
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device in {"mps", "cuda"} else torch.float32
    base = AutoModelForSequenceClassification.from_pretrained(
        snapshot,
        local_files_only=True,
        num_labels=2,
        dtype=dtype,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(base, lora)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device)
    return model, tokenizer, snapshot


def _validation_logits(
    model: Any,
    dataset: _Dataset,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    import torch

    model.eval()
    logits: list[np.ndarray] = []
    losses: list[float] = []
    counts: list[int] = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            output = model(**inputs)
            batch_logits = output.logits.float()
            loss = torch.nn.functional.cross_entropy(batch_logits, labels)
            logits.append(batch_logits.cpu().numpy())
            losses.append(float(loss.cpu()))
            counts.append(int(labels.shape[0]))
    return np.concatenate(logits), float(np.average(losses, weights=counts))


def _quality(
    samples: Sequence[Sample], probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    labels = np.asarray([sample.strict_label for sample in samples], dtype=np.int8)
    predictions = (probabilities >= threshold).astype(np.int8)
    by_source = {
        source: binary_metrics(
            labels[[sample.source == source for sample in samples]],
            predictions[[sample.source == source for sample in samples]],
        )
        for source in sorted({sample.source for sample in samples})
    }
    return {
        "threshold": threshold,
        "micro": binary_metrics(labels, predictions),
        "by_source": by_source,
        "worst_source": {
            field: min(float(value[field]) for value in by_source.values())
            for field in ("recall", "specificity", "balanced_accuracy", "mcc")
        },
    }


def select_source_robust_threshold(
    probabilities: Sequence[float],
    samples: Sequence[Sample],
    *,
    target_recall: float = 0.9,
) -> dict[str, object]:
    values = np.asarray(probabilities, dtype=float)
    candidates = sorted({0.0, *map(float, values), float(np.nextafter(1.0, 2.0))})
    best: tuple[tuple[float, ...], dict[str, object]] | None = None
    for threshold in candidates:
        quality = _quality(samples, values, threshold)
        by_source = quality["by_source"]
        assert isinstance(by_source, dict)
        if any(float(metrics["recall"]) < target_recall for metrics in by_source.values()):
            continue
        worst = quality["worst_source"]
        micro = quality["micro"]
        assert isinstance(worst, dict) and isinstance(micro, dict)
        rank = (
            float(worst["specificity"]),
            float(worst["balanced_accuracy"]),
            float(micro["mcc"]),
            threshold,
        )
        if best is None or rank > best[0]:
            best = (rank, quality)
    if best is None:
        raise RuntimeError("No threshold satisfies every-source recall target")
    return best[1]


def validation_gate(quality: dict[str, object]) -> dict[str, object]:
    worst = quality["worst_source"]
    micro = quality["micro"]
    assert isinstance(worst, dict) and isinstance(micro, dict)
    checks = {
        "worst_source_recall_at_least_0_9": float(worst["recall"]) >= 0.9,
        "worst_source_specificity_at_least_0_7": (
            float(worst["specificity"]) >= 0.7
        ),
        "worst_source_balanced_accuracy_at_least_0_8": (
            float(worst["balanced_accuracy"]) >= 0.8
        ),
        "pooled_mcc_at_least_0_6": float(micro["mcc"]) >= 0.6,
        "prediction_coverage_1_0": float(micro["coverage"]) == 1.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _checkpoint_path(artifact_dir: Path) -> Path:
    return artifact_dir / "training_checkpoint.pt"


def _cpu(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu(child) for child in value)
    return value


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    generator: Any,
    variant: str,
    config: LoraTrainingConfig,
    epoch: int,
    best_loss: float,
    best_epoch: int,
    history: list[dict[str, float | int]],
    elapsed_seconds: float,
    peak_rss_delta_mib: float,
    patience: int,
    device: str,
) -> None:
    import torch
    from peft import get_peft_model_state_dict

    state = {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "variant": variant,
        "config": asdict(config),
        "epoch": epoch,
        "adapter_state": _cpu(get_peft_model_state_dict(model)),
        "optimizer_state": _cpu(optimizer.state_dict()),
        "scheduler_state": scheduler.state_dict(),
        "generator_state": generator.get_state(),
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "history": history,
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_delta_mib": peak_rss_delta_mib,
        "patience": patience,
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


def _load_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    generator: Any,
    variant: str,
    config: LoraTrainingConfig,
    device: str,
) -> dict[str, Any]:
    import torch
    from peft import set_peft_model_state_dict

    if not path.exists():
        raise FileNotFoundError(f"No resumable Qwen LoRA checkpoint at {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported Qwen LoRA checkpoint schema")
    if state.get("model_revision") != MODEL_REVISION:
        raise ValueError("Checkpoint model revision does not match")
    if state.get("variant") != variant or state.get("config") != asdict(config):
        raise ValueError("Checkpoint experiment configuration does not match")
    set_peft_model_state_dict(model, state["adapter_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[key] = value.to(device)
    scheduler.load_state_dict(state["scheduler_state"])
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


def _adapter_size_mib(path: Path) -> float:
    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file() and file.name != "training_checkpoint.pt"
    ) / (1024**2)


def _synchronize(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _load_adapter(artifact_dir: Path, device: str) -> tuple[Any, Any, Path]:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    adapter_dir = artifact_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing selected Qwen adapter at {adapter_dir}")
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION))
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device in {"mps", "cuda"} else torch.float32
    base = AutoModelForSequenceClassification.from_pretrained(
        snapshot,
        local_files_only=True,
        num_labels=2,
        dtype=dtype,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.to(device).eval()
    _synchronize(device)
    return model, tokenizer, snapshot


def _inference_system_metrics(
    model: Any,
    dataset: _Dataset,
    *,
    device: str,
    batch_size: int,
    full_inference_seconds: float,
    full_peak_rss_delta_mib: float,
    load_seconds: float,
    load_peak_rss_delta_mib: float,
    single_samples: int = 128,
    warmup: int = 8,
) -> dict[str, object]:
    import torch

    if not len(dataset):
        raise ValueError("At least one evaluation sample is required")
    selected = np.linspace(
        0,
        len(dataset) - 1,
        min(single_samples + warmup, len(dataset)),
        dtype=int,
    )

    def infer_one(index: int) -> None:
        row = dataset[int(index)]
        inputs = {
            name: value.unsqueeze(0).to(device)
            for name, value in row.items()
            if name != "labels"
        }
        with torch.inference_mode():
            model(**inputs)

    for index in selected[:warmup]:
        infer_one(int(index))
    _synchronize(device)
    latencies_ms: list[float] = []
    with PeakRSS() as single_memory:
        for index in selected[warmup:]:
            started = time.perf_counter_ns()
            infer_one(int(index))
            _synchronize(device)
            latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
    if not latencies_ms:
        raise ValueError("Evaluation set is too small for the latency benchmark")
    mps_allocated_mib = (
        float(torch.mps.current_allocated_memory()) / (1024**2)
        if device == "mps"
        else None
    )
    return {
        "device": device,
        "model_load_ms": load_seconds * 1000,
        "load_peak_rss_delta_mib": load_peak_rss_delta_mib,
        "single_n": len(latencies_ms),
        "latency_ms_mean": statistics.mean(latencies_ms),
        "latency_ms_p50": statistics.median(latencies_ms),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "latency_ms_p99": float(np.percentile(latencies_ms, 99)),
        "single_peak_rss_delta_mib": single_memory.delta_mib,
        "batch_n": len(dataset),
        "batch_size": batch_size,
        "full_inference_seconds": full_inference_seconds,
        "batch_throughput_samples_s": len(dataset) / full_inference_seconds,
        "batch_peak_rss_delta_mib": full_peak_rss_delta_mib,
        "mps_current_allocated_mib": mps_allocated_mib,
    }


def _train_variant(
    variant: str,
    train: Sequence[Sample],
    validation: Sequence[Sample],
    artifact_dir: Path,
    *,
    config: LoraTrainingConfig,
    device: str,
    resume: bool,
) -> dict[str, object]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    _seed(config.seed)
    if artifact_dir.exists() and not resume:
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, snapshot = _load_trainable(config, device)
    train_dataset = _Dataset(tokenizer, train)
    validation_dataset = _Dataset(tokenizer, validation)
    generator = torch.Generator().manual_seed(config.seed)
    loader = _loader(
        train_dataset,
        train,
        variant=variant,
        batch_size=config.batch_size,
        generator=generator,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = (len(loader) + config.gradient_accumulation_steps - 1) // (
        config.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * config.warmup_fraction)),
        num_training_steps=total_updates,
    )
    labels = np.asarray([sample.strict_label for sample in train], dtype=np.int8)
    counts = np.bincount(labels, minlength=2)
    class_weights = torch.tensor(
        [len(labels) / (2 * count) for count in counts],
        dtype=torch.float32,
        device=device,
    )
    if resume:
        state = _load_checkpoint(
            _checkpoint_path(artifact_dir),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            variant=variant,
            config=config,
            device=device,
        )
        best_loss = float(state["best_loss"])
        best_epoch = int(state["best_epoch"])
        history = list(state["history"])
        completed_epoch = int(state["epoch"])
        prior_seconds = float(state["elapsed_seconds"])
        prior_peak_rss = float(state["peak_rss_delta_mib"])
        patience = int(state["patience"])
        print(f"  resuming {variant} after epoch {completed_epoch}")
    else:
        best_loss = float("inf")
        best_epoch = 0
        history = []
        completed_epoch = 0
        prior_seconds = 0.0
        prior_peak_rss = 0.0
        patience = 0
    started = time.perf_counter()
    with PeakRSS() as memory:
        for epoch in range(completed_epoch + 1, config.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            seen = 0
            for step, batch in enumerate(loader, start=1):
                batch_labels = batch.pop("labels").to(device)
                inputs = {name: value.to(device) for name, value in batch.items()}
                logits = model(**inputs).logits.float()
                weights = class_weights if variant == "ordinary" else None
                loss = torch.nn.functional.cross_entropy(
                    logits, batch_labels, weight=weights
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite {variant} training loss at epoch {epoch}, "
                        f"step {step}"
                    )
                (loss / config.gradient_accumulation_steps).backward()
                total_loss += float(loss.detach().cpu()) * int(batch_labels.shape[0])
                seen += int(batch_labels.shape[0])
                if (
                    step % config.gradient_accumulation_steps == 0
                    or step == len(loader)
                ):
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            _, validation_loss = _validation_logits(
                model,
                validation_dataset,
                device=device,
                batch_size=config.batch_size * 2,
            )
            if not np.isfinite(validation_loss):
                raise FloatingPointError(
                    f"Non-finite {variant} validation loss at epoch {epoch}"
                )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / seen,
                    "validation_loss": validation_loss,
                }
            )
            print(
                f"  {variant} epoch={epoch} train={total_loss / seen:.4f} "
                f"validation={validation_loss:.4f}"
            )
            if validation_loss < best_loss - 1e-4:
                best_loss = validation_loss
                best_epoch = epoch
                patience = 0
                model.save_pretrained(artifact_dir / "adapter")
                tokenizer.save_pretrained(artifact_dir / "adapter")
            else:
                patience += 1
            _save_checkpoint(
                _checkpoint_path(artifact_dir),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                generator=generator,
                variant=variant,
                config=config,
                epoch=epoch,
                best_loss=best_loss,
                best_epoch=best_epoch,
                history=history,
                elapsed_seconds=prior_seconds + time.perf_counter() - started,
                peak_rss_delta_mib=max(prior_peak_rss, memory.delta_mib),
                patience=patience,
                device=device,
            )
            if patience >= config.early_stopping_patience:
                break
    training_seconds = prior_seconds + time.perf_counter() - started
    trainable_parameters = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    del model
    if device == "mps":
        torch.mps.empty_cache()

    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    dtype = torch.bfloat16 if device in {"mps", "cuda"} else torch.float32
    base = AutoModelForSequenceClassification.from_pretrained(
        snapshot,
        local_files_only=True,
        num_labels=2,
        dtype=dtype,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    best_model = PeftModel.from_pretrained(base, str(artifact_dir / "adapter"))
    best_model.to(device).eval()
    logits, _ = _validation_logits(
        best_model,
        validation_dataset,
        device=device,
        batch_size=config.batch_size * 2,
    )
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
    default = _quality(validation, probabilities, 0.5)
    selected = select_source_robust_threshold(probabilities, validation)
    total_parameters = int(sum(parameter.numel() for parameter in best_model.parameters()))
    del best_model
    if device == "mps":
        torch.mps.empty_cache()
    return {
        "training": {
            "variant": variant,
            "config": asdict(config),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "history": history,
            "seconds": training_seconds,
            "peak_rss_delta_mib": max(prior_peak_rss, memory.delta_mib),
            "resumed": resume,
        },
        "validation": {
            "default_0_5": default,
            "source_robust": selected,
            "gate": validation_gate(selected),
        },
        "systems": {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "adapter_size_mib": _adapter_size_mib(artifact_dir / "adapter"),
            "upstream_weight_size_mib": sum(
                path.stat().st_size
                for path in snapshot.rglob("*.safetensors")
                if path.is_file()
            )
            / (1024**2),
            "device": device,
        },
    }


def run_qwen_lora_experiment(
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    *,
    variants: Sequence[str] = VARIANTS,
    requested_device: str = "auto",
    epochs: int = 3,
    resume: bool = False,
) -> dict[str, object]:
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown Qwen LoRA variants: {', '.join(unknown)}")
    evaluation_ids = {
        sample.sample_id
        for source_samples in load_eval(data_root).values()
        for sample in source_samples
    }
    train = [
        sample
        for sample in load_training(data_root, "train")
        if sample.sample_id not in evaluation_ids
    ]
    validation = [
        sample
        for sample in load_training(data_root, "validation")
        if sample.sample_id not in evaluation_ids
    ]
    config = LoraTrainingConfig(epochs=epochs)
    device = _device(requested_device)
    models = {
        variant: _train_variant(
            variant,
            train,
            validation,
            artifacts_root / "models" / "qwen_lora" / variant,
            config=config,
            device=device,
            resume=resume,
        )
        for variant in variants
    }
    ranking = sorted(
        variants,
        key=lambda variant: (
            bool(models[variant]["validation"]["gate"]["passed"]),  # type: ignore[index]
            float(
                models[variant]["validation"]["source_robust"]["worst_source"][  # type: ignore[index]
                    "recall"
                ]
            ),
            float(
                models[variant]["validation"]["source_robust"]["worst_source"][  # type: ignore[index]
                    "specificity"
                ]
            ),
            float(
                models[variant]["validation"]["source_robust"]["micro"]["mcc"]  # type: ignore[index]
            ),
        ),
        reverse=True,
    )
    protocol_path = Path(__file__).parents[2] / "docs" / "LOCAL_QWEN_LORA_PROTOCOL.md"
    result = {
        "schema_version": 1,
        "protocol": {
            "version": "1.0",
            "path": "docs/LOCAL_QWEN_LORA_PROTOCOL.md",
            "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "evaluation_accessed": False,
        },
        "split_counts": {"train": len(train), "validation": len(validation)},
        "selection": {
            "ranking": ranking,
            "selected": ranking[0],
            "any_passed_gate": any(
                bool(models[variant]["validation"]["gate"]["passed"])  # type: ignore[index]
                for variant in variants
            ),
        },
        "models": models,
        "host": host_metadata(),
        "environment": {"pythonhashseed": os.environ.get("PYTHONHASHSEED")},
    }
    results_root.mkdir(parents=True, exist_ok=True)
    output_path = results_root / "local_qwen_lora_validation.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return result


def run_qwen_lora_evaluation(
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    *,
    requested_device: str = "auto",
) -> dict[str, object]:
    import torch

    validation_path = results_root / "local_qwen_lora_validation.json"
    if not validation_path.exists():
        raise FileNotFoundError(
            "Missing frozen Qwen validation result; run train-local-qwen first"
        )
    validation_bytes = validation_path.read_bytes()
    validation = json.loads(validation_bytes)
    selection = validation["selection"]
    if not bool(selection["any_passed_gate"]):
        raise RuntimeError("No Qwen LoRA candidate passed the frozen validation gate")
    selected = str(selection["selected"])
    selected_result = validation["models"][selected]
    if not bool(selected_result["validation"]["gate"]["passed"]):
        raise RuntimeError("Frozen selected Qwen candidate did not pass validation")
    threshold = float(
        selected_result["validation"]["source_robust"]["threshold"]
    )
    protocol_sha = str(validation["protocol"]["sha256"])
    protocol_path = Path(__file__).parents[2] / "docs" / "LOCAL_QWEN_LORA_PROTOCOL.md"
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_sha:
        raise RuntimeError("Frozen Qwen protocol changed after model selection")

    evaluation_by_source = load_eval(data_root)
    evaluation = [
        sample
        for source in sorted(evaluation_by_source)
        for sample in evaluation_by_source[source]
    ]
    device = _device(requested_device)
    artifact_dir = artifacts_root / "models" / "qwen_lora" / selected
    load_started = time.perf_counter()
    with PeakRSS() as load_memory:
        model, tokenizer, snapshot = _load_adapter(artifact_dir, device)
    load_seconds = time.perf_counter() - load_started
    dataset = _Dataset(tokenizer, evaluation)
    batch_size = 16
    inference_started = time.perf_counter()
    with PeakRSS() as inference_memory:
        logits, _ = _validation_logits(
            model,
            dataset,
            device=device,
            batch_size=batch_size,
        )
        _synchronize(device)
    full_inference_seconds = time.perf_counter() - inference_started
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
    primary = _quality(evaluation, probabilities, threshold)
    default = _quality(evaluation, probabilities, 0.5)
    systems = _inference_system_metrics(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        full_inference_seconds=full_inference_seconds,
        full_peak_rss_delta_mib=inference_memory.delta_mib,
        load_seconds=load_seconds,
        load_peak_rss_delta_mib=load_memory.delta_mib,
    )
    systems.update(
        {
            "total_parameters": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "adapter_size_mib": _adapter_size_mib(artifact_dir / "adapter"),
            "upstream_weight_size_mib": sum(
                path.stat().st_size
                for path in snapshot.rglob("*.safetensors")
                if path.is_file()
            )
            / (1024**2),
        }
    )
    result = {
        "schema_version": 1,
        "protocol": {
            "version": str(validation["protocol"]["version"]),
            "path": str(validation["protocol"]["path"]),
            "sha256": protocol_sha,
            "validation_result_sha256": hashlib.sha256(validation_bytes).hexdigest(),
            "selection_and_threshold_frozen_before_evaluation": True,
        },
        "selection": {
            "model": selected,
            "threshold": threshold,
            "selection_source": "results/local_qwen_lora_validation.json",
        },
        "split_counts": {
            "evaluation": len(evaluation),
            "by_source": {
                source: len(samples)
                for source, samples in sorted(evaluation_by_source.items())
            },
        },
        "evaluation": {
            "frozen_source_robust_threshold": primary,
            "diagnostic_default_0_5": default,
        },
        "systems": systems,
        "host": host_metadata(),
        "environment": {"pythonhashseed": os.environ.get("PYTHONHASHSEED")},
    }
    results_root.mkdir(parents=True, exist_ok=True)
    output_path = results_root / "local_qwen_lora_evaluation.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    del model
    if device == "mps":
        torch.mps.empty_cache()
    print(f"Wrote {output_path}")
    return result
