from __future__ import annotations

import platform
import statistics
import sys
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import psutil
import sklearn


class PeakRSS(AbstractContextManager["PeakRSS"]):
    def __init__(self, interval_seconds: float = 0.005) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "PeakRSS":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)

    @property
    def delta_mib(self) -> float:
        return max(0.0, (self.peak - self.baseline) / (1024**2))


def host_metadata() -> dict[str, object]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "Apple Silicon",
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_gib": round(psutil.virtual_memory().total / (1024**3), 2),
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
    }


def fit_with_metrics(model: Any, texts: Sequence[str], labels: Sequence[int]) -> dict[str, float]:
    started = time.perf_counter()
    with PeakRSS() as memory:
        model.fit(texts, labels)
    return {
        "train_seconds": time.perf_counter() - started,
        "train_peak_rss_delta_mib": memory.delta_mib,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values), quantile))


def benchmark_model(
    model: Any,
    model_path: Path,
    texts: Sequence[str],
    *,
    train_metrics: dict[str, float],
    single_samples: int = 256,
    warmup: int = 16,
    batch_repeats: int = 5,
) -> dict[str, object]:
    if not texts:
        raise ValueError("At least one benchmark sample is required")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    artifact_bytes = model_path.stat().st_size

    load_times: list[float] = []
    loaded = None
    for _ in range(5):
        started = time.perf_counter_ns()
        loaded = joblib.load(model_path)
        load_times.append((time.perf_counter_ns() - started) / 1e6)
    assert loaded is not None

    sample_count = min(single_samples, len(texts))
    selected = [texts[index] for index in np.linspace(0, len(texts) - 1, sample_count, dtype=int)]
    for text in selected[:warmup]:
        loaded.predict([text])

    latencies_ms: list[float] = []
    with PeakRSS() as single_memory:
        for text in selected:
            started = time.perf_counter_ns()
            loaded.predict([text])
            latencies_ms.append((time.perf_counter_ns() - started) / 1e6)

    batch_times: list[float] = []
    with PeakRSS() as batch_memory:
        for _ in range(batch_repeats):
            started = time.perf_counter()
            loaded.predict(texts)
            batch_times.append(time.perf_counter() - started)
    median_batch_seconds = statistics.median(batch_times)

    return {
        **train_metrics,
        "model_size_mib": artifact_bytes / (1024**2),
        "load_ms_p50": statistics.median(load_times),
        "single_n": len(latencies_ms),
        "latency_ms_mean": statistics.mean(latencies_ms),
        "latency_ms_p50": statistics.median(latencies_ms),
        "latency_ms_p95": _percentile(latencies_ms, 95),
        "latency_ms_p99": _percentile(latencies_ms, 99),
        "single_peak_rss_delta_mib": single_memory.delta_mib,
        "batch_n": len(texts),
        "batch_repeats": batch_repeats,
        "batch_throughput_samples_s": len(texts) / median_batch_seconds,
        "batch_peak_rss_delta_mib": batch_memory.delta_mib,
    }

