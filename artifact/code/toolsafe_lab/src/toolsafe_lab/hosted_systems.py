from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from toolsafe_lab.data import load_eval
from toolsafe_lab.hosted_cascade import _local_routes
from toolsafe_lab.llm_api import (
    MODEL_SPECS,
    _read_jsonl,
    select_samples,
    summarize_model,
)
from toolsafe_lab.llm_prompt import prompt_spec


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize_direct_latency(
    *,
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    model_ids: Sequence[str],
    prompt_version: str,
    run_label: str = "latency-c1-n256",
    limit: int = 256,
    seed: int = 260110156,
) -> dict[str, object]:
    evaluation = load_eval(data_root)
    all_samples = [
        sample for source_samples in evaluation.values() for sample in source_samples
    ]
    samples = select_samples(all_samples, limit=limit, seed=seed)
    prompt = prompt_spec(prompt_version)
    routes = _local_routes(
        data_root=data_root,
        results_root=results_root,
        artifacts_root=artifacts_root,
        samples=all_samples,
        include_semantic=False,
    )["hybrid_tfidf_linearsvc"]

    local_rows = []
    with (results_root / "systems_performance.csv").open(encoding="utf-8") as handle:
        import csv

        local_rows = list(csv.DictReader(handle))
    hybrid_system = next(
        row
        for row in local_rows
        if row["model"] == "hybrid_tfidf_linearsvc"
        and row["source"] == "local_measurement"
    )
    local_mean_ms = float(hybrid_system["latency_ms_mean"])

    summaries: list[dict[str, object]] = []
    for model_id in model_ids:
        spec = MODEL_SPECS[model_id]
        record_path = (
            artifacts_root
            / "api_runs"
            / "eval"
            / prompt_version
            / run_label
            / f"{model_id}.jsonl"
        )
        if not record_path.exists():
            print(f"{spec.name}: no direct latency records yet")
            continue
        records = _read_jsonl(record_path)
        summary = summarize_model(
            spec=spec,
            samples=samples,
            records=records,
            prompt=prompt,
            run_label=run_label,
        )
        latest = {
            str(record["sample_id"]): record
            for record in records
            if record.get("sample_id") in {sample.sample_id for sample in samples}
        }
        paired = [
            (sample, latest[sample.sample_id])
            for sample in samples
            if sample.sample_id in latest
            and latest[sample.sample_id].get("status") == "ok"
            and latest[sample.sample_id].get("latency_ms") is not None
        ]
        composed_latencies = [
            local_mean_ms
            + (
                float(record["latency_ms"])
                if routes[sample.sample_id] == "defer"
                else 0.0
            )
            for sample, record in paired
        ]
        paired_ids = {sample.sample_id for sample, _ in paired}
        paired_cascade_cost = sum(
            float(record.get("estimated_cost_usd", 0.0))
            for sample_id, record in latest.items()
            if sample_id in paired_ids and routes[sample_id] == "defer"
        )
        monitor_cost = sum(
            float(record.get("estimated_cost_usd", 0.0))
            for record in latest.values()
        )
        cascade_cost = sum(
            float(record.get("estimated_cost_usd", 0.0))
            for sample_id, record in latest.items()
            if routes[sample_id] == "defer"
        )
        cascade_system = {
            "timing_semantics": "offline_composition_of_measured_components",
            "paired_successful_samples": len(paired),
            "local_latency_ms_fixed_at_measured_mean": local_mean_ms,
            "monitor_call_reduction": (
                sum(routes[sample.sample_id] == "allow" for sample in samples)
                / len(samples)
            ),
            "paired_monitor_call_reduction": (
                sum(routes[sample.sample_id] == "allow" for sample, _ in paired)
                / len(paired)
                if paired
                else 0.0
            ),
            "latency_ms_mean": statistics.mean(composed_latencies),
            "latency_ms_p50": statistics.median(composed_latencies),
            "latency_ms_p95": _percentile(composed_latencies, 95),
            "latency_ms_p99": _percentile(composed_latencies, 99),
            "observed_standard_cost_usd": cascade_cost,
            "cost_per_1k_successful_composed_decisions_usd": (
                paired_cascade_cost * 1000 / len(paired) if paired else None
            ),
            "observed_cost_reduction": (
                1.0 - cascade_cost / monitor_cost if monitor_cost else 0.0
            ),
        }
        summary["hybrid_cascade_simulation"] = cascade_system
        summaries.append(summary)
        system = summary["system_row"]
        print(
            f"{spec.name}: n={system.get('single_n', 0)} "
            f"p50={float(system.get('latency_ms_p50', 0)):.1f} ms; "
            f"cascade mean={cascade_system['latency_ms_mean']:.1f} ms"
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "eval",
        "selection": {"selected": len(samples), "available": len(all_samples), "seed": seed},
        "prompt_version": prompt_version,
        "run_label": run_label,
        "summaries": summaries,
    }
    path = results_root / "llm_eval" / "direct_latency_n256.json"
    _write_json(path, payload)
    print(f"Wrote {path}")
    return payload
