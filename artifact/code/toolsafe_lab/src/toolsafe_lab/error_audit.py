from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from toolsafe_lab.data import load_eval
from toolsafe_lab.metrics import strict_value


HEADLINE_MODEL = "hybrid_tfidf_linearsvc"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _taxonomy(row: dict[str, object]) -> str:
    if row["category"] == "cascade_false_negative":
        return (
            "local_safe_boundary_leak"
            if row["route"] == "allow"
            else "downstream_monitor_miss"
        )
    if row["category"] == "cascade_false_positive":
        return "downstream_monitor_false_alarm"
    if row["category"] == "cascade_ts_guard_disagreement":
        return "local_allow_overrode_monitor"
    if row["category"] == "ts_guard_best_hosted_disagreement":
        return "monitor_policy_disagreement"
    return "agreement_control"


def _sample_order(category: str, sample_id: str) -> str:
    return hashlib.sha256(f"{category}:{sample_id}".encode()).hexdigest()


def generate_error_audit(
    *,
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    prompt_version: str,
    limit_per_category: int = 50,
) -> dict[str, object]:
    evaluation = load_eval(data_root)
    samples = [
        sample for source_samples in evaluation.values() for sample in source_samples
    ]

    headline: dict[str, dict[str, str]] = {}
    with gzip.open(
        results_root / "cascade" / "ts_guard_primary_predictions.csv.gz",
        "rt",
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            if row["model"] == HEADLINE_MODEL:
                headline[row["sample_id"]] = row

    hosted = json.loads(
        (results_root / "cascade" / "hosted_cascades.json").read_text(
            encoding="utf-8"
        )
    )
    best_id, best_monitor = max(
        hosted["monitors"].items(),
        key=lambda item: float(item[1]["micro"]["f1"]),
    )
    hosted_records = {
        str(record["sample_id"]): record
        for record in _read_jsonl(
            artifacts_root
            / "api_runs"
            / "eval"
            / prompt_version
            / f"{best_id}.jsonl"
        )
    }

    candidates: dict[str, list[dict[str, object]]] = {
        "cascade_false_negative": [],
        "cascade_false_positive": [],
        "cascade_ts_guard_disagreement": [],
        "ts_guard_best_hosted_disagreement": [],
        "agreement_control": [],
    }
    for sample in samples:
        local = headline[sample.sample_id]
        label = sample.strict_label
        ts_guard = int(local["monitor_prediction"])
        cascade = int(local["cascade_prediction"])
        hosted_record = hosted_records.get(sample.sample_id)
        hosted_prediction = (
            strict_value(hosted_record.get("prediction"))
            if hosted_record is not None and hosted_record.get("status") == "ok"
            else None
        )
        hosted_fail_closed = 1 if hosted_prediction is None else hosted_prediction
        base = {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "subset": sample.subset,
            "trajectory_id": sample.trajectory_id,
            "segment_id": sample.segment_id,
            "label_exact": sample.label,
            "label_strict": label,
            "route": local["route"],
            "local_probability": float(local["local_probability"]),
            "ts_guard_prediction": ts_guard,
            "cascade_prediction": cascade,
            "best_hosted_model": best_monitor["model"],
            "best_hosted_prediction": hosted_fail_closed,
            "best_hosted_valid": hosted_prediction is not None,
        }
        if label == 1 and cascade == 0:
            candidates["cascade_false_negative"].append(base)
        if label == 0 and cascade == 1:
            candidates["cascade_false_positive"].append(base)
        if cascade != ts_guard:
            candidates["cascade_ts_guard_disagreement"].append(base)
        if ts_guard != hosted_fail_closed:
            candidates["ts_guard_best_hosted_disagreement"].append(base)
        if cascade == ts_guard == hosted_fail_closed == label:
            candidates["agreement_control"].append(base)

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    requested = [
        "cascade_false_negative",
        "cascade_false_positive",
        "cascade_ts_guard_disagreement",
        "ts_guard_best_hosted_disagreement",
        "agreement_control",
    ]
    for category in requested:
        ordered = sorted(
            candidates[category],
            key=lambda row: _sample_order(category, str(row["sample_id"])),
        )
        for row in ordered:
            sample_id = str(row["sample_id"])
            if sample_id in selected_ids:
                continue
            annotated = {**row, "category": category}
            annotated["taxonomy"] = _taxonomy(annotated)
            selected.append(annotated)
            selected_ids.add(sample_id)
            if sum(item["category"] == category for item in selected) == limit_per_category:
                break

    audit_root = results_root / "error_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    manifest_path = audit_root / "sample_manifest.csv"
    fields = (
        "sample_id",
        "category",
        "taxonomy",
        "source",
        "subset",
        "trajectory_id",
        "segment_id",
        "label_exact",
        "label_strict",
        "route",
        "local_probability",
        "ts_guard_prediction",
        "cascade_prediction",
        "best_hosted_model",
        "best_hosted_prediction",
        "best_hosted_valid",
    )
    temporary = manifest_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(selected)
    temporary.replace(manifest_path)

    payload: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sampling": (
            "Deterministic SHA-256 order within prespecified categories; "
            "sample IDs deduplicated across categories in listed priority order."
        ),
        "limit_per_category": limit_per_category,
        "best_hosted_monitor": {
            "model_id": best_id,
            "model": best_monitor["model"],
            "selection_rule": "highest pooled fail-closed F1 among completed batches",
        },
        "candidate_counts": {
            category: len(rows) for category, rows in candidates.items()
        },
        "selected_counts": dict(Counter(row["category"] for row in selected)),
        "taxonomy_counts": dict(Counter(row["taxonomy"] for row in selected)),
        "source_counts": dict(Counter(row["source"] for row in selected)),
        "manifest": str(manifest_path.relative_to(results_root.parent)),
        "interpretation_guardrail": (
            "This is a model-generated routing/error taxonomy, not a judgment "
            "that benchmark labels are wrong. Label-error claims require human "
            "double annotation and inter-annotator agreement."
        ),
    }
    summary_path = audit_root / "summary.json"
    temporary_json = summary_path.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_json.replace(summary_path)
    print(
        f"Wrote {manifest_path} ({len(selected)} unique audit examples) "
        f"and {summary_path}"
    )
    return payload
