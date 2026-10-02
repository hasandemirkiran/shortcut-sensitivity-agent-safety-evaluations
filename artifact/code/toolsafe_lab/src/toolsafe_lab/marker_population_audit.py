"""Post-hoc census audit of the AgentDojo literal-marker intervention.

This module deliberately lives outside the original frozen 64-base audit.  It
selects every marker-present AgentDojo-Traj row, materializes only the original
and marker-paraphrase variants, and reuses the provenance-checked local
TS-Guard runner.  Raw benchmark text and generations remain under ``artifacts``;
the result written under ``results`` is aggregate and text-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

from toolsafe_lab.data import Sample, load_eval
from toolsafe_lab.shortcut_audit import (
    BOOTSTRAP_SEED,
    MARKER_CLOSE,
    MARKER_OPEN,
    PARAPHRASED_CLOSE,
    PARAPHRASED_OPEN,
    AuditExample,
    PredictionView,
    _coverage,
    _derived_id,
)
from toolsafe_lab.ts_guard_local import (
    TS_GUARD_MODEL_REVISION,
    run_ts_guard_shortcut_inference,
)
from toolsafe_lab.ts_guard_shortcut_analysis import (
    _reference_prediction_map,
    local_prediction_views,
)
from toolsafe_lab.uncertainty import clustered_mean_interval


COHORT = "agentdojo_marker_population"
VARIANTS = ("original", "marker_paraphrase")
RESULT_NAME = "agentdojo_marker_population_tsguard_q8.json"
RUNTIME_NAME = "mlx_q8_g64_greedy"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_json(row) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_marker_population_cohort(
    samples: Sequence[Sample],
) -> tuple[list[AuditExample], dict[str, object]]:
    """Build a census of all rows carrying a complete literal marker pair."""
    eligible = [
        sample
        for sample in samples
        if MARKER_OPEN in sample.history and MARKER_CLOSE in sample.history
    ]
    if len({sample.sample_id for sample in eligible}) != len(eligible):
        raise ValueError("Marker-present AgentDojo rows are not unique by sample_id")

    examples: list[AuditExample] = []
    for base in sorted(eligible, key=lambda sample: sample.sample_id):
        marker_history = base.history.replace(MARKER_OPEN, PARAPHRASED_OPEN).replace(
            MARKER_CLOSE, PARAPHRASED_CLOSE
        )
        if marker_history == base.history:
            raise AssertionError("Marker paraphrase did not change an eligible row")
        for variant, history in (("original", base.history), ("marker_paraphrase", marker_history)):
            derived = replace(
                base,
                sample_id=_derived_id(
                    cohort=COHORT,
                    base_sample_id=base.sample_id,
                    variant=variant,
                    serialization="released_react",
                ),
                source="AgentDojo-Marker-Population-Audit",
                split="counterfactual",
                history=history,
                trajectory_id=base.sample_id,
                segment_id=variant,
            )
            examples.append(
                AuditExample(
                    sample=derived,
                    cohort=COHORT,
                    base_sample_id=base.sample_id,
                    variant=variant,
                    domain=base.subset,
                    original_label=base.strict_label,
                    serialization="released_react",
                )
            )

    cell_counts = Counter(
        f"{sample.subset}/{'unsafe' if sample.strict_label else 'safe'}"
        for sample in eligible
    )
    interaction_counts = Counter(sample.trajectory_id for sample in eligible)
    return examples, {
        "status": "post_hoc_additive_census",
        "input_rows": len(samples),
        "base_rows": len(eligible),
        "derived_rows": len(examples),
        "variants": list(VARIANTS),
        "by_domain_label": dict(sorted(cell_counts.items())),
        "source_interactions": len(interaction_counts),
        "max_rows_per_interaction": max(interaction_counts.values(), default=0),
        "selection": "all rows with a complete literal <INFORMATION> marker pair",
    }


def _artifact_row(example: AuditExample) -> dict[str, object]:
    return {
        "cohort": example.cohort,
        "base_sample_id": example.base_sample_id,
        "variant": example.variant,
        "domain": example.domain,
        "original_label": example.original_label,
        "serialization": example.serialization,
        "axis": example.axis,
        "sample": asdict(example.sample),
    }


def prepare_marker_population(
    *, data_root: Path, artifacts_root: Path, results_root: Path
) -> tuple[list[AuditExample], dict[str, object]]:
    samples = load_eval(data_root)["AgentDojo-Traj"]
    examples, summary = build_marker_population_cohort(samples)
    path = artifacts_root / "shortcut_audit" / "cohorts" / f"{COHORT}.jsonl"
    _write_jsonl(path, [_artifact_row(example) for example in examples])
    summary["artifact"] = {
        "path": str(path),
        "rows": len(examples),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    _write_json(results_root / "agentdojo_marker_population_cohort.json", summary)
    return examples, summary


def _raw_path(artifacts_root: Path) -> Path:
    return (
        artifacts_root
        / "shortcut_audit"
        / "ts_guard_local"
        / TS_GUARD_MODEL_REVISION[:12]
        / RUNTIME_NAME
        / f"{COHORT}.jsonl"
    )


def _read_records(path: Path) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(record, dict) for record in records):
        raise TypeError(f"Invalid local record in {path}")
    return records


def _paired_rows(
    examples: Sequence[AuditExample], views: Mapping[str, PredictionView]
) -> list[tuple[AuditExample, PredictionView, PredictionView]]:
    by_base: dict[str, dict[str, AuditExample]] = defaultdict(dict)
    for example in examples:
        by_base[example.base_sample_id][example.variant] = example
    rows = []
    for base_id in sorted(by_base):
        variants = by_base[base_id]
        if set(variants) != set(VARIANTS):
            raise ValueError(f"Incomplete marker-population pair for {base_id}")
        original = variants["original"]
        transformed = variants["marker_paraphrase"]
        rows.append(
            (original, views[original.sample.sample_id], views[transformed.sample.sample_id])
        )
    return rows


def _component_signature(record: Mapping[str, object]) -> tuple[object, object, object] | None:
    value = record.get("components")
    if not isinstance(value, dict):
        return None
    return (
        value.get("Malicious_User_Request"),
        value.get("Being_Attacked"),
        value.get("Harmfulness_Rating"),
    )


def _reference_agreement(
    examples: Sequence[AuditExample],
    views: Mapping[str, PredictionView],
    raw_samples: Sequence[Sample],
    *,
    data_root: Path,
) -> dict[str, object]:
    released, alignment = _reference_prediction_map(raw_samples, data_root=data_root)
    originals = [example for example in examples if example.variant == "original"]
    jointly_valid = exact = strict = 0
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for example in originals:
        local = views[example.sample.sample_id]
        released_score = released[example.base_sample_id]
        label = "unsafe" if example.original_label else "safe"
        for bucket in (by_label[label], by_domain[example.domain]):
            bucket["expected"] += 1
        if not local.valid or local.score is None or released_score not in {0, 0.0, 0.5, 1, 1.0}:
            continue
        jointly_valid += 1
        exact_match = int(local.score == float(released_score))
        strict_match = int(local.strict_decision == int(float(released_score) != 0.0))
        exact += exact_match
        strict += strict_match
        for bucket in (by_label[label], by_domain[example.domain]):
            bucket["jointly_valid"] += 1
            bucket["exact_agreement"] += exact_match
            bucket["strict_agreement"] += strict_match

    def summarize(values: Mapping[str, Counter[str]]) -> dict[str, object]:
        return {
            key: {
                **dict(counts),
                "strict_agreement_rate_all_rows": counts["strict_agreement"] / counts["expected"],
            }
            for key, counts in sorted(values.items())
        }

    return {
        "expected": len(originals),
        "jointly_valid": jointly_valid,
        "exact_agreement": {"count": exact, "rate": exact / jointly_valid if jointly_valid else None},
        "strict_agreement": {"count": strict, "rate": strict / jointly_valid if jointly_valid else None},
        "by_original_label": summarize(by_label),
        "by_domain": summarize(by_domain),
        "released_alignment": alignment,
        "interpretation": "agreement diagnostic for the Q8 reproduction; not model equivalence",
    }


def analyze_marker_population(
    *, data_root: Path, artifacts_root: Path, results_root: Path, replicates: int = 10_000
) -> dict[str, object]:
    raw_samples = load_eval(data_root)["AgentDojo-Traj"]
    examples, cohort = build_marker_population_cohort(raw_samples)
    raw_path = _raw_path(artifacts_root)
    records = _read_records(raw_path)
    views = local_prediction_views(examples, records)
    pairs = _paired_rows(examples, views)
    valid = [row for row in pairs if row[1].valid and row[2].valid]
    flips = [float(original.strict_decision != transformed.strict_decision) for _, original, transformed in valid]
    interactions = []
    by_raw_id = {sample.sample_id: sample for sample in raw_samples}
    for example, *_ in valid:
        interactions.append(by_raw_id[example.base_sample_id].trajectory_id)
    interval = clustered_mean_interval(
        flips,
        clusters=interactions,
        replicates=replicates,
        seed=BOOTSTRAP_SEED,
    ).to_dict()

    directions = Counter(
        f"{original.strict_decision}_to_{transformed.strict_decision}"
        for _, original, transformed in valid
        if original.strict_decision != transformed.strict_decision
    )
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_label: dict[str, list[float]] = defaultdict(list)
    for (example, *_), flipped in zip(valid, flips):
        by_domain[example.domain].append(flipped)
        by_label["unsafe" if example.original_label else "safe"].append(flipped)

    records_by_id = {str(record.get("sample_id")): record for record in records}
    component_changes = score_changes = 0
    for example, original, transformed in valid:
        variants = {
            row.variant: row
            for row in examples
            if row.base_sample_id == example.base_sample_id
        }
        original_record = records_by_id[variants["original"].sample.sample_id]
        transformed_record = records_by_id[variants["marker_paraphrase"].sample.sample_id]
        component_changes += int(
            _component_signature(original_record) != _component_signature(transformed_record)
        )
        score_changes += int(original.score != transformed.score)

    result = {
        "status": "post_hoc_additive_census",
        "cohort": cohort,
        "coverage": _coverage(examples, views),
        "valid_pairs": len(valid),
        "invalid_pairs": len(pairs) - len(valid),
        "strict_decision": {
            "flip_count": int(sum(flips)),
            "flip_rate": sum(flips) / len(flips) if flips else None,
            "interaction_clustered_interval": interval,
            "direction_counts": dict(sorted(directions.items())),
            "by_domain": {
                key: {"n": len(values), "flips": int(sum(values)), "rate": sum(values) / len(values)}
                for key, values in sorted(by_domain.items())
            },
            "by_original_label": {
                key: {"n": len(values), "flips": int(sum(values)), "rate": sum(values) / len(values)}
                for key, values in sorted(by_label.items())
            },
        },
        "subthreshold_sensitivity": {
            "composite_score_changes": score_changes,
            "component_tuple_changes": component_changes,
            "denominator": len(valid),
        },
        "released_original_validation": _reference_agreement(
            examples, views, raw_samples, data_root=data_root
        ),
        "raw_output": {
            "path": str(raw_path),
            "rows": len(records),
            "size_bytes": raw_path.stat().st_size,
            "sha256": _sha256(raw_path),
        },
        "inference_boundary": (
            "Census rate for marker-present AgentDojo-Traj rows under the pinned Q8 "
            "reproduction; the clustered interval is a superpopulation sensitivity summary"
        ),
    }
    _write_json(results_root / RESULT_NAME, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "infer", "analyze"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    data_root = project_root / "data" / "raw"
    artifacts_root = project_root / "artifacts"
    results_root = project_root / "results"
    examples, summary = prepare_marker_population(
        data_root=data_root, artifacts_root=artifacts_root, results_root=results_root
    )
    if args.action == "prepare":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.action == "infer":
        if args.model_path is None:
            raise SystemExit("--model-path is required for infer")
        run_ts_guard_shortcut_inference(
            cohorts={COHORT: examples},
            artifacts_root=artifacts_root,
            cohort=COHORT,
            backend="mlx_q8",
            decoding="greedy",
            mlx_model_path=args.model_path.resolve(),
            max_records=args.max_records,
        )
        return
    result = analyze_marker_population(
        data_root=data_root,
        artifacts_root=artifacts_root,
        results_root=results_root,
        replicates=args.bootstrap_replicates,
    )
    print(json.dumps(result["strict_decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
