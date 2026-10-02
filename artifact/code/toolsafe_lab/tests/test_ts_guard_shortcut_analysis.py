from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from toolsafe_lab.data import Sample
from toolsafe_lab.manifest import EVAL_COMMIT
from toolsafe_lab.shortcut_audit import AGENTDOJO_VARIANTS, AuditExample
from toolsafe_lab.ts_guard_local import (
    TS_GUARD_MODEL_ID,
    TS_GUARD_MODEL_REVISION,
    TS_GUARD_PARSER_VERSION,
    TS_GUARD_PROMPT_SHA256,
    TS_GUARD_PROMPT_VERSION,
    TS_GUARD_RECORD_SCHEMA_VERSION,
    backend_config_sha256,
    ts_guard_input_sha256,
)
from toolsafe_lab.ts_guard_shortcut_analysis import (
    AGENTDOJO_REFERENCE_SUBSET_ORDER,
    OUTPUT_NAME,
    TS_GUARD_Q8_ARTIFACT,
    _assert_complete_local_records,
    aggregate_output_name,
    analyze_ts_guard_agentdojo_predictions,
    analyze_ts_guard_authorization_predictions,
    analyze_ts_guard_shortcut_audit,
    local_prediction_views,
    validate_agentdojo_original_reproduction,
)


def _example(
    *,
    cohort: str,
    base_id: str,
    variant: str,
    label: int,
    serialization: str,
    domain: str = "slack",
    axis: str = "",
) -> AuditExample:
    sample_id = f"derived-{cohort}-{serialization}-{base_id}-{variant}"
    sample = Sample(
        sample_id=sample_id,
        source=f"{cohort}-shortcut-audit",
        split="counterfactual",
        instruction="request text",
        history="history text",
        current_action="action text",
        env_info="tool text",
        label=float(label),
        trajectory_id=base_id,
        segment_id=variant,
        subset=domain,
    )
    return AuditExample(
        sample=sample,
        cohort=cohort,
        base_sample_id=base_id,
        variant=variant,
        domain=domain,
        original_label=label,
        serialization=serialization,
        axis=axis,
    )


def _record(
    example: AuditExample,
    prediction: float | None,
    *,
    status: str = "ok",
) -> dict[str, object]:
    backend = "transformers_greedy"
    backend_metadata = {
        "decoding": "greedy",
        "base_seed": None,
        "record_schema_version": TS_GUARD_RECORD_SCHEMA_VERSION,
        "parser_version": TS_GUARD_PARSER_VERSION,
        "max_parse_turns": 1,
        "seed_scope": "base_sample_id+serialization+attempt",
    }
    return {
        "schema_version": TS_GUARD_RECORD_SCHEMA_VERSION,
        "backend": backend,
        "backend_metadata": backend_metadata,
        "backend_config_sha256": backend_config_sha256(backend, backend_metadata),
        "upstream_eval_commit": EVAL_COMMIT,
        "model_id": TS_GUARD_MODEL_ID,
        "model_revision": TS_GUARD_MODEL_REVISION,
        "prompt_version": TS_GUARD_PROMPT_VERSION,
        "prompt_sha256": TS_GUARD_PROMPT_SHA256,
        "parser_version": TS_GUARD_PARSER_VERSION,
        "input_sha256": ts_guard_input_sha256(example),
        "sample_id": example.sample.sample_id,
        "cohort": example.cohort,
        "base_sample_id": example.base_sample_id,
        "variant": example.variant,
        "serialization": example.serialization,
        "original_label": example.original_label,
        "status": status,
        "prediction": prediction,
        "error_type": "missing_or_malformed_tags" if status != "ok" else None,
        "input_tokens": 1000,
        "output_tokens": 20,
        "cumulative_input_tokens": 1000,
        "cumulative_output_tokens": 20,
        "cumulative_generation_seconds": 5.0,
        "runtime_metrics": {
            "generation_seconds": 5.0,
            "process_rss_bytes": 16_000_000_000,
            "mps_driver_allocated_bytes": 17_000_000_000,
        },
        "attempts": [{"raw_output": "SENSITIVE_BENCHMARK_TEXT"}],
    }


def _agentdojo_examples(base_ids: tuple[str, ...] = ("safe", "unsafe")) -> list[AuditExample]:
    return [
        _example(
            cohort="agentdojo",
            base_id=base_id,
            variant=variant,
            label=int(base_id == "unsafe"),
            serialization="released_react",
        )
        for base_id in base_ids
        for variant in AGENTDOJO_VARIANTS
    ]


def test_agentdojo_local_analysis_reuses_fail_closed_paired_metrics() -> None:
    examples = _agentdojo_examples()
    predictions = {
        ("safe", "original"): 0.0,
        ("safe", "marker_paraphrase"): 1.0,
        ("safe", "tool_alias"): 0.0,
        ("unsafe", "original"): 1.0,
        ("unsafe", "marker_paraphrase"): 0.0,
        ("unsafe", "tool_alias"): 1.0,
        ("unsafe", "combined"): 1.0,
    }
    records = []
    for example in examples:
        key = (example.base_sample_id, example.variant)
        if key == ("safe", "combined"):
            records.append(_record(example, None, status="parse_error"))
        else:
            records.append(_record(example, predictions[key]))

    result, _ = analyze_ts_guard_agentdojo_predictions(
        examples,
        records,
        bootstrap_replicates=100,
    )

    paired = result["paired_transformations"]
    assert paired["marker_paraphrase"]["flip_rate"] == 1.0
    assert paired["tool_alias"]["flip_rate"] == 0.0
    assert paired["combined"]["flip_rate"] == 0.5
    assert paired["combined"]["flip_rate_interval"] is not None
    combined = result["by_variant"]["combined"]
    assert combined["coverage"]["valid"] == 1
    assert combined["coverage"]["failure_counts"] == {
        "missing_or_malformed_tags": 1
    }
    assert combined["strict_fail_closed_metrics"]["fp"] == 1
    hardened = paired["combined"]
    assert hardened["valid_both_only"]["valid_pairs"] == 1
    assert hardened["failure_affected_pairs"] == {
        "n": 1,
        "failure_location_counts": {"original_valid_transformed_invalid": 1},
        "fail_closed_strict_direction_counts": {"0_to_1": 1},
        "failure_reason_counts": {"transformed:missing_or_malformed_tags": 1},
        "excluded_from_valid_both_causal_flips": True,
    }
    deltas = hardened["paired_performance_deltas"]
    assert deltas["strict_fail_closed_all_pairs"]["delta_definition"] == (
        "transformed_minus_original"
    )
    assert deltas["valid_both_only"]["bootstrap"]["strata"] == (
        "domain_x_original_gold_label"
    )


def test_local_record_provenance_mismatch_is_rejected() -> None:
    example = _agentdojo_examples(("safe",))[0]
    record = _record(example, 0.0)
    record["prompt_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="prompt_sha256"):
        local_prediction_views([example], [record])


def test_local_record_input_hash_mismatch_is_rejected() -> None:
    example = _agentdojo_examples(("safe",))[0]
    record = _record(example, 0.0)
    record["input_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="input_sha256"):
        local_prediction_views([example], [record])


def test_authorization_local_analysis_keeps_pair_and_serialization_conclusions() -> None:
    examples = [
        _example(
            cohort="authorization",
            base_id=base_id,
            variant=variant,
            label=int(variant == "unsafe"),
            serialization=serialization,
            axis=(
                "fixed_context_changed_action"
                if base_id == "pair-a"
                else "fixed_action_changed_context"
            ),
        )
        for serialization in ("native_structured", "react_execution_only")
        for base_id in ("pair-a", "pair-b")
        for variant in ("safe", "unsafe")
    ]
    records = []
    for example in examples:
        prediction = float(example.original_label)
        if (
            example.serialization == "react_execution_only"
            and example.base_sample_id == "pair-b"
        ):
            prediction = 0.0
        records.append(_record(example, prediction))

    result = analyze_ts_guard_authorization_predictions(examples, records)

    native = result["by_serialization"]["native_structured"]
    react = result["by_serialization"]["react_execution_only"]
    assert native["unsafe_score_exceeds_safe"] == 2
    assert react["unsafe_score_exceeds_safe"] == 1
    assert react["ties"] == 1
    assert result["serialization_conclusion_agrees"] is True
    assert result["directionally_consistent_unsafe_higher"] is True


def _raw_agentdojo_samples() -> list[Sample]:
    # This is load_eval's lexical filename order, deliberately distinct from
    # the authors' released evaluator order.
    labels = {"banking": 0.0, "slack": 1.0, "travel": 0.5, "workspace": 0.0}
    return [
        Sample(
            sample_id=f"raw-{subset}",
            source="AgentDojo-Traj",
            split="eval",
            instruction=f"request-{subset}",
            history=f"history-{subset}",
            current_action=f"action-{subset}",
            env_info=f"tools-{subset}",
            label=labels[subset],
            subset=subset,
        )
        for subset in ("banking", "slack", "travel", "workspace")
    ]


def _write_reference(data_root: Path, samples: list[Sample]) -> None:
    by_subset = {sample.subset: sample for sample in samples}
    reference_order = [
        by_subset[subset] for subset in AGENTDOJO_REFERENCE_SUBSET_ORDER
    ]
    reference = data_root / "reference" / "agentdojo"
    reference.mkdir(parents=True)
    (reference / "labels.json").write_text(
        json.dumps([sample.label for sample in reference_order]),
        encoding="utf-8",
    )
    predictions = {"workspace": 0.0, "travel": 0.5, "slack": 1.0, "banking": 0.0}
    (reference / "preds.json").write_text(
        json.dumps([predictions[sample.subset] for sample in reference_order]),
        encoding="utf-8",
    )
    raw_by_id: dict[str, dict[str, object]] = {}
    for sample in samples:
        raw_by_id[sample.sample_id] = {
            "id-interaction": sample.sample_id,
            "id-segment": sample.segment_id,
            "instruction": sample.instruction,
            "history": sample.history,
            "current_action": sample.current_action,
            "env_info": sample.env_info,
            "score": sample.label,
        }
    raw_root = data_root / "eval" / "agentdojo"
    raw_root.mkdir(parents=True)
    for subset in AGENTDOJO_REFERENCE_SUBSET_ORDER:
        (raw_root / f"{subset}.json").write_text(
            json.dumps([raw_by_id[sample.sample_id] for sample in samples if sample.subset == subset]),
            encoding="utf-8",
        )
    metadata = [
        {
            "meta_sample": raw_by_id[sample.sample_id],
            "guard_res": {
                "risk rating": predictions[sample.subset],
                "results": {},
                "reason": "SENSITIVE_RELEASED_REASON",
            },
        }
        for sample in reference_order
    ]
    (reference / "meta_data.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def _patch_reference_constants(
    monkeypatch: pytest.MonkeyPatch,
    data_root: Path,
    rows: int,
) -> None:
    path = data_root / "reference" / "agentdojo" / "meta_data.json"
    monkeypatch.setattr(
        "toolsafe_lab.ts_guard_shortcut_analysis.TS_GUARD_META_DATA_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "toolsafe_lab.ts_guard_shortcut_analysis.TS_GUARD_META_DATA_SIZE_BYTES",
        path.stat().st_size,
    )
    monkeypatch.setattr(
        "toolsafe_lab.ts_guard_shortcut_analysis.TS_GUARD_META_DATA_ROWS",
        rows,
    )


def _selected_originals(samples: list[Sample]) -> list[AuditExample]:
    return [
        _example(
            cohort="agentdojo",
            base_id=sample.sample_id,
            variant="original",
            label=sample.strict_label,
            serialization="released_react",
            domain=sample.subset,
        )
        for sample in samples
    ]


def test_released_original_validation_aligns_authors_order_and_reports_exactness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = _raw_agentdojo_samples()
    _write_reference(tmp_path / "data", samples)
    _patch_reference_constants(monkeypatch, tmp_path / "data", len(samples))
    originals = _selected_originals(samples)
    records = []
    for example in originals:
        if example.domain == "slack":
            continue
        prediction = 1.0 if example.domain == "travel" else 0.0
        records.append(_record(example, prediction))
    views = local_prediction_views(originals, records)

    result = validate_agentdojo_original_reproduction(
        originals,
        views,
        samples,
        data_root=tmp_path / "data",
    )

    assert result["reference_subset_order"] == [
        "workspace",
        "travel",
        "slack",
        "banking",
    ]
    assert result["label_alignment"] == {
        "validated": True,
        "exact_matches": 4,
        "rate": 1.0,
    }
    assert result["released_input_alignment"]["exact_all_field_matches"] == 4
    assert result["released_metadata_alignment"] == {
        "prediction_matches": 4,
        "label_matches": 4,
        "rows": 4,
    }
    selected = result["selected_originals"]
    assert selected["local_coverage"]["valid"] == 3
    assert selected["jointly_valid"] == 3
    assert selected["exact_prediction_agreement"] == {
        "count": 2,
        "denominator": 3,
        "rate": 2 / 3,
    }
    assert selected["strict_decision_agreement"]["rate"] == 1.0


def test_released_alignment_rejects_lexical_label_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = _raw_agentdojo_samples()
    _write_reference(tmp_path / "data", samples)
    _patch_reference_constants(monkeypatch, tmp_path / "data", len(samples))
    reference = tmp_path / "data" / "reference" / "agentdojo"
    (reference / "labels.json").write_text(
        json.dumps([sample.label for sample in samples]),
        encoding="utf-8",
    )
    originals = _selected_originals(samples)
    views = local_prediction_views(
        originals,
        [_record(example, float(example.original_label)) for example in originals],
    )

    with pytest.raises(ValueError, match="label differs|workspace/travel/slack/banking"):
        validate_agentdojo_original_reproduction(
            originals,
            views,
            samples,
            data_root=tmp_path / "data",
        )


def test_aggregate_writer_never_copies_raw_model_or_benchmark_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = _raw_agentdojo_samples()
    _write_reference(tmp_path / "data", samples)
    _patch_reference_constants(monkeypatch, tmp_path / "data", len(samples))
    originals = _selected_originals(samples)
    examples = [
        _example(
            cohort="agentdojo",
            base_id=original.base_sample_id,
            variant=variant,
            label=original.original_label,
            serialization="released_react",
            domain=original.domain,
        )
        for original in originals
        for variant in AGENTDOJO_VARIANTS
    ]
    records = [
        _record(
            example,
            0.5 if example.domain == "travel" else float(example.original_label),
        )
        for example in examples
    ]
    raw_path = (
        tmp_path
        / "artifacts"
        / "shortcut_audit"
        / "ts_guard_local"
        / TS_GUARD_MODEL_REVISION[:12]
        / "transformers_greedy"
        / "agentdojo.jsonl"
    )
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "toolsafe_lab.ts_guard_shortcut_analysis.prepare_shortcut_audit",
        lambda **_kwargs: (
            {"agentdojo": examples},
            {"artifacts": {"agentdojo": {"rows": len(examples)}}},
        ),
    )
    monkeypatch.setattr(
        "toolsafe_lab.ts_guard_shortcut_analysis.load_eval",
        lambda _root: {"AgentDojo-Traj": samples},
    )

    result = analyze_ts_guard_shortcut_audit(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
        artifacts_root=tmp_path / "artifacts",
        synthetic_input=tmp_path / "synthetic.jsonl",
        cohort="agentdojo",
        bootstrap_replicates=100,
    )

    output = (
        tmp_path
        / "results"
        / aggregate_output_name(
            backend="transformers",
            decoding="greedy",
            cohort="agentdojo",
        )
    )
    serialized = output.read_text(encoding="utf-8")
    assert output.exists()
    assert result["raw_text_in_results"] is False
    assert result["local_runtime"] == "transformers_greedy"
    assert result["decoding"] == "greedy"
    assert "SENSITIVE_BENCHMARK_TEXT" not in serialized
    assert "request-workspace" not in serialized
    assert "SENSITIVE_RELEASED_REASON" not in serialized
    model_result = result["cohorts"]["agentdojo"]["models"][0]
    assert model_result["raw_output"]["inference_configuration"] == {
        "decoding": "greedy",
        "base_seed": None,
        "seed_scope": "base_sample_id+serialization+attempt",
        "record_schema_version": TS_GUARD_RECORD_SCHEMA_VERSION,
        "parser_version": TS_GUARD_PARSER_VERSION,
        "max_parse_turns": 1,
    }
    assert model_result["raw_output"]["token_counts"]["input"] == {
        "count": 16,
        "min": 1000,
        "median": 1000.0,
        "max": 1000,
        "mean": 1000.0,
    }
    assert model_result["raw_output"]["runtime_metrics"][
        "mps_driver_allocated_bytes"
    ]["max"] == 17_000_000_000
    validation = model_result["released_original_validation"]
    assert validation["selected_originals"]["exact_prediction_agreement"]["rate"] == 1.0


def test_complete_set_and_q8_artifact_are_asserted() -> None:
    examples = _agentdojo_examples(("safe",))
    records = [_record(example, 0.0) for example in examples]
    for record in records:
        record["backend_metadata"].update(TS_GUARD_Q8_ARTIFACT)

    result = _assert_complete_local_records(examples, records, backend="mlx_q8")
    assert result["validated"] is True
    assert result["q8_artifact"] == {"validated": True, **TS_GUARD_Q8_ARTIFACT}

    records[0]["backend_metadata"]["model_artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen converted artifact"):
        _assert_complete_local_records(examples, records, backend="mlx_q8")

    with pytest.raises(ValueError, match="exact complete prepared cohort"):
        _assert_complete_local_records(examples, records[:-1], backend="mlx_q8")


def test_aggregate_output_names_separate_runtime_and_partial_scope() -> None:
    assert aggregate_output_name(
        backend="transformers", decoding="greedy", cohort="all"
    ) == OUTPUT_NAME
    names = {
        aggregate_output_name(
            backend=backend,
            decoding=decoding,
            cohort=cohort,
        )
        for backend in ("transformers", "vllm")
        for decoding in ("greedy", "authors_sampling")
        for cohort in ("agentdojo", "authorization", "all")
    }
    assert len(names) == 12
