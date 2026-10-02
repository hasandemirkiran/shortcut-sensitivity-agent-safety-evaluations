from __future__ import annotations

from pathlib import Path

import pytest

from toolsafe_lab.llm_prompt import prompt_spec, ts_guard_composite_score
from toolsafe_lab.openrouter_luna_shortcut_analysis import (
    ANALYSIS_VERSION,
    DEFAULT_AGGREGATE_PATH,
    analyze_luna_records,
    parser,
)
from toolsafe_lab.openrouter_luna_shortcut_runner import (
    MODEL_ID,
    PROMPT_VERSION,
    PROTOCOL_VERSION,
    build_interleaved_plan,
)

from test_openrouter_shortcut_analysis import _fixed_rules, _source_interactions
from test_openrouter_shortcut_runner import _examples


def _assessment(block: int) -> dict[str, object]:
    return {
        "malicious_user_request": False,
        "third_party_attack": bool(block),
        "current_action_harmfulness": 1.0 if block else 0.0,
    }


def _records(
    outcomes: dict[tuple[str, str], list[int]],
) -> tuple[list[object], list[dict[str, object]]]:
    examples = _examples(("base-a", "base-b"))
    calls = build_interleaved_plan(examples)
    records: list[dict[str, object]] = []
    for call in calls:
        assessment = _assessment(
            outcomes[(call.base_sample_id, call.condition)][call.repetition]
        )
        score = ts_guard_composite_score(
            malicious_user_request=bool(assessment["malicious_user_request"]),
            third_party_attack=bool(assessment["third_party_attack"]),
            current_action_harmfulness=float(assessment["current_action_harmfulness"]),
        )
        records.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "cohort_sha256": "a" * 64,
                "model": MODEL_ID,
                "prompt_sha256": prompt_spec(PROMPT_VERSION).sha256,
                "rendered_input_sha256": call.rendered_input_sha256,
                "request_sha256": call.request_sha256,
                "job_id": call.job_id,
                "condition": call.condition,
                "repetition": call.repetition,
                "base_sample_id": call.base_sample_id,
                "status": "ok",
                "assessment": assessment,
                "prediction": score,
                "strict_decision": int(score != 0.0),
                "resolved_model": MODEL_ID,
                "resolved_provider": "OpenAI",
                "attempts": 1,
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        )
    return calls, records


def test_luna_wrapper_runs_shared_estimands_checks_and_luna_cost_rates() -> None:
    examples = _examples(("base-a", "base-b"))
    outcomes = {
        ("base-a", "original"): [0, 0, 0],
        ("base-a", "marker_paraphrase"): [1, 1, 1],
        ("base-a", "neutral_json_formatting_placebo"): [0, 0, 1],
        ("base-a", "tool_alias"): [0, 0, 0],
        ("base-a", "combined"): [1, 1, 0],
        ("base-b", "original"): [1, 1, 1],
        ("base-b", "marker_paraphrase"): [1, 1, 1],
        ("base-b", "neutral_json_formatting_placebo"): [1, 1, 1],
        ("base-b", "tool_alias"): [0, 0, 0],
        ("base-b", "combined"): [1, 1, 1],
    }
    calls, records = _records(outcomes)

    result = analyze_luna_records(
        calls=calls,
        records=records,
        cohort_sha256="a" * 64,
        source_interaction_by_base=_source_interactions(examples),
        fixed_rule_by_base=_fixed_rules(examples),
        bootstrap_replicates=100,
    )

    contrasts = result["marginal_block_probability_contrasts"]
    assert contrasts["marker_paraphrase"]["point"] == 0.5
    assert contrasts["neutral_json_formatting_placebo"]["point"] == 1 / 6
    assert contrasts["tool_alias"]["point"] == -0.5
    assert contrasts["combined"]["point"] == 1 / 3
    assert result["placebo_comparisons"]["marker_paraphrase_minus_placebo"][
        "point"
    ] == pytest.approx(1 / 3)
    assert result["analysis_version"] == ANALYSIS_VERSION
    assert result["protocol_version"] == PROTOCOL_VERSION
    assert result["planned_calls"] == 30
    assert result["repetitions_per_condition"] == 3
    assert result["raw_benchmark_text_in_summary"] is False
    implementation = result["analysis_implementation_sha256"]
    assert set(implementation) == {"luna_wrapper", "shared_estimands"}
    assert all(len(value) == 64 for value in implementation.values())
    provenance = result["provider_model_cost_provenance"]
    assert provenance["requested_model"] == MODEL_ID
    assert provenance["pinned_upstream_provider"] == "openai"
    assert provenance["resolved_provider_counts"] == {"OpenAI": 30}
    assert provenance["observed_or_token_estimated_cost_usd"] == pytest.approx(
        30 * (100 * 0.20 + 20 * 1.20) / 1_000_000
    )


def test_luna_analysis_rejects_haiku_provenance() -> None:
    examples = _examples(("base-a", "base-b"))
    outcomes = {
        (base, condition): [0, 0, 0]
        for base in ("base-a", "base-b")
        for condition in (
            "original",
            "marker_paraphrase",
            "neutral_json_formatting_placebo",
            "tool_alias",
            "combined",
        )
    }
    calls, records = _records(outcomes)
    records[0]["model"] = "anthropic/claude-haiku-4.5"

    with pytest.raises(ValueError, match="Record provenance mismatch"):
        analyze_luna_records(
            calls=calls,
            records=records,
            cohort_sha256="a" * 64,
            source_interaction_by_base=_source_interactions(examples),
            fixed_rule_by_base=_fixed_rules(examples),
            bootstrap_replicates=100,
        )


def test_luna_analysis_has_separate_records_and_aggregate_defaults() -> None:
    args = parser().parse_args([])

    assert args.records == Path(
        "artifacts/shortcut_audit/openrouter/gpt-5.6-luna-authors-v2-k3.jsonl"
    )
    assert args.output == DEFAULT_AGGREGATE_PATH
    assert args.output == Path(
        "results/benchmark_shortcut_audit.openrouter_gpt_5_6_luna_k3.json"
    )
    assert "haiku" not in args.records.as_posix()
    assert "haiku" not in args.output.as_posix()
