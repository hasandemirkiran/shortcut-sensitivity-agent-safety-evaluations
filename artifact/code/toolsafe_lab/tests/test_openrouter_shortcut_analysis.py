from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from toolsafe_lab.llm_prompt import prompt_spec, ts_guard_composite_score
from toolsafe_lab.openrouter_shortcut_analysis import (
    _raw_records_artifact_provenance,
    analyze_openrouter_records,
)
from toolsafe_lab.openrouter_shortcut_runner import (
    MODEL_ID,
    PROMPT_VERSION,
    PROTOCOL_VERSION,
    build_interleaved_plan,
)
from toolsafe_lab.shortcut_audit import AuditExample

from test_openrouter_shortcut_runner import _examples


def _assessment(block: int) -> dict[str, object]:
    if block:
        return {
            "malicious_user_request": False,
            "third_party_attack": True,
            "current_action_harmfulness": 1.0,
        }
    return {
        "malicious_user_request": False,
        "third_party_attack": False,
        "current_action_harmfulness": 0.0,
    }


def _records(
    examples: list[AuditExample],
    outcomes: dict[tuple[str, str], list[int]],
) -> tuple[list[object], list[dict[str, object]]]:
    calls = build_interleaved_plan(examples)
    records = []
    for call in calls:
        block = outcomes[(call.base_sample_id, call.condition)][call.repetition]
        assessment = _assessment(block)
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
                "resolved_provider": "Anthropic",
                "attempts": 1,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "cost": 0.001,
                },
            }
        )
    return calls, records


def _source_interactions(
    examples: list[AuditExample],
    *,
    shared: bool = False,
) -> dict[str, str]:
    base_ids = sorted({example.base_sample_id for example in examples})
    if shared:
        return {base_id: "source-interaction-shared" for base_id in base_ids}
    return {base_id: f"source-interaction-{base_id}" for base_id in base_ids}


def _fixed_rules(examples: list[AuditExample]) -> dict[str, int]:
    base_ids = sorted({example.base_sample_id for example in examples})
    return {base_id: index % 2 for index, base_id in enumerate(base_ids)}


def _set_original_labels(
    examples: list[AuditExample],
    labels: dict[str, int],
) -> list[AuditExample]:
    return [
        replace(example, original_label=labels.get(example.base_sample_id, 0))
        for example in examples
    ]


def test_repeated_analysis_uses_marginal_probability_not_one_draw_flips() -> None:
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
    calls, records = _records(examples, outcomes)

    result = analyze_openrouter_records(
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
    assert contrasts["marker_paraphrase"]["interval"]["clusters"] == 2
    assert contrasts["marker_paraphrase"]["by_original_fixed_rule"] == {
        "rule_negative": {
            "base_steps": 1,
            "source_interactions": 1,
            "point": 1.0,
        },
        "rule_positive": {
            "base_steps": 1,
            "source_interactions": 1,
            "point": 0.0,
        },
    }
    accuracy = result["repeated_condition_accuracy_contrasts"]
    assert result["by_condition"]["original"]["strict_fail_closed_accuracy"] == 0.5
    assert accuracy["marker_paraphrase"]["point"] == -0.5
    assert accuracy["neutral_json_formatting_placebo"]["point"] == pytest.approx(-1 / 6)
    assert accuracy["tool_alias"]["point"] == 0.5
    assert accuracy["combined"]["point"] == pytest.approx(-1 / 3)
    assert accuracy["marker_paraphrase"]["by_domain"]["slack"] == {
        "bases": 2,
        "point": -0.5,
    }
    assert accuracy["marker_paraphrase"]["by_original_label"]["safe"] == {
        "bases": 2,
        "point": -0.5,
    }
    assert result["base_steps"] == 2
    assert result["source_interactions"] == 2
    assert result["placebo_comparisons"]["marker_paraphrase_minus_placebo"][
        "point"
    ] == pytest.approx(1 / 3)
    assert (
        "one-draw"
        in result["secondary_one_draw_flip_diagnostics"]["marker_paraphrase"]["interpretation"]
    )


def test_marginal_and_placebo_robustness_diagnostics() -> None:
    examples = _set_original_labels(
        _examples(("base-a", "base-b", "base-c", "base-d")),
        {"base-c": 1, "base-d": 1},
    )
    outcomes = {
        (base, condition): [0, 0, 0]
        for base in ("base-a", "base-b", "base-c", "base-d")
        for condition in (
            "original",
            "marker_paraphrase",
            "neutral_json_formatting_placebo",
            "tool_alias",
            "combined",
        )
    }
    for base in ("base-a", "base-b", "base-c", "base-d"):
        outcomes[(base, "marker_paraphrase")] = [1, 1, 1]
    calls, records = _records(examples, outcomes)

    result = analyze_openrouter_records(
        calls=calls,
        records=records,
        cohort_sha256="a" * 64,
        source_interaction_by_base=_source_interactions(examples),
        fixed_rule_by_base=_fixed_rules(examples),
        bootstrap_replicates=100,
    )

    marker = result["marginal_block_probability_contrasts"]["marker_paraphrase"]
    assert marker["point"] == 1.0
    assert marker["by_original_label"]["safe"]["point"] == 1.0
    assert marker["by_original_label"]["safe"]["interval"]["clusters"] == 2
    assert marker["by_original_label"]["unsafe"]["point"] == 1.0
    assert marker["by_original_label"]["unsafe"]["interval"]["clusters"] == 2
    assert marker["source_interaction_sign_diagnostic"] == {
        "interpretation": (
            "exploratory exact two-sided sign test after equally averaging base-step "
            "contrasts within each upstream source interaction; zero interactions are "
            "excluded from the binomial test"
        ),
        "source_interactions": 4,
        "positive": 4,
        "negative": 0,
        "tie": 0,
        "nonzero": 4,
        "two_sided_exact_p": 0.125,
        "zero_tolerance": 1e-12,
    }
    order = marker["signed_order_diagnostics"]
    assert [order["by_repetition"][str(index)]["point"] for index in range(3)] == [
        1.0,
        1.0,
        1.0,
    ]
    assert [
        order["by_chronological_schedule_third"][third]["point"]
        for third in ("first", "middle", "last")
    ] == [1.0, 1.0, 1.0]
    assert order["pairs_crossing_chronological_third_boundaries"] == 0

    marker_placebo = result["placebo_comparisons"]["marker_paraphrase_minus_placebo"]
    assert marker_placebo["by_original_label"]["safe"]["interval"]["clusters"] == 2
    assert marker_placebo["source_interaction_sign_diagnostic"]["two_sided_exact_p"] == 0.125
    assert marker_placebo["signed_order_diagnostics"]["all_pairs_point"] == 1.0


def test_raw_records_artifact_provenance_is_text_free(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_bytes(b'{"job_id":"one"}\n')

    assert _raw_records_artifact_provenance(
        records_path,
        artifact_path="artifacts/records.jsonl",
        parsed_records=1,
    ) == {
        "artifact_path": "artifacts/records.jsonl",
        "bytes": 17,
        "sha256": "6bd7b564293bfb8fb84edea70e28f798ed3056c099e893fe0038b3f16cd257b2",
        "parsed_records": 1,
        "raw_benchmark_text_embedded_in_aggregate": False,
    }


def test_analysis_reports_stability_components_unanimity_and_cost() -> None:
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
    for condition in (
        "original",
        "marker_paraphrase",
        "neutral_json_formatting_placebo",
        "combined",
    ):
        outcomes[("base-b", condition)] = [1, 1, 1]
    outcomes[("base-a", "marker_paraphrase")] = [1, 1, 1]
    outcomes[("base-a", "neutral_json_formatting_placebo")] = [0, 1, 0]
    outcomes[("base-a", "combined")] = [1, 1, 1]
    outcomes[("base-b", "tool_alias")] = [0, 0, 0]
    calls, records = _records(examples, dict(outcomes))

    result = analyze_openrouter_records(
        calls=calls,
        records=records,
        cohort_sha256="a" * 64,
        source_interaction_by_base=_source_interactions(examples),
        fixed_rule_by_base=_fixed_rules(examples),
        bootstrap_replicates=100,
    )

    stability = result["within_condition_stability"]
    assert stability["original"]["fail_closed_all_three_unanimous_rate"] == 1.0
    assert (
        stability["neutral_json_formatting_placebo"]["fail_closed_all_three_unanimous_rate"] == 0.5
    )
    unanimous = result["strict_all_three_unanimous_shifts"]
    assert unanimous["marker_paraphrase"]["counts"]["0_to_1"] == 1
    assert unanimous["tool_alias"]["counts"]["1_to_0"] == 1
    components = result["component_and_ordinal_contrasts"]["marker_paraphrase"]
    assert components["differences_condition_minus_original"]["third_party_attack"]["point"] == 0.5
    provenance = result["provider_model_cost_provenance"]
    assert provenance["resolved_provider_counts"] == {"Anthropic": 30}
    assert provenance["observed_or_token_estimated_cost_usd"] == 0.03
    assert result["raw_benchmark_text_in_summary"] is False


def test_analysis_fails_closed_but_excludes_failure_from_component_contrast() -> None:
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
    calls, records = _records(examples, outcomes)
    failed = next(
        row
        for row in records
        if row["base_sample_id"] == "base-a"
        and row["condition"] == "marker_paraphrase"
        and row["repetition"] == 0
    )
    failed.update(status="error", error_type="http_error")
    failed.pop("assessment")
    failed.pop("prediction")
    failed.pop("strict_decision")

    result = analyze_openrouter_records(
        calls=calls,
        records=records,
        cohort_sha256="a" * 64,
        source_interaction_by_base=_source_interactions(examples),
        fixed_rule_by_base=_fixed_rules(examples),
        bootstrap_replicates=100,
    )

    marker = result["by_condition"]["marker_paraphrase"]
    assert marker["coverage"]["valid"] == 5
    assert marker["strict_fail_closed_block_rate"] == 1 / 6
    assert (
        result["component_and_ordinal_contrasts"]["marker_paraphrase"][
            "complete_valid_six_call_bases"
        ]
        == 1
    )
    assert result["analysis_status"] == "partial_or_failed"


def test_every_interval_clusters_by_upstream_source_interaction() -> None:
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
    calls, records = _records(examples, outcomes)

    result = analyze_openrouter_records(
        calls=calls,
        records=records,
        cohort_sha256="a" * 64,
        source_interaction_by_base=_source_interactions(examples, shared=True),
        fixed_rule_by_base=_fixed_rules(examples),
        bootstrap_replicates=100,
    )

    assert result["base_steps"] == 2
    assert result["source_interactions"] == 1
    assert (
        result["by_condition"]["original"]["strict_fail_closed_block_rate_interval"]["clusters"]
        == 1
    )
    assert (
        result["marginal_block_probability_contrasts"]["marker_paraphrase"]["interval"]["clusters"]
        == 1
    )
    assert (
        result["repeated_condition_accuracy_contrasts"]["marker_paraphrase"]["interval"]["clusters"]
        == 1
    )
    assert (
        result["within_condition_stability"]["original"][
            "fail_closed_mean_pairwise_disagreement_interval"
        ]["clusters"]
        == 1
    )
    assert (
        result["component_and_ordinal_contrasts"]["marker_paraphrase"][
            "differences_condition_minus_original"
        ]["malicious_user_request"]["interval"]["clusters"]
        == 1
    )
    assert (
        result["placebo_comparisons"]["marker_paraphrase_minus_placebo"]["interval"]["clusters"]
        == 1
    )


def test_analysis_requires_exact_source_interaction_coverage() -> None:
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
    calls, records = _records(examples, outcomes)

    with pytest.raises(ValueError, match="exactly cover"):
        analyze_openrouter_records(
            calls=calls,
            records=records,
            cohort_sha256="a" * 64,
            source_interaction_by_base={"base-a": "interaction-a"},
            fixed_rule_by_base=_fixed_rules(examples),
            bootstrap_replicates=100,
        )

    with pytest.raises(ValueError, match="unexpected=.*base-c"):
        analyze_openrouter_records(
            calls=calls,
            records=records,
            cohort_sha256="a" * 64,
            source_interaction_by_base={
                "base-a": "interaction-a",
                "base-b": "interaction-b",
                "base-c": "interaction-c",
            },
            fixed_rule_by_base=_fixed_rules(examples),
            bootstrap_replicates=100,
        )

    with pytest.raises(ValueError, match="Fixed-rule mapping must exactly cover"):
        analyze_openrouter_records(
            calls=calls,
            records=records,
            cohort_sha256="a" * 64,
            source_interaction_by_base=_source_interactions(examples),
            fixed_rule_by_base={"base-a": 0},
            bootstrap_replicates=100,
        )
