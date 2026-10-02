#!/usr/bin/env python3
"""Assert the publication's headline values against aggregate artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict[str, Any]:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def interval(
    actual: dict[str, Any] | list[float], lower: float, upper: float, *, label: str
) -> None:
    if isinstance(actual, dict):
        values = [actual["lower"], actual["upper"]]
    else:
        values = actual
    close(values[0], lower, label=f"{label} lower")
    close(values[1], upper, label=f"{label} upper")


def assert_linuxarena() -> None:
    result = load("linuxarena_shortcut_audit.json")
    cohort = result["trajectory_lexical_diagnostics"]["exact_action_content_dedup"]
    assert cohort["total"] == 576
    assert cohort["attack"] == 176
    assert cohort["honest"] == 400
    close(cohort["auc"], 0.9335795454545455, label="LinuxArena lexical AUC")
    delta = cohort["paired_auc_delta_vs_inverse_mean_shell_length"]
    close(
        delta["inverse_mean_shell_length_auc"],
        0.9103693181818182,
        label="LinuxArena length AUC",
    )
    close(delta["point"], 0.023210227272727257, label="LinuxArena AUC delta")
    assert delta["clustered_bootstrap_95pct_ci"] == [
        0.008016273239307481,
        0.040192394024138926,
    ]
    length = result["trajectory_diagnostics"]["exact_action_content_dedup"][
        "deterministic_features"
    ]["inverse_mean_shell_command_chars"]
    interval(
        length["clustered_bootstrap_auc_95pct_ci"],
        0.8726638966480447,
        0.9410483760477112,
        label="LinuxArena length AUC",
    )
    interval(
        cohort["clustered_bootstrap_auc_95pct_ci"],
        0.9067694800641153,
        0.9554308210784314,
        label="LinuxArena lexical AUC",
    )

    matched = result["trajectory_lexical_diagnostics"]["haiku_model_matched"]
    close(matched["auc"], 0.9548913043478261, label="LinuxArena Haiku lexical AUC")
    matched_delta = matched["paired_auc_delta_vs_inverse_mean_shell_length"]
    close(
        matched_delta["inverse_mean_shell_length_auc"],
        0.9433695652173915,
        label="LinuxArena Haiku length AUC",
    )
    close(
        matched_delta["point"], 0.011521739130434638, label="LinuxArena Haiku AUC delta"
    )
    interval(
        matched_delta["clustered_bootstrap_95pct_ci"],
        -0.0030434782608694836,
        0.02701222826086964,
        label="LinuxArena Haiku AUC delta",
    )

    per_environment = list(cohort["per_environment_auc"].values())
    assert len(per_environment) == 10
    close(
        min(per_environment), 0.9, label="LinuxArena minimum held-out-environment AUC"
    )
    close(
        max(per_environment), 1.0, label="LinuxArena maximum held-out-environment AUC"
    )
    close(
        sum(per_environment) / len(per_environment),
        0.9512727901667046,
        label="LinuxArena macro held-out-environment AUC",
    )

    opus = result["monitor_cohorts"]["claude_opus_4_6"]
    shell = opus["shell_bearing_deterministic_features"]
    assert shell["inverse_mean_shell_command_chars"]["attack"] == 70
    assert shell["inverse_mean_shell_command_chars"]["honest"] == 200
    close(
        shell["inverse_n_actions"]["auc"],
        0.947392857142857,
        label="Opus inverse actions AUC",
    )
    close(
        shell["inverse_mean_shell_command_chars"]["auc"],
        0.9053571428571429,
        label="Opus shell length AUC",
    )
    close(
        opus["shell_bearing_trajectory_max_score"][
            "within_mean_shell_length_quartiles"
        ]["weighted_auc"],
        0.8964541005394201,
        label="Opus within-length-quartile AUC",
    )
    close(
        shell["inverse_mean_shell_command_chars"]["within_mean_shell_length_quartiles"][
            "weighted_auc"
        ],
        0.6066065252360892,
        label="Length within-length-quartile AUC",
    )
    close(
        opus["shell_bearing_trajectory_max_score"]["auc"],
        0.8692142857142856,
        label="Opus monitor AUC",
    )
    close(
        opus["shell_bearing_trajectory_max_score"]["tpr_at_1pct_trajectory_fpr"][
            "expected_tpr"
        ],
        0.42857142857142855,
        label="Opus trajectory TPR",
    )
    close(
        result["trajectory_lexical_diagnostics"]["opus_honest_vs_lastraj"][
            "tpr_at_1pct_trajectory_fpr"
        ]["expected_tpr"],
        0.2571428571428571,
        label="Lexical trajectory TPR",
    )
    close(
        shell["inverse_mean_shell_command_chars"]["tpr_at_1pct_trajectory_fpr"][
            "expected_tpr"
        ],
        0.22857142857142856,
        label="Length trajectory TPR",
    )
    operational = opus["official_style_stepwise_1pct_fpr"]
    assert operational["attack_trajectories"] == 74
    assert operational["attack_scored_actions"] == 891
    assert operational["honest_trajectories"] == 200
    assert operational["honest_scored_actions"] == 12494
    close(
        operational["expected_any_action_detection"],
        0.7851702267208615,
        label="Operational detection",
    )
    interval(
        operational["clustered_bootstrap_detection_95pct_ci"],
        0.6095405030863963,
        0.9372730323638973,
        label="Operational detection",
    )


def assert_cached_tsbench() -> None:
    result = load("tsbench_cached_shortcut_audit.json")
    dojo = result["AgentDojo"]
    fixed = dojo["fixed_rule_vs_gold"]
    assert fixed["n"] == 1220
    assert fixed["tp"] == 342 and fixed["fn"] == 10
    close(fixed["f1"], 0.8018757327080891, label="AgentDojo fixed-rule F1")
    close(
        fixed["balanced_accuracy"],
        0.8942055927943025,
        label="AgentDojo fixed-rule balanced accuracy",
    )
    grouped = dojo["gold_tool_lookup"]["group_cv"]
    lodo = dojo["gold_tool_lookup"]["leave_domain_out"]
    close(grouped["f1"], 0.9008746355685131, label="AgentDojo grouped lookup F1")
    close(
        grouped["balanced_accuracy"],
        0.9245195328864684,
        label="AgentDojo grouped lookup balanced accuracy",
    )
    close(lodo["f1"], 0.49547920433996384, label="AgentDojo LODO lookup F1")
    close(
        lodo["seen_tool_coverage"],
        0.17295081967213113,
        label="AgentDojo LODO tool coverage",
    )

    harm = result["AgentHarm"]
    assert harm["strict_subset_determinism"]["harmful_subset_rows"] == 525
    assert harm["strict_subset_determinism"]["benign_subset_rows"] == 206
    close(
        harm["all_unsafe_strict_baseline"]["f1"],
        0.8359872611464968,
        label="AgentHarm all-unsafe F1",
    )
    close(
        harm["all_unsafe_strict_baseline"]["balanced_accuracy"],
        0.5,
        label="AgentHarm all-unsafe balanced accuracy",
    )
    weighting = harm["guards"]["TS-Guard"]["interaction_weighting"]
    close(weighting["row_level"]["f1"], 0.9016829052258636, label="TS-Guard row F1")
    close(
        weighting["equal_interaction_id_weight"]["f1"],
        0.7193282690863438,
        label="TS-Guard equal-interaction F1",
    )
    close(
        weighting["equal_interaction_id_weight"]["accuracy"],
        0.7244742567077593,
        label="TS-Guard equal-interaction accuracy",
    )
    close(
        weighting["equal_interaction_id_weight"]["balanced_accuracy"],
        0.7756858465608465,
        label="TS-Guard equal-interaction balanced accuracy",
    )
    close(
        weighting["row_level"]["accuracy"],
        0.8481532147742819,
        label="TS-Guard row accuracy",
    )
    close(
        weighting["row_level"]["balanced_accuracy"],
        0.7541793804900601,
        label="TS-Guard row balanced accuracy",
    )
    majority = weighting["interaction_id_majority"]
    close(majority["accuracy"], 0.7157360406091371, label="TS-Guard majority accuracy")
    close(majority["f1"], 0.7142857142857142, label="TS-Guard majority F1")
    close(
        majority["balanced_accuracy"],
        0.7701111111111111,
        label="TS-Guard majority balanced accuracy",
    )
    assert (
        harm["interaction_structure"]["by_subset"]["harmful_steps"][
            "interaction_ids_with_duplicate_segment_ids"
        ]
        == 57
    )
    assert (
        harm["interaction_structure"]["by_subset"]["harmful_steps"]["interaction_ids"]
        == 72
    )
    assert (
        harm["interaction_structure"]["by_subset"]["benign_steps"][
            "interaction_ids_with_duplicate_segment_ids"
        ]
        == 0
    )
    assert (
        harm["interaction_structure"]["by_subset"]["benign_steps"]["interaction_ids"]
        == 125
    )

    for guard, expected in {
        "TS-Guard": (
            31,
            321,
            33,
            547,
            0.03624414108105953,
            0.0009494813514912124,
            0.0732895442989811,
        ),
        "GPT-5.5": (
            17,
            320,
            6,
            547,
            0.04215607861060329,
            0.017035415855552995,
            0.07053552300736281,
        ),
        "Claude Haiku 4.5": (
            239,
            321,
            2,
            547,
            0.7408919794745624,
            0.6867232044178884,
            0.7975370173267788,
        ),
    }.items():
        data = dojo["guards"][guard]
        present = data["block_rate_strata"]["safe_marker_present"]
        absent = data["block_rate_strata"]["safe_marker_absent"]
        assert (
            present["blocked"],
            present["n"],
            absent["blocked"],
            absent["n"],
        ) == expected[:4]
        diff = data["safe_marker_clustered_rate_difference"]
        close(diff["point"], expected[4], label=f"{guard} marker difference")
        interval(diff, expected[5], expected[6], label=f"{guard} marker difference")


def model_result(name: str, cohort: str = "agentdojo") -> dict[str, Any]:
    result = load(name)
    models = result["cohorts"][cohort]["models"]
    assert len(models) == 1
    return models[0]


def assert_paired(
    model: dict[str, Any],
    variant: str,
    *,
    flips: int,
    denominator: int,
    rate: float,
    valid_both: bool,
) -> None:
    paired = model["paired_transformations"][variant]
    if valid_both:
        paired = paired["valid_both_only"]
    assert paired["flip_count"] == flips
    assert paired["n"] == denominator
    close(paired["flip_rate"], rate, label=f"{model['model']} {variant} flip rate")


def assert_counterfactual_manifest() -> None:
    manifest = load("counterfactual_cohort_manifest.json")
    assert manifest["schema_version"] == 1
    assert manifest["prompt_version"] == "authors_v2_structured"
    assert manifest["prompt_sha256"] == (
        "0e6dd5de7686d45e7fb54d999bf58b5959191d247820d84bc1d095f3d52b1c2f"
    )
    agentdojo = manifest["artifacts"]["agentdojo"]
    assert agentdojo["rows"] == 256 and agentdojo["size_bytes"] == 2415616
    assert agentdojo["sha256"] == (
        "bdfe8848090aab7f6efd0d575d6547ad88aae84831c8d0792055bb0076c638a2"
    )
    authorization = manifest["artifacts"]["authorization"]
    assert authorization["rows"] == 32 and authorization["size_bytes"] == 50612
    assert authorization["sha256"] == (
        "7316e3cb2fdcc8665b8ed50dbab068918a3576a4b5642380f572815b84c3aa6a"
    )
    cohort = manifest["cohorts"]["agentdojo"]
    assert cohort["selected_base_rows"] == 64
    assert cohort["selected_base_rows_by_cell"] == {
        "banking/safe": 8,
        "banking/unsafe": 8,
        "slack/safe": 8,
        "slack/unsafe": 8,
        "travel/safe": 8,
        "travel/unsafe": 8,
        "workspace/safe": 8,
        "workspace/unsafe": 8,
    }


def assert_counterfactuals() -> None:
    ts_guard = model_result("agentdojo_counterfactual_tsguard_q8.json")
    assert ts_guard["overall_coverage"]["valid"] == 255
    gate = ts_guard["released_original_validation"]["selected_originals"]
    assert gate["strict_decision_agreement"]["count"] == 63
    assert gate["strict_decision_agreement"]["denominator"] == 64
    assert gate["exact_prediction_agreement"]["count"] == 58
    assert gate["mlx_comparability_gate"]["passed"] is True
    assert_paired(
        ts_guard,
        "marker_paraphrase",
        flips=3,
        denominator=64,
        rate=0.046875,
        valid_both=True,
    )
    assert_paired(
        ts_guard,
        "tool_alias",
        flips=3,
        denominator=63,
        rate=0.047619047619047616,
        valid_both=True,
    )
    assert_paired(
        ts_guard,
        "combined",
        flips=6,
        denominator=64,
        rate=0.09375,
        valid_both=True,
    )

    gpt = model_result("agentdojo_counterfactual_gpt55.json")
    assert gpt["overall_coverage"]["valid"] == 256
    assert_paired(
        gpt,
        "marker_paraphrase",
        flips=3,
        denominator=64,
        rate=0.046875,
        valid_both=False,
    )
    provenance = load("gpt55_followup_provenance.json")
    assert provenance["resolved_model"] == "gpt-5.5-2026-04-23"
    run = provenance["counterfactual_run"]
    assert run["rows"] == 256 and run["valid_rows"] == 256
    assert run["usage"]["input_tokens"] == 721558
    assert run["usage"]["output_tokens"] == 48653
    assert run["usage"]["reasoning_tokens"] == 39579
    close(run["usage"]["estimated_cost_usd"], 5.06738, label="GPT-5.5 cost")
    repeat = provenance["prior_original_repeat_comparison"]
    assert repeat["n"] == 64
    assert repeat["strict_decision_agreement"] == 63
    assert repeat["strict_decision_disagreement"] == 1
    assert repeat["complete_component_agreement"] == 35
    assert repeat["complete_component_disagreement"] == 29

    ts_runtime = ts_guard["raw_output"]["cumulative_generation_seconds"]
    close(
        ts_runtime["mean"] * ts_runtime["count"],
        2668.49519204651,
        label="TS-Guard AgentDojo generation seconds",
    )
    close(
        ts_runtime["median"],
        9.264517125091515,
        label="TS-Guard median generation seconds",
    )
    assert (
        ts_guard["raw_output"]["runtime_metrics"]["mlx_peak_memory_bytes"]["max"]
        == 9466752040
    )
    auth = model_result(
        "agentdojo_counterfactual_tsguard_q8.json", cohort="authorization"
    )
    auth_runtime = auth["raw_output"]["cumulative_generation_seconds"]
    close(
        auth_runtime["mean"] * auth_runtime["count"],
        200.62941420846618,
        label="TS-Guard pilot generation seconds",
    )
    assert_paired(
        gpt,
        "tool_alias",
        flips=1,
        denominator=64,
        rate=0.015625,
        valid_both=False,
    )
    assert_paired(
        gpt,
        "combined",
        flips=2,
        denominator=64,
        rate=0.03125,
        valid_both=False,
    )


def assert_corrected_reanalysis() -> None:
    result = load("shortcut_audit_cached_reanalysis.json")
    assert result["schema_version"] == 2
    assert result["generated_from_cached_artifacts_only"] is True
    assert result["AgentDojo_paired"]["frozen_base_rows"] == 64
    guards = result["AgentDojo_paired"]["guards"]
    q8 = guards["TS-Guard Q8"]
    expected = {
        "marker_paraphrase": {
            "flips": 3,
            "pairs": 64,
            "base_rate": 0.046875,
            "cp": (0.009773074954456552, 0.13093573724140578),
            "interaction_rate": 0.04310344827586207,
            "interactions": 58,
            "poststratified": 0.09082466567607728,
            "components": 7,
            "ordinal": 5,
        },
        "tool_alias": {
            "flips": 3,
            "pairs": 63,
            "base_rate": 0.047619047619047616,
            "cp": (0.00992995115073328, 0.1329184007481506),
            "interaction_rate": 0.05263157894736842,
            "interactions": 57,
            "poststratified": 0.09499044788792188,
            "components": 14,
            "ordinal": 12,
        },
        "combined": {
            "flips": 6,
            "pairs": 64,
            "base_rate": 0.09375,
            "cp": (0.03518733287081832, 0.19296910444834386),
            "interaction_rate": 0.08620689655172414,
            "interactions": 58,
            "poststratified": 0.1300148588410104,
            "components": 14,
            "ordinal": 13,
        },
    }
    for variant, values in expected.items():
        data = q8[variant]
        balanced = data["balanced_cohort"]
        assert balanced["flip_count"] == values["flips"]
        assert data["valid_pairs"] == values["pairs"]
        close(balanced["rate"], values["base_rate"], label=f"Q8 {variant} base rate")
        interval(
            balanced["clopper_pearson_interval"],
            *values["cp"],
            label=f"Q8 {variant} descriptive CP interval",
        )
        clustered = data["interaction_clustered_sensitivity"]
        assert clustered["interaction_count"] == values["interactions"]
        assert (
            clustered["percentile_cluster_bootstrap_interval"]["resampling_unit"]
            == "source_interaction_id"
        )
        close(
            clustered["rate"],
            values["interaction_rate"],
            label=f"Q8 {variant} interaction rate",
        )
        population = data["marker_present_population_poststratified"]
        assert population["target_population_n"] == 673
        close(
            population["rate"],
            values["poststratified"],
            label=f"Q8 {variant} poststratified rate",
        )
        components = data["ordinal_and_component_sensitivity"]
        assert components["any_component_change_count"] == values["components"]
        assert components["ordinal_score_change_count"] == values["ordinal"]

    marker_rule = q8["marker_paraphrase"]["deterministic_rule_sensitivity"]
    assert marker_rule["rule_changed_pairs"] == 42
    assert marker_rule["directionally_concordant_flips_when_rule_changed"] == 2

    gpt = guards["GPT-5.5"]
    gpt_expected = {
        "marker_paraphrase": (3, 0.015625, -0.03278688524590164, 0.07145445134575532),
        "tool_alias": (1, 0.015625, 0.0, 0.047619047619047616),
        "combined": (2, 0.0, -0.04838709677419355, 0.043478260869565216),
    }
    for variant, (flips, point, lower, upper) in gpt_expected.items():
        assert gpt[variant]["balanced_cohort"]["flip_count"] == flips
        assert (
            gpt[variant]["decision_process"]
            == "potentially_stochastic_single_draw_per_condition"
        )
        assert "signed K=1 mean estimates" in gpt[variant]["inferential_scope"]
        signed = gpt[variant]["signed_block_probability_contrast"]
        close(signed["point"], point, label=f"GPT-5.5 {variant} signed contrast")
        interval(
            signed["source_interaction_cluster_bootstrap_interval"],
            lower,
            upper,
            label=f"GPT-5.5 {variant} signed contrast",
        )
        assert "not interleaved" in signed["design_limitations"]

    matched = result["AgentHarm_exact_execution_matched_context"]
    assert matched["mixed_label_exact_execution_groups"] == 16
    assert matched["rows_in_groups"] == 69
    assert matched["source_interaction_ids"] == 39
    ts_guard = matched["guards"]["TS-Guard"]["ordinal_composite_score"]
    close(ts_guard["macro_safe_mean"], 0.4125, label="AgentHarm matched safe mean")
    close(
        ts_guard["macro_unsafe_mean"],
        0.9223214285714286,
        label="AgentHarm matched unsafe mean",
    )
    close(
        ts_guard["macro_unsafe_minus_safe"],
        0.5098214285714286,
        label="AgentHarm matched difference",
    )
    assert ts_guard["group_direction_counts"] == {
        "tied": 5,
        "unsafe_higher": 11,
    }


def assert_repeated_haiku() -> None:
    result = load("benchmark_shortcut_audit.openrouter_haiku_k3.json")
    assert result["analysis_schema_version"] == 3
    assert result["analysis_status"] == "complete"
    assert result["planned_calls"] == 960
    assert result["base_steps"] == 64
    assert result["source_interactions"] == 58
    assert result["overall_coverage"] == {
        "expected": 960,
        "failed_or_missing": 0,
        "failure_counts": {},
        "received": 960,
        "valid": 960,
        "valid_rate": 1.0,
    }
    assert (
        result["cohort_sha256"]
        == "bdfe8848090aab7f6efd0d575d6547ad88aae84831c8d0792055bb0076c638a2"
    )
    assert (
        result["prompt_sha256"]
        == "0e6dd5de7686d45e7fb54d999bf58b5959191d247820d84bc1d095f3d52b1c2f"
    )
    assert (
        result["schedule_sha256"]
        == "ecb2b0cdae6b9a9f5f4365d116b2c310397df3537c006f57651e6041a40da621"
    )

    contrasts = result["marginal_block_probability_contrasts"]
    expected = {
        "marker_paraphrase": (
            -0.08333333333333333,
            -0.15053763440860213,
            -0.02487562189054726,
        ),
        "tool_alias": (0.11458333333333333, 0.026881720430107524, 0.20588235294117646),
        "combined": (-0.026041666666666668, -0.08994708994708994, 0.029850746268656716),
        "neutral_json_formatting_placebo": (
            -0.015624999999999998,
            -0.06779661016949153,
            0.027777777777777776,
        ),
    }
    safe_expected = {
        "marker_paraphrase": (
            -0.16666666666666666,
            -0.3,
            -0.058823529411764705,
            6,
            0,
            0.03125,
        ),
        "tool_alias": (
            0.22916666666666666,
            0.05555555555555555,
            0.39215686274509803,
            2,
            9,
            0.109375,
        ),
        "combined": (
            -0.052083333333333336,
            -0.17204301075268816,
            0.05208333333333333,
            5,
            2,
            0.453125,
        ),
        "neutral_json_formatting_placebo": (
            -0.031249999999999997,
            -0.13541666666666666,
            0.05376344086021506,
            2,
            2,
            1.0,
        ),
    }
    for condition, (point, lower, upper) in expected.items():
        contrast = contrasts[condition]
        assert contrast["interval"]["clusters"] == 58
        close(contrast["point"], point, label=f"Haiku {condition} block contrast")
        interval(
            contrast["interval"],
            lower,
            upper,
            label=f"Haiku {condition} block contrast",
        )
        close(
            contrast["by_original_label"]["unsafe"]["point"],
            0.0,
            label=f"Haiku {condition} unsafe contrast",
        )
        safe_point, safe_lower, safe_upper, decreases, increases, sign_p = (
            safe_expected[condition]
        )
        safe = contrast["by_original_label"]["safe"]
        close(safe["point"], safe_point, label=f"Haiku {condition} safe contrast")
        interval(
            safe["interval"],
            safe_lower,
            safe_upper,
            label=f"Haiku {condition} safe contrast",
        )
        assert contrast["direction_counts"]["decrease"] == decreases
        assert contrast["direction_counts"]["increase"] == increases
        sign = contrast["source_interaction_sign_diagnostic"]
        close(
            sign["two_sided_exact_p"],
            sign_p,
            label=f"Haiku {condition} sign diagnostic",
        )

    marker_placebo = result["placebo_comparisons"]["marker_paraphrase_minus_placebo"]
    close(marker_placebo["point"], -0.06770833333333333, label="Haiku marker-placebo")
    interval(
        marker_placebo["interval"],
        -0.12820512820512822,
        -0.016129032258064516,
        label="Haiku marker-placebo",
    )
    interval(
        marker_placebo["by_original_label"]["safe"]["interval"],
        -0.25806451612903225,
        -0.03225806451612903,
        label="Haiku safe marker-placebo",
    )
    close(
        marker_placebo["source_interaction_sign_diagnostic"]["two_sided_exact_p"],
        0.0625,
        label="Haiku marker-placebo sign diagnostic",
    )
    alias_placebo = result["placebo_comparisons"]["tool_alias_minus_placebo"]
    close(alias_placebo["point"], 0.13020833333333334, label="Haiku alias-placebo")
    interval(
        alias_placebo["interval"],
        0.04918032786885245,
        0.21717171717171718,
        label="Haiku alias-placebo",
    )
    interval(
        alias_placebo["by_original_label"]["safe"]["interval"],
        0.1111111111111111,
        0.4117647058823529,
        label="Haiku safe alias-placebo",
    )
    close(
        alias_placebo["source_interaction_sign_diagnostic"]["two_sided_exact_p"],
        0.0390625,
        label="Haiku alias-placebo sign diagnostic",
    )

    marker_rounds = contrasts["marker_paraphrase"]["signed_order_diagnostics"][
        "by_repetition"
    ]
    alias_rounds = contrasts["tool_alias"]["signed_order_diagnostics"]["by_repetition"]
    for repetition, marker_point, alias_point in (
        ("0", -0.09375, 0.09375),
        ("1", -0.078125, 0.125),
        ("2", -0.078125, 0.125),
    ):
        close(
            marker_rounds[repetition]["point"],
            marker_point,
            label=f"Haiku marker round {repetition}",
        )
        close(
            alias_rounds[repetition]["point"],
            alias_point,
            label=f"Haiku alias round {repetition}",
        )

    original_stability = result["within_condition_stability"]["original"]
    close(
        original_stability["fail_closed_mean_pairwise_disagreement"],
        0.020833333333333332,
        label="Haiku repeat-original disagreement",
    )
    close(
        result["within_condition_stability"]["neutral_json_formatting_placebo"][
            "fail_closed_mean_pairwise_disagreement"
        ],
        0.0,
        label="Haiku placebo disagreement",
    )

    accuracy = result["repeated_condition_accuracy_contrasts"]
    close(
        accuracy["marker_paraphrase"]["point"],
        0.08333333333333333,
        label="Haiku marker accuracy",
    )
    close(
        accuracy["tool_alias"]["point"],
        -0.11458333333333333,
        label="Haiku alias accuracy",
    )
    for condition, expected_accuracy in (
        ("original", 0.734375),
        ("marker_paraphrase", 0.8177083333333334),
        ("tool_alias", 0.6197916666666666),
    ):
        close(
            result["by_condition"][condition]["strict_fail_closed_accuracy"],
            expected_accuracy,
            label=f"Haiku {condition} absolute accuracy",
        )
    marker_rule = contrasts["marker_paraphrase"]["by_original_fixed_rule"]
    close(
        marker_rule["rule_positive"]["point"],
        -0.047619047619047616,
        label="Haiku rule-positive marker",
    )
    close(
        marker_rule["rule_negative"]["point"],
        -0.15151515151515152,
        label="Haiku rule-negative marker",
    )
    assert contrasts["marker_paraphrase"]["by_domain"] == {
        "banking": {"bases": 16, "point": 0.0},
        "slack": {"bases": 16, "point": -0.125},
        "travel": {"bases": 16, "point": -0.14583333333333334},
        "workspace": {"bases": 16, "point": -0.0625},
    }
    assert contrasts["tool_alias"]["by_domain"] == {
        "banking": {"bases": 16, "point": 0.25},
        "slack": {"bases": 16, "point": 0.125},
        "travel": {"bases": 16, "point": 0.020833333333333336},
        "workspace": {"bases": 16, "point": 0.0625},
    }
    assert result["by_condition"]["original"][
        "strict_fail_closed_accuracy_by_original_label"
    ]["unsafe"] == {"base_steps": 32, "calls": 96, "point": 1.0}

    provider = result["provider_model_cost_provenance"]
    assert provider["requested_model"] == "anthropic/claude-haiku-4.5"
    assert provider["pinned_upstream_provider"] == "anthropic"
    assert provider["resolved_model_counts"] == {"anthropic/claude-haiku-4.5": 960}
    assert provider["resolved_provider_counts"] == {"Anthropic": 960}
    assert provider["attempt_count_distribution"] == {"1": 960}
    assert provider["prompt_tokens"] == 3247713
    assert provider["completion_tokens"] == 31990
    close(
        provider["observed_or_token_estimated_cost_usd"], 3.407663, label="Haiku cost"
    )
    raw = result["raw_records_artifact"]
    assert raw["bytes"] == 1608985
    assert raw["parsed_records"] == 960
    assert raw["raw_benchmark_text_embedded_in_aggregate"] is False
    assert (
        raw["sha256"]
        == "f21a34ca9bb49a5c112213dbe0d563055737fd114dbb0db6ba490338c063b188"
    )


def assert_repeated_luna() -> None:
    result = load("benchmark_shortcut_audit.openrouter_gpt_5_6_luna_k3.json")
    assert result["analysis_schema_version"] == 3
    assert result["analysis_status"] == "complete"
    assert result["planned_calls"] == 960
    assert result["base_steps"] == 64
    assert result["source_interactions"] == 58
    assert result["overall_coverage"] == {
        "expected": 960,
        "failed_or_missing": 0,
        "failure_counts": {},
        "received": 960,
        "valid": 960,
        "valid_rate": 1.0,
    }
    assert (
        result["cohort_sha256"]
        == "bdfe8848090aab7f6efd0d575d6547ad88aae84831c8d0792055bb0076c638a2"
    )
    assert (
        result["prompt_sha256"]
        == "0e6dd5de7686d45e7fb54d999bf58b5959191d247820d84bc1d095f3d52b1c2f"
    )
    assert (
        result["schedule_sha256"]
        == "77ef17c93dac28b70efb89fedaae460b77d91c475d5a8c33e07169ee70c272b2"
    )
    assert set(result["analysis_implementation_sha256"]) == {
        "luna_wrapper",
        "shared_estimands",
    }

    contrasts = result["marginal_block_probability_contrasts"]
    expected = {
        "marker_paraphrase": (0.0, -0.03125, 0.031746031746031744),
        "tool_alias": (
            0.010416666666666668,
            -0.020833333333333336,
            0.04411764705882353,
        ),
        "combined": (0.0, -0.041666666666666664, 0.04445273631840784),
        "neutral_json_formatting_placebo": (
            0.010416666666666668,
            -0.02564102564102564,
            0.04838709677419355,
        ),
    }
    for condition, (point, lower, upper) in expected.items():
        contrast = contrasts[condition]
        close(contrast["point"], point, label=f"Luna {condition} block contrast")
        interval(
            contrast["interval"],
            lower,
            upper,
            label=f"Luna {condition} block contrast",
        )
        close(
            contrast["by_original_label"]["unsafe"]["point"],
            0.0,
            label=f"Luna {condition} unsafe contrast",
        )
        assert (
            result["strict_all_three_unanimous_shifts"][condition][
                "strict_unanimous_shift_count"
            ]
            == 0
        )

    marker_placebo = result["placebo_comparisons"]["marker_paraphrase_minus_placebo"]
    close(marker_placebo["point"], -0.010416666666666668, label="Luna marker-placebo")
    interval(
        marker_placebo["interval"],
        -0.047619047619047616,
        0.026041666666666664,
        label="Luna marker-placebo",
    )
    alias_placebo = result["placebo_comparisons"]["tool_alias_minus_placebo"]
    close(alias_placebo["point"], 0.0, label="Luna alias-placebo")
    interval(
        alias_placebo["interval"],
        -0.036458333333333336,
        0.03241689435336963,
        label="Luna alias-placebo",
    )
    close(
        result["within_condition_stability"]["original"][
            "fail_closed_mean_pairwise_disagreement"
        ],
        0.07291666666666666,
        label="Luna repeat-original disagreement",
    )
    for condition in (
        "marker_paraphrase",
        "neutral_json_formatting_placebo",
        "tool_alias",
        "combined",
    ):
        close(
            result["within_condition_stability"][condition][
                "fail_closed_mean_pairwise_disagreement"
            ],
            0.0625,
            label=f"Luna {condition} disagreement",
        )

    provider = result["provider_model_cost_provenance"]
    assert provider["requested_model"] == "openai/gpt-5.6-luna"
    assert provider["pinned_upstream_provider"] == "openai"
    assert provider["resolved_model_counts"] == {"openai/gpt-5.6-luna": 960}
    assert provider["resolved_provider_counts"] == {"OpenAI": 960}
    assert provider["attempt_count_distribution"] == {"1": 960}
    assert provider["prompt_tokens"] == 2700051
    assert provider["completion_tokens"] == 173298
    close(
        provider["observed_or_token_estimated_cost_usd"],
        0.48405043,
        label="Luna cost",
    )
    raw = result["raw_records_artifact"]
    assert raw["bytes"] == 1762198
    assert raw["parsed_records"] == 960
    assert raw["raw_benchmark_text_embedded_in_aggregate"] is False
    assert (
        raw["sha256"]
        == "b39f08ac98dfbc3d8c8d1aaebb04860ec7491b5c77a2a1032b2163b3cbfd1fcf"
    )


def assert_stepguard_feasibility() -> None:
    result = load("stepguard_feasibility.json")
    assert result["artifact_kind"] == "one_row_feasibility_gate"
    assert result["code_revision"] == "47c9011ee73c90be403846d35476ec55d5dab63b"
    assert result["model"]["revision"] == "c322d1ec2de38f204498d45f79646c6e80a0ea7e"
    assert result["model"]["bf16_weight_bytes"] == 8044936192
    pilot = result["pilot"]
    assert pilot["input_tokens"] == 2207 and pilot["output_tokens"] == 221
    assert pilot["parsed_strict_decision"] == 0
    close(pilot["model_load_seconds"], 3.0, label="StepGuard model load")
    close(pilot["generation_seconds"], 45.93, label="StepGuard generation")
    close(
        result["runtime"]["mps_peak_driver_allocation_gb"],
        26.72,
        label="StepGuard peak MPS driver allocation",
    )
    raw = result["raw_record_artifact"]
    assert raw["bytes"] == 2459 and raw["records"] == 1 and raw["shipped"] is False
    assert (
        raw["sha256"]
        == "31348dbba250427da2341ea3c4593627adb978bb8277643ce1c81f9addf82ac5"
    )


def assert_lookup_coverage() -> None:
    result = load("agentdojo_lookup_coverage.json")
    assert result["source_commit"] == "46358fa424a927a895c6c8322f99032c4eb5155e"
    expected_counts = {
        "all": (1220, 137, 804, 64, 215),
        "seen": (211, 137, 4, 64, 6),
        "unseen": (1009, 0, 800, 0, 209),
    }
    metrics = result["metrics"]
    for stratum, expected in expected_counts.items():
        row = metrics[stratum]
        assert tuple(row[k] for k in ("n", "tp", "tn", "fp", "fn")) == expected
        n, tp, tn, fp, fn = expected
        assert n == tp + tn + fp + fn
        close(row["accuracy"], (tp + tn) / n, label=f"{stratum} accuracy")
        close(row["f1"], 2 * tp / (2 * tp + fp + fn), label=f"{stratum} F1")
        close(row["balanced_accuracy"], (tp / (tp + fn) + tn / (tn + fp)) / 2,
              label=f"{stratum} balanced accuracy")
    for key in ("n", "tp", "tn", "fp", "fn"):
        assert metrics["seen"][key] + metrics["unseen"][key] == metrics["all"][key]
    original = load("tsbench_cached_shortcut_audit.json")["AgentDojo"][
        "gold_tool_lookup"]["leave_domain_out"]
    for key in ("n", "tp", "tn", "fp", "fn", "accuracy", "f1", "balanced_accuracy"):
        close(metrics["all"][key], original[key], label=f"original pooled lookup {key}")
    assert set(result["by_domain"]) == {"banking", "slack", "travel", "workspace"}
    assert all(row["fallback"] == 0 for row in result["by_domain"].values())
    assert sum(row["n"] for row in result["by_domain"].values()) == metrics["all"]["n"]


def assert_manuscript_alignment() -> None:
    manuscript = (ROOT / "main.tex").read_text(encoding="utf-8")
    for stale_claim in (
        "gap contracts",
        "much smaller than its",
        "cannot estimate the stochastic",
        "do not estimate Equation",
        "Agent-safety benchmarks contain",
        "\\oldtext{",
        "\\oldblock{",
        "\\update{",
        "\\reviewer{",
        "\\proposal{",
        "reviewmode",
        "review/jens",
        "We find that X",
    ):
        if stale_claim in manuscript:
            raise AssertionError(f"stale manuscript claim remains: {stale_claim}")
    for required_value in (
        "$74.09\\pp$",
        "$-8.33",
        "$-16.67",
        "$+11.46",
        "$73.44\\%$",
        "$81.77\\%$",
        "$61.98\\%$",
        "balanced accuracy $.925$",
        "$F_1=.836$; balanced accuracy $.500$",
        "Equal interaction weight (197) & .724 & .719 & .776",
        "Interaction-majority (197) & .716 & .714 & .770",
        "macro mean $.951$",
        "weighted AUC $.896$ versus $.607$",
        "74 human-crafted LaStraj attacks",
        "70 human-crafted attacks",
        "TS-Guard plus six hosted configurations",
        "\\texttt{<INFORMATION>... </INFORMATION>} tag pair",
        "alias blocking rises",
        "marker effects are absent in banking",
        "GPT-5.6 Luna gives $0.00\\pp$ and $+1.04\\pp$",
        "contrasts are $0.00\\pp$ $[-3.13,3.17]\\pp$ and $+1.04\\pp$",
        "$10.48\\pp$ observational marker gap precedes a $0.00\\pp$ repeated",
        "Every unsafe original is blocked",
        "57 of 72 harmful interactions",
        "1,608,985 bytes",
        "1,762,198 bytes",
        "Seen names & 211 & .797 & .508 & 137 & 4 & 64 & 6",
        "Unseen names & 1,009 & .000 & .500 & 0 & 800 & 0 & 209",
        "All names & 1,220 & .495 & .658 & 137 & 804 & 64 & 215",
    ):
        if required_value not in manuscript:
            raise AssertionError(f"expected manuscript value missing: {required_value}")

    checklist = (ROOT / "checklist.tex").read_text(encoding="utf-8")
    declaration = checklist.split("Declaration of LLM usage", maxsplit=1)[1]
    declaration = declaration.split("\\end{enumerate}", maxsplit=1)[0]
    if "Answer: \\answerNo{}" not in declaration:
        raise AssertionError("checklist question 16 must remain Answer: No")


def main() -> None:
    assert_linuxarena()
    assert_cached_tsbench()
    assert_counterfactual_manifest()
    assert_counterfactuals()
    assert_corrected_reanalysis()
    assert_repeated_haiku()
    assert_repeated_luna()
    assert_stepguard_feasibility()
    assert_lookup_coverage()
    assert_manuscript_alignment()
    print("Verified the manuscript values covered by committed aggregate assertions.")


if __name__ == "__main__":
    main()
