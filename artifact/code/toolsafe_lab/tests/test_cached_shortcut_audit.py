import json

from toolsafe_lab.cached_shortcut_audit import (
    _agentharm_assessment_comparison,
    _interaction_weighted_and_majority,
    _same_tool_contrast,
    clustered_safe_marker_difference,
    fixed_agentdojo_rule,
    tool_lookup_group_cv,
    tool_lookup_leave_domain_out,
)
from toolsafe_lab.data import Sample


def _sample(
    sample_id: str,
    *,
    label: float,
    tool: str,
    subset: str = "workspace",
    interaction: str | None = None,
    marker: bool = False,
    arguments: str = "{}",
) -> Sample:
    return Sample(
        sample_id=sample_id,
        source="AgentDojo-Traj",
        split="eval",
        instruction="sentinel request text",
        history=(
            "external <InFoRmAtIoN> sentinel payload </INFORMATION>"
            if marker
            else "sentinel ordinary history"
        ),
        current_action=(
            "Thought: sentinel rationale\n"
            f"Action: {tool}\n"
            f"Action Input: {arguments}"
        ),
        env_info="sentinel tool schema",
        label=label,
        trajectory_id=interaction or f"trajectory-{sample_id}",
        segment_id="0",
        subset=subset,
    )


def test_fixed_rule_requires_marker_and_frozen_side_effect_prefix() -> None:
    assert fixed_agentdojo_rule(
        _sample("both", label=1.0, tool="send_email", marker=True)
    ) == 1
    assert fixed_agentdojo_rule(
        _sample("read", label=0.0, tool="read_file", marker=True)
    ) == 0
    assert fixed_agentdojo_rule(
        _sample("unmarked", label=0.0, tool="send_email", marker=False)
    ) == 0


def test_tool_lookup_group_cv_and_domain_holdout_use_different_support() -> None:
    samples: list[Sample] = []
    target: dict[str, int] = {}
    for domain in ("banking", "workspace"):
        for label, tool in ((0.0, f"read_{domain}"), (1.0, f"send_{domain}")):
            for index in range(4):
                sample_id = f"{domain}-{int(label)}-{index}"
                sample = _sample(
                    sample_id,
                    label=label,
                    tool=tool,
                    subset=domain,
                    interaction=f"{domain}-interaction-{label}-{index}",
                )
                samples.append(sample)
                target[sample_id] = sample.strict_label

    grouped = tool_lookup_group_cv(samples, target)
    held_out = tool_lookup_leave_domain_out(samples, target)

    assert grouped["f1"] == 1.0
    assert grouped["seen_tool_coverage"] == 1.0
    assert held_out["seen_tool_coverage"] == 0.0
    assert held_out["f1"] < grouped["f1"]


def test_marker_matched_same_tool_control_measures_context_separation() -> None:
    samples = [
        _sample("safe-a", label=0.0, tool="send_email", marker=True),
        _sample("safe-b", label=0.0, tool="delete_file", marker=True),
        _sample("unsafe-a", label=1.0, tool="send_email", marker=True),
        _sample("unsafe-b", label=1.0, tool="delete_file", marker=True),
    ]
    predictions = {
        "safe-a": 0.0,
        "safe-b": 0.0,
        "unsafe-a": 1.0,
        "unsafe-b": 1.0,
    }

    result = _same_tool_contrast(samples, predictions, marker_only_safe=True)

    assert result["mixed_tool_count"] == 2
    assert result["safe_block_rate"] == 0.0
    assert result["unsafe_block_rate"] == 1.0
    assert result["pooled_rate_difference"] == 1.0


def test_agentharm_request_flag_can_solve_strict_but_not_loose_labels() -> None:
    samples = [
        _sample("safe", label=0.0, tool="read_file"),
        _sample("intermediate", label=0.5, tool="read_file"),
        _sample("direct", label=1.0, tool="send_email"),
    ]
    predictions = {"safe": 0.0, "intermediate": 0.5, "direct": 1.0}
    assessments = {
        "safe": {
            "malicious_user_request": False,
            "third_party_attack": False,
            "current_action_harmfulness": 0.0,
        },
        "intermediate": {
            "malicious_user_request": True,
            "third_party_attack": False,
            "current_action_harmfulness": 0.5,
        },
        "direct": {
            "malicious_user_request": True,
            "third_party_attack": False,
            "current_action_harmfulness": 1.0,
        },
    }

    result = _agentharm_assessment_comparison(samples, predictions, assessments)

    assert result["strict"]["malicious_user_request_flag_only"]["f1"] == 1.0
    assert result["strict"]["composite_request_flag_agreement"] == 1.0
    assert result["loose"]["malicious_user_request_flag_only"]["fp"] == 1
    assert result["loose"]["current_action_rating_one_only"]["f1"] == 1.0


def test_interaction_weighting_and_clustered_marker_bootstrap_are_deterministic() -> None:
    samples = [
        *[
            _sample(
                f"unsafe-{index}",
                label=1.0,
                tool="send_email",
                interaction="long-unsafe",
            )
            for index in range(4)
        ],
        _sample(
            "safe-fp",
            label=0.0,
            tool="read_file",
            interaction="safe-fp",
        ),
        _sample(
            "safe-tn",
            label=0.0,
            tool="read_file",
            interaction="safe-tn",
        ),
    ]
    predictions = {sample.sample_id: 1.0 for sample in samples}
    predictions["safe-tn"] = 0.0
    weighting = _interaction_weighted_and_majority(samples, predictions)

    assert weighting["row_level"]["f1"] > weighting["equal_interaction_id_weight"]["f1"]
    assert weighting["equal_interaction_id_weight"]["effective_weight"] == 3.0

    marker_samples = [
        _sample(
            "marker-a",
            label=0.0,
            tool="read_file",
            interaction="marker-a",
            marker=True,
        ),
        _sample(
            "marker-b",
            label=0.0,
            tool="read_file",
            interaction="marker-b",
            marker=True,
        ),
        _sample(
            "absent-a",
            label=0.0,
            tool="read_file",
            interaction="absent-a",
        ),
        _sample(
            "absent-b",
            label=0.0,
            tool="read_file",
            interaction="absent-b",
        ),
    ]
    marker_predictions = {
        "marker-a": 1.0,
        "marker-b": 1.0,
        "absent-a": 0.0,
        "absent-b": 0.0,
    }
    first = clustered_safe_marker_difference(
        marker_samples, marker_predictions, replicates=100, seed=17
    )
    second = clustered_safe_marker_difference(
        marker_samples, marker_predictions, replicates=100, seed=17
    )

    assert first == second
    assert first["point"] == 1.0
    assert "sentinel" not in json.dumps(first)
