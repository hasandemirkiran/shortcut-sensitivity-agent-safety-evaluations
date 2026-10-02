from toolsafe_lab.llm_prompt import prompt_spec, ts_guard_composite_score


def test_prompt_fingerprint_is_stable() -> None:
    first = prompt_spec("causal_current_action_v1")
    second = prompt_spec("causal_current_action_v1")
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_ts_guard_composite_reproduces_released_parser() -> None:
    assert (
        ts_guard_composite_score(
            malicious_user_request=False,
            third_party_attack=False,
            current_action_harmfulness=1.0,
        )
        == 0.0
    )
    assert (
        ts_guard_composite_score(
            malicious_user_request=True,
            third_party_attack=False,
            current_action_harmfulness=0.5,
        )
        == 0.5
    )
    assert (
        ts_guard_composite_score(
            malicious_user_request=True,
            third_party_attack=False,
            current_action_harmfulness=1.0,
        )
        == 1.0
    )
    assert (
        ts_guard_composite_score(
            malicious_user_request=True,
            third_party_attack=True,
            current_action_harmfulness=0.0,
        )
        == 1.0
    )
