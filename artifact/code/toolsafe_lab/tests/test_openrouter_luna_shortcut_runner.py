from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from toolsafe_lab.openrouter_luna_shortcut_runner import (
    DEFAULT_CONCURRENCY,
    DEFAULT_COST_LIMIT_USD,
    DEFAULT_MAX_ATTEMPTS,
    INPUT_USD_PER_MTOK,
    MAX_TOKENS,
    MODEL_ID,
    OUTPUT_USD_PER_MTOK,
    PINNED_PROVIDER,
    PROTOCOL_VERSION,
    REASONING_EFFORT,
    REPETITIONS,
    TEMPERATURE_POLICY,
    _request_body,
    build_interleaved_plan,
    build_manifest,
    load_openrouter_key,
    parser,
    projected_cost,
    run_openrouter_plan,
)
from toolsafe_lab.openrouter_shortcut_runner import (
    build_interleaved_plan as build_haiku_interleaved_plan,
)
from toolsafe_lab.openrouter_shortcut_runner import parser as haiku_parser

from test_openrouter_shortcut_runner import _examples


def test_luna_request_is_strict_pinned_and_medium_reasoning() -> None:
    body = _request_body("inert benchmark payload")

    assert body["model"] == "openai/gpt-5.6-luna" == MODEL_ID
    assert "temperature" not in body
    assert TEMPERATURE_POLICY == "provider_default_explicit_temperature_omitted"
    assert body["max_tokens"] == 2048 == MAX_TOKENS
    assert body["reasoning"] == {"effort": "medium", "exclude": True}
    assert REASONING_EFFORT == "medium"
    assert body["provider"] == {
        "only": ["openai"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    assert PINNED_PROVIDER == "openai"
    assert body["plugins"] == []
    assert "tools" not in body
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False


def test_luna_plan_is_exact_haiku_schedule_with_separate_960_jobs() -> None:
    examples = _examples(tuple(f"base-{index:02d}" for index in range(64)))
    haiku = build_haiku_interleaved_plan(examples)
    luna = build_interleaved_plan(list(reversed(examples)))

    assert len(luna) == 960 == 64 * 5 * REPETITIONS
    assert [
        (
            row.schedule_index,
            row.round_index,
            row.block_id,
            row.condition,
            row.record_variant,
            row.repetition,
            row.base_sample_id,
            row.source_sample_id,
            row.rendered_input_sha256,
        )
        for row in luna
    ] == [
        (
            row.schedule_index,
            row.round_index,
            row.block_id,
            row.condition,
            row.record_variant,
            row.repetition,
            row.base_sample_id,
            row.source_sample_id,
            row.rendered_input_sha256,
        )
        for row in haiku
    ]
    assert {row.job_id for row in luna}.isdisjoint(row.job_id for row in haiku)
    assert all(luna_row.request_sha256 != haiku_row.request_sha256 for luna_row, haiku_row in zip(luna, haiku, strict=True))
    assert PROTOCOL_VERSION != "OPENROUTER_HAIKU_AGENTDOJO_K3_v1.0"


def test_luna_pricing_manifest_and_default_paths_are_separate(tmp_path: Path) -> None:
    examples = _examples(tuple(f"base-{index:02d}" for index in range(64)))
    calls = build_interleaved_plan(examples)
    prior = tmp_path / "gpt-5.6-luna.jsonl"
    with prior.open("w", encoding="utf-8") as handle:
        for index in range(64):
            handle.write(
                json.dumps(
                    {
                        "status": "ok",
                        "sample_id": f"base-{index:02d}",
                        "usage": {"input_tokens": 1000, "output_tokens": 100},
                    }
                )
                + "\n"
            )
    projection = projected_cost(calls, prior_usage_path=prior)

    assert INPUT_USD_PER_MTOK == 0.20
    assert OUTPUT_USD_PER_MTOK == 1.20
    assert DEFAULT_COST_LIMIT_USD == 3.0
    assert projection["method"] == "prior_first_party_luna_usage_for_same_64_originals"
    assert projection["matched_prior_bases"] == 64
    assert projection["expected_list_price_usd"] == pytest.approx(0.3072)
    assert projection["projected_max_output_tokens"] == 960 * 2048

    cohort = tmp_path / "agentdojo.jsonl"
    cohort.write_text("frozen cohort placeholder\n", encoding="utf-8")
    manifest = build_manifest(
        calls=calls,
        cohort_path=cohort,
        cohort_sha256="a" * 64,
        prior_usage_path=prior,
    )
    assert manifest["planned_calls"] == 960
    assert manifest["reasoning"] == {"effort": "medium", "exclude": True}
    assert manifest["max_output_tokens_per_call"] == 2048
    assert manifest["temperature_policy"] == TEMPERATURE_POLICY
    assert len(manifest["schedule_planner_sha256"]) == 64
    assert manifest["raw_benchmark_text_in_manifest"] is False
    assert "only valid status=ok records are complete" in manifest["resume_policy"]
    assert "Send the requested message" not in json.dumps(manifest)

    luna_args = parser().parse_args([])
    haiku_args = haiku_parser().parse_args([])
    assert luna_args.output != haiku_args.output
    assert luna_args.output.with_suffix(".manifest.json") != haiku_args.output.with_suffix(
        ".manifest.json"
    )
    assert luna_args.prior_usage_file.name == "gpt-5.6-luna.jsonl"
    assert luna_args.concurrency == DEFAULT_CONCURRENCY == 2
    assert luna_args.max_attempts == DEFAULT_MAX_ATTEMPTS == 3


def test_mocked_luna_transport_resumes_and_persists_no_prompt_or_key(tmp_path: Path) -> None:
    calls = build_interleaved_plan(_examples(("base-a",)))
    output = tmp_path / "luna.jsonl"
    secret = "dummy-openrouter-test-key"
    seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen += 1
        assert request.headers["authorization"] == f"Bearer {secret}"
        body = json.loads(request.content)
        assert body["model"] == MODEL_ID
        assert "temperature" not in body
        assert body["reasoning"] == {"effort": "medium", "exclude": True}
        assert body["provider"]["only"] == ["openai"]
        return httpx.Response(
            200,
            json={
                "id": f"generation-{seen}",
                "model": MODEL_ID,
                "provider": "OpenAI",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "malicious_user_request": False,
                                    "third_party_attack": True,
                                    "current_action_harmfulness": 0.5,
                                }
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 80,
                    "total_tokens": 180,
                    "cost": 0.000116,
                },
            },
        )

    async def run_twice() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            first = await run_openrouter_plan(
                calls=calls,
                cohort_sha256="a" * 64,
                output_path=output,
                api_key=secret,
                client=client,
            )
            second = await run_openrouter_plan(
                calls=calls,
                cohort_sha256="a" * 64,
                output_path=output,
                api_key=secret,
                client=client,
            )
        return first, second

    first, second = asyncio.run(run_twice())

    assert seen == len(calls)
    assert first == second
    assert all(row["status"] == "ok" for row in first)
    assert all(row["resolved_provider"] == "OpenAI" for row in first)
    assert all(row["reasoning_effort"] == "medium" for row in first)
    persisted = output.read_text(encoding="utf-8")
    assert secret not in persisted
    assert "Send the requested message" not in persisted
    assert "untrusted text" not in persisted


def test_failed_terminal_pilot_is_retried_on_resume(tmp_path: Path) -> None:
    calls = build_interleaved_plan(_examples(("base-a",)))
    call = calls[0]
    output = tmp_path / "luna.jsonl"
    failed = {
        "protocol_version": PROTOCOL_VERSION,
        "cohort_sha256": "a" * 64,
        "model": MODEL_ID,
        "prompt_sha256": (
            "0e6dd5de7686d45e7fb54d999bf58b5959191d247820d84bc1d095f3d52b1c2f"
        ),
        "rendered_input_sha256": call.rendered_input_sha256,
        "request_sha256": call.request_sha256,
        "job_id": call.job_id,
        "status": "error",
        "error_type": "http_error",
        "http_status": 401,
        "terminal": True,
    }
    output.write_text(json.dumps(failed) + "\n", encoding="utf-8")
    seen = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen += 1
        return httpx.Response(
            200,
            json={
                "id": "generation-retry",
                "model": MODEL_ID,
                "provider": "OpenAI",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "malicious_user_request": False,
                                    "third_party_attack": False,
                                    "current_action_harmfulness": 0.0,
                                }
                            ),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    async def run() -> list[dict[str, object]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_openrouter_plan(
                calls=calls,
                cohort_sha256="a" * 64,
                output_path=output,
                api_key="dummy",
                max_calls=1,
                client=client,
            )

    records = asyncio.run(run())

    assert seen == 1
    assert len(records) == 2
    assert records[0]["status"] == "error"
    assert records[1]["status"] == "ok"
    assert records[0]["job_id"] == records[1]["job_id"] == call.job_id


def test_luna_key_loader_requires_private_file_permissions(tmp_path: Path) -> None:
    path = tmp_path / "keys.txt"
    path.write_text("OPENROUTER_API_KEY=dummy-openrouter-key\n", encoding="utf-8")
    path.chmod(0o600)
    assert load_openrouter_key(path) == "dummy-openrouter-key"

    path.chmod(0o644)
    with pytest.raises(PermissionError, match="chmod 600"):
        load_openrouter_key(path)


def test_resume_makes_no_request_when_prior_observed_cost_reached_cap(
    tmp_path: Path,
) -> None:
    calls = build_interleaved_plan(_examples(("base-a",)))
    call = calls[0]
    output = tmp_path / "luna.jsonl"
    prior = {
        "protocol_version": PROTOCOL_VERSION,
        "cohort_sha256": "a" * 64,
        "model": MODEL_ID,
        "prompt_sha256": (
            "0e6dd5de7686d45e7fb54d999bf58b5959191d247820d84bc1d095f3d52b1c2f"
        ),
        "rendered_input_sha256": call.rendered_input_sha256,
        "request_sha256": call.request_sha256,
        "job_id": call.job_id,
        "status": "ok",
        "usage": {"cost": 3.0},
    }
    output.write_text(json.dumps(prior) + "\n", encoding="utf-8")
    seen = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen += 1
        raise AssertionError("Observed cost cap must stop before another request")

    async def run() -> list[dict[str, object]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_openrouter_plan(
                calls=calls,
                cohort_sha256="a" * 64,
                output_path=output,
                api_key="dummy",
                client=client,
            )

    records = asyncio.run(run())

    assert seen == 0
    assert records == [prior]


@pytest.mark.parametrize(
    ("concurrency", "max_attempts"),
    [(3, 3), (2, 4)],
)
def test_luna_runner_enforces_concurrency_and_attempt_caps(
    tmp_path: Path,
    concurrency: int,
    max_attempts: int,
) -> None:
    with pytest.raises(ValueError):
        asyncio.run(
            run_openrouter_plan(
                calls=[],
                cohort_sha256="a" * 64,
                output_path=tmp_path / "unused.jsonl",
                api_key="dummy",
                concurrency=concurrency,
                max_attempts=max_attempts,
            )
        )
