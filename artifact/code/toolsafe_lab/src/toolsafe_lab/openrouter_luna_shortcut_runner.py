"""Repeated, interleaved OpenRouter GPT-5.6 Luna AgentDojo inference.

The frozen Haiku planner remains the source of the K=3 schedule semantics. This
module rebinds that schedule to a separate Luna protocol and request envelope,
so Haiku job IDs, manifests, and result records are never modified. Candidate
tool calls are untrusted text: no tools are exposed or executed, and raw
benchmark prompts are never written to result records or manifests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Mapping, Sequence

import httpx

import toolsafe_lab.openrouter_shortcut_runner as haiku_runner
from toolsafe_lab.llm_prompt import ASSESSMENT_SCHEMA, prompt_spec
from toolsafe_lab.openrouter_shortcut_runner import (
    CONDITIONS,
    REPETITIONS,
    SCHEDULE_SEED,
    PlannedCall,
    _canonical,
    _load_prior_usage,
    _parse_response,
    _provider_error,
    _repair_and_read_jsonl,
    _schedule_sha256,
    _sha256_path,
    _sha256_text,
    build_interleaved_plan as build_haiku_interleaved_plan,
    load_frozen_agentdojo_cohort,
    load_openrouter_key as _load_openrouter_key,
)
from toolsafe_lab.shortcut_audit import AuditExample


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "OPENROUTER_GPT_5_6_LUNA_AGENTDOJO_K3_v1.0"
PROMPT_VERSION = "authors_v2_structured"
MODEL_ID = "openai/gpt-5.6-luna"
PINNED_PROVIDER = "openai"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REASONING_EFFORT = "medium"
REASONING_EXCLUDE = True
MAX_TOKENS = 2048
TEMPERATURE_POLICY = "provider_default_explicit_temperature_omitted"
DEFAULT_CONCURRENCY = 2
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_COST_LIMIT_USD = 3.0
INPUT_USD_PER_MTOK = 0.20
OUTPUT_USD_PER_MTOK = 1.20

DEFAULT_COHORT_PATH = Path("artifacts/shortcut_audit/cohorts/agentdojo.jsonl")
DEFAULT_PRIOR_USAGE_PATH = Path(
    "artifacts/api_runs/eval/authors_v2_structured/gpt-5.6-luna.jsonl"
)
DEFAULT_OUTPUT_PATH = Path(
    "artifacts/shortcut_audit/openrouter/gpt-5.6-luna-authors-v2-k3.jsonl"
)
DEFAULT_KEYS_PATH = Path.home() / "Desktop/api_keys.txt"


def load_openrouter_key(path: Path) -> str:
    """Load the key with the shared parser while requiring private file permissions."""
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(
            f"OpenRouter key file must not be group/world accessible: {path} "
            f"has mode {mode:04o}; run chmod 600 on it first"
        )
    return _load_openrouter_key(path)


def request_settings() -> dict[str, object]:
    """Return the complete non-message OpenRouter Luna request configuration."""
    return {
        "model": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "reasoning": {
            "effort": REASONING_EFFORT,
            "exclude": REASONING_EXCLUDE,
        },
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "toolsafe_step_assessment",
                "strict": True,
                "schema": ASSESSMENT_SCHEMA,
            },
        },
        "provider": {
            "only": [PINNED_PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        },
        "plugins": [],
    }


def _request_body(rendered_input: str) -> dict[str, object]:
    prompt = prompt_spec(PROMPT_VERSION)
    return {
        **request_settings(),
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": rendered_input},
        ],
    }


def build_interleaved_plan(
    examples: Sequence[AuditExample],
    *,
    schedule_seed: int = SCHEDULE_SEED,
    repetitions: int = REPETITIONS,
) -> list[PlannedCall]:
    """Bind the exact frozen Haiku K=3 schedule to Luna requests and job IDs."""
    scheduled = build_haiku_interleaved_plan(
        examples,
        schedule_seed=schedule_seed,
        repetitions=repetitions,
    )
    calls: list[PlannedCall] = []
    for source in scheduled:
        request_sha = _sha256_text(_canonical(_request_body(source.rendered_input)))
        identity = {
            "protocol_version": PROTOCOL_VERSION,
            "schedule_seed": schedule_seed,
            "round_index": source.round_index,
            "condition": source.condition,
            "base_sample_id": source.base_sample_id,
            "rendered_input_sha256": source.rendered_input_sha256,
            "request_sha256": request_sha,
        }
        calls.append(
            replace(
                source,
                job_id=_sha256_text(_canonical(identity))[:24],
                request_sha256=request_sha,
            )
        )
    if len({call.job_id for call in calls}) != len(calls):
        raise AssertionError("Planned Luna OpenRouter job IDs are not unique")
    return calls


def projected_cost(
    calls: Sequence[PlannedCall],
    *,
    prior_usage_path: Path,
) -> dict[str, object]:
    """Estimate Luna spend from first-party usage for these frozen originals."""
    base_ids = {call.base_sample_id for call in calls}
    prior = _load_prior_usage(prior_usage_path, base_ids)
    if set(prior) == base_ids:
        input_tokens = sum(prior[call.base_sample_id][0] for call in calls)
        expected_output_tokens = sum(prior[call.base_sample_id][1] for call in calls)
        method = "prior_first_party_luna_usage_for_same_64_originals"
    else:
        input_tokens = sum(
            (len(call.rendered_input.encode("utf-8")) + 3) // 4 for call in calls
        )
        expected_output_tokens = 256 * len(calls)
        method = "utf8_bytes_div_4_plus_256_output_tokens_per_call"
    expected_usd = (
        input_tokens * INPUT_USD_PER_MTOK
        + expected_output_tokens * OUTPUT_USD_PER_MTOK
    ) / 1_000_000
    max_output_tokens = len(calls) * MAX_TOKENS
    single_attempt_cap_usd = (
        input_tokens * INPUT_USD_PER_MTOK
        + max_output_tokens * OUTPUT_USD_PER_MTOK
    ) / 1_000_000
    return {
        "method": method,
        "matched_prior_bases": len(prior),
        "projected_input_tokens": input_tokens,
        "projected_expected_output_tokens": expected_output_tokens,
        "projected_max_output_tokens": max_output_tokens,
        "expected_list_price_usd": expected_usd,
        "single_attempt_output_cap_list_price_usd": single_attempt_cap_usd,
        "rates_usd_per_million_tokens": {
            "input": INPUT_USD_PER_MTOK,
            "output": OUTPUT_USD_PER_MTOK,
        },
        "notes": [
            "List-price projection excludes prompt-cache discounts.",
            "The output-cap figure assumes every successful call uses all 2048 tokens.",
            "HTTP retries are not included; the runner caps attempts and observed spend.",
        ],
    }


def build_manifest(
    *,
    calls: Sequence[PlannedCall],
    cohort_path: Path,
    cohort_sha256: str,
    prior_usage_path: Path,
) -> dict[str, object]:
    prompt = prompt_spec(PROMPT_VERSION)
    core = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cohort_path": str(cohort_path.resolve()),
        "cohort_sha256": cohort_sha256,
        "cohort_base_samples": len({call.base_sample_id for call in calls}),
        "conditions": list(CONDITIONS),
        "repetitions_per_condition": REPETITIONS,
        "planned_calls": len(calls),
        "schedule_seed": SCHEDULE_SEED,
        "schedule_sha256": _schedule_sha256(calls),
        "schedule_design": (
            "exact Haiku K=3 schedule: for each repetition, shuffle bases; within "
            "each base, shuffle all five conditions; concurrency does not exceed two"
        ),
        "schedule_reference_protocol": "OPENROUTER_HAIKU_AGENTDOJO_K3_v1.0",
        "schedule_planner_sha256": _sha256_path(Path(haiku_runner.__file__).resolve()),
        "model": MODEL_ID,
        "provider": "openrouter",
        "pinned_upstream_provider": PINNED_PROVIDER,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "request_settings": request_settings(),
        "request_settings_sha256": _sha256_text(_canonical(request_settings())),
        "reasoning": {
            "effort": REASONING_EFFORT,
            "exclude": REASONING_EXCLUDE,
        },
        "max_output_tokens_per_call": MAX_TOKENS,
        "temperature_policy": TEMPERATURE_POLICY,
        "runner_sha256": _sha256_path(Path(__file__).resolve()),
        "raw_benchmark_text_in_manifest": False,
        "primary_estimand": (
            "base-equal marginal block-probability difference: condition minus original"
        ),
        "stability_diagnostic": (
            "within-condition repeated-call disagreement; not subtracted as a causal effect"
        ),
        "resume_policy": (
            "only valid status=ok records are complete; failed pilot or run records remain "
            "in the append-only artifact and are retried on the next invocation"
        ),
        "cost_projection": projected_cost(calls, prior_usage_path=prior_usage_path),
    }
    return {**core, "manifest_sha256": _sha256_text(_canonical(core))}


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError(f"Existing manifest is not an object: {path}")
        if existing.get("manifest_sha256") != manifest.get("manifest_sha256"):
            raise ValueError("Existing Luna OpenRouter manifest does not match current plan")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed_job_ids(
    records: Sequence[Mapping[str, object]],
    calls: Sequence[PlannedCall],
    *,
    cohort_sha256: str,
) -> set[str]:
    expected = {call.job_id: call for call in calls}
    latest: dict[str, Mapping[str, object]] = {}
    for record in records:
        job_id = str(record.get("job_id", ""))
        if job_id not in expected:
            raise ValueError(f"Output contains a job outside this frozen Luna plan: {job_id}")
        call = expected[job_id]
        provenance = {
            "protocol_version": PROTOCOL_VERSION,
            "cohort_sha256": cohort_sha256,
            "model": MODEL_ID,
            "prompt_sha256": prompt_spec(PROMPT_VERSION).sha256,
            "rendered_input_sha256": call.rendered_input_sha256,
            "request_sha256": call.request_sha256,
        }
        mismatches = [key for key, value in provenance.items() if record.get(key) != value]
        if mismatches:
            raise ValueError(
                f"Resume provenance mismatch for {job_id}: {', '.join(mismatches)}"
            )
        latest[job_id] = record
    return {
        job_id
        for job_id, record in latest.items()
        if record.get("status") == "ok"
    }


def _base_record(
    call: PlannedCall,
    *,
    cohort_sha256: str,
    attempts: int,
    started: float,
    concurrency: int,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "provider": "openrouter",
        "pinned_upstream_provider": PINNED_PROVIDER,
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_spec(PROMPT_VERSION).sha256,
        "cohort_sha256": cohort_sha256,
        "job_id": call.job_id,
        "schedule_index": call.schedule_index,
        "round_index": call.round_index,
        "block_id": call.block_id,
        "condition": call.condition,
        "variant": call.record_variant,
        "repetition": call.repetition,
        "base_sample_id": call.base_sample_id,
        "source_sample_id": call.source_sample_id,
        "domain": call.domain,
        "original_label": call.original_label,
        "rendered_input_sha256": call.rendered_input_sha256,
        "request_sha256": call.request_sha256,
        "request_settings_sha256": _sha256_text(_canonical(request_settings())),
        "reasoning_effort": REASONING_EFFORT,
        "reasoning_excluded_from_response": REASONING_EXCLUDE,
        "max_output_tokens": MAX_TOKENS,
        "temperature_policy": TEMPERATURE_POLICY,
        "attempts": attempts,
        "concurrency": concurrency,
        "latency_ms": (time.monotonic() - started) * 1000,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


async def _request_one(
    *,
    client: httpx.AsyncClient,
    api_key: str,
    call: PlannedCall,
    cohort_sha256: str,
    max_attempts: int,
    concurrency: int,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, object]:
    started = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "ToolSafe-Lab Luna shortcut audit",
                    "X-OpenRouter-Metadata": "enabled",
                },
                json=_request_body(call.rendered_input),
            )
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt < max_attempts:
                await sleep(min(8.0, 0.75 * 2 ** (attempt - 1)))
                continue
            return {
                **_base_record(
                    call,
                    cohort_sha256=cohort_sha256,
                    attempts=attempt,
                    started=started,
                    concurrency=concurrency,
                ),
                "status": "error",
                "error_type": "network_error",
                "terminal": False,
            }

        retryable = response.status_code in {
            408,
            409,
            425,
            429,
            500,
            502,
            503,
            504,
            524,
            529,
        }
        if response.is_error:
            if retryable and attempt < max_attempts:
                retry_after = response.headers.get("retry-after")
                try:
                    delay = (
                        float(retry_after)
                        if retry_after
                        else 0.75 * 2 ** (attempt - 1)
                    )
                except ValueError:
                    delay = 0.75 * 2 ** (attempt - 1)
                await sleep(min(15.0, max(0.0, delay)))
                continue
            try:
                error_data: object = response.json()
            except (json.JSONDecodeError, ValueError):
                error_data = None
            error_type, error_code = _provider_error(error_data)
            return {
                **_base_record(
                    call,
                    cohort_sha256=cohort_sha256,
                    attempts=attempt,
                    started=started,
                    concurrency=concurrency,
                ),
                "status": "error",
                "error_type": "http_error",
                "http_status": response.status_code,
                "provider_error_type": error_type,
                "provider_error_code": error_code,
                "terminal": not retryable,
            }

        try:
            data = response.json()
            assessment, prediction, strict_decision, metadata = _parse_response(data)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return {
                **_base_record(
                    call,
                    cohort_sha256=cohort_sha256,
                    attempts=attempt,
                    started=started,
                    concurrency=concurrency,
                ),
                "status": "error",
                "error_type": str(exc),
                "http_status": response.status_code,
                "terminal": True,
            }
        resolved_provider = metadata.get("resolved_provider")
        resolved_model = metadata.get("resolved_model")
        if not isinstance(resolved_model, str) or "gpt-5.6-luna" not in resolved_model.lower():
            return {
                **_base_record(
                    call,
                    cohort_sha256=cohort_sha256,
                    attempts=attempt,
                    started=started,
                    concurrency=concurrency,
                ),
                "status": "error",
                "error_type": "unexpected_resolved_model",
                "http_status": response.status_code,
                "resolved_model": resolved_model,
                "resolved_provider": resolved_provider,
                "terminal": True,
            }
        if resolved_provider is None:
            return {
                **_base_record(
                    call,
                    cohort_sha256=cohort_sha256,
                    attempts=attempt,
                    started=started,
                    concurrency=concurrency,
                ),
                "status": "error",
                "error_type": "missing_resolved_provider",
                "http_status": response.status_code,
                "resolved_model": metadata.get("resolved_model"),
                "terminal": True,
            }
        if PINNED_PROVIDER not in str(resolved_provider).lower():
            return {
                **_base_record(
                    call,
                    cohort_sha256=cohort_sha256,
                    attempts=attempt,
                    started=started,
                    concurrency=concurrency,
                ),
                "status": "error",
                "error_type": "unexpected_resolved_provider",
                "http_status": response.status_code,
                "resolved_model": metadata.get("resolved_model"),
                "resolved_provider": resolved_provider,
                "terminal": True,
            }
        return {
            **_base_record(
                call,
                cohort_sha256=cohort_sha256,
                attempts=attempt,
                started=started,
                concurrency=concurrency,
            ),
            "status": "ok",
            "terminal": True,
            "assessment": assessment,
            "prediction": prediction,
            "strict_decision": strict_decision,
            **metadata,
        }
    raise AssertionError("Unreachable request state")


def _record_cost(record: Mapping[str, object]) -> float:
    usage = record.get("usage")
    if not isinstance(usage, dict):
        return 0.0
    if isinstance(usage.get("cost"), (int, float)):
        return float(usage["cost"])
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(prompt_tokens, (int, float)) and isinstance(
        completion_tokens, (int, float)
    ):
        return (
            float(prompt_tokens) * INPUT_USD_PER_MTOK
            + float(completion_tokens) * OUTPUT_USD_PER_MTOK
        ) / 1_000_000
    return 0.0


def _observed_cost(records: Iterable[Mapping[str, object]]) -> float:
    return sum(_record_cost(record) for record in records)


async def run_openrouter_plan(
    *,
    calls: Sequence[PlannedCall],
    cohort_sha256: str,
    output_path: Path,
    api_key: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_calls: int | None = None,
    max_total_cost_usd: float = DEFAULT_COST_LIMIT_USD,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, object]]:
    if not 1 <= concurrency <= DEFAULT_CONCURRENCY:
        raise ValueError("Luna OpenRouter audit concurrency must be one or two")
    if not 1 <= max_attempts <= DEFAULT_MAX_ATTEMPTS:
        raise ValueError("Luna OpenRouter audit max attempts must be between one and three")
    if max_total_cost_usd <= 0:
        raise ValueError("Luna OpenRouter audit cost limit must be positive")

    previous = _repair_and_read_jsonl(output_path)
    completed = _completed_job_ids(previous, calls, cohort_sha256=cohort_sha256)
    pending = [call for call in calls if call.job_id not in completed]
    if max_calls is not None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        pending = pending[:max_calls]
    print(f"OpenRouter GPT-5.6 Luna: {len(completed)} cached, {len(pending)} pending this run")
    if not pending:
        return previous

    output_path.parent.mkdir(parents=True, exist_ok=True)
    owned_client = client is None
    if client is None:
        timeout = httpx.Timeout(connect=30.0, read=180.0, write=60.0, pool=30.0)
        limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)
        client = httpx.AsyncClient(timeout=timeout, limits=limits)

    queue: asyncio.Queue[PlannedCall] = asyncio.Queue()
    for call in pending:
        queue.put_nowait(call)
    lock = asyncio.Lock()
    new_records: list[dict[str, object]] = []
    observed_cost = _observed_cost(previous)
    stop_for_cost = observed_cost >= max_total_cost_usd

    async def worker() -> None:
        nonlocal observed_cost, stop_for_cost
        while True:
            async with lock:
                if stop_for_cost or queue.empty():
                    return
                call = queue.get_nowait()
            record = await _request_one(
                client=client,  # type: ignore[arg-type]
                api_key=api_key,
                call=call,
                cohort_sha256=cohort_sha256,
                max_attempts=max_attempts,
                concurrency=concurrency,
            )
            async with lock:
                new_records.append(record)
                observed_cost += _record_cost(record)
                with output_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                if observed_cost >= max_total_cost_usd:
                    stop_for_cost = True
                done = len(new_records)
                if done == 1 or done % 25 == 0 or done == len(pending):
                    valid = sum(row.get("status") == "ok" for row in new_records)
                    print(
                        f"  {done}/{len(pending)} completed; {valid} valid; "
                        f"provider-reported spend ${observed_cost:.4f}"
                    )

    try:
        await asyncio.gather(*(worker() for _ in range(concurrency)))
    finally:
        if owned_client:
            await client.aclose()
    if stop_for_cost:
        print(
            "Stopped before scheduling remaining calls at observed cost limit "
            f"${max_total_cost_usd:.2f}; rerun to resume only after reviewing spend."
        )
    return [*previous, *new_records]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Dry-run or execute the frozen repeated OpenRouter GPT-5.6 Luna audit"
    )
    value.add_argument("--cohort-file", type=Path, default=DEFAULT_COHORT_PATH)
    value.add_argument("--prior-usage-file", type=Path, default=DEFAULT_PRIOR_USAGE_PATH)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    value.add_argument(
        "--keys-file",
        type=Path,
        default=DEFAULT_KEYS_PATH,
    )
    value.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    value.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    value.add_argument("--max-calls", type=int)
    value.add_argument(
        "--max-total-cost-usd",
        type=float,
        default=DEFAULT_COST_LIMIT_USD,
    )
    value.add_argument(
        "--execute",
        action="store_true",
        help="Actually call OpenRouter; without this flag only the Luna manifest is prepared",
    )
    return value


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    cohort_path = args.cohort_file.resolve()
    prior_usage_path = args.prior_usage_file.resolve()
    output_path = args.output.resolve()
    examples, cohort_sha256 = load_frozen_agentdojo_cohort(cohort_path)
    calls = build_interleaved_plan(examples)
    manifest = build_manifest(
        calls=calls,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        prior_usage_path=prior_usage_path,
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    _write_manifest(manifest_path, manifest)
    cost = manifest["cost_projection"]
    assert isinstance(cost, dict)
    print(
        f"Prepared {len(calls)} Luna calls ({REPETITIONS} per condition across 64 bases); "
        f"expected list-price spend ${float(cost['expected_list_price_usd']):.2f}; "
        "single-attempt output-cap projection "
        f"${float(cost['single_attempt_output_cap_list_price_usd']):.2f}."
    )
    print(f"Manifest: {manifest_path}")
    if not args.execute:
        print("Dry run only: no API key was read and no provider request was made.")
        return

    api_key = load_openrouter_key(args.keys_file.resolve())
    asyncio.run(
        run_openrouter_plan(
            calls=calls,
            cohort_sha256=cohort_sha256,
            output_path=output_path,
            api_key=api_key,
            concurrency=args.concurrency,
            max_attempts=args.max_attempts,
            max_calls=args.max_calls,
            max_total_cost_usd=args.max_total_cost_usd,
        )
    )


if __name__ == "__main__":
    main()
