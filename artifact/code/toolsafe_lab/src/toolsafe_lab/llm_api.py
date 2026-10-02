from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import platform
import random
import re
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import httpx
import numpy as np

from toolsafe_lab.data import Sample, load_eval, load_training
from toolsafe_lab.experiment import QUALITY_FIELDS, SYSTEM_FIELDS
from toolsafe_lab.llm_prompt import (
    ASSESSMENT_SCHEMA,
    DEFAULT_PROMPT_VERSION,
    PromptSpec,
    prompt_spec,
    render_sample,
    ts_guard_composite_score,
)
from toolsafe_lab.metrics import binary_metrics


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    api_model: str
    reasoning_effort: str
    input_usd_per_mtok: float
    cached_input_usd_per_mtok: float
    cache_write_usd_per_mtok: float | None
    output_usd_per_mtok: float
    adaptive_thinking: bool = False


class ProviderResponseError(ValueError):
    def __init__(
        self,
        category: str,
        *,
        stop_reason: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        super().__init__(category)
        self.category = category
        self.stop_reason = stop_reason
        self.usage = usage or {}


MODEL_SPECS = {
    "gpt-5.6-terra": ModelSpec(
        "GPT-5.6 Terra",
        "openai",
        "gpt-5.6-terra",
        "medium",
        2.5,
        0.25,
        3.125,
        15.0,
    ),
    "gpt-5.6-luna": ModelSpec(
        "GPT-5.6 Luna",
        "openai",
        "gpt-5.6-luna",
        "medium",
        1.0,
        0.1,
        1.25,
        6.0,
    ),
    "gpt-5.6-sol": ModelSpec(
        "GPT-5.6 Sol",
        "openai",
        "gpt-5.6-sol",
        "medium",
        # Official promotional price recorded 2026-08-29.
        4.0,
        0.4,
        None,
        20.0,
    ),
    "gpt-5.5": ModelSpec(
        "GPT-5.5",
        "openai",
        "gpt-5.5",
        "medium",
        5.0,
        0.5,
        None,
        30.0,
    ),
    "claude-opus-4.8": ModelSpec(
        "Claude Opus 4.8",
        "anthropic",
        "claude-opus-4-8",
        "medium",
        5.0,
        0.5,
        None,
        25.0,
        adaptive_thinking=True,
    ),
    "claude-sonnet-5": ModelSpec(
        "Claude Sonnet 5",
        "anthropic",
        "claude-sonnet-5",
        "medium",
        # Introductory first-party price through 2026-08-31.
        2.0,
        0.2,
        None,
        10.0,
        adaptive_thinking=True,
    ),
    "claude-haiku-4.5": ModelSpec(
        "Claude Haiku 4.5",
        "anthropic",
        "claude-haiku-4-5-20251001",
        "unsupported",
        1.0,
        0.1,
        None,
        5.0,
    ),
}

# GPT-5.6 Sol was added only as an explicit post-hoc shortcut-audit replication.
# Preserve the pre-existing defaults for all broader hosted-evaluation commands.
DEFAULT_HOSTED_MODEL_IDS = tuple(
    model_id for model_id in MODEL_SPECS if model_id != "gpt-5.6-sol"
)


def load_api_keys(path: Path) -> dict[str, str]:
    aliases = {
        "openai": "openai",
        "openai_api_key": "openai",
        "claude": "anthropic",
        "anthropic": "anthropic",
        "anthropic_api_key": "anthropic",
    }
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pair: list[str] | None = None
        for separator in ("=", ":"):
            if separator in line:
                pair = line.split(separator, 1)
                break
        if pair is None:
            parts = line.split(None, 1)
            if len(parts) == 2:
                pair = parts
        if pair is None:
            continue
        raw_name, raw_value = pair
        normalized = raw_name.strip().lower()
        if normalized not in aliases:
            continue
        value = raw_value.strip().strip("\"'")
        if value:
            values[aliases[normalized]] = value
    return values


def select_samples(
    samples: Sequence[Sample],
    *,
    limit: int | None,
    seed: int,
) -> list[Sample]:
    if limit is None or limit >= len(samples):
        return list(samples)
    if limit <= 0:
        raise ValueError("Sample limit must be positive")

    buckets: dict[tuple[str, float], list[Sample]] = defaultdict(list)
    for sample in samples:
        buckets[(sample.source, sample.label)].append(sample)
    for key, bucket in buckets.items():
        bucket.sort(
            key=lambda sample: hashlib.sha256(
                f"{seed}:{key}:{sample.sample_id}".encode()
            ).hexdigest()
        )

    selected: list[Sample] = []
    positions = {key: 0 for key in buckets}
    keys = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            position = positions[key]
            bucket = buckets[key]
            if position >= len(bucket):
                continue
            selected.append(bucket[position])
            positions[key] += 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected


def _validate_assessment(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Assessment is not an object")
    expected = {
        "malicious_user_request",
        "third_party_attack",
        "current_action_harmfulness",
    }
    if set(value) != expected:
        raise ValueError("Assessment fields do not match the required schema")
    malicious = value["malicious_user_request"]
    attacked = value["third_party_attack"]
    harmfulness = value["current_action_harmfulness"]
    if not isinstance(malicious, bool) or not isinstance(attacked, bool):
        raise ValueError("Assessment booleans are invalid")
    if isinstance(harmfulness, bool):
        raise ValueError("Harmfulness rating is invalid")
    try:
        rating = float(harmfulness)
    except (TypeError, ValueError) as exc:
        raise ValueError("Harmfulness rating is invalid") from exc
    if rating not in {0.0, 0.5, 1.0}:
        raise ValueError("Harmfulness rating is outside the label set")
    return {
        "malicious_user_request": malicious,
        "third_party_attack": attacked,
        "current_action_harmfulness": rating,
    }


def _openai_request(spec: ModelSpec, prompt: PromptSpec, sample: Sample) -> dict[str, object]:
    return {
        "model": spec.api_model,
        "instructions": prompt.system,
        "input": render_sample(sample),
        "reasoning": {"effort": spec.reasoning_effort},
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "toolsafe_step_assessment",
                "strict": True,
                "schema": ASSESSMENT_SCHEMA,
            },
        },
        "max_output_tokens": 2048,
        "service_tier": "default",
        "store": False,
    }


def _anthropic_request(
    spec: ModelSpec, prompt: PromptSpec, sample: Sample
) -> dict[str, object]:
    output_config: dict[str, object] = {
        "format": {"type": "json_schema", "schema": ASSESSMENT_SCHEMA}
    }
    if spec.reasoning_effort != "unsupported":
        output_config["effort"] = spec.reasoning_effort
    request: dict[str, object] = {
        "model": spec.api_model,
        "max_tokens": 2048,
        "system": prompt.system,
        "messages": [{"role": "user", "content": render_sample(sample)}],
        "output_config": output_config,
    }
    if spec.adaptive_thinking:
        request["thinking"] = {"type": "adaptive"}
    return request


def _extract_openai(data: dict[str, object]) -> tuple[dict[str, object], dict[str, int], str]:
    usage_value = data.get("usage")
    usage = usage_value if isinstance(usage_value, dict) else {}
    input_details = usage.get("input_tokens_details")
    input_details = input_details if isinstance(input_details, dict) else {}
    output_details = usage.get("output_tokens_details")
    output_details = output_details if isinstance(output_details, dict) else {}
    normalized = {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "cached_input_tokens": int(input_details.get("cached_tokens", 0)),
        "cache_write_tokens": int(input_details.get("cache_write_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "reasoning_tokens": int(output_details.get("reasoning_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }
    status = str(data.get("status", "unknown"))
    if status != "completed":
        incomplete = data.get("incomplete_details")
        reason = (
            str(incomplete.get("reason"))
            if isinstance(incomplete, dict) and incomplete.get("reason")
            else status
        )
        raise ProviderResponseError(
            "incomplete_response", stop_reason=reason, usage=normalized
        )

    output = data.get("output")
    if not isinstance(output, list):
        raise ProviderResponseError("missing_output", usage=normalized)
    text: str | None = None
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "refusal":
                raise ProviderResponseError(
                    "refusal", stop_reason="refusal", usage=normalized
                )
            if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                text = str(block["text"])
                break
    if text is None:
        raise ProviderResponseError("missing_output_text", usage=normalized)
    try:
        assessment = _validate_assessment(json.loads(text))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProviderResponseError("invalid_structured_output", usage=normalized) from exc
    return assessment, normalized, status


def _extract_anthropic(
    data: dict[str, object],
) -> tuple[dict[str, object], dict[str, int], str]:
    stop_reason = str(data.get("stop_reason", "unknown"))
    usage_value = data.get("usage")
    usage = usage_value if isinstance(usage_value, dict) else {}
    input_tokens = int(usage.get("input_tokens", 0))
    cached = int(usage.get("cache_read_input_tokens", 0))
    cache_write = int(usage.get("cache_creation_input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    normalized = {
        "input_tokens": input_tokens + cached + cache_write,
        "cached_input_tokens": cached,
        "cache_write_tokens": cache_write,
        "output_tokens": output_tokens,
        "reasoning_tokens": 0,
        "total_tokens": input_tokens + cached + cache_write + output_tokens,
    }
    if stop_reason == "refusal":
        raise ProviderResponseError(
            "refusal", stop_reason=stop_reason, usage=normalized
        )
    if stop_reason == "max_tokens":
        raise ProviderResponseError(
            "max_tokens", stop_reason=stop_reason, usage=normalized
        )
    content = data.get("content")
    if not isinstance(content, list):
        raise ProviderResponseError(
            "missing_content", stop_reason=stop_reason, usage=normalized
        )
    text = next(
        (
            str(block["text"])
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ),
        None,
    )
    if text is None:
        raise ProviderResponseError(
            "missing_text", stop_reason=stop_reason, usage=normalized
        )
    try:
        assessment = _validate_assessment(json.loads(text))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProviderResponseError(
            "invalid_structured_output", stop_reason=stop_reason, usage=normalized
        ) from exc
    return assessment, normalized, stop_reason


def _estimate_cost(spec: ModelSpec, usage: dict[str, int]) -> float:
    cached = usage["cached_input_tokens"]
    cache_write = usage["cache_write_tokens"]
    total_input = usage["input_tokens"]
    uncached = max(0, total_input - cached - cache_write)
    cache_write_rate = spec.cache_write_usd_per_mtok or spec.input_usd_per_mtok
    input_cost = (
        uncached * spec.input_usd_per_mtok
        + cached * spec.cached_input_usd_per_mtok
        + cache_write * cache_write_rate
    ) / 1_000_000
    output_cost = usage["output_tokens"] * spec.output_usd_per_mtok / 1_000_000
    return input_cost + output_cost


def _error_record(
    *,
    spec: ModelSpec,
    prompt: PromptSpec,
    sample: Sample,
    attempts: int,
    started: float,
    run_id: str,
    concurrency: int,
    error_type: str,
    http_status: int | None = None,
    terminal: bool = True,
    provider_error_type: str | None = None,
    provider_error_code: str | None = None,
    stop_reason: str | None = None,
    usage: dict[str, int] | None = None,
    queued_at: float | None = None,
    run_label: str = "quality",
) -> dict[str, object]:
    finished = time.time()
    record: dict[str, object] = {
        "run_id": run_id,
        "run_label": run_label,
        "provider": spec.provider,
        "model": spec.name,
        "api_model": spec.api_model,
        "reasoning_effort": spec.reasoning_effort,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "sample_id": sample.sample_id,
        "dataset": sample.source,
        "split": sample.split,
        "gold_label": sample.label,
        "status": "error",
        "error_type": error_type,
        "http_status": http_status,
        "terminal": terminal,
        "provider_error_type": provider_error_type,
        "provider_error_code": provider_error_code,
        "stop_reason": stop_reason,
        "attempts": attempts,
        "concurrency": concurrency,
        "latency_ms": (finished - started) * 1000,
        "client_queue_ms": (started - queued_at) * 1000 if queued_at else 0.0,
        "timing_semantics": "api_round_trip_excluding_client_queue",
        "started_unix": started,
        "finished_unix": finished,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if usage:
        record["usage"] = usage
        record["estimated_cost_usd"] = _estimate_cost(spec, usage)
    return record


async def _request_one(
    *,
    client: httpx.AsyncClient,
    api_key: str,
    spec: ModelSpec,
    prompt: PromptSpec,
    sample: Sample,
    semaphore: asyncio.Semaphore,
    max_attempts: int,
    run_id: str,
    concurrency: int,
    run_label: str,
) -> dict[str, object]:
    queued_at = time.time()
    async with semaphore:
        started = time.time()
        for attempt in range(1, max_attempts + 1):
            try:
                if spec.provider == "openai":
                    response = await client.post(
                        "https://api.openai.com/v1/responses",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=_openai_request(spec, prompt, sample),
                    )
                else:
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                        },
                        json=_anthropic_request(spec, prompt, sample),
                    )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == max_attempts:
                    return _error_record(
                        spec=spec,
                        prompt=prompt,
                        sample=sample,
                        attempts=attempt,
                        started=started,
                        run_id=run_id,
                        concurrency=concurrency,
                        error_type=type(exc).__name__,
                        terminal=False,
                        queued_at=queued_at,
                        run_label=run_label,
                    )
                await asyncio.sleep(min(30.0, 0.75 * 2 ** (attempt - 1)) + random.random())
                continue

            retryable = response.status_code in {408, 409, 429} or response.status_code >= 500
            if response.is_error:
                if retryable and attempt < max_attempts:
                    retry_after = response.headers.get("retry-after")
                    try:
                        delay = float(retry_after) if retry_after else 0.75 * 2 ** (attempt - 1)
                    except ValueError:
                        delay = 0.75 * 2 ** (attempt - 1)
                    await asyncio.sleep(min(60.0, delay) + random.random())
                    continue
                provider_error_type = None
                provider_error_code = None
                try:
                    error_payload = response.json()
                    error_value = (
                        error_payload.get("error")
                        if isinstance(error_payload, dict)
                        else None
                    )
                    if isinstance(error_value, dict):
                        if error_value.get("type") is not None:
                            provider_error_type = str(error_value["type"])
                        if error_value.get("code") is not None:
                            provider_error_code = str(error_value["code"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                return _error_record(
                    spec=spec,
                    prompt=prompt,
                    sample=sample,
                    attempts=attempt,
                    started=started,
                    run_id=run_id,
                    concurrency=concurrency,
                    error_type="http_error",
                    http_status=response.status_code,
                    terminal=not retryable,
                    provider_error_type=provider_error_type,
                    provider_error_code=provider_error_code,
                    queued_at=queued_at,
                    run_label=run_label,
                )

            try:
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Provider response is not an object")
                if spec.provider == "openai":
                    assessment, usage, stop_reason = _extract_openai(data)
                else:
                    assessment, usage, stop_reason = _extract_anthropic(data)
                prediction = ts_guard_composite_score(
                    malicious_user_request=bool(assessment["malicious_user_request"]),
                    third_party_attack=bool(assessment["third_party_attack"]),
                    current_action_harmfulness=float(
                        assessment["current_action_harmfulness"]
                    ),
                )
            except ProviderResponseError as exc:
                return _error_record(
                    spec=spec,
                    prompt=prompt,
                    sample=sample,
                    attempts=attempt,
                    started=started,
                    run_id=run_id,
                    concurrency=concurrency,
                    error_type=exc.category,
                    http_status=response.status_code,
                    terminal=True,
                    stop_reason=exc.stop_reason,
                    usage=exc.usage,
                    queued_at=queued_at,
                    run_label=run_label,
                )
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                return _error_record(
                    spec=spec,
                    prompt=prompt,
                    sample=sample,
                    attempts=attempt,
                    started=started,
                    run_id=run_id,
                    concurrency=concurrency,
                    error_type=type(exc).__name__,
                    http_status=response.status_code,
                    terminal=True,
                    queued_at=queued_at,
                    run_label=run_label,
                )

            finished = time.time()
            return {
                "run_id": run_id,
                "run_label": run_label,
                "provider": spec.provider,
                "model": spec.name,
                "api_model": spec.api_model,
                "reasoning_effort": spec.reasoning_effort,
                "prompt_version": prompt.version,
                "prompt_sha256": prompt.sha256,
                "sample_id": sample.sample_id,
                "dataset": sample.source,
                "split": sample.split,
                "gold_label": sample.label,
                "status": "ok",
                "assessment": assessment,
                "prediction": prediction,
                "response_id": data.get("id"),
                "resolved_model": data.get("model"),
                "stop_reason": stop_reason,
                "attempts": attempt,
                "concurrency": concurrency,
                "latency_ms": (finished - started) * 1000,
                "client_queue_ms": (started - queued_at) * 1000,
                "timing_semantics": "api_round_trip_excluding_client_queue",
                "usage": usage,
                "estimated_cost_usd": _estimate_cost(spec, usage),
                "started_unix": started,
                "finished_unix": finished,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
    raise AssertionError("Unreachable request state")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    return records


async def evaluate_model(
    *,
    spec: ModelSpec,
    samples: Sequence[Sample],
    api_key: str,
    prompt: PromptSpec,
    output_path: Path,
    concurrency: int,
    max_attempts: int,
    run_label: str,
) -> list[dict[str, object]]:
    if concurrency <= 0:
        raise ValueError("Concurrency must be positive")
    previous = _read_jsonl(output_path)
    completed = {
        str(record["sample_id"])
        for record in previous
        if (record.get("status") == "ok" or record.get("terminal") is True)
        and record.get("prompt_sha256") == prompt.sha256
        and record.get("api_model") == spec.api_model
    }
    pending = [sample for sample in samples if sample.sample_id not in completed]
    print(
        f"{spec.name}: {len(completed & {sample.sample_id for sample in samples})} "
        f"cached, {len(pending)} pending"
    )
    if not pending:
        return previous

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    timeout = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=60.0)
    limits = httpx.Limits(
        max_connections=max(concurrency * 2, 4),
        max_keepalive_connections=max(concurrency, 2),
    )
    semaphore = asyncio.Semaphore(concurrency)
    new_records: list[dict[str, object]] = []
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [
            asyncio.create_task(
                _request_one(
                    client=client,
                    api_key=api_key,
                    spec=spec,
                    prompt=prompt,
                    sample=sample,
                    semaphore=semaphore,
                    max_attempts=max_attempts,
                    run_id=run_id,
                    concurrency=concurrency,
                    run_label=run_label,
                )
            )
            for sample in pending
        ]
        with output_path.open("a", encoding="utf-8") as handle:
            for index, task in enumerate(asyncio.as_completed(tasks), start=1):
                record = await task
                new_records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if index == 1 or index % max(1, min(25, len(tasks) // 10)) == 0:
                    successful = sum(item.get("status") == "ok" for item in new_records)
                    spend = sum(
                        float(item.get("estimated_cost_usd", 0)) for item in new_records
                    )
                    print(
                        f"  {index}/{len(tasks)} complete; {successful} valid; "
                        f"estimated spend ${spend:.4f}"
                    )
    return [*previous, *new_records]


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def summarize_model(
    *,
    spec: ModelSpec,
    samples: Sequence[Sample],
    records: Sequence[dict[str, object]],
    prompt: PromptSpec,
    run_label: str,
) -> dict[str, object]:
    expected = {sample.sample_id: sample for sample in samples}
    latest: dict[str, dict[str, object]] = {}
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if (
            sample_id in expected
            and record.get("prompt_sha256") == prompt.sha256
            and record.get("api_model") == spec.api_model
        ):
            latest[sample_id] = record
    valid = [record for record in latest.values() if record.get("status") == "ok"]

    quality_rows = []
    for dataset in sorted({sample.source for sample in samples}):
        dataset_samples = [sample for sample in samples if sample.source == dataset]
        dataset_valid = [
            latest[sample.sample_id]
            for sample in dataset_samples
            if sample.sample_id in latest and latest[sample.sample_id].get("status") == "ok"
        ]
        if not dataset_valid:
            continue
        labels = [int(float(record["gold_label"]) != 0.0) for record in dataset_valid]
        predictions = [
            int(float(record["prediction"]) != 0.0) for record in dataset_valid
        ]
        metrics = binary_metrics(
            labels, predictions, total_predictions=len(dataset_samples)
        )
        quality_rows.append(
            {
                "model": spec.name,
                "dataset": dataset,
                "evaluation_mode": "strict",
                "source": "api_measurement",
                **metrics,
            }
        )

    usage_fields = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    )
    usage = {
        field: sum(
            int(record.get("usage", {}).get(field, 0))
            for record in latest.values()
            if isinstance(record.get("usage"), dict)
        )
        for field in usage_fields
    }
    timed = [
        record
        for record in valid
        if record.get("timing_semantics") == "api_round_trip_excluding_client_queue"
        and record.get("latency_ms") is not None
    ]
    latencies = [float(record["latency_ms"]) for record in timed]
    queue_times = [float(record.get("client_queue_ms", 0)) for record in timed]
    cost = sum(
        float(record.get("estimated_cost_usd", 0)) for record in latest.values()
    )
    attempts = sum(int(record.get("attempts", 1)) for record in latest.values())
    run_spans: dict[str, list[float]] = defaultdict(list)
    for record in latest.values():
        run_id = str(record.get("run_id", "unknown"))
        run_spans[run_id].extend(
            [float(record.get("started_unix", 0)), float(record.get("finished_unix", 0))]
        )
    active_wall_seconds = sum(
        max(values) - min(values) for values in run_spans.values() if values
    )
    concurrency_values = sorted(
        {int(record.get("concurrency", 1)) for record in latest.values()}
    )
    system_row: dict[str, object] = {
        "model": spec.name,
        "variant": (
            f"zero-shot {prompt.version}; "
            f"{spec.reasoning_effort} reasoning effort; structured JSON"
        ),
        "source": "api_measurement",
        "hardware": (
            f"{spec.provider} hosted API; default service tier; "
            f"client {platform.system()} {platform.machine()}"
        ),
        "provider": spec.provider,
        "api_model": spec.api_model,
        "reasoning_effort": spec.reasoning_effort,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "run_label": run_label,
        "requests": len(latest),
        "successful_requests": len(valid),
        "failed_requests": len(latest) - len(valid),
        "request_coverage": len(valid) / len(samples) if samples else 0.0,
        "request_retries": max(0, attempts - len(latest)),
        "refused_requests": sum(
            record.get("error_type") == "refusal" for record in latest.values()
        ),
        "truncated_requests": sum(
            record.get("error_type") in {"max_tokens", "incomplete_response"}
            for record in latest.values()
        ),
        "http_error_requests": sum(
            record.get("error_type") == "http_error" for record in latest.values()
        ),
        "structured_output_error_requests": sum(
            record.get("error_type") == "invalid_structured_output"
            for record in latest.values()
        ),
        "concurrency": ",".join(str(value) for value in concurrency_values),
        "active_wall_seconds": active_wall_seconds,
        "wall_throughput_samples_s": (
            len(valid) / active_wall_seconds if active_wall_seconds > 0 else 0.0
        ),
        **usage,
        "estimated_cost_usd": cost,
        "cost_per_1k_predictions_usd": cost * 1000 / len(valid) if valid else math.nan,
        "timing_semantics": (
            "api_round_trip_excluding_client_queue"
            if valid
            and all(
                record.get("timing_semantics")
                == "api_round_trip_excluding_client_queue"
                for record in valid
            )
            else "legacy_completion_time_including_client_queue"
        ),
    }
    if latencies:
        system_row.update(
            {
                "single_n": len(latencies),
                "latency_ms_mean": statistics.mean(latencies),
                "latency_ms_p50": statistics.median(latencies),
                "latency_ms_p95": _percentile(latencies, 95),
                "latency_ms_p99": _percentile(latencies, 99),
                "client_queue_ms_mean": statistics.mean(queue_times),
                "client_queue_ms_p50": statistics.median(queue_times),
                "client_queue_ms_p95": _percentile(queue_times, 95),
            }
        )

    macro = {}
    for metric in ("accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc"):
        values = [float(row[metric]) for row in quality_rows]
        macro[metric] = statistics.mean(values) if values else None
    return {
        "model": spec.name,
        "provider": spec.provider,
        "api_model": spec.api_model,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "run_label": run_label,
        "requested_samples": len(samples),
        "successful_samples": len(valid),
        "quality_rows": quality_rows,
        "macro": macro,
        "system_row": system_row,
    }


def project_full_evaluation(
    *,
    spec: ModelSpec,
    selected_samples: Sequence[Sample],
    full_samples: Sequence[Sample],
    records: Sequence[dict[str, object]],
    prompt: PromptSpec,
    observed_throughput: float,
) -> dict[str, object]:
    expected = {sample.sample_id: sample for sample in selected_samples}
    latest: dict[str, dict[str, object]] = {}
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if (
            sample_id in expected
            and record.get("status") == "ok"
            and record.get("prompt_sha256") == prompt.sha256
            and record.get("api_model") == spec.api_model
        ):
            latest[sample_id] = record
    paired = [
        (sample, latest[sample.sample_id])
        for sample in selected_samples
        if sample.sample_id in latest
    ]
    if len(paired) < 2:
        return {"available": False, "reason": "fewer than two successful pilot calls"}

    input_lengths = np.asarray(
        [len(render_sample(sample)) for sample, _ in paired], dtype=float
    )
    input_tokens = np.asarray(
        [int(record["usage"]["input_tokens"]) for _, record in paired],  # type: ignore[index]
        dtype=float,
    )
    slope, intercept = np.polyfit(input_lengths, input_tokens, 1)
    full_lengths = np.asarray(
        [len(render_sample(sample)) for sample in full_samples], dtype=float
    )
    projected_input = float(np.maximum(intercept + slope * full_lengths, 0).sum())
    average_output = statistics.mean(
        int(record["usage"]["output_tokens"])  # type: ignore[index]
        for _, record in paired
    )
    average_reasoning = statistics.mean(
        int(record["usage"]["reasoning_tokens"])  # type: ignore[index]
        for _, record in paired
    )
    projected_output = average_output * len(full_samples)
    projected_reasoning = average_reasoning * len(full_samples)
    # Conservative standard-tier estimate: assume no prompt-cache discount.
    standard_cost = (
        projected_input * spec.input_usd_per_mtok
        + projected_output * spec.output_usd_per_mtok
    ) / 1_000_000
    wall_hours = (
        len(full_samples) / observed_throughput / 3600
        if observed_throughput > 0
        else None
    )
    return {
        "available": True,
        "pricing_as_of": "2026-07-23",
        "samples": len(full_samples),
        "pilot_successes": len(paired),
        "projected_input_tokens": round(projected_input),
        "projected_output_tokens": round(projected_output),
        "projected_reasoning_tokens": round(projected_reasoning),
        "projected_standard_api_cost_usd": standard_cost,
        "projected_batch_api_cost_usd": standard_cost * 0.5,
        "projected_direct_wall_hours_at_observed_throughput": wall_hours,
        "input_token_regression": {
            "tokens_per_input_character": float(slope),
            "fixed_tokens": float(intercept),
        },
        "assumptions": [
            "Input tokens are projected by a linear fit on pilot API usage.",
            "Output and reasoning tokens use the pilot per-request mean.",
            "Standard cost assumes no prompt-cache discount.",
            "Batch cost applies the providers' published 50% batch discount.",
            "Wall time extrapolates observed pilot throughput and is not an SLA.",
        ],
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fields: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def publish_summaries(results_root: Path, summaries: Sequence[dict[str, object]]) -> None:
    model_names = {str(summary["model"]) for summary in summaries}
    quality = [
        row
        for row in _read_csv(results_root / "predictive_quality.csv")
        if not (
            row.get("source") in {"api_measurement", "api_batch_measurement"}
            and row.get("model") in model_names
        )
    ]
    systems = [
        row
        for row in _read_csv(results_root / "systems_performance.csv")
        if not (
            row.get("source") in {"api_measurement", "api_batch_measurement"}
            and row.get("model") in model_names
        )
    ]
    for summary in summaries:
        quality.extend(summary["quality_rows"])  # type: ignore[arg-type]
        systems.append(summary["system_row"])  # type: ignore[arg-type]
    _write_csv(results_root / "predictive_quality.csv", quality, QUALITY_FIELDS)
    _write_csv(results_root / "systems_performance.csv", systems, SYSTEM_FIELDS)


def run_llm_evaluation(
    *,
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    keys_file: Path,
    model_ids: Sequence[str],
    split: str,
    limit: int | None,
    seed: int,
    concurrency: int,
    max_attempts: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    publish: bool = False,
    run_label: str = "quality",
) -> list[dict[str, object]]:
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", run_label) is None:
        raise ValueError("Run label must contain only lowercase letters, digits, _ or -")
    if split == "validation":
        all_samples = load_training(data_root, "validation")
    elif split == "eval":
        datasets = load_eval(data_root)
        all_samples = [sample for samples in datasets.values() for sample in samples]
    else:
        raise ValueError(f"Unsupported split: {split}")
    samples = select_samples(all_samples, limit=limit, seed=seed)
    if publish and (split != "eval" or len(samples) != len(all_samples)):
        raise ValueError("Publishing requires the complete evaluation split")
    if publish and run_label != "quality":
        raise ValueError("Only the quality run can be published")

    keys = load_api_keys(keys_file)
    prompt = prompt_spec(prompt_version)
    if split == "eval":
        projection_samples = all_samples
    else:
        projection_samples = [
            sample for dataset in load_eval(data_root).values() for sample in dataset
        ]
    summaries = []
    for model_id in model_ids:
        spec = MODEL_SPECS[model_id]
        try:
            api_key = keys[spec.provider]
        except KeyError as exc:
            raise ValueError(f"No {spec.provider} API key found in {keys_file}") from exc
        run_directory = artifacts_root / "api_runs" / split / prompt.version
        if run_label != "quality":
            run_directory = run_directory / run_label
        output_path = run_directory / f"{model_id}.jsonl"
        records = asyncio.run(
            evaluate_model(
                spec=spec,
                samples=samples,
                api_key=api_key,
                prompt=prompt,
                output_path=output_path,
                concurrency=concurrency,
                max_attempts=max_attempts,
                run_label=run_label,
            )
        )
        summary = summarize_model(
            spec=spec,
            samples=samples,
            records=records,
            prompt=prompt,
            run_label=run_label,
        )
        summary["full_eval_projection"] = project_full_evaluation(
            spec=spec,
            selected_samples=samples,
            full_samples=projection_samples,
            records=records,
            prompt=prompt,
            observed_throughput=float(
                summary["system_row"]["wall_throughput_samples_s"]  # type: ignore[index]
            ),
        )
        summaries.append(summary)
        macro = summary["macro"]
        system = summary["system_row"]
        print(
            f"  macro strict F1={float(macro['f1'] or 0) * 100:.2f}; "
            f"coverage={float(system['request_coverage']) * 100:.2f}%; "
            f"p50={float(system.get('latency_ms_p50', 0)):.1f} ms; "
            f"cost=${float(system['estimated_cost_usd']):.4f}"
        )

    projected_standard = sum(
        float(summary["full_eval_projection"].get("projected_standard_api_cost_usd", 0))  # type: ignore[union-attr]
        for summary in summaries
    )
    projected_batch = sum(
        float(summary["full_eval_projection"].get("projected_batch_api_cost_usd", 0))  # type: ignore[union-attr]
        for summary in summaries
    )
    print(
        f"Projected complete eval cost for selected models: "
        f"${projected_standard:.2f} standard or ${projected_batch:.2f} batch"
    )

    results_root.mkdir(parents=True, exist_ok=True)
    selection_name = "full" if len(samples) == len(all_samples) else f"n{len(samples)}"
    summary_filename = f"{split}-{selection_name}-{prompt.version}"
    if run_label != "quality":
        summary_filename += f"-{run_label}"
    summary_path = results_root / "llm_pilots" / f"{summary_filename}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "split": split,
                "selection": {
                    "selected": len(samples),
                    "available": len(all_samples),
                    "seed": seed,
                },
                "prompt_version": prompt.version,
                "prompt_sha256": prompt.sha256,
                "run_label": run_label,
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    if publish:
        publish_summaries(results_root, summaries)
        print("Published complete API measurements to the two result tables")
    return summaries
