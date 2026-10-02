from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import httpx

from toolsafe_lab.data import Sample, load_eval
from toolsafe_lab.llm_api import (
    MODEL_SPECS,
    ModelSpec,
    ProviderResponseError,
    _anthropic_request,
    _estimate_cost,
    _extract_anthropic,
    _extract_openai,
    _openai_request,
    load_api_keys,
    publish_summaries,
    summarize_model,
)
from toolsafe_lab.llm_prompt import (
    DEFAULT_PROMPT_VERSION,
    PromptSpec,
    prompt_spec,
    ts_guard_composite_score,
)


TERMINAL_OPENAI = {"completed", "failed", "expired", "cancelled"}
TERMINAL_ANTHROPIC = {"ended"}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object in {path}")
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _headers(provider: str, api_key: str) -> dict[str, str]:
    if provider == "openai":
        return {"Authorization": f"Bearer {api_key}"}
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def _raise_http(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        message = response.text[:1_000]
        raise RuntimeError(
            f"Provider returned HTTP {response.status_code}: {message}"
        ) from exc


def _state_path(batch_root: Path, model_id: str) -> Path:
    return batch_root / f"{model_id}.state.json"


def _prepare_openai_input(
    path: Path,
    spec: ModelSpec,
    prompt: PromptSpec,
    samples: Sequence[Sample],
) -> None:
    _write_jsonl(
        path,
        (
            {
                "custom_id": sample.sample_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": _openai_request(spec, prompt, sample),
            }
            for sample in samples
        ),
    )


def _submit_openai(
    *,
    client: httpx.Client,
    api_key: str,
    spec: ModelSpec,
    prompt: PromptSpec,
    samples: Sequence[Sample],
    batch_root: Path,
) -> dict[str, object]:
    input_path = batch_root / f"{spec.api_model}.input.jsonl"
    _prepare_openai_input(input_path, spec, prompt, samples)
    with input_path.open("rb") as handle:
        upload = client.post(
            "https://api.openai.com/v1/files",
            headers=_headers("openai", api_key),
            data={"purpose": "batch"},
            files={"file": (input_path.name, handle, "application/jsonl")},
        )
    _raise_http(upload)
    file_data = upload.json()
    create = client.post(
        "https://api.openai.com/v1/batches",
        headers={
            **_headers("openai", api_key),
            "content-type": "application/json",
        },
        json={
            "input_file_id": file_data["id"],
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "metadata": {
                "description": "ToolSafe-Lab frozen TS-Bench evaluation",
                "model": spec.api_model,
                "prompt_sha256": prompt.sha256,
            },
        },
    )
    _raise_http(create)
    batch = create.json()
    return {
        "provider": "openai",
        "model": spec.name,
        "api_model": spec.api_model,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "samples": len(samples),
        "input_file_id": file_data["id"],
        "batch_id": batch["id"],
        "status": batch["status"],
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "provider_state": batch,
    }


def _submit_anthropic(
    *,
    client: httpx.Client,
    api_key: str,
    spec: ModelSpec,
    prompt: PromptSpec,
    samples: Sequence[Sample],
) -> dict[str, object]:
    requests = [
        {
            "custom_id": sample.sample_id,
            "params": _anthropic_request(spec, prompt, sample),
        }
        for sample in samples
    ]
    create = client.post(
        "https://api.anthropic.com/v1/messages/batches",
        headers=_headers("anthropic", api_key),
        json={"requests": requests},
    )
    _raise_http(create)
    batch = create.json()
    return {
        "provider": "anthropic",
        "model": spec.name,
        "api_model": spec.api_model,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "samples": len(samples),
        "batch_id": batch["id"],
        "status": batch["processing_status"],
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "provider_state": batch,
    }


def submit_batches(
    *,
    data_root: Path,
    artifacts_root: Path,
    keys_file: Path,
    model_ids: Sequence[str],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, object]]:
    evaluation = load_eval(data_root)
    samples = [sample for source_samples in evaluation.values() for sample in source_samples]
    prompt = prompt_spec(prompt_version)
    keys = load_api_keys(keys_file)
    batch_root = artifacts_root / "api_batches" / "eval" / prompt.version
    batch_root.mkdir(parents=True, exist_ok=True)
    states: list[dict[str, object]] = []
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        for model_id in model_ids:
            spec = MODEL_SPECS[model_id]
            state_path = _state_path(batch_root, model_id)
            if state_path.exists():
                state = _read_json(state_path)
                print(
                    f"{spec.name}: existing batch {state['batch_id']} "
                    f"status={state['status']}"
                )
                states.append(state)
                continue
            api_key = keys[spec.provider]
            print(f"{spec.name}: submitting {len(samples)} frozen evaluation requests")
            if spec.provider == "openai":
                state = _submit_openai(
                    client=client,
                    api_key=api_key,
                    spec=spec,
                    prompt=prompt,
                    samples=samples,
                    batch_root=batch_root,
                )
            else:
                state = _submit_anthropic(
                    client=client,
                    api_key=api_key,
                    spec=spec,
                    prompt=prompt,
                    samples=samples,
                )
            _write_json(state_path, state)
            print(f"  batch={state['batch_id']} status={state['status']}")
            states.append(state)
    return states


def refresh_batch_states(
    *,
    artifacts_root: Path,
    keys_file: Path,
    model_ids: Sequence[str],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, object]]:
    keys = load_api_keys(keys_file)
    prompt = prompt_spec(prompt_version)
    batch_root = artifacts_root / "api_batches" / "eval" / prompt.version
    states: list[dict[str, object]] = []
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        for model_id in model_ids:
            spec = MODEL_SPECS[model_id]
            state_path = _state_path(batch_root, model_id)
            if not state_path.exists():
                raise FileNotFoundError(f"No submitted batch for {model_id}")
            state = _read_json(state_path)
            api_key = keys[spec.provider]
            if spec.provider == "openai":
                response = client.get(
                    f"https://api.openai.com/v1/batches/{state['batch_id']}",
                    headers=_headers("openai", api_key),
                )
                _raise_http(response)
                provider_state = response.json()
                status = provider_state["status"]
            else:
                response = client.get(
                    f"https://api.anthropic.com/v1/messages/batches/{state['batch_id']}",
                    headers=_headers("anthropic", api_key),
                )
                _raise_http(response)
                provider_state = response.json()
                status = provider_state["processing_status"]
            state["status"] = status
            state["provider_state"] = provider_state
            state["refreshed_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(state_path, state)
            counts = provider_state.get("request_counts", {})
            print(f"{spec.name}: status={status} counts={counts}")
            states.append(state)
    return states


def _success_record(
    *,
    spec: ModelSpec,
    prompt: PromptSpec,
    sample: Sample,
    batch_id: str,
    response: dict[str, object],
) -> dict[str, object]:
    if spec.provider == "openai":
        assessment, usage, stop_reason = _extract_openai(response)
    else:
        assessment, usage, stop_reason = _extract_anthropic(response)
    prediction = ts_guard_composite_score(
        malicious_user_request=bool(assessment["malicious_user_request"]),
        third_party_attack=bool(assessment["third_party_attack"]),
        current_action_harmfulness=float(assessment["current_action_harmfulness"]),
    )
    return {
        "run_id": batch_id,
        "run_label": "quality-batch",
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
        "response_id": response.get("id"),
        "resolved_model": response.get("model"),
        "stop_reason": stop_reason,
        "attempts": 1,
        "concurrency": 0,
        "latency_ms": None,
        "client_queue_ms": 0.0,
        "timing_semantics": "batch_turnaround_not_online_latency",
        "usage": usage,
        "estimated_cost_usd": _estimate_cost(spec, usage) * 0.5,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def _failed_record(
    *,
    spec: ModelSpec,
    prompt: PromptSpec,
    sample: Sample,
    batch_id: str,
    error_type: str,
    details: object,
) -> dict[str, object]:
    return {
        "run_id": batch_id,
        "run_label": "quality-batch",
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
        "terminal": True,
        "batch_error": details,
        "attempts": 1,
        "concurrency": 0,
        "latency_ms": None,
        "timing_semantics": "batch_turnaround_not_online_latency",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def _convert_openai_results(
    *,
    lines: Sequence[dict[str, object]],
    spec: ModelSpec,
    prompt: PromptSpec,
    samples_by_id: dict[str, Sample],
    batch_id: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in lines:
        sample = samples_by_id[str(line["custom_id"])]
        response_wrapper = line.get("response")
        if (
            isinstance(response_wrapper, dict)
            and int(response_wrapper.get("status_code", 0)) == 200
            and isinstance(response_wrapper.get("body"), dict)
        ):
            try:
                records.append(
                    _success_record(
                        spec=spec,
                        prompt=prompt,
                        sample=sample,
                        batch_id=batch_id,
                        response=response_wrapper["body"],  # type: ignore[arg-type]
                    )
                )
            except ProviderResponseError as exc:
                records.append(
                    _failed_record(
                        spec=spec,
                        prompt=prompt,
                        sample=sample,
                        batch_id=batch_id,
                        error_type=exc.category,
                        details={"stop_reason": exc.stop_reason, "usage": exc.usage},
                    )
                )
        else:
            records.append(
                _failed_record(
                    spec=spec,
                    prompt=prompt,
                    sample=sample,
                    batch_id=batch_id,
                    error_type="batch_request_error",
                    details=line.get("error") or response_wrapper,
                )
            )
    return records


def _convert_anthropic_results(
    *,
    lines: Sequence[dict[str, object]],
    spec: ModelSpec,
    prompt: PromptSpec,
    samples_by_id: dict[str, Sample],
    batch_id: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in lines:
        sample = samples_by_id[str(line["custom_id"])]
        result = line.get("result")
        if (
            isinstance(result, dict)
            and result.get("type") == "succeeded"
            and isinstance(result.get("message"), dict)
        ):
            try:
                records.append(
                    _success_record(
                        spec=spec,
                        prompt=prompt,
                        sample=sample,
                        batch_id=batch_id,
                        response=result["message"],  # type: ignore[arg-type]
                    )
                )
            except ProviderResponseError as exc:
                records.append(
                    _failed_record(
                        spec=spec,
                        prompt=prompt,
                        sample=sample,
                        batch_id=batch_id,
                        error_type=exc.category,
                        details={"stop_reason": exc.stop_reason, "usage": exc.usage},
                    )
                )
        else:
            error_type = (
                str(result.get("type", "batch_request_error"))
                if isinstance(result, dict)
                else "batch_request_error"
            )
            records.append(
                _failed_record(
                    spec=spec,
                    prompt=prompt,
                    sample=sample,
                    batch_id=batch_id,
                    error_type=error_type,
                    details=result,
                )
            )
    return records


def _parse_jsonl(text: str) -> list[dict[str, object]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError("Batch result line is not an object")
        rows.append(value)
    return rows


def collect_batches(
    *,
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    keys_file: Path,
    model_ids: Sequence[str],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    publish: bool = False,
) -> list[dict[str, object]]:
    evaluation = load_eval(data_root)
    samples = [sample for source_samples in evaluation.values() for sample in source_samples]
    samples_by_id = {sample.sample_id: sample for sample in samples}
    prompt = prompt_spec(prompt_version)
    keys = load_api_keys(keys_file)
    batch_root = artifacts_root / "api_batches" / "eval" / prompt.version
    summaries: list[dict[str, object]] = []
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        for model_id in model_ids:
            spec = MODEL_SPECS[model_id]
            state = _read_json(_state_path(batch_root, model_id))
            status = str(state["status"])
            terminal = (
                status in TERMINAL_OPENAI
                if spec.provider == "openai"
                else status in TERMINAL_ANTHROPIC
            )
            if not terminal:
                print(f"{spec.name}: batch still {status}; skipping collection")
                continue
            provider_state = state["provider_state"]
            if not isinstance(provider_state, dict):
                raise TypeError("Provider batch state is invalid")
            api_key = keys[spec.provider]
            if spec.provider == "openai":
                if status != "completed":
                    raise RuntimeError(f"{spec.name} batch ended with status {status}")
                output_file_id = provider_state.get("output_file_id")
                if not output_file_id:
                    raise RuntimeError(f"{spec.name} batch has no output file")
                response = client.get(
                    f"https://api.openai.com/v1/files/{output_file_id}/content",
                    headers=_headers("openai", api_key),
                )
                _raise_http(response)
                lines = _parse_jsonl(response.text)
                records = _convert_openai_results(
                    lines=lines,
                    spec=spec,
                    prompt=prompt,
                    samples_by_id=samples_by_id,
                    batch_id=str(state["batch_id"]),
                )
            else:
                response = client.get(
                    f"https://api.anthropic.com/v1/messages/batches/{state['batch_id']}/results",
                    headers=_headers("anthropic", api_key),
                )
                _raise_http(response)
                lines = _parse_jsonl(response.text)
                records = _convert_anthropic_results(
                    lines=lines,
                    spec=spec,
                    prompt=prompt,
                    samples_by_id=samples_by_id,
                    batch_id=str(state["batch_id"]),
                )
            if len(records) != len(samples):
                print(
                    f"{spec.name}: collected {len(records)}/{len(samples)} results; "
                    "coverage will expose missing requests"
                )
            output_path = (
                artifacts_root
                / "api_runs"
                / "eval"
                / prompt.version
                / f"{model_id}.jsonl"
            )
            _write_jsonl(output_path, records)
            summary = summarize_model(
                spec=spec,
                samples=samples,
                records=records,
                prompt=prompt,
                run_label="quality-batch",
            )
            for row in summary["quality_rows"]:  # type: ignore[union-attr]
                row["source"] = "api_batch_measurement"
            system = summary["system_row"]
            system["source"] = "api_batch_measurement"  # type: ignore[index]
            system["variant"] = (  # type: ignore[index]
                f"zero-shot {prompt.version}; {spec.reasoning_effort} reasoning; "
                "structured JSON; provider Batch API"
            )
            system["timing_semantics"] = "batch_turnaround_not_online_latency"  # type: ignore[index]
            summaries.append(summary)
            print(
                f"{spec.name}: collected {summary['successful_samples']}/{len(samples)} "
                f"valid; batch cost=${float(system['estimated_cost_usd']):.2f}"  # type: ignore[index]
            )

    summary_path = results_root / "llm_eval" / "eval-full-batch.json"
    existing: dict[str, object] = {}
    if summary_path.exists():
        existing = _read_json(summary_path)
    by_model = {
        str(summary["api_model"]): summary
        for summary in existing.get("summaries", [])  # type: ignore[union-attr]
        if isinstance(summary, dict)
    }
    by_model.update({str(summary["api_model"]): summary for summary in summaries})
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "eval",
        "selection": {"selected": len(samples), "available": len(samples)},
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "run_label": "quality-batch",
        "summaries": list(by_model.values()),
    }
    _write_json(summary_path, payload)
    if publish and summaries:
        publish_summaries(results_root, list(by_model.values()))
    return summaries
