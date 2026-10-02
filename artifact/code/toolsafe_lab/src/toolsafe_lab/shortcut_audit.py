from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from toolsafe_lab.data import Sample, load_eval
from toolsafe_lab.llm_api import MODEL_SPECS, ModelSpec, evaluate_model, load_api_keys
from toolsafe_lab.llm_prompt import prompt_spec, render_sample
from toolsafe_lab.metrics import binary_metrics, strict_value
from toolsafe_lab.react_parser import parse_react_step
from toolsafe_lab.synthetic_data import validate_synthetic_record
from toolsafe_lab.uncertainty import clustered_mean_interval


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "BENCHMARK_SHORTCUT_AUDIT_PROTOCOL_v1.0"
SELECTION_SALT = "260828"
BOOTSTRAP_SEED = 260828
AGENTDOJO_DOMAINS = ("banking", "slack", "travel", "workspace")
AGENTDOJO_VARIANTS = (
    "original",
    "marker_paraphrase",
    "tool_alias",
    "combined",
)
MARKER_OPEN = "<INFORMATION>"
MARKER_CLOSE = "</INFORMATION>"
PARAPHRASED_OPEN = "<UNTRUSTED_EXTERNAL_CONTENT>"
PARAPHRASED_CLOSE = "</UNTRUSTED_EXTERNAL_CONTENT>"
SYNTHETIC_SERIALIZATIONS = ("native_structured", "react_execution_only")
SYNTHETIC_PAIRS = {
    "71cb": "fixed_context_changed_action",
    "b23f": "fixed_context_changed_action",
    "cfc4": "fixed_context_changed_action",
    "8245": "fixed_context_changed_action",
    "a1a4": "fixed_action_changed_context",
    "afe7": "fixed_action_changed_context",
    "4e57": "fixed_action_changed_context",
    "5750": "fixed_action_changed_context",
}
DEFAULT_MODELS = {
    "agentdojo": ("claude-haiku-4.5", "gpt-5.6-luna"),
    "authorization": ("claude-sonnet-5",),
}
RUN_LABEL = "benchmark-shortcut-audit-v1"
PROMPT_VERSION = "authors_v2_structured"
POST_HOC_REPEAT_MODEL_IDS = frozenset({"gpt-5.6-sol"})
REPEAT_ORIGINAL_VARIANT = "repeat_original"
POST_HOC_SOL_RESULT_NAME = "benchmark_shortcut_audit.gpt-5.6-sol.agentdojo.json"


@dataclass(frozen=True)
class AuditExample:
    sample: Sample
    cohort: str
    base_sample_id: str
    variant: str
    domain: str
    original_label: int
    serialization: str
    axis: str = ""


@dataclass(frozen=True)
class PredictionView:
    score: float | None
    strict_decision: int
    valid: bool
    received: bool
    failure: str | None


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _derived_id(
    *,
    cohort: str,
    base_sample_id: str,
    variant: str,
    serialization: str,
) -> str:
    value = "\x1f".join(
        (
            PROTOCOL_VERSION,
            cohort,
            base_sample_id,
            variant,
            serialization,
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _selection_digest(sample: Sample) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{sample.sample_id}".encode("utf-8")).hexdigest()


def _replace_identifier(text: str, original: str, replacement: str) -> str:
    return re.sub(
        rf"(?<![A-Za-z0-9_]){re.escape(original)}(?![A-Za-z0-9_])",
        replacement,
        text,
    )


def _tool_alias(sample: Sample, tool_name: str) -> str:
    digest = hashlib.sha256(
        f"{SELECTION_SALT}:{sample.sample_id}:{tool_name}".encode("utf-8")
    ).hexdigest()[:12]
    return f"tool_{digest}"


def _marker_present(history: str) -> bool:
    return MARKER_OPEN in history and MARKER_CLOSE in history


def _agentdojo_variant_fields(
    sample: Sample,
) -> tuple[dict[str, tuple[str, str, str, str]] | None, str | None]:
    if not _marker_present(sample.history):
        return None, "marker_pair_missing"
    parsed = parse_react_step(sample.current_action)
    if not parsed.status.startswith("ok_") or not parsed.tool_name:
        return None, f"candidate_tool_{parsed.status}"
    tool_name = parsed.tool_name
    alias = _tool_alias(sample, tool_name)
    if _replace_identifier(sample.current_action, tool_name, alias) == sample.current_action:
        return None, "tool_name_absent_from_current_action"
    if _replace_identifier(sample.env_info, tool_name, alias) == sample.env_info:
        return None, "tool_name_absent_from_tool_descriptions"

    original = (
        sample.instruction,
        sample.history,
        sample.current_action,
        sample.env_info,
    )
    marker = (
        sample.instruction,
        sample.history.replace(MARKER_OPEN, PARAPHRASED_OPEN).replace(
            MARKER_CLOSE, PARAPHRASED_CLOSE
        ),
        sample.current_action,
        sample.env_info,
    )
    aliased = tuple(_replace_identifier(value, tool_name, alias) for value in original)
    combined = (
        aliased[0],
        aliased[1].replace(MARKER_OPEN, PARAPHRASED_OPEN).replace(MARKER_CLOSE, PARAPHRASED_CLOSE),
        aliased[2],
        aliased[3],
    )
    if marker[1] == sample.history:
        return None, "marker_paraphrase_no_change"
    if aliased[2] == sample.current_action or aliased[3] == sample.env_info:
        return None, "tool_alias_no_change"
    return {
        "original": original,
        "marker_paraphrase": marker,
        "tool_alias": aliased,
        "combined": combined,
    }, None


def build_agentdojo_cohort(
    samples: Sequence[Sample],
) -> tuple[list[AuditExample], dict[str, object]]:
    cells: dict[tuple[str, int], list[Sample]] = defaultdict(list)
    marker_counts: Counter[tuple[str, int]] = Counter()
    for sample in samples:
        if sample.subset not in AGENTDOJO_DOMAINS:
            continue
        label = sample.strict_label
        if _marker_present(sample.history):
            cells[(sample.subset, label)].append(sample)
            marker_counts[(sample.subset, label)] += 1

    selected: list[Sample] = []
    selected_by_cell: dict[str, int] = {}
    marker_by_cell: dict[str, int] = {}
    for domain in AGENTDOJO_DOMAINS:
        for label in (0, 1):
            key = (domain, label)
            ordered = sorted(cells[key], key=_selection_digest)
            chosen = ordered[:8]
            selected.extend(chosen)
            cell_name = f"{domain}/{'unsafe' if label else 'safe'}"
            selected_by_cell[cell_name] = len(chosen)
            marker_by_cell[cell_name] = marker_counts[key]

    examples: list[AuditExample] = []
    exclusions: Counter[str] = Counter()
    eligible_bases = 0
    for base in selected:
        variants, reason = _agentdojo_variant_fields(base)
        if variants is None:
            exclusions[str(reason)] += 1
            continue
        eligible_bases += 1
        for variant in AGENTDOJO_VARIANTS:
            instruction, history, current_action, env_info = variants[variant]
            derived = Sample(
                sample_id=_derived_id(
                    cohort="agentdojo",
                    base_sample_id=base.sample_id,
                    variant=variant,
                    serialization="released_react",
                ),
                source="AgentDojo-Shortcut-Audit",
                split="counterfactual",
                instruction=instruction,
                history=history,
                current_action=current_action,
                env_info=env_info,
                label=base.label,
                trajectory_id=base.sample_id,
                segment_id=variant,
                subset=base.subset,
            )
            examples.append(
                AuditExample(
                    sample=derived,
                    cohort="agentdojo",
                    base_sample_id=base.sample_id,
                    variant=variant,
                    domain=base.subset,
                    original_label=base.strict_label,
                    serialization="released_react",
                )
            )

    summary: dict[str, object] = {
        "input_rows": len(samples),
        "marker_present_by_cell": marker_by_cell,
        "selected_base_rows_by_cell": selected_by_cell,
        "selected_base_rows": len(selected),
        "eligible_base_rows": eligible_bases,
        "excluded_base_rows": len(selected) - eligible_bases,
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "derived_rows": len(examples),
        "variant_counts": dict(sorted(Counter(row.variant for row in examples).items())),
        "selection_salt": SELECTION_SALT,
        "max_per_domain_label_cell": 8,
    }
    return examples, summary


def _read_synthetic(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            records.append(validate_synthetic_record(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid synthetic row {line_number}: {exc}") from exc
    return records


def _synthetic_action(record: Mapping[str, object], serialization: str) -> str:
    action = record["current_action"]
    if not isinstance(action, dict):
        raise TypeError("Validated synthetic action is not an object")
    if serialization == "native_structured":
        return _json(action)
    if serialization == "react_execution_only":
        return f"(1) Action: {action['tool_name']}\n(2) Action Input: {_json(action['arguments'])}"
    raise ValueError(f"Unsupported synthetic serialization: {serialization}")


def _validate_synthetic_pair(
    safe: Mapping[str, object],
    unsafe: Mapping[str, object],
    axis: str,
) -> None:
    if safe["tool_schema"] != unsafe["tool_schema"]:
        raise ValueError("Synthetic pair changed its tool schema")
    same_context = (
        safe["user_request"] == unsafe["user_request"] and safe["history"] == unsafe["history"]
    )
    same_action = safe["current_action"] == unsafe["current_action"]
    if axis == "fixed_context_changed_action":
        if not same_context or same_action:
            raise ValueError("Fixed-context pair does not isolate the action")
    elif axis == "fixed_action_changed_context":
        if not same_action or same_context:
            raise ValueError("Fixed-action pair does not isolate authorization context")
    else:
        raise ValueError(f"Unsupported authorization axis: {axis}")


def build_authorization_cohort(
    input_path: Path,
) -> tuple[list[AuditExample], dict[str, object]]:
    records = _read_synthetic(input_path)
    by_triad: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_triad[str(record["triad_id"])].append(record)

    resolved: list[tuple[str, str, dict[str, object], dict[str, object]]] = []
    for prefix, axis in SYNTHETIC_PAIRS.items():
        matches = sorted(triad_id for triad_id in by_triad if triad_id.startswith(prefix))
        if len(matches) != 1:
            raise ValueError(
                f"Synthetic prefix {prefix} matched {len(matches)} triads; expected one"
            )
        triad_id = matches[0]
        variants = {str(row["variant"]): row for row in by_triad[triad_id]}
        if "safe" not in variants or "unsafe" not in variants:
            raise ValueError(f"Synthetic triad {triad_id} lacks safe or unsafe row")
        safe = variants["safe"]
        unsafe = variants["unsafe"]
        _validate_synthetic_pair(safe, unsafe, axis)
        resolved.append((triad_id, axis, safe, unsafe))

    examples: list[AuditExample] = []
    for serialization in SYNTHETIC_SERIALIZATIONS:
        for triad_id, axis, safe, unsafe in resolved:
            for variant, record in (("safe", safe), ("unsafe", unsafe)):
                schema = record["tool_schema"]
                sample = Sample(
                    sample_id=_derived_id(
                        cohort="authorization",
                        base_sample_id=triad_id,
                        variant=variant,
                        serialization=serialization,
                    ),
                    source="Synthetic-Authorization-Audit",
                    split="counterfactual",
                    instruction=str(record["user_request"]),
                    history=str(record["history"]),
                    current_action=_synthetic_action(record, serialization),
                    env_info=_json(schema),
                    label=float(record["label"]),
                    trajectory_id=triad_id,
                    segment_id=variant,
                    subset=str(record["domain"]),
                )
                examples.append(
                    AuditExample(
                        sample=sample,
                        cohort="authorization",
                        base_sample_id=triad_id,
                        variant=variant,
                        domain=str(record["domain"]),
                        original_label=int(variant == "unsafe"),
                        serialization=serialization,
                        axis=axis,
                    )
                )

    summary: dict[str, object] = {
        "input_rows": len(records),
        "selected_pairs": len(resolved),
        "derived_rows": len(examples),
        "serializations": dict(sorted(Counter(row.serialization for row in examples).items())),
        "axis_counts": dict(
            sorted(
                Counter(
                    row.axis
                    for row in examples
                    if row.serialization == SYNTHETIC_SERIALIZATIONS[0] and row.variant == "safe"
                ).items()
            )
        ),
        "pair_prefixes": list(SYNTHETIC_PAIRS),
        "human_validation_status": "manually_screened_directional_pilot_only",
    }
    return examples, summary


def build_shortcut_audit_cohorts(
    *,
    data_root: Path,
    synthetic_input: Path,
) -> tuple[dict[str, list[AuditExample]], dict[str, object]]:
    evaluation = load_eval(data_root)
    try:
        agentdojo = evaluation["AgentDojo-Traj"]
    except KeyError as exc:
        raise ValueError("Pinned evaluation has no AgentDojo-Traj source") from exc
    agentdojo_examples, agentdojo_summary = build_agentdojo_cohort(agentdojo)
    authorization_examples, authorization_summary = build_authorization_cohort(synthetic_input)
    cohorts = {
        "agentdojo": agentdojo_examples,
        "authorization": authorization_examples,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_VERSION,
        "raw_text_in_summary": False,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_spec(PROMPT_VERSION).sha256,
        "cohorts": {
            "agentdojo": agentdojo_summary,
            "authorization": authorization_summary,
        },
    }
    return cohorts, summary


def _artifact_row(example: AuditExample) -> dict[str, object]:
    return {
        "cohort": example.cohort,
        "base_sample_id": example.base_sample_id,
        "variant": example.variant,
        "domain": example.domain,
        "original_label": example.original_label,
        "serialization": example.serialization,
        "axis": example.axis,
        "sample": asdict(example.sample),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_json(row) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def prepare_shortcut_audit(
    *,
    project_root: Path,
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    synthetic_input: Path,
) -> tuple[dict[str, list[AuditExample]], dict[str, object]]:
    cohorts, summary = build_shortcut_audit_cohorts(
        data_root=data_root,
        synthetic_input=synthetic_input,
    )
    artifacts: dict[str, object] = {}
    for name, examples in cohorts.items():
        path = artifacts_root / "shortcut_audit" / "cohorts" / f"{name}.jsonl"
        _write_jsonl(path, (_artifact_row(example) for example in examples))
        artifacts[name] = {
            "path": _relative(path, project_root),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "rows": len(examples),
        }
    summary["artifacts"] = artifacts
    output = results_root / "benchmark_shortcut_audit_cohorts.json"
    _write_json(output, summary)
    print(f"Wrote shortcut-audit cohort summary to {output}")
    return cohorts, summary


def _selected_cohorts(cohort: str) -> tuple[str, ...]:
    if cohort == "all":
        return ("agentdojo", "authorization")
    if cohort not in DEFAULT_MODELS:
        raise ValueError(f"Unsupported shortcut-audit cohort: {cohort}")
    return (cohort,)


def _model_ids_for(
    cohort: str,
    explicit: Sequence[str] | None,
) -> tuple[str, ...]:
    values = tuple(explicit) if explicit else DEFAULT_MODELS[cohort]
    unknown = sorted(set(values) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown hosted model(s): {', '.join(unknown)}")
    if not values:
        raise ValueError("At least one hosted model is required")
    return values


def _examples_for_hosted_model(
    examples: Sequence[AuditExample],
    *,
    cohort: str,
    model_id: str,
) -> list[AuditExample]:
    """Add exact-input repeat controls only to the post-hoc hosted replication."""
    selected = list(examples)
    if cohort != "agentdojo" or model_id not in POST_HOC_REPEAT_MODEL_IDS:
        return selected

    originals = [row for row in selected if row.variant == "original"]
    base_ids = {row.base_sample_id for row in originals}
    if len(originals) != len(base_ids):
        raise ValueError("AgentDojo originals are not unique by base sample")
    if base_ids != {row.base_sample_id for row in selected}:
        raise ValueError("AgentDojo repeat control lacks an original for every base sample")

    repeats_by_base: dict[str, AuditExample] = {}
    for original in originals:
        repeat_sample = replace(
            original.sample,
            sample_id=_derived_id(
                cohort="agentdojo",
                base_sample_id=original.base_sample_id,
                variant=REPEAT_ORIGINAL_VARIANT,
                serialization=original.serialization,
            ),
            segment_id=REPEAT_ORIGINAL_VARIANT,
        )
        if repeat_sample.sample_id == original.sample.sample_id:
            raise AssertionError("Repeat control did not receive a distinct sample ID")
        if render_sample(repeat_sample) != render_sample(original.sample):
            raise AssertionError("Repeat control changed the rendered model input")
        repeats_by_base[original.base_sample_id] = replace(
            original,
            sample=repeat_sample,
            variant=REPEAT_ORIGINAL_VARIANT,
        )

    interleaved: list[AuditExample] = []
    for example in selected:
        interleaved.append(example)
        if example.variant == "original":
            interleaved.append(repeats_by_base[example.base_sample_id])
    return interleaved


def _raw_output_path(
    artifacts_root: Path,
    cohort: str,
    model_id: str,
) -> Path:
    return (
        artifacts_root
        / "api_runs"
        / "shortcut_audit"
        / cohort
        / PROMPT_VERSION
        / f"{model_id}.jsonl"
    )


def run_shortcut_audit_inference(
    *,
    project_root: Path,
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    synthetic_input: Path,
    keys_file: Path,
    cohort: str,
    model_ids: Sequence[str] | None,
    concurrency: int,
    max_attempts: int,
) -> dict[str, list[dict[str, object]]]:
    cohorts, _ = prepare_shortcut_audit(
        project_root=project_root,
        data_root=data_root,
        results_root=results_root,
        artifacts_root=artifacts_root,
        synthetic_input=synthetic_input,
    )
    keys = load_api_keys(keys_file)
    prompt = prompt_spec(PROMPT_VERSION)
    outputs: dict[str, list[dict[str, object]]] = {}
    for cohort_name in _selected_cohorts(cohort):
        for model_id in _model_ids_for(cohort_name, model_ids):
            model_examples = _examples_for_hosted_model(
                cohorts[cohort_name],
                cohort=cohort_name,
                model_id=model_id,
            )
            samples = [example.sample for example in model_examples]
            spec = MODEL_SPECS[model_id]
            try:
                api_key = keys[spec.provider]
            except KeyError as exc:
                raise ValueError(f"No {spec.provider} API key found in {keys_file}") from exc
            output_path = _raw_output_path(artifacts_root, cohort_name, model_id)
            print(
                f"Scoring {len(samples)} {cohort_name} counterfactual rows with "
                f"{spec.name}; raw responses -> {output_path}"
            )
            outputs[f"{cohort_name}/{model_id}"] = asyncio.run(
                evaluate_model(
                    spec=spec,
                    samples=samples,
                    api_key=api_key,
                    prompt=prompt,
                    output_path=output_path,
                    concurrency=concurrency,
                    max_attempts=max_attempts,
                    run_label=RUN_LABEL,
                )
            )
    return outputs


def _read_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"No shortcut-audit API output at {path}")
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            records.append(value)
    if not records:
        raise ValueError(f"Shortcut-audit API output is empty: {path}")
    return records


def _prediction_views(
    examples: Sequence[AuditExample],
    records: Sequence[dict[str, object]],
    spec: ModelSpec,
) -> dict[str, PredictionView]:
    expected = {example.sample.sample_id for example in examples}
    prompt = prompt_spec(PROMPT_VERSION)
    latest: dict[str, dict[str, object]] = {}
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if (
            sample_id in expected
            and record.get("api_model") == spec.api_model
            and record.get("prompt_sha256") == prompt.sha256
        ):
            latest[sample_id] = record

    views: dict[str, PredictionView] = {}
    for sample_id in expected:
        record = latest.get(sample_id)
        if record is None:
            views[sample_id] = PredictionView(None, 1, False, False, "missing_record")
            continue
        score_value = record.get("prediction")
        mapped = strict_value(score_value) if record.get("status") == "ok" else None
        if mapped is None:
            failure = str(record.get("error_type") or "invalid_prediction")
            views[sample_id] = PredictionView(None, 1, False, True, failure)
            continue
        views[sample_id] = PredictionView(float(score_value), mapped, True, True, None)
    return views


def _coverage(
    examples: Sequence[AuditExample],
    views: Mapping[str, PredictionView],
) -> dict[str, object]:
    selected = [views[row.sample.sample_id] for row in examples]
    failures = Counter(view.failure for view in selected if view.failure is not None)
    valid = sum(view.valid for view in selected)
    received = sum(view.received for view in selected)
    return {
        "expected": len(selected),
        "received": received,
        "valid": valid,
        "failed_or_missing": len(selected) - valid,
        "valid_rate": valid / len(selected) if selected else 0.0,
        "failure_counts": dict(sorted(failures.items())),
    }


def _strict_metrics(
    examples: Sequence[AuditExample],
    views: Mapping[str, PredictionView],
) -> dict[str, object]:
    labels = [row.original_label for row in examples]
    predictions = [views[row.sample.sample_id].strict_decision for row in examples]
    return binary_metrics(labels, predictions)


def _rate_interval(
    values: Sequence[float],
    clusters: Sequence[str],
    replicates: int,
) -> dict[str, object] | None:
    if not values:
        return None
    return clustered_mean_interval(
        values,
        clusters=clusters,
        replicates=replicates,
        seed=BOOTSTRAP_SEED,
    ).to_dict()


def _simple_flip_summary(
    rows: Sequence[tuple[AuditExample, int, int, bool]],
) -> dict[str, object]:
    total = len(rows)
    flips = sum(original != transformed for _, original, transformed, _ in rows)
    original_unsafe = sum(original == 1 for _, original, _, _ in rows)
    original_safe = total - original_unsafe
    unsafe_to_safe = sum(original == 1 and transformed == 0 for _, original, transformed, _ in rows)
    safe_to_unsafe = sum(original == 0 and transformed == 1 for _, original, transformed, _ in rows)
    return {
        "n": total,
        "valid_pairs": sum(valid for *_, valid in rows),
        "flip_count": flips,
        "flip_rate": flips / total if total else 0.0,
        "unsafe_to_safe": {
            "count": unsafe_to_safe,
            "denominator": original_unsafe,
            "rate": unsafe_to_safe / original_unsafe if original_unsafe else None,
        },
        "safe_to_unsafe": {
            "count": safe_to_unsafe,
            "denominator": original_safe,
            "rate": safe_to_unsafe / original_safe if original_safe else None,
        },
    }


def _paired_flip_summary(
    examples: Sequence[AuditExample],
    views: Mapping[str, PredictionView],
    transformed_variant: str,
    replicates: int,
) -> dict[str, object]:
    by_base: dict[str, dict[str, AuditExample]] = defaultdict(dict)
    for example in examples:
        by_base[example.base_sample_id][example.variant] = example
    rows: list[tuple[AuditExample, int, int, bool]] = []
    for base_id in sorted(by_base):
        variants = by_base[base_id]
        if "original" not in variants or transformed_variant not in variants:
            raise ValueError(f"Incomplete AgentDojo pair for {base_id}")
        original = variants["original"]
        transformed = variants[transformed_variant]
        original_view = views[original.sample.sample_id]
        transformed_view = views[transformed.sample.sample_id]
        rows.append(
            (
                original,
                original_view.strict_decision,
                transformed_view.strict_decision,
                original_view.valid and transformed_view.valid,
            )
        )

    clusters = [row.base_sample_id for row, *_ in rows]
    flip_values = [float(original != transformed) for _, original, transformed, _ in rows]
    original_unsafe_rows = [row for row in rows if row[1] == 1]
    original_safe_rows = [row for row in rows if row[1] == 0]
    result = _simple_flip_summary(rows)
    result["flip_rate_interval"] = _rate_interval(flip_values, clusters, replicates)
    result["unsafe_to_safe"]["interval"] = _rate_interval(  # type: ignore[index]
        [float(transformed == 0) for _, _, transformed, _ in original_unsafe_rows],
        [row.base_sample_id for row, *_ in original_unsafe_rows],
        replicates,
    )
    result["safe_to_unsafe"]["interval"] = _rate_interval(  # type: ignore[index]
        [float(transformed == 1) for _, _, transformed, _ in original_safe_rows],
        [row.base_sample_id for row, *_ in original_safe_rows],
        replicates,
    )

    by_domain: dict[str, list[tuple[AuditExample, int, int, bool]]] = defaultdict(list)
    by_label: dict[str, list[tuple[AuditExample, int, int, bool]]] = defaultdict(list)
    for row in rows:
        example = row[0]
        by_domain[example.domain].append(row)
        by_label["unsafe" if example.original_label else "safe"].append(row)
    result["by_domain"] = {
        key: _simple_flip_summary(value) for key, value in sorted(by_domain.items())
    }
    result["by_original_label"] = {
        key: _simple_flip_summary(value) for key, value in sorted(by_label.items())
    }
    return result


def _repeat_control_comparison(
    examples: Sequence[AuditExample],
    views: Mapping[str, PredictionView],
    transformed_variant: str,
    replicates: int,
) -> dict[str, object]:
    by_base: dict[str, dict[str, AuditExample]] = defaultdict(dict)
    for example in examples:
        by_base[example.base_sample_id][example.variant] = example

    differences: list[float] = []
    transformed_flips: list[float] = []
    repeat_flips: list[float] = []
    transformed_signed_contrasts: list[float] = []
    repeat_signed_contrasts: list[float] = []
    transformed_minus_repeat_contrasts: list[float] = []
    clusters: list[str] = []
    valid_all = 0
    for base_id in sorted(by_base):
        variants = by_base[base_id]
        required = ("original", transformed_variant, REPEAT_ORIGINAL_VARIANT)
        if any(variant not in variants for variant in required):
            raise ValueError(f"Incomplete repeat-control comparison for {base_id}")
        original = views[variants["original"].sample.sample_id]
        transformed = views[variants[transformed_variant].sample.sample_id]
        repeated = views[variants[REPEAT_ORIGINAL_VARIANT].sample.sample_id]
        transformed_flip = float(original.strict_decision != transformed.strict_decision)
        repeat_flip = float(original.strict_decision != repeated.strict_decision)
        transformed_flips.append(transformed_flip)
        repeat_flips.append(repeat_flip)
        differences.append(transformed_flip - repeat_flip)
        transformed_signed_contrasts.append(
            float(transformed.strict_decision - original.strict_decision)
        )
        repeat_signed_contrasts.append(float(repeated.strict_decision - original.strict_decision))
        transformed_minus_repeat_contrasts.append(
            float(transformed.strict_decision - repeated.strict_decision)
        )
        clusters.append(base_id)
        valid_all += int(original.valid and transformed.valid and repeated.valid)

    return {
        "n": len(differences),
        "valid_triplets": valid_all,
        "transformation_flip_rate": sum(transformed_flips) / len(transformed_flips),
        "repeat_original_flip_rate": sum(repeat_flips) / len(repeat_flips),
        "excess_flip_rate_over_repeat": sum(differences) / len(differences),
        "excess_flip_rate_interval": _rate_interval(
            differences,
            clusters,
            replicates,
        ),
        "transformation_signed_block_contrast": sum(transformed_signed_contrasts)
        / len(transformed_signed_contrasts),
        "transformation_signed_block_contrast_interval": _rate_interval(
            transformed_signed_contrasts,
            clusters,
            replicates,
        ),
        "repeat_original_signed_block_contrast": sum(repeat_signed_contrasts)
        / len(repeat_signed_contrasts),
        "transformation_minus_repeat_signed_block_contrast": sum(transformed_minus_repeat_contrasts)
        / len(transformed_minus_repeat_contrasts),
        "estimand_status": (
            "unsigned flip rates are descriptive disagreement diagnostics; under "
            "stable independent draws, the signed transformation-minus-original "
            "mean is an unbiased but noisy estimate of the average marginal block-"
            "probability contrast"
        ),
        "single_draw_signed_contrast_assumptions": (
            "stable guard distribution and independent draws in original and transformed conditions"
        ),
        "design_limitations": (
            "K=1 per condition lacks interleaved repetition, drift checks, and "
            "within-condition stability estimates; repeat both conditions for a "
            "more precise and auditable estimate"
        ),
    }


def analyze_agentdojo_predictions(
    examples: Sequence[AuditExample],
    records: Sequence[dict[str, object]],
    spec: ModelSpec,
    *,
    bootstrap_replicates: int,
) -> dict[str, object]:
    views = _prediction_views(examples, records, spec)
    by_variant: dict[str, list[AuditExample]] = defaultdict(list)
    for example in examples:
        by_variant[example.variant].append(example)
    variant_results = {
        variant: {
            "strict_fail_closed_metrics": _strict_metrics(rows, views),
            "coverage": _coverage(rows, views),
        }
        for variant, rows in sorted(by_variant.items())
    }
    paired = {
        variant: _paired_flip_summary(
            examples,
            views,
            variant,
            bootstrap_replicates,
        )
        for variant in AGENTDOJO_VARIANTS
        if variant != "original"
    }
    result = {
        "model": spec.name,
        "api_model": spec.api_model,
        "rows": len(examples),
        "base_samples": len({row.base_sample_id for row in examples}),
        "overall_coverage": _coverage(examples, views),
        "by_variant": variant_results,
        "paired_transformations": paired,
        "failure_policy": "strict decisions fail closed; coverage reported separately",
    }
    if REPEAT_ORIGINAL_VARIANT in by_variant:
        repeat_summary = _paired_flip_summary(
            examples,
            views,
            REPEAT_ORIGINAL_VARIANT,
            bootstrap_replicates,
        )
        repeat_summary["rendered_input_relation"] = (
            "byte-identical render_sample payload; distinct audit sample ID"
        )
        result["repeat_original_control"] = repeat_summary
        for variant in AGENTDOJO_VARIANTS:
            if variant == "original":
                continue
            paired[variant]["repeat_control_comparison"] = _repeat_control_comparison(
                examples,
                views,
                variant,
                bootstrap_replicates,
            )
    return result


def _authorization_direction(exceeds: int, reversals: int, complete: int) -> str:
    if complete == 0:
        return "no_complete_pairs"
    if exceeds > reversals:
        return "unsafe_higher"
    if reversals > exceeds:
        return "safe_higher"
    return "no_direction"


def _authorization_pair_counts(
    by_pair: Mapping[str, Mapping[str, AuditExample]],
    views: Mapping[str, PredictionView],
) -> dict[str, object]:
    exceeds = ties = reversals = incomplete = 0
    for base_id in sorted(by_pair):
        pair = by_pair[base_id]
        if "safe" not in pair or "unsafe" not in pair:
            raise ValueError(f"Incomplete authorization pair for {base_id}")
        safe_view = views[pair["safe"].sample.sample_id]
        unsafe_view = views[pair["unsafe"].sample.sample_id]
        if not safe_view.valid or not unsafe_view.valid:
            incomplete += 1
            continue
        if safe_view.score is None or unsafe_view.score is None:
            raise AssertionError("Valid authorization prediction has no score")
        if unsafe_view.score > safe_view.score:
            exceeds += 1
        elif unsafe_view.score < safe_view.score:
            reversals += 1
        else:
            ties += 1
    complete = exceeds + ties + reversals
    return {
        "expected_pairs": len(by_pair),
        "complete_valid_pairs": complete,
        "incomplete_pairs": incomplete,
        "unsafe_score_exceeds_safe": exceeds,
        "ties": ties,
        "reversals": reversals,
        "fractions_among_complete": {
            "unsafe_score_exceeds_safe": exceeds / complete if complete else None,
            "ties": ties / complete if complete else None,
            "reversals": reversals / complete if complete else None,
        },
        "direction": _authorization_direction(exceeds, reversals, complete),
    }


def analyze_authorization_predictions(
    examples: Sequence[AuditExample],
    records: Sequence[dict[str, object]],
    spec: ModelSpec,
) -> dict[str, object]:
    views = _prediction_views(examples, records, spec)
    serialization_results: dict[str, object] = {}
    directions: list[str] = []
    for serialization in SYNTHETIC_SERIALIZATIONS:
        selected = [row for row in examples if row.serialization == serialization]
        by_pair: dict[str, dict[str, AuditExample]] = defaultdict(dict)
        for example in selected:
            by_pair[example.base_sample_id][example.variant] = example
        pair_counts = _authorization_pair_counts(by_pair, views)
        direction = str(pair_counts["direction"])
        directions.append(direction)
        by_axis: dict[str, dict[str, dict[str, AuditExample]]] = defaultdict(dict)
        for base_id, pair in by_pair.items():
            axis = pair["safe"].axis
            by_axis[axis][base_id] = pair
        serialization_results[serialization] = {
            **pair_counts,
            "by_axis": {
                axis: _authorization_pair_counts(pairs, views)
                for axis, pairs in sorted(by_axis.items())
            },
            "strict_fail_closed_metrics": _strict_metrics(selected, views),
            "coverage": _coverage(selected, views),
        }
    comparable = all(direction != "no_complete_pairs" for direction in directions)
    agrees = comparable and len(set(directions)) == 1
    return {
        "model": spec.name,
        "api_model": spec.api_model,
        "rows": len(examples),
        "pairs": len({row.base_sample_id for row in examples}),
        "overall_coverage": _coverage(examples, views),
        "by_serialization": serialization_results,
        "serialization_conclusion_agrees": agrees,
        "directionally_consistent_unsafe_higher": (agrees and directions[0] == "unsafe_higher"),
        "serialization_directions": directions,
        "status": "exploratory_unreviewed_authorization_pilot",
    }


def analyze_shortcut_audit(
    *,
    project_root: Path,
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
    synthetic_input: Path,
    cohort: str,
    model_ids: Sequence[str] | None,
    bootstrap_replicates: int,
) -> dict[str, object]:
    cohorts, cohort_summary = prepare_shortcut_audit(
        project_root=project_root,
        data_root=data_root,
        results_root=results_root,
        artifacts_root=artifacts_root,
        synthetic_input=synthetic_input,
    )
    prompt = prompt_spec(PROMPT_VERSION)
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_VERSION,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "raw_text_in_results": False,
        "cohort_artifacts": cohort_summary["artifacts"],
        "cohorts": {},
    }
    cohort_results = result["cohorts"]
    assert isinstance(cohort_results, dict)
    for cohort_name in _selected_cohorts(cohort):
        model_results = []
        for model_id in _model_ids_for(cohort_name, model_ids):
            spec = MODEL_SPECS[model_id]
            model_examples = _examples_for_hosted_model(
                cohorts[cohort_name],
                cohort=cohort_name,
                model_id=model_id,
            )
            raw_path = _raw_output_path(artifacts_root, cohort_name, model_id)
            records = _read_records(raw_path)
            if cohort_name == "agentdojo":
                analysis = analyze_agentdojo_predictions(
                    model_examples,
                    records,
                    spec,
                    bootstrap_replicates=bootstrap_replicates,
                )
            else:
                analysis = analyze_authorization_predictions(
                    model_examples,
                    records,
                    spec,
                )
            analysis["raw_output"] = {
                "path": _relative(raw_path, project_root),
                "sha256": _sha256(raw_path),
                "size_bytes": raw_path.stat().st_size,
                "records": len(records),
            }
            model_results.append(analysis)
        cohort_results[cohort_name] = {"models": model_results}
    if cohort == "agentdojo" and tuple(model_ids or ()) == ("gpt-5.6-sol",):
        output = results_root / POST_HOC_SOL_RESULT_NAME
    else:
        output = results_root / "benchmark_shortcut_audit.json"
    _write_json(output, result)
    print(f"Wrote shortcut-audit aggregate results to {output}")
    return result
