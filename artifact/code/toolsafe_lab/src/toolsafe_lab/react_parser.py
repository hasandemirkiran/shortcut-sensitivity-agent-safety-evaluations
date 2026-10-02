from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from toolsafe_lab.data import Sample, load_eval, load_training


MARKER_PREFIX = r"^[ \t]*(?:(?:\(\d+\)|\d+[.)])[ \t]*)?(?:\*{0,2})"
THOUGHT_MARKER = re.compile(
    MARKER_PREFIX + r"thought(?:\*{0,2})[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
ACTION_MARKER = re.compile(
    MARKER_PREFIX + r"action(?:\*{0,2})[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
ACTION_INPUT_MARKER = re.compile(
    MARKER_PREFIX + r"action[ \t_]+input(?:\*{0,2})[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class ParsedAction:
    status: str
    preamble: str
    thought: str
    tool_name: str
    argument_text: str
    normalized_arguments: str
    arguments_json: Any
    suffix: str
    thought_removed_text: str
    execution_only_text: str
    thought_marker_count: int
    action_marker_count: int
    action_input_marker_count: int
    thought_before_action: bool
    action_before_input: bool
    json_like_braces: bool
    argument_json_parseable: bool
    argument_json_exact: bool
    argument_json_object: bool


def _json_argument(value: str) -> tuple[bool, bool, Any, str, str]:
    candidate = value.strip()
    if not candidate:
        return False, False, None, "", ""

    if candidate.startswith("```"):
        opening_end = candidate.find("\n")
        closing_start = candidate.rfind("```")
        if opening_end >= 0 and closing_start > opening_end:
            fenced_value = candidate[opening_end + 1 : closing_start].strip()
            suffix = candidate[closing_start + 3 :].strip()
            try:
                parsed = json.loads(fenced_value)
            except (json.JSONDecodeError, TypeError):
                return False, False, None, candidate, ""
            normalized = json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            return True, not suffix, parsed, normalized, suffix

    leading = len(candidate) - len(candidate.lstrip())
    try:
        parsed, end = json.JSONDecoder().raw_decode(candidate.lstrip())
    except (json.JSONDecodeError, TypeError):
        return False, False, None, candidate, ""
    suffix = candidate[leading + end :].strip()
    normalized = json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return True, not suffix, parsed, normalized, suffix


def parse_react_step(text: str) -> ParsedAction:
    thoughts = list(THOUGHT_MARKER.finditer(text))
    actions = list(ACTION_MARKER.finditer(text))
    action_inputs = list(ACTION_INPUT_MARKER.finditer(text))
    action_input = action_inputs[0] if action_inputs else None
    preceding_actions = (
        [candidate for candidate in actions if candidate.start() < action_input.start()]
        if action_input is not None
        else []
    )
    action = preceding_actions[-1] if preceding_actions else (actions[0] if actions else None)
    preceding_thoughts = (
        [candidate for candidate in thoughts if candidate.start() < action.start()]
        if action is not None
        else []
    )
    thought = (
        preceding_thoughts[0]
        if preceding_thoughts
        else (thoughts[0] if thoughts else None)
    )

    boundary = thought or action or action_input
    preamble = text[: boundary.start()].strip() if boundary is not None else text.strip()

    rationale = ""
    if thought is not None and action is not None and thought.start() < action.start():
        rationale = text[thought.end() : action.start()].strip()
    elif thought is not None:
        rationale = text[thought.end() :].strip()

    tool_name = ""
    if action is not None:
        tool_suffix = text[action.end() :]
        tool_name = tool_suffix.splitlines()[0].strip() if tool_suffix else ""

    argument_text = ""
    if action_input is not None:
        argument_text = text[action_input.end() :].strip()

    json_parseable, json_exact, arguments_json, normalized, suffix = _json_argument(
        argument_text
    )
    if not json_parseable:
        normalized = argument_text

    action_before_input = (
        action is not None
        and action_input is not None
        and action.start() < action_input.start()
    )
    thought_before_action = (
        thought is not None and action is not None and thought.start() < action.start()
    )

    if action is None:
        status = "missing_action"
    elif action_input is None:
        status = "missing_action_input"
    elif not action_before_input:
        status = "out_of_order_markers"
    elif not tool_name:
        status = "empty_tool_name"
    elif not argument_text:
        status = "empty_arguments"
    elif json_parseable and isinstance(arguments_json, dict):
        status = "ok_json_object"
    elif json_parseable:
        status = "ok_json_value"
    else:
        status = "ok_text_arguments"

    thought_removed_text = text[action.start() :].strip() if action is not None else ""
    execution_parts = ["[TOOL_NAME]", tool_name, "[ARGUMENTS]", normalized]
    execution_only_text = "\n".join(execution_parts).strip()

    return ParsedAction(
        status=status,
        preamble=preamble,
        thought=rationale,
        tool_name=tool_name,
        argument_text=argument_text,
        normalized_arguments=normalized,
        arguments_json=arguments_json if json_parseable else None,
        suffix=suffix,
        thought_removed_text=thought_removed_text,
        execution_only_text=execution_only_text,
        thought_marker_count=len(thoughts),
        action_marker_count=len(actions),
        action_input_marker_count=len(action_inputs),
        thought_before_action=thought_before_action,
        action_before_input=action_before_input,
        json_like_braces="{" in text and "}" in text,
        argument_json_parseable=json_parseable,
        argument_json_exact=json_exact,
        argument_json_object=json_parseable and isinstance(arguments_json, dict),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_id(sample: Sample) -> str:
    provenance = "\x1f".join(
        (
            sample.split,
            sample.source,
            sample.subset,
            sample.trajectory_id,
            sample.segment_id,
            sample.sample_id,
        )
    )
    return hashlib.sha256(provenance.encode("utf-8")).hexdigest()[:24]


def _write_deterministic_jsonl_gz(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for row in rows:
                    text.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    text.write("\n")


def _derived_row(sample: Sample) -> dict[str, object]:
    parsed = parse_react_step(sample.current_action)
    return {
        "row_id": _row_id(sample),
        "sample_id": sample.sample_id,
        "source": sample.source,
        "split": sample.split,
        "trajectory_id": sample.trajectory_id,
        "segment_id": sample.segment_id,
        "subset": sample.subset,
        "label": sample.label,
        "original": {
            "instruction": sample.instruction,
            "history": sample.history,
            "current_action": sample.current_action,
            "env_info": sample.env_info,
        },
        "parsed_action": asdict(parsed),
    }


def _parser_profile(samples: Sequence[Sample]) -> dict[str, object]:
    parsed = [parse_react_step(sample.current_action) for sample in samples]
    statuses = Counter(item.status for item in parsed)
    row_ids = {_row_id(sample) for sample in samples}
    return {
        "row_count": len(samples),
        "unique_sample_ids": len({sample.sample_id for sample in samples}),
        "unique_row_ids": len(row_ids),
        "one_to_one": len(samples) == len(row_ids),
        "status_counts": dict(sorted(statuses.items())),
        "status_rates": {
            status: count / len(samples) if samples else 0.0
            for status, count in sorted(statuses.items())
        },
        "coverage": {
            "tool_name": sum(bool(item.tool_name) for item in parsed) / len(samples)
            if samples
            else 0.0,
            "argument_text": sum(bool(item.argument_text) for item in parsed)
            / len(samples)
            if samples
            else 0.0,
            "json_parseable": sum(item.argument_json_parseable for item in parsed)
            / len(samples)
            if samples
            else 0.0,
            "json_object": sum(item.argument_json_object for item in parsed) / len(samples)
            if samples
            else 0.0,
            "thought_removed": sum(bool(item.thought_removed_text) for item in parsed)
            / len(samples)
            if samples
            else 0.0,
            "execution_view_emitted": sum(
                bool(item.execution_only_text) for item in parsed
            )
            / len(samples)
            if samples
            else 0.0,
            "complete_action_and_input": sum(
                item.action_before_input and bool(item.tool_name)
                for item in parsed
            )
            / len(samples)
            if samples
            else 0.0,
            "parser_ok": sum(item.status.startswith("ok_") for item in parsed)
            / len(samples)
            if samples
            else 0.0,
        },
        "multiple_markers": {
            "thought": sum(item.thought_marker_count > 1 for item in parsed),
            "action": sum(item.action_marker_count > 1 for item in parsed),
            "action_input": sum(item.action_input_marker_count > 1 for item in parsed),
        },
    }


def _group_samples(samples: Iterable[Sample]) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[f"{sample.split}/{sample.source}"].append(sample)
        groups[f"{sample.split}/ALL"].append(sample)
    return dict(groups)


def run_react_parser(
    *,
    data_root: Path,
    results_root: Path,
    artifacts_root: Path,
) -> dict[str, object]:
    loaded_train = load_training(data_root, "train")
    loaded_validation = load_training(data_root, "validation")
    evaluation_by_source = load_eval(data_root)
    evaluation = [
        sample
        for source_samples in evaluation_by_source.values()
        for sample in source_samples
    ]
    evaluation_ids = {sample.sample_id for sample in evaluation}
    train = [
        sample for sample in loaded_train if sample.sample_id not in evaluation_ids
    ]
    validation = [
        sample
        for sample in loaded_validation
        if sample.sample_id not in evaluation_ids
    ]
    by_split = {
        "train": train,
        "validation": validation,
        "eval": evaluation,
    }

    artifact_records: dict[str, object] = {}
    for split, samples in by_split.items():
        path = artifacts_root / "derived" / "react" / f"{split}.jsonl.gz"
        _write_deterministic_jsonl_gz(
            path,
            (_derived_row(sample) for sample in samples),
        )
        artifact_records[split] = {
            "path": str(path.relative_to(artifacts_root.parent)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": len(samples),
        }

    all_samples = [*train, *validation, *evaluation]
    groups = _group_samples(all_samples)
    result = {
        "schema_version": 1,
        "scope": {
            "description": (
                "Deterministic one-to-one parsing of classifier rows into rationale, "
                "tool name, arguments, suffix, and execution-only representations."
            ),
            "raw_text_in_summary": False,
            "derived_artifacts_are_git_ignored": True,
            "parser_failure_policy": (
                "Preserve the row and emit an explicit status; never restore Thought "
                "text into the execution-only representation."
            ),
        },
        "excluded_evaluation_duplicates": {
            "train": len(loaded_train) - len(train),
            "validation": len(loaded_validation) - len(validation),
        },
        "artifacts": artifact_records,
        "groups": {
            key: _parser_profile(samples)
            for key, samples in sorted(groups.items())
        },
    }
    output_path = results_root / "react_parser_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote parser summary to {output_path}")
    return result
