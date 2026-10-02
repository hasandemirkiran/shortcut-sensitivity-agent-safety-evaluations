from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from toolsafe_lab.data import FIELD_ORDER, Sample, load_eval, load_training
from toolsafe_lab.react_parser import parse_react_step

EVAL_LAYOUT = {
    "AgentHarm-Traj": "eval/agentharm",
    "ASB-Traj": "eval/asb",
    "AgentDojo-Traj": "eval/agentdojo",
}


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _percentile(values: Sequence[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _length_summary(values: Sequence[str]) -> dict[str, float | int]:
    lengths = [len(value) for value in values]
    return _numeric_summary(lengths)


def _numeric_summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {
            "min": 0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0,
            "mean": 0.0,
        }
    return {
        "min": min(values),
        "p50": round(_percentile(values, 0.50), 6),
        "p95": round(_percentile(values, 0.95), 6),
        "max": max(values),
        "mean": round(sum(values) / len(values), 6),
    }


def action_shape(text: str) -> dict[str, bool | int]:
    parsed = parse_react_step(text)

    return {
        "thought_marker": parsed.thought_marker_count > 0,
        "action_marker": parsed.action_marker_count > 0,
        "action_input_marker": parsed.action_input_marker_count > 0,
        "thought_before_action": parsed.thought_before_action,
        "action_before_input": parsed.action_before_input,
        "json_like_braces": parsed.json_like_braces,
        "tool_name_nonempty": bool(parsed.tool_name),
        "argument_text_nonempty": bool(parsed.argument_text),
        "argument_json_exact": parsed.argument_json_exact,
        "argument_json_prefix": parsed.argument_json_parseable,
        "argument_json_object": parsed.argument_json_object,
        "tool_name_characters": len(parsed.tool_name),
        "argument_characters": len(parsed.argument_text),
    }


def _field_profile(samples: Sequence[Sample], field: str) -> dict[str, object]:
    raw_values = [getattr(sample, field) for sample in samples]
    values = [value if isinstance(value, str) else str(value) for value in raw_values]
    empty = sum(not value.strip() for value in values)
    type_counts = Counter(type(value).__name__ for value in raw_values)
    return {
        "post_loader_type_counts": dict(sorted(type_counts.items())),
        "empty_count": empty,
        "empty_rate": _rate(empty, len(samples)),
        "character_length": _length_summary(values),
    }


def audit_samples(samples: Sequence[Sample]) -> dict[str, object]:
    shapes = [action_shape(sample.current_action) for sample in samples]
    boolean_keys = (
        "thought_marker",
        "action_marker",
        "action_input_marker",
        "thought_before_action",
        "action_before_input",
        "json_like_braces",
        "tool_name_nonempty",
        "argument_text_nonempty",
        "argument_json_exact",
        "argument_json_prefix",
        "argument_json_object",
    )
    format_profile: dict[str, object] = {}
    for key in boolean_keys:
        count = sum(bool(shape[key]) for shape in shapes)
        format_profile[key] = {
            "count": count,
            "rate": _rate(count, len(samples)),
        }

    format_profile["tool_name_character_length"] = _numeric_summary(
        [int(shape["tool_name_characters"]) for shape in shapes]
    )
    format_profile["argument_character_length"] = _numeric_summary(
        [int(shape["argument_characters"]) for shape in shapes]
    )

    label_counts = Counter(str(sample.label) for sample in samples)
    return {
        "row_count": len(samples),
        "label_counts": dict(sorted(label_counts.items())),
        "fields": {
            field: _field_profile(samples, field)
            for field in FIELD_ORDER
        },
        "current_action_format": format_profile,
    }


def _read_list(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"Expected JSON list in {path}")
    return value


def _raw_type_profiles(data_root: Path) -> dict[str, object]:
    profiles: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {field: Counter() for field in FIELD_ORDER}
    )
    row_counts: Counter[str] = Counter()

    for source, relative_directory in EVAL_LAYOUT.items():
        for path in sorted((data_root / relative_directory).glob("*.json")):
            for row in _read_list(path):
                key = f"eval/{source}"
                row_counts[key] += 1
                for field in FIELD_ORDER:
                    profiles[key][field][type(row.get(field)).__name__] += 1

    for split in ("train", "validation"):
        for path in sorted((data_root / split).glob("*.json")):
            source = "ASB-Traj" if "_asb-" in path.name else "AgentAlign-Traj"
            key = f"{split}/{source}"
            for row in _read_list(path):
                extra = row.get("extra_info")
                if not isinstance(extra, dict):
                    raise TypeError(f"Unexpected extra_info schema in {path}")
                row_counts[key] += 1
                raw_fields = {
                    "instruction": extra.get("user_request"),
                    "history": extra.get("history"),
                    "current_action": extra.get("current_action"),
                    "env_info": extra.get("env_info"),
                }
                for field, value in raw_fields.items():
                    profiles[key][field][type(value).__name__] += 1

    return {
        key: {
            "raw_row_count": row_counts[key],
            "field_type_counts": {
                field: dict(sorted(counts.items()))
                for field, counts in sorted(fields.items())
            },
        }
        for key, fields in sorted(profiles.items())
    }


def _group_samples(samples: Iterable[Sample]) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[f"{sample.split}/{sample.source}"].append(sample)
        groups[f"{sample.split}/ALL"].append(sample)
        groups["all/ALL"].append(sample)
    return dict(groups)


def run_representation_audit(
    *,
    data_root: Path,
    results_root: Path,
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
    all_samples = [*train, *validation, *evaluation]
    groups = _group_samples(all_samples)
    profiles = {
        key: audit_samples(samples)
        for key, samples in sorted(groups.items())
    }

    result = {
        "schema_version": 1,
        "scope": {
            "description": (
                "Aggregate audit of the normalized, deduplicated rows consumed by "
                "ToolSafe-Lab classifiers; raw source types are reported separately."
            ),
            "field_order": list(FIELD_ORDER),
            "current_action_interpretation": (
                "Released candidate-step text, potentially containing ReAct Thought, "
                "Action, and Action Input sections; not assumed to be a typed API call."
            ),
            "raw_text_included": False,
        },
        "classifier_row_counts": {
            "train": len(train),
            "validation": len(validation),
            "eval": len(evaluation),
            "all": len(all_samples),
        },
        "excluded_evaluation_duplicates": {
            "train": len(loaded_train) - len(train),
            "validation": len(loaded_validation) - len(validation),
        },
        "raw_source_profiles": _raw_type_profiles(data_root),
        "groups": profiles,
    }

    output_path = results_root / "representation_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote representation audit to {output_path}")
    return result
