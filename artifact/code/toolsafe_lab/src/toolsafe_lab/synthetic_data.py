from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from toolsafe_lab.data import Sample, load_eval, load_training


SCHEMA_VERSION = 1
LABEL_BY_VARIANT = {"safe": 0.0, "controversial": 0.5, "unsafe": 1.0}
SPLITS = {"train", "validation"}
AUDIT_VALUES = {
    "authorization": {"authorized", "ambiguous", "unauthorized"},
    "history_provenance": {"trusted", "untrusted_benign", "untrusted_malicious"},
    "request_alignment": {"aligned", "ambiguous", "misaligned"},
    "consequence": {"benign", "sensitive", "harmful"},
}
REQUIRED_KEYS = {
    "schema_version",
    "record_id",
    "triad_id",
    "split",
    "domain",
    "category",
    "variant",
    "label",
    "user_request",
    "history",
    "current_action",
    "tool_schema",
    "audit",
    "provenance",
}
THOUGHT_PATTERN = re.compile(r"(?im)^\s*(?:thought|analysis)\s*:")


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_json_type(value: Any, expected: str, field: str) -> None:
    valid = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }
    if expected not in valid:
        raise ValueError(f"{field} uses unsupported JSON type {expected}")
    if not valid[expected]:
        raise ValueError(f"{field} does not match JSON type {expected}")


def validate_synthetic_record(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Synthetic record must be an object")
    if set(value) != REQUIRED_KEYS:
        missing = sorted(REQUIRED_KEYS - set(value))
        extra = sorted(set(value) - REQUIRED_KEYS)
        raise ValueError(f"Synthetic record keys differ: missing={missing}, extra={extra}")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported synthetic record schema version")
    for field in ("record_id", "triad_id", "domain", "category"):
        _require_string(value[field], field)
    split = _require_string(value["split"], "split")
    if split not in SPLITS:
        raise ValueError(f"Unsupported synthetic split: {split}")
    variant = _require_string(value["variant"], "variant")
    if variant not in LABEL_BY_VARIANT:
        raise ValueError(f"Unsupported counterfactual variant: {variant}")
    if isinstance(value["label"], bool):
        raise ValueError("Synthetic label cannot be boolean")
    try:
        label = float(value["label"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Synthetic label must be numeric") from exc
    if label != LABEL_BY_VARIANT[variant]:
        raise ValueError("Synthetic label does not match counterfactual variant")
    user_request = _require_string(value["user_request"], "user_request")
    history = _require_string(value["history"], "history")
    if THOUGHT_PATTERN.search(user_request) or THOUGHT_PATTERN.search(history):
        raise ValueError("Production inputs cannot contain generated Thought fields")

    action = value["current_action"]
    if not isinstance(action, dict) or set(action) != {"tool_name", "arguments"}:
        raise ValueError("current_action must contain tool_name and arguments")
    tool_name = _require_string(action["tool_name"], "current_action.tool_name")
    arguments = action["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError("current_action.arguments must be an object")

    schema = value["tool_schema"]
    if not isinstance(schema, dict) or set(schema) != {
        "name",
        "description",
        "parameters",
    }:
        raise ValueError("tool_schema must contain name, description, and parameters")
    if _require_string(schema["name"], "tool_schema.name") != tool_name:
        raise ValueError("Candidate action tool does not match tool schema")
    _require_string(schema["description"], "tool_schema.description")
    parameters = schema["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("tool_schema.parameters must be an object")
    properties = parameters.get("properties")
    required = parameters.get("required", [])
    if parameters.get("type") != "object" or not isinstance(properties, dict):
        raise ValueError("tool_schema.parameters must be an object JSON schema")
    if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
        raise ValueError("tool_schema required fields must be strings")
    if set(arguments) - set(properties):
        raise ValueError("Candidate action contains arguments absent from tool schema")
    if set(required) - set(arguments):
        raise ValueError("Candidate action omits a required argument")
    for key, argument in arguments.items():
        property_schema = properties[key]
        if not isinstance(property_schema, dict) or not isinstance(
            property_schema.get("type"), str
        ):
            raise ValueError(f"tool_schema property {key} lacks a JSON type")
        _validate_json_type(argument, property_schema["type"], f"argument {key}")

    audit = value["audit"]
    if not isinstance(audit, dict) or set(audit) != set(AUDIT_VALUES):
        raise ValueError("audit fields do not match the frozen schema")
    for field, allowed in AUDIT_VALUES.items():
        if audit[field] not in allowed:
            raise ValueError(f"Unsupported audit value for {field}")

    provenance = value["provenance"]
    expected_provenance = {
        "provider",
        "model",
        "prompt_version",
        "prompt_sha256",
        "request_id",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance:
        raise ValueError("provenance fields do not match the frozen schema")
    for field in expected_provenance:
        _require_string(provenance[field], f"provenance.{field}")
    prompt_sha = str(provenance["prompt_sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", prompt_sha):
        raise ValueError("provenance.prompt_sha256 is not a SHA-256 hex digest")
    return value


def synthetic_text(record: dict[str, object]) -> str:
    action = record["current_action"]
    schema = record["tool_schema"]
    return "\n".join(
        (
            "[USER_REQUEST]",
            str(record["user_request"]),
            "[INTERACTION_HISTORY]",
            str(record["history"]),
            "[CURRENT_ACTION]",
            json.dumps(action, ensure_ascii=False, sort_keys=True),
            "[TOOL_DESCRIPTIONS]",
            json.dumps(schema, ensure_ascii=False, sort_keys=True),
        )
    )


def _sample_text(sample: Sample) -> str:
    return sample.text


def normalized_fingerprint(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def benchmark_texts_without_labels(data_root: Path) -> list[str]:
    samples: list[Sample] = []
    for split in ("train", "validation"):
        samples.extend(load_training(data_root, split))
    for source_samples in load_eval(data_root).values():
        samples.extend(source_samples)
    return [_sample_text(sample) for sample in samples]


def maximum_benchmark_similarities(
    synthetic_texts: Sequence[str],
    benchmark_texts: Sequence[str],
    *,
    batch_size: int = 64,
) -> np.ndarray:
    if not synthetic_texts:
        return np.asarray([], dtype=float)
    if not benchmark_texts:
        return np.zeros(len(synthetic_texts), dtype=float)
    vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(5, 5),
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
    )
    benchmark = vectorizer.transform(benchmark_texts)
    maxima: list[np.ndarray] = []
    for start in range(0, len(synthetic_texts), batch_size):
        candidate = vectorizer.transform(synthetic_texts[start : start + batch_size])
        similarities = candidate @ benchmark.T
        maxima.append(np.asarray(similarities.max(axis=1).toarray()).ravel())
    return np.concatenate(maxima)


def validate_triads(records: Sequence[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["triad_id"]), []).append(record)
    invalid: dict[str, str] = {}
    for triad_id, rows in grouped.items():
        variants = Counter(str(row["variant"]) for row in rows)
        if variants != Counter(LABEL_BY_VARIANT.keys()):
            invalid[triad_id] = "triad must contain exactly one row per variant"
            continue
        invariants = {
            (
                str(row["split"]),
                str(row["domain"]),
                json.dumps(row["tool_schema"], sort_keys=True),
            )
            for row in rows
        }
        if len(invariants) != 1:
            invalid[triad_id] = "triad split, domain, or tool schema changed"
    return {
        "triad_count": len(grouped),
        "valid_triad_count": len(grouped) - len(invalid),
        "valid_fraction": (len(grouped) - len(invalid)) / len(grouped) if grouped else 0.0,
        "invalid": invalid,
    }


def _read_jsonl(path: Path) -> list[object]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_synthetic_file(
    input_path: Path,
    data_root: Path,
    *,
    near_duplicate_threshold: float = 0.90,
) -> dict[str, object]:
    raw = _read_jsonl(input_path)
    accepted_schema: list[dict[str, object]] = []
    schema_errors: list[dict[str, object]] = []
    for index, value in enumerate(raw):
        try:
            accepted_schema.append(validate_synthetic_record(value))
        except ValueError as exc:
            schema_errors.append({"line": index + 1, "error": str(exc)})
    texts = [synthetic_text(row) for row in accepted_schema]
    benchmark_texts = benchmark_texts_without_labels(data_root)
    benchmark_hashes = {normalized_fingerprint(text) for text in benchmark_texts}
    exact_benchmark = [
        index
        for index, text in enumerate(texts)
        if normalized_fingerprint(text) in benchmark_hashes
    ]
    seen: dict[str, int] = {}
    exact_synthetic: list[dict[str, int]] = []
    for index, text in enumerate(texts):
        fingerprint = normalized_fingerprint(text)
        if fingerprint in seen:
            exact_synthetic.append({"first": seen[fingerprint], "duplicate": index})
        else:
            seen[fingerprint] = index
    similarities = maximum_benchmark_similarities(texts, benchmark_texts)
    near_benchmark = [
        index
        for index, score in enumerate(similarities)
        if score >= near_duplicate_threshold
    ]
    rejected = set(exact_benchmark) | set(near_benchmark)
    rejected.update(pair["duplicate"] for pair in exact_synthetic)
    accepted = [
        row for index, row in enumerate(accepted_schema) if index not in rejected
    ]
    triad_audit = validate_triads(accepted)
    invalid_triads = set(triad_audit["invalid"])
    training_eligible = [
        row for row in accepted if str(row["triad_id"]) not in invalid_triads
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "input_rows": len(raw),
        "schema_valid_rows": len(accepted_schema),
        "schema_valid_fraction": len(accepted_schema) / len(raw) if raw else 0.0,
        "schema_errors": schema_errors,
        "exact_benchmark_duplicate_indices": exact_benchmark,
        "exact_synthetic_duplicates": exact_synthetic,
        "near_benchmark_duplicate_indices": near_benchmark,
        "near_duplicate_threshold": near_duplicate_threshold,
        "maximum_benchmark_similarity": {
            "max": float(similarities.max()) if len(similarities) else 0.0,
            "p95": float(np.percentile(similarities, 95)) if len(similarities) else 0.0,
        },
        "accepted_rows": len(accepted),
        "triads_after_filtering": triad_audit,
        "training_eligible_rows_before_human_review": len(training_eligible),
        "accepted_record_ids": [str(row["record_id"]) for row in accepted],
        "training_eligible_record_ids_before_human_review": [
            str(row["record_id"]) for row in training_eligible
        ],
    }


def write_synthetic_audit(
    input_path: Path,
    data_root: Path,
    results_root: Path,
) -> dict[str, object]:
    result = audit_synthetic_file(input_path, data_root)
    results_root.mkdir(parents=True, exist_ok=True)
    output_path = results_root / "synthetic_data_audit.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return result
