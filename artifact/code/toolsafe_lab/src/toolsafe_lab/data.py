from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FIELD_ORDER = ("instruction", "history", "current_action", "env_info")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source: str
    split: str
    instruction: str
    history: str
    current_action: str
    env_info: str
    label: float
    trajectory_id: str = ""
    segment_id: str = ""
    subset: str = ""
    malicious_user_request: bool | None = None
    prompt_injection: bool | None = None

    @property
    def text(self) -> str:
        # Explicit field markers make the serialization inspectable and keep the
        # current action distinguishable from potentially injected history.
        return "\n".join(
            (
                "[USER_REQUEST]",
                self.instruction,
                "[INTERACTION_HISTORY]",
                self.history,
                "[CURRENT_ACTION]",
                self.current_action,
                "[TOOL_DESCRIPTIONS]",
                self.env_info,
            )
        )

    @property
    def strict_label(self) -> int:
        return int(self.label != 0.0)

    @property
    def loose_label(self) -> int:
        return int(self.label == 1.0)

    @property
    def exact_label(self) -> int:
        return {0.0: 0, 0.5: 1, 1.0: 2}[self.label]

    @property
    def auxiliary_labels(self) -> dict[str, int]:
        if self.malicious_user_request is None or self.prompt_injection is None:
            raise ValueError(
                f"Sample {self.sample_id} has no released auxiliary training labels"
            )
        return {
            "strict": self.strict_label,
            "harmfulness": self.exact_label,
            "malicious_user_request": int(self.malicious_user_request),
            "prompt_injection": int(self.prompt_injection),
        }


def _string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sample_id(fields: Iterable[str]) -> str:
    joined = "\x1f".join(fields)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:20]


def _from_eval(row: dict[str, object], source: str, subset: str) -> Sample:
    fields = tuple(_string(row.get(name)) for name in FIELD_ORDER)
    label = float(row["score"])
    interaction_id = _string(row.get("id-interaction"))
    segment_id = _string(row.get("id-segment"))
    return Sample(
        _sample_id(fields),
        source,
        "eval",
        *fields,
        label,
        trajectory_id=f"{source}:{subset}:{interaction_id}",
        segment_id=segment_id,
        subset=subset,
    )


def _from_training(
    row: dict[str, object], source: str, split: str, subset: str
) -> Sample:
    extra = row["extra_info"]
    reward = row["reward_model"]
    if not isinstance(extra, dict) or not isinstance(reward, dict):
        raise TypeError("Unexpected upstream training schema")
    truth = reward["ground_truth"]
    if not isinstance(truth, dict):
        raise TypeError("Unexpected upstream ground-truth schema")
    fields = (
        _string(extra.get("user_request")),
        _string(extra.get("history")),
        _string(extra.get("current_action")),
        _string(extra.get("env_info")),
    )
    label = float(truth["Harmfulness_Rating"])
    row_index = _string(extra.get("index"))
    return Sample(
        _sample_id(fields),
        source,
        split,
        *fields,
        label,
        trajectory_id=f"{source}:{subset}:{row_index}",
        segment_id=row_index,
        subset=subset,
        malicious_user_request=bool(truth["Malicious_User_Request"]),
        prompt_injection=bool(truth["Prompt_Injection"]),
    )


def _read_list(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"Expected a JSON list in {path}")
    return value


def load_eval(data_root: Path) -> dict[str, list[Sample]]:
    layout = _eval_layout(data_root)
    datasets: dict[str, list[Sample]] = {}
    for source, paths in layout.items():
        if not paths:
            raise FileNotFoundError(f"No files found for {source}; run `toolsafe-lab fetch`")
        datasets[source] = [
            _from_eval(row, source, path.stem)
            for path in paths
            for row in _read_list(path)
        ]
    return datasets


def _eval_layout(data_root: Path) -> dict[str, list[Path]]:
    return {
        "AgentHarm-Traj": sorted((data_root / "eval" / "agentharm").glob("*.json")),
        "ASB-Traj": sorted((data_root / "eval" / "asb").glob("*.json")),
        "AgentDojo-Traj": sorted((data_root / "eval" / "agentdojo").glob("*.json")),
    }


def load_eval_sample_ids(data_root: Path) -> set[str]:
    """Hash evaluation inputs without loading their labels."""
    sample_ids: set[str] = set()
    for source, paths in _eval_layout(data_root).items():
        if not paths:
            raise FileNotFoundError(f"No files found for {source}; run `toolsafe-lab fetch`")
        for path in paths:
            for row in _read_list(path):
                fields = tuple(_string(row.get(name)) for name in FIELD_ORDER)
                sample_ids.add(_sample_id(fields))
    return sample_ids


def load_training(data_root: Path, split: str) -> list[Sample]:
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported split: {split}")
    paths = sorted((data_root / split).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No {split} files; run `toolsafe-lab fetch`")
    samples: list[Sample] = []
    for path in paths:
        source = "ASB-Traj" if "_asb-" in path.name else "AgentAlign-Traj"
        samples.extend(
            _from_training(row, source, split, path.stem)
            for row in _read_list(path)
        )
    return deduplicate(samples)


def deduplicate(samples: Iterable[Sample]) -> list[Sample]:
    unique: dict[str, Sample] = {}
    for sample in samples:
        existing = unique.get(sample.sample_id)
        if existing is not None and existing.label != sample.label:
            raise ValueError(f"Conflicting labels for duplicate {sample.sample_id}")
        unique[sample.sample_id] = sample
    return list(unique.values())


def split_summary(samples: Iterable[Sample]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for sample in samples:
        bucket = summary.setdefault(
            sample.source, {"total": 0, "safe": 0, "controversial": 0, "unsafe": 0}
        )
        bucket["total"] += 1
        key = {0.0: "safe", 0.5: "controversial", 1.0: "unsafe"}[sample.label]
        bucket[key] += 1
    return summary
