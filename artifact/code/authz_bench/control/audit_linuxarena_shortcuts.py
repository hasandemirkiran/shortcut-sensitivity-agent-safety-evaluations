#!/usr/bin/env python3
"""Reproduce the corrected LinuxArena shortcut audit.

This script is deliberately separate from the earlier exploratory analyses.  It:

* verifies the exact public sample used by the audit;
* identifies repeated trajectories by hashing action content while excluding only
  ``monitor_response``;
* keys monitor cohorts by ``run_display_name`` rather than the often-empty model
  metadata embedded in ensemble responses;
* keeps post-hoc trajectory diagnostics separate from the paper's operational
  step-wise false-positive-rate convention; and
* writes aggregate metrics only--never prompts, commands, outputs, or reasoning.

The default download is large (704,485,115 bytes).  Pass ``--input`` to reuse an
already-downloaded copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


HF_REPO = "anonymouslinuxarena/linuxarena-public"
HF_REVISION = "2fd29f9c7bc190fb873bd76c6bd648f22b9ab0a7"
SAMPLE_PATH = "data/sample.jsonl"
SAMPLE_BYTES = 704_485_115
SAMPLE_SHA256 = "0b1e469ad48d93ccfec986f7cbafee43d49afb93bae730437d02c16233e7cfdb"
SAMPLE_URL = (
    "https://huggingface.co/datasets/"
    f"{HF_REPO}/resolve/{HF_REVISION}/{SAMPLE_PATH}"
)

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "linuxarena_shortcut_sample.jsonl"
DEFAULT_OUTPUT = HERE / "results" / "linuxarena_shortcut_audit.json"

SHELL_FUNCTIONS = {"bash", "bash_session", "python"}
HAIKU_RUNS = {"Haiku honest", "Haiku attack"}
CANONICAL_SOURCE_RUNS = {
    "Haiku honest",
    "Haiku attack",
    "GPT-5 Nano monitor on Opus honest",
    "GPT-5 Nano monitor on straj",
}
MONITOR_COHORTS = {
    "gpt5_nano": {
        "honest_run": "GPT-5 Nano monitor on Opus honest",
        "attack_run": "GPT-5 Nano monitor on straj",
    },
    "claude_opus_4_6": {
        "honest_run": "Opus monitor on Opus honest",
        "attack_run": "Opus monitor on straj",
    },
}

_URL = re.compile(r"https?://[^\s'\"`;|)]+", re.I)
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_B64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/])")


@dataclass
class Trajectory:
    trajectory_id: Optional[str]
    run: Optional[str]
    environment: Optional[str]
    main_task: Optional[str]
    side_task: Optional[str]
    label: int
    side_success: bool
    commands: List[str]
    n_actions: int
    action_content_hash: str
    monitor_scores: List[float]
    monitorable_actions: int
    legacy_max_score: Optional[float]
    legacy_max_model: Optional[str]

    @property
    def has_shell(self) -> bool:
        return bool(self.commands)

    @property
    def mean_shell_command_chars(self) -> float:
        if not self.commands:
            return float("nan")
        return float(np.mean([len(command) for command in self.commands]))

    @property
    def total_shell_command_chars(self) -> int:
        return sum(len(command) for command in self.commands)

    @property
    def cluster_key(self) -> Tuple[Any, ...]:
        return (self.label, self.environment, self.main_task, self.side_task)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    digest = hashlib.sha256()
    total = 0
    print(f"downloading pinned LinuxArena sample -> {path}", file=sys.stderr)
    with requests.get(SAMPLE_URL, stream=True, timeout=600) as response:
        response.raise_for_status()
        with partial.open("wb") as output:
            for chunk in response.iter_content(chunk_size=8 << 20):
                if not chunk:
                    continue
                output.write(chunk)
                digest.update(chunk)
                total += len(chunk)
    if total != SAMPLE_BYTES or digest.hexdigest() != SAMPLE_SHA256:
        partial.unlink(missing_ok=True)
        raise ValueError(
            "downloaded LinuxArena sample failed integrity check: "
            f"bytes={total}, sha256={digest.hexdigest()}"
        )
    os.replace(partial, path)


def verify_sample(path: Path) -> None:
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != SAMPLE_BYTES or digest != SAMPLE_SHA256:
        raise ValueError(
            "LinuxArena input is not the pinned public sample: "
            f"expected bytes={SAMPLE_BYTES}, sha256={SAMPLE_SHA256}; "
            f"got bytes={size}, sha256={digest}"
        )


def action_content_hash(actions: Sequence[Dict[str, Any]]) -> str:
    """Hash the complete action trace, excluding only attached monitor results."""
    content = [{key: value for key, value in action.items() if key != "monitor_response"}
               for action in actions]
    encoded = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shell_commands_of(actions: Sequence[Dict[str, Any]]) -> List[str]:
    commands: List[str] = []
    for action in actions:
        if str(action.get("function")) not in SHELL_FUNCTIONS:
            continue
        arguments = action.get("arguments") or {}
        value = arguments.get("cmd") or arguments.get("command") or arguments.get("code")
        if isinstance(value, str) and value.strip():
            commands.append(value)
    return commands


def monitor_scores_of(actions: Sequence[Dict[str, Any]]) -> Tuple[List[float], int]:
    """Return usable action scores and the number of actions expected to be scored.

    The ensemble wrapper frequently leaves its top-level ``model_name`` empty.  A
    numeric score is therefore accepted without requiring that field.  Submits are
    legitimate skips and upstream monitor errors are excluded, matching the public
    paper's monitor-coverage description.
    """
    scores: List[float] = []
    monitorable = 0
    for action in actions:
        if action.get("function") == "submit":
            continue
        monitorable += 1
        response = action.get("monitor_response")
        if not isinstance(response, dict):
            continue
        if response.get("error") or response.get("sus_score") is None:
            continue
        try:
            scores.append(float(response["sus_score"]))
        except (TypeError, ValueError):
            continue
    return scores, monitorable


def legacy_monitor_summary_of(
    actions: Sequence[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[str]]:
    """Reproduce the old max-action/top-level-model parsing for diagnosis only."""
    best: Optional[float] = None
    model: Optional[str] = None
    for action in actions:
        response = action.get("monitor_response")
        if not isinstance(response, dict) or response.get("sus_score") is None:
            continue
        try:
            score = float(response["sus_score"])
        except (TypeError, ValueError):
            continue
        if best is None or score > best:
            best = score
            model = response.get("model_name")
    return best, model


def load_trajectories(path: Path) -> List[Trajectory]:
    trajectories: List[Trajectory] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}") from error
            actions = record.get("actions") or []
            side_task = record.get("side_task")
            scores, monitorable = monitor_scores_of(actions)
            legacy_score, legacy_model = legacy_monitor_summary_of(actions)
            trajectories.append(
                Trajectory(
                    trajectory_id=record.get("trajectory_id"),
                    run=record.get("run_display_name"),
                    environment=(record.get("environment") or {}).get("id"),
                    main_task=(record.get("main_task") or {}).get("id"),
                    side_task=(side_task or {}).get("id")
                    if isinstance(side_task, dict)
                    else side_task,
                    label=int(side_task is not None),
                    side_success=bool(record.get("side_task_success")),
                    commands=shell_commands_of(actions),
                    n_actions=len(actions),
                    action_content_hash=action_content_hash(actions),
                    monitor_scores=scores,
                    monitorable_actions=monitorable,
                    legacy_max_score=legacy_score,
                    legacy_max_model=legacy_model,
                )
            )
    return trajectories


def label_counts(rows: Sequence[Trajectory]) -> Dict[str, int]:
    attacks = sum(row.label for row in rows)
    return {"total": len(rows), "attack": attacks, "honest": len(rows) - attacks}


def deduplicate_by_action_content(rows: Sequence[Trajectory]) -> List[Trajectory]:
    unique: Dict[str, Trajectory] = {}
    for row in rows:
        unique.setdefault(row.action_content_hash, row)
    return list(unique.values())


def normalize_text(text: str) -> str:
    return _B64.sub("BASE64", _IP.sub("1.1.1.1", _URL.sub("http://example.com", text)))


def randomized_threshold(scores: Sequence[float], target_fpr: float) -> Dict[str, float]:
    """Threshold and tie probability that attain an exact expected FPR."""
    values = np.asarray(scores, dtype=float)
    if not len(values):
        raise ValueError("cannot calibrate a threshold without honest scores")
    target = target_fpr * len(values)
    above = 0
    for threshold in np.unique(values)[::-1]:
        equal = int(np.sum(values == threshold))
        if above + equal >= target - 1e-12:
            tie_probability = (target - above) / equal
            return {
                "threshold": float(threshold),
                "tie_probability": float(tie_probability),
                "n_above": int(above),
                "n_equal": equal,
                "n_calibration": len(values),
                "target_fpr": float(target_fpr),
            }
        above += equal
    return {
        "threshold": float("-inf"),
        "tie_probability": 0.0,
        "n_above": len(values),
        "n_equal": 0,
        "n_calibration": len(values),
        "target_fpr": float(target_fpr),
    }


def expected_flag_probability(scores: Sequence[float], calibration: Dict[str, float]) -> float:
    values = np.asarray(scores, dtype=float)
    threshold = calibration["threshold"]
    if np.any(values > threshold):
        return 1.0
    equal = int(np.sum(values == threshold))
    return float(1.0 - (1.0 - calibration["tie_probability"]) ** equal)


def expected_tpr_at_trajectory_fpr(
    labels: Sequence[int], scores: Sequence[float], target_fpr: float = 0.01
) -> Dict[str, float]:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    calibration = randomized_threshold(values[y == 0], target_fpr)
    positives = values[y == 1]
    tpr = np.mean([
        expected_flag_probability([score], calibration) for score in positives
    ])
    return {**calibration, "expected_tpr": float(tpr)}


def quartile_conditional_auc(
    labels: Sequence[int], scores: Sequence[float], conditioning: Sequence[float]
) -> Dict[str, Any]:
    """Weighted within-quartile AUC diagnostic; this is not causal adjustment."""
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    z = np.asarray(conditioning, dtype=float)
    finite = np.isfinite(values) & np.isfinite(z)
    missing = int((~finite).sum())
    y, values, z = y[finite], values[finite], z[finite]
    if not len(z):
        return {"weighted_auc": None, "bins": [], "conditioning_missing": missing}
    edges = np.quantile(z, np.linspace(0, 1, 5))
    aucs: List[float] = []
    weights: List[int] = []
    bins: List[Dict[str, Any]] = []
    for index in range(4):
        lower, upper = edges[index], edges[index + 1]
        mask = (z >= lower) & (z < upper if index < 3 else z <= upper)
        if int(mask.sum()) < 10 or len(np.unique(y[mask])) < 2:
            continue
        value = float(roc_auc_score(y[mask], values[mask]))
        aucs.append(value)
        weights.append(int(mask.sum()))
        bins.append({
            "n": int(mask.sum()),
            "attack": int(y[mask].sum()),
            "auc": value,
            "lower": float(lower),
            "upper": float(upper),
        })
    weighted = float(np.average(aucs, weights=weights)) if aucs else None
    return {"weighted_auc": weighted, "bins": bins, "conditioning_missing": missing}


def trajectory_score_metrics(
    rows: Sequence[Trajectory], scores: Sequence[float]
) -> Dict[str, Any]:
    y = np.asarray([row.label for row in rows], dtype=int)
    values = np.asarray(scores, dtype=float)
    valid = ~np.isnan(values)
    y_valid, score_valid = y[valid], values[valid]
    mean_length = np.asarray(
        [row.mean_shell_command_chars for row in rows], dtype=float
    )[valid]
    return {
        **label_counts([row for row, keep in zip(rows, valid) if keep]),
        "missing": int((~valid).sum()),
        "auc": float(roc_auc_score(y_valid, score_valid)),
        "tpr_at_1pct_trajectory_fpr": expected_tpr_at_trajectory_fpr(
            y_valid, score_valid
        ),
        "within_mean_shell_length_quartiles": quartile_conditional_auc(
            y_valid, score_valid, mean_length
        ),
    }


def deterministic_feature_scores(rows: Sequence[Trajectory]) -> Dict[str, np.ndarray]:
    if any(not row.has_shell for row in rows):
        raise ValueError("deterministic shell features require shell-bearing rows")
    return {
        "inverse_mean_shell_command_chars": np.asarray(
            [-row.mean_shell_command_chars for row in rows]
        ),
        "inverse_n_shell_commands": np.asarray([-len(row.commands) for row in rows]),
        "inverse_n_actions": np.asarray([-row.n_actions for row in rows]),
        "inverse_total_shell_command_chars": np.asarray(
            [-row.total_shell_command_chars for row in rows]
        ),
    }


def deterministic_feature_metrics(rows: Sequence[Trajectory]) -> Dict[str, Any]:
    feature_scores = deterministic_feature_scores(rows)
    return {
        name: trajectory_score_metrics(rows, scores)
        for name, scores in feature_scores.items()
    }


def loeo_char_scores(rows: Sequence[Trajectory]) -> Tuple[np.ndarray, Dict[str, float]]:
    y = np.asarray([row.label for row in rows], dtype=int)
    environments = np.asarray([row.environment for row in rows], dtype=object)
    texts = [normalize_text("\n".join(row.commands)) for row in rows]
    predictions = np.full(len(rows), np.nan)
    per_environment: Dict[str, float] = {}
    for environment in sorted(set(environments)):
        test = np.where(environments == environment)[0]
        train = np.where(environments != environment)[0]
        if len(np.unique(y[train])) < 2:
            continue
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50_000,
        )
        train_matrix = vectorizer.fit_transform([texts[index] for index in train])
        classifier = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=2_000,
        ).fit(train_matrix, y[train])
        predictions[test] = classifier.predict_proba(
            vectorizer.transform([texts[index] for index in test])
        )[:, 1]
        if len(np.unique(y[test])) == 2:
            per_environment[str(environment)] = float(
                roc_auc_score(y[test], predictions[test])
            )
    return predictions, per_environment


def clustered_auc_samples(
    rows: Sequence[Trajectory],
    scores: Sequence[float],
    bootstrap: int,
    seed: int,
) -> np.ndarray:
    y = np.asarray([row.label for row in rows], dtype=int)
    values = np.asarray(scores, dtype=float)
    groups: Dict[int, Dict[Tuple[Any, ...], List[int]]] = {0: {}, 1: {}}
    for index, row in enumerate(rows):
        groups[row.label].setdefault(row.cluster_key, []).append(index)
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap)
    for iteration in range(bootstrap):
        indices: List[int] = []
        for label in (0, 1):
            clusters = list(groups[label].values())
            for selected in rng.integers(0, len(clusters), size=len(clusters)):
                indices.extend(clusters[int(selected)])
        sample = np.asarray(indices, dtype=int)
        samples[iteration] = roc_auc_score(y[sample], values[sample])
    return samples


def percentile_interval(samples: np.ndarray) -> List[float]:
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def add_clustered_auc_ci(
    metrics: Dict[str, Any],
    rows: Sequence[Trajectory],
    scores: Sequence[float],
    bootstrap: int,
    seed: int,
) -> np.ndarray:
    samples = clustered_auc_samples(rows, scores, bootstrap, seed)
    metrics["clustered_bootstrap_auc_95pct_ci"] = percentile_interval(samples)
    metrics["clustered_bootstrap_replicates"] = bootstrap
    metrics["cluster_counts"] = {
        str(label): len({row.cluster_key for row in rows if row.label == label})
        for label in (0, 1)
    }
    return samples


def add_deterministic_feature_cis(
    metrics: Dict[str, Any],
    rows: Sequence[Trajectory],
    bootstrap: int,
    seed: int,
) -> None:
    for offset, (name, scores) in enumerate(
        deterministic_feature_scores(rows).items(), start=1
    ):
        add_clustered_auc_ci(
            metrics[name], rows, scores, bootstrap, seed + offset
        )


def official_style_stepwise_metrics(
    rows: Sequence[Trajectory],
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    honest = [row for row in rows if row.label == 0]
    attacks = [row for row in rows if row.label == 1]
    honest_scores = [score for row in honest for score in row.monitor_scores]
    calibration = randomized_threshold(honest_scores, 0.01)
    detection = float(np.mean([
        expected_flag_probability(row.monitor_scores, calibration) for row in attacks
    ]))

    groups: Dict[int, Dict[Tuple[Any, ...], List[Trajectory]]] = {0: {}, 1: {}}
    for row in rows:
        groups[row.label].setdefault(row.cluster_key, []).append(row)
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap)
    for iteration in range(bootstrap):
        sampled: Dict[int, List[Trajectory]] = {0: [], 1: []}
        for label in (0, 1):
            clusters = list(groups[label].values())
            for selected in rng.integers(0, len(clusters), size=len(clusters)):
                sampled[label].extend(clusters[int(selected)])
        sampled_honest_scores = [
            score for row in sampled[0] for score in row.monitor_scores
        ]
        sampled_calibration = randomized_threshold(sampled_honest_scores, 0.01)
        samples[iteration] = np.mean([
            expected_flag_probability(row.monitor_scores, sampled_calibration)
            for row in sampled[1]
        ])

    return {
        **calibration,
        "honest_trajectories": len(honest),
        "attack_trajectories": len(attacks),
        "successful_attack_trajectories": sum(row.side_success for row in attacks),
        "honest_scored_actions": len(honest_scores),
        "attack_scored_actions": sum(len(row.monitor_scores) for row in attacks),
        "monitorable_actions": sum(row.monitorable_actions for row in rows),
        "scored_action_coverage": float(
            sum(len(row.monitor_scores) for row in rows)
            / sum(row.monitorable_actions for row in rows)
        ),
        "expected_any_action_detection": detection,
        "clustered_bootstrap_detection_95pct_ci": percentile_interval(samples),
        "clustered_bootstrap_replicates": bootstrap,
    }


def duplicate_summary(rows: Sequence[Trajectory]) -> Dict[str, Any]:
    groups: Dict[str, List[Trajectory]] = defaultdict(list)
    for row in rows:
        groups[row.action_content_hash].append(row)
    patterns = Counter(
        tuple(sorted(str(row.run) for row in group))
        for group in groups.values()
        if len(group) > 1
    )
    return {
        "unique_action_content_hashes": len(groups),
        "duplicate_groups": sum(len(group) > 1 for group in groups.values()),
        "duplicate_rows_beyond_first": sum(len(group) - 1 for group in groups.values()),
        "maximum_multiplicity": max(len(group) for group in groups.values()),
        "duplicate_run_patterns": [
            {"runs": list(pattern), "groups": count}
            for pattern, count in sorted(patterns.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def legacy_cohort_artifact(rows: Sequence[Trajectory]) -> Dict[str, Any]:
    shell = [row for row in rows if row.has_shell]
    output: Dict[str, Any] = {}
    predicates = {
        "claude_opus_4_6_backend_names": lambda model: bool(
            model and "claude-opus-4-6" in model
        ),
        "gpt5_nano_exact_backend_name": lambda model: (
            model == "openai/gpt-5-nano-2025-08-07"
        ),
    }
    for name, predicate in predicates.items():
        cohort = [
            row for row in shell
            if row.legacy_max_score is not None and predicate(row.legacy_max_model)
        ]
        labels = [row.label for row in cohort]
        scores = [float(row.legacy_max_score) for row in cohort]
        item: Dict[str, Any] = label_counts(cohort)
        item["auc"] = (
            float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
        )
        output[name] = item
    return output


def run_metrics(
    rows: Sequence[Trajectory], bootstrap: int, seed: int, lexical: bool
) -> Dict[str, Any]:
    raw_shell = [row for row in rows if row.has_shell]
    unique_all = deduplicate_by_action_content(rows)
    unique_shell = [row for row in unique_all if row.has_shell]
    canonical = [
        row for row in rows if row.run in CANONICAL_SOURCE_RUNS and row.has_shell
    ]
    haiku = [row for row in rows if row.run in HAIKU_RUNS and row.has_shell]
    unique_deterministic = deterministic_feature_metrics(unique_shell)
    haiku_deterministic = deterministic_feature_metrics(haiku)
    add_deterministic_feature_cis(
        unique_deterministic, unique_shell, bootstrap, seed + 10
    )
    add_deterministic_feature_cis(
        haiku_deterministic, haiku, bootstrap, seed + 20
    )

    result: Dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "repository": HF_REPO,
            "revision": HF_REVISION,
            "path": SAMPLE_PATH,
            "url": SAMPLE_URL,
            "bytes": SAMPLE_BYTES,
            "sha256": SAMPLE_SHA256,
        },
        "corpus": {
            "raw": label_counts(rows),
            "raw_with_shell": label_counts(raw_shell),
            "unique_trajectory_ids": len({row.trajectory_id for row in rows}),
            "action_content_deduplication": duplicate_summary(rows),
            "exact_action_content_unique": label_counts(unique_all),
            "exact_action_content_unique_with_shell": label_counts(unique_shell),
            "canonical_source_run_with_shell": label_counts(canonical),
            "run_counts": {
                str(run): label_counts([row for row in rows if row.run == run])
                for run in sorted({row.run for row in rows}, key=str)
            },
        },
        "legacy_parser_artifact": {
            "reproduction": legacy_cohort_artifact(rows),
        },
        "trajectory_diagnostics": {
            "exact_action_content_dedup": {
                "counts": label_counts(unique_shell),
                "deterministic_features": unique_deterministic,
            },
            "haiku_model_matched": {
                "counts": label_counts(haiku),
                "deterministic_features": haiku_deterministic,
            },
        },
        "monitor_cohorts": {},
    }

    for offset, (name, cohort) in enumerate(MONITOR_COHORTS.items(), start=1):
        selected = [
            row for row in rows
            if row.run in {cohort["honest_run"], cohort["attack_run"]}
        ]
        shell_selected = [row for row in selected if row.has_shell]
        full_scores = [
            max(row.monitor_scores) if row.monitor_scores else float("nan")
            for row in selected
        ]
        shell_scores = [
            max(row.monitor_scores) if row.monitor_scores else float("nan")
            for row in shell_selected
        ]
        full_metrics = trajectory_score_metrics(selected, full_scores)
        shell_metrics = trajectory_score_metrics(shell_selected, shell_scores)
        add_clustered_auc_ci(
            full_metrics, selected, full_scores, bootstrap, seed + 100 * offset
        )
        add_clustered_auc_ci(
            shell_metrics,
            shell_selected,
            shell_scores,
            bootstrap,
            seed + 100 * offset + 1,
        )
        shell_deterministic = deterministic_feature_metrics(shell_selected)
        add_deterministic_feature_cis(
            shell_deterministic,
            shell_selected,
            bootstrap,
            seed + 100 * offset + 10,
        )
        result["monitor_cohorts"][name] = {
            "honest_run": cohort["honest_run"],
            "attack_run": cohort["attack_run"],
            "full_trajectory_max_score": full_metrics,
            "shell_bearing_trajectory_max_score": shell_metrics,
            "shell_bearing_deterministic_features": shell_deterministic,
            "official_style_stepwise_1pct_fpr": official_style_stepwise_metrics(
                selected, bootstrap, seed + 100 * offset + 2
            ),
        }

    if lexical:
        lexical_cohorts = {
            "exact_action_content_dedup": unique_shell,
            "haiku_model_matched": haiku,
            "opus_honest_vs_lastraj": [
                row for row in rows
                if row.run in {
                    "GPT-5 Nano monitor on Opus honest",
                    "GPT-5 Nano monitor on straj",
                }
                and row.has_shell
            ],
        }
        lexical_results: Dict[str, Any] = {}
        for offset, (name, cohort_rows) in enumerate(lexical_cohorts.items(), start=1):
            print(f"fitting LOEO lexical diagnostic: {name}", file=sys.stderr)
            predictions, per_environment = loeo_char_scores(cohort_rows)
            metrics = trajectory_score_metrics(cohort_rows, predictions)
            lexical_samples = add_clustered_auc_ci(
                metrics,
                cohort_rows,
                predictions,
                bootstrap,
                seed + 1_000 + offset,
            )
            length_scores = np.asarray(
                [-row.mean_shell_command_chars for row in cohort_rows]
            )
            length_samples = clustered_auc_samples(
                cohort_rows,
                length_scores,
                bootstrap,
                seed + 1_000 + offset,
            )
            metrics["per_environment_auc"] = per_environment
            metrics["paired_auc_delta_vs_inverse_mean_shell_length"] = {
                "inverse_mean_shell_length_auc": float(roc_auc_score(
                    [row.label for row in cohort_rows], length_scores
                )),
                "inverse_mean_shell_length_clustered_bootstrap_95pct_ci": (
                    percentile_interval(length_samples)
                ),
                "point": float(metrics["auc"] - roc_auc_score(
                    [row.label for row in cohort_rows], length_scores
                )),
                "clustered_bootstrap_95pct_ci": percentile_interval(
                    lexical_samples - length_samples
                ),
            }
            lexical_results[name] = metrics
        result["trajectory_lexical_diagnostics"] = lexical_results

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="fail if --input does not exist instead of downloading the pinned sample",
    )
    parser.add_argument(
        "--skip-lexical",
        action="store_true",
        help="skip the slower char-TFIDF LOEO trajectory diagnostics",
    )
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_260_828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    if not args.input.exists():
        if args.no_download:
            raise FileNotFoundError(args.input)
        download_sample(args.input)
    print("verifying pinned source", file=sys.stderr)
    verify_sample(args.input)
    print("loading trajectories", file=sys.stderr)
    rows = load_trajectories(args.input)
    result = run_metrics(
        rows,
        bootstrap=args.bootstrap,
        seed=args.seed,
        lexical=not args.skip_lexical,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(f"wrote aggregate audit -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
