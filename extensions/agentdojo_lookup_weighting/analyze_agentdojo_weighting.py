"""Reproduce AgentDojo LODO results and also evaluate interaction weighting"""
import hashlib
import json
import sys
import tempfile
import urllib.request
import random
from collections import defaultdict
from pathlib import Path
from statistics import quantiles


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(tempfile.mkdtemp(prefix="lookup-coverage-"))
sys.path.insert(0, str(PROJECT / "artifact/code/toolsafe_lab/src"))
from toolsafe_lab.manifest import ASSETS, EVAL_COMMIT
from toolsafe_lab.data import _from_eval
from toolsafe_lab.react_parser import parse_react_step
from sklearn.model_selection import GroupKFold
from interaction_weighting import (
    interaction_weighted_metrics,
    interaction_majority_metrics,
    metrics_from_counts,
)
from cluster_bootstrap import resample_interactions


def _tool_name(sample):
    return parse_react_step(sample.current_action).tool_name.strip().casefold()

def _lookup_predictions(train, test, target):
    fallback = int(sum(target[s.sample_id] for s in train) / len(train) >= .5)
    by_tool = defaultdict(list)
    for sample in train:
        by_tool[_tool_name(sample)].append(target[sample.sample_id])
    majorities = {name: int(sum(values) / len(values) >= .5) for name, values in by_tool.items()}
    return [majorities.get(_tool_name(s), fallback) for s in test]

def binary_metrics(gold, prediction):
    tp = sum(y == 1 and p == 1 for y, p in zip(gold, prediction))
    tn = sum(y == 0 and p == 0 for y, p in zip(gold, prediction))
    fp = sum(y == 0 and p == 1 for y, p in zip(gold, prediction))
    fn = sum(y == 1 and p == 0 for y, p in zip(gold, prediction))
    return {"n": len(gold), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp+tn)/len(gold), "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
            "balanced_accuracy": .5*(tp/(tp+fn)+tn/(tn+fp))}

samples = []
for asset in ASSETS:
    if not asset.destination.startswith("eval/agentdojo/"):
        continue
    path = ROOT / "pinned-data" / asset.destination
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(asset.url, timeout=45) as response:
            path.write_bytes(response.read())
    content = path.read_bytes()
    assert len(content) == asset.size
    assert hashlib.sha256(content).hexdigest() == asset.sha256
    samples.extend(_from_eval(row, "AgentDojo-Traj", path.stem) for row in json.loads(content))

interaction_tasks = defaultdict(set)

for sample in samples:
    interaction_tasks[sample.trajectory_id].add(
        (sample.subset, sample.instruction)
    )

assert all(len(tasks) == 1 for tasks in interaction_tasks.values())

target = {s.sample_id: s.strict_label for s in samples}
evaluated_rows = []

for domain in sorted({s.subset for s in samples}):
    train = [s for s in samples if s.subset != domain]
    test = [s for s in samples if s.subset == domain]

    predictions = _lookup_predictions(train, test, target)

    for sample, prediction in zip(test, predictions):
        evaluated_rows.append({
            "trajectory_id": sample.trajectory_id,
            "domain": domain,
            "gold": target[sample.sample_id],
            "prediction": prediction,
            "task_id": (sample.subset, sample.instruction),
        })

metrics = {
    "all": binary_metrics(
        [row["gold"] for row in evaluated_rows],
        [row["prediction"] for row in evaluated_rows],
    )
}


cached = json.loads((PROJECT / "results/tsbench_cached_shortcut_audit.json").read_text())
original = cached["AgentDojo"]["gold_tool_lookup"]["leave_domain_out"]
for key in ("n", "tp", "tn", "fp", "fn", "f1", "balanced_accuracy", "accuracy"):
    assert abs(metrics["all"][key] - original[key]) < 1e-12, key


# Reproduce the original interaction-grouped five-fold CV
grouped_rows = []
groups = [sample.trajectory_id for sample in samples]

for train_indices, test_indices in GroupKFold(n_splits=5).split(
    samples, groups=groups
):
    train = [samples[i] for i in train_indices]
    test = [samples[i] for i in test_indices]

    predictions = _lookup_predictions(train, test, target)

    for sample, prediction in zip(test, predictions):
        grouped_rows.append({
            "trajectory_id": sample.trajectory_id,
            "domain": sample.subset,
            "gold": target[sample.sample_id],
            "prediction": prediction,
        })

grouped_metrics = binary_metrics(
    [row["gold"] for row in grouped_rows],
    [row["prediction"] for row in grouped_rows],
)

original_grouped = cached["AgentDojo"]["gold_tool_lookup"]["group_cv"]

for key in ("n", "tp", "tn", "fp", "fn", "f1", "balanced_accuracy", "accuracy"):
    assert abs(grouped_metrics[key] - original_grouped[key]) < 1e-12, key

print(f"Grouped-CV F1: {grouped_metrics['f1']:.6f}")


# Interaction-weighted evaluation
if any(row["trajectory_id"].endswith(":") for row in evaluated_rows):
    raise ValueError("Missing interaction identifiers")

weighted_counts, weighted_f1, total_weight = (
    interaction_weighted_metrics(evaluated_rows)
)

n_interactions = len({
    row["trajectory_id"] for row in evaluated_rows
})

assert len(evaluated_rows) == metrics["all"]["n"]
assert abs(total_weight - n_interactions) < 1e-9

print(f"Step-weighted F1: {metrics['all']['f1']:.6f}")
print(f"Interaction-weighted F1: {weighted_f1:.6f}")


# Interaction weighting while preserving domain proportions
_, domain_preserving_f1, domain_total_weight = (
    interaction_weighted_metrics(
        evaluated_rows,
        preserve_domain_shares=True
    )
)

assert abs(domain_total_weight - len(evaluated_rows)) < 1e-9

print(f"Domain-preserving interaction F1: {domain_preserving_f1:.6f}")

extension_result = {
    "source_commit": EVAL_COMMIT,
    "n_steps": len(evaluated_rows),
    "n_interactions": n_interactions,
    "f1": {
        "step_weighted": metrics["all"]["f1"],
        "interaction_weighted": weighted_f1,
        "domain_preserving": domain_preserving_f1,
    },
}


unit_metrics = {
    "rows": metrics_from_counts(metrics["all"]),
    "equal_interaction_weight": metrics_from_counts(weighted_counts),
    "interaction_majority": interaction_majority_metrics(evaluated_rows),
}

assert abs(
    unit_metrics["equal_interaction_weight"]["f1"] - weighted_f1
) < 1e-12

assert sum(
    unit_metrics["interaction_majority"][key]
    for key in ("tp", "tn", "fp", "fn")
) == n_interactions

extension_result["evaluation_units"] = unit_metrics

task_counts, task_f1, task_total_weight = (
    interaction_weighted_metrics(
        evaluated_rows,
        group_key="task_id"
    )
)

n_tasks = len({row["task_id"] for row in evaluated_rows})

assert n_tasks == 58
assert abs(task_total_weight - n_tasks) < 1e-9

extension_result["task_weighting"] = {
    "n_tasks": n_tasks,
    "metrics": metrics_from_counts(task_counts),
}

print(f"Task-weighted LODO F1: {task_f1:.6f}")

print(json.dumps({"evaluation_units": unit_metrics}, indent=2))



# Evaluate grouped-CV predictions at all three units
grouped_weighted_counts, grouped_weighted_f1, grouped_total_weight = (
    interaction_weighted_metrics(grouped_rows)
)

assert len(grouped_rows) == len(evaluated_rows)
assert abs(grouped_total_weight - n_interactions) < 1e-9

grouped_unit_metrics = {
    "rows": metrics_from_counts(grouped_metrics),
    "equal_interaction_weight": metrics_from_counts(grouped_weighted_counts),
    "interaction_majority": interaction_majority_metrics(grouped_rows),
}

assert abs(
    grouped_unit_metrics["equal_interaction_weight"]["f1"]
    - grouped_weighted_f1
) < 1e-12

assert sum(
    grouped_unit_metrics["interaction_majority"][key]
    for key in ("tp", "tn", "fp", "fn")
) == n_interactions

extension_result["group_cv"] = {
    "folds": 5,
    "evaluation_units": grouped_unit_metrics,
}

print(json.dumps({"group_cv": extension_result["group_cv"]}, indent=2))



# Examine domain composition and held-out performance
domain_composition = {}

for domain in sorted({row["domain"] for row in evaluated_rows}):
    domain_rows = [
        row for row in evaluated_rows
        if row["domain"] == domain
    ]

    domain_interactions = {
        row["trajectory_id"] for row in domain_rows
    }

    domain_metrics = binary_metrics(
        [row["gold"] for row in domain_rows],
        [row["prediction"] for row in domain_rows],
    )

    _, domain_weighted_f1, _ = interaction_weighted_metrics(
        domain_rows
    )

    domain_composition[domain] = {
        "n_steps": len(domain_rows),
        "step_share": len(domain_rows) / len(evaluated_rows),
        "n_interactions": len(domain_interactions),
        "interaction_share": len(domain_interactions) / n_interactions,
        "step_f1": domain_metrics["f1"],
        "interaction_f1": domain_weighted_f1,
    }

assert sum(
    item["n_steps"] for item in domain_composition.values()
) == len(evaluated_rows)

assert sum(
    item["n_interactions"] for item in domain_composition.values()
) == n_interactions

extension_result["domain_composition"] = domain_composition

print(json.dumps({"domain_composition": domain_composition}, indent=2))


# Paired domain-stratified interaction bootstrap
rng = random.Random(260828)
n_bootstrap = 10000

global_differences = []
domain_preserving_differences = []

for _ in range(n_bootstrap):
    boot_rows = resample_interactions(evaluated_rows, rng)

    step_f1 = binary_metrics(
        [r["gold"] for r in boot_rows],
        [r["prediction"] for r in boot_rows]
    )["f1"]

    _, global_f1, _ = interaction_weighted_metrics(boot_rows)

    _, preserved_f1, _ = interaction_weighted_metrics(
        boot_rows,
        preserve_domain_shares=True
    )

    global_differences.append(global_f1 - step_f1)
    domain_preserving_differences.append(preserved_f1 - step_f1)


def confidence_interval(values):
    cutoffs = quantiles(values, n=40, method="inclusive")
    return [cutoffs[0], cutoffs[-1]]


extension_result["bootstrap"] = {
    "replicates": n_bootstrap,
    "seed": 260828,
    "global_minus_step": {
        "difference": weighted_f1 - metrics["all"]["f1"],
        "ci95": confidence_interval(global_differences)
    },
    "domain_preserving_minus_step": {
        "difference": domain_preserving_f1 - metrics["all"]["f1"],
        "ci95": confidence_interval(domain_preserving_differences)
    }
}


# Save complete interaction-weighting experiment
output_path = (
    Path(__file__).resolve().parent
    / "agentdojo_interaction_weighting.json"
)
with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
    output_file.write(json.dumps(extension_result, indent=2) + "\n")
