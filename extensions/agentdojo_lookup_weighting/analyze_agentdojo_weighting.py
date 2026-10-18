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
from interaction_weighting import interaction_weighted_metrics
from cluster_bootstrap import resample_interactions


def _tool_name(sample):
    return parse_react_step(sample.current_action).tool_name.strip().casefold()

def _lookup_predictions(train, test, target):
    fallback = int(sum(target[s.sample_id] for s in train) / len(train) >= .5)
    by_tool = defaultdict(list)
    for sample in train:
        by_tool[_tool_name(sample)].append(target[sample.sample_id])
    majorities = {name: int(sum(values) / len(values) >= .5) for name, values in by_tool.items()}
    return [majorities.get(_tool_name(s), fallback) for s in test], None

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


target = {s.sample_id: s.strict_label for s in samples}
evaluated_rows = []

for domain in sorted({s.subset for s in samples}):
    train = [s for s in samples if s.subset != domain]
    test = [s for s in samples if s.subset == domain]

    predictions, _ = _lookup_predictions(train, test, target)

    for sample, prediction in zip(test, predictions):
        evaluated_rows.append({
            "trajectory_id": sample.trajectory_id,
            "domain": domain,
            "gold": target[sample.sample_id],
            "prediction": prediction,
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


# Interaction-weighted evaluation
if any(row["trajectory_id"].endswith(":") for row in evaluated_rows):
    raise ValueError("Missing interaction identifiers")

_, weighted_f1, total_weight = (
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


# Paired domain-stratified interaction bootstrap
rng = random.Random(42)
n_bootstrap = 2000

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
    "seed": 42,
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
output_path.write_text(
    json.dumps(extension_result, indent=2) + "\n",
    encoding="utf-8"
)