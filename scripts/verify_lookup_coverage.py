"""Recompute only the existing LODO diagnostic, split by vocabulary coverage."""
import hashlib
import json
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(tempfile.mkdtemp(prefix="lookup-coverage-"))
sys.path.insert(0, str(PROJECT / "artifact/code/toolsafe_lab/src"))
from toolsafe_lab.manifest import ASSETS, EVAL_COMMIT
from toolsafe_lab.data import _from_eval
from toolsafe_lab.react_parser import parse_react_step

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
provenance = []
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
    provenance.append({"url": asset.url, "sha256": asset.sha256, "bytes": asset.size})

target = {s.sample_id: s.strict_label for s in samples}
strata = {name: {"gold": [], "prediction": []} for name in ("all", "seen", "unseen")}
by_domain = {}
for domain in sorted({s.subset for s in samples}):
    train = [s for s in samples if s.subset != domain]
    test = [s for s in samples if s.subset == domain]
    predictions, _ = _lookup_predictions(train, test, target)
    known = {_tool_name(s) for s in train}
    fallback = int(sum(target[s.sample_id] for s in train) / len(train) >= .5)
    by_domain[domain] = {"fallback": fallback, "n": len(test)}
    for sample, prediction in zip(test, predictions):
        names = ("all", "seen" if _tool_name(sample) in known else "unseen")
        for name in names:
            strata[name]["gold"].append(target[sample.sample_id])
            strata[name]["prediction"].append(prediction)

metrics = {name: binary_metrics(value["gold"], value["prediction"]) for name, value in strata.items()}
cached = json.loads((PROJECT / "results/tsbench_cached_shortcut_audit.json").read_text())
original = cached["AgentDojo"]["gold_tool_lookup"]["leave_domain_out"]
for key in ("n", "tp", "tn", "fp", "fn", "f1", "balanced_accuracy", "accuracy"):
    assert abs(metrics["all"][key] - original[key]) < 1e-12, key
result = {"source_commit": EVAL_COMMIT, "method": "unchanged exact-casefolded-tool-majority LODO, pooled held-out rows stratified by name seen in training domains", "unseen_fallback": "training-domain majority strict label; safe in all four folds", "ties": "unsafe", "metrics": metrics, "by_domain": by_domain, "inputs": provenance}
(PROJECT / "results/agentdojo_lookup_coverage.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"metrics": metrics, "by_domain": by_domain}, indent=2))
