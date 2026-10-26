"""Evaluate binary predictions with interaction and task weighting"""
from collections import Counter, defaultdict

def interaction_weighted_metrics(rows, *, preserve_domain_shares=False, group_key="trajectory_id"):
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset")

    sizes = Counter(row[group_key] for row in rows)

    if preserve_domain_shares:
        domain_steps = Counter(row["domain"] for row in rows)
        domain_interactions = defaultdict(set)

        for row in rows:
            domain_interactions[row["domain"]].add(
                row[group_key]
            )

    counts = {
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
    }

    for row in rows:
        gold = row["gold"]
        pred = row["prediction"]

        if gold not in (0, 1) or pred not in (0, 1):
            raise ValueError("Labels and predictions must be binary")

        weight = 1.0 / sizes[row[group_key]]

        if preserve_domain_shares:
            domain = row["domain"]
            weight *= (
                domain_steps[domain]
                / len(domain_interactions[domain])
            )

        if gold == 1 and pred == 1:
            counts["tp"] += weight
        elif gold == 0 and pred == 0:
            counts["tn"] += weight
        elif gold == 0 and pred == 1:
            counts["fp"] += weight
        else:
            counts["fn"] += weight

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0

    total_weight = sum(counts.values())

    return counts, f1, total_weight

def metrics_from_counts(counts):
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    total = tp + tn + fp + fn
    if (total == 0) or (tp + fn == 0) or (tn + fp == 0):
        raise ValueError("Both label classes are required")

    f1_denominator = 2 * tp + fp + fn

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total,
        "f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
        "balanced_accuracy": 0.5 * (tp/(tp + fn) + tn/(tn + fp)),
    }

def interaction_majority_metrics(rows):
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset")

    interactions = defaultdict(list)
    for row in rows:
        if row["gold"] not in (0, 1) or row["prediction"] not in (0, 1):
            raise ValueError("Labels and predictions must be binary")
        interactions[row["trajectory_id"]].append(row)

    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for steps in interactions.values():
        n = len(steps)
        gold = int(2 * sum(row["gold"] for row in steps) >= n)
        pred = int(2 * sum(row["prediction"] for row in steps) >= n)

        if gold == 1 and pred == 1:
            counts["tp"] += 1
        elif gold == 0 and pred == 0:
            counts["tn"] += 1
        elif gold == 0 and pred == 1:
            counts["fp"] += 1
        else:
            counts["fn"] += 1

    return metrics_from_counts(counts)
