from collections import Counter, defaultdict

def interaction_weighted_metrics(rows, *, preserve_domain_shares=False):
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset")

    sizes = Counter(row["trajectory_id"] for row in rows)

    if preserve_domain_shares:
        domain_steps = Counter(row["domain"] for row in rows)
        domain_interactions = defaultdict(set)

        for row in rows:
            domain_interactions[row["domain"]].add(
                row["trajectory_id"]
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

        weight = 1.0 / sizes[row["trajectory_id"]]

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