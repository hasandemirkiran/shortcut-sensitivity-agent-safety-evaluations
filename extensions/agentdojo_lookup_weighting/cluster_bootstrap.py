
from collections import defaultdict


def resample_interactions(rows, rng):
    groups = defaultdict(lambda: defaultdict(list))

    for row in rows:
        groups[row["domain"]][row["trajectory_id"]].append(row)

    resampled = []

    for domain in sorted(groups):
        interactions = groups[domain]
        ids = sorted(interactions)

        for draw_index in range(len(ids)):
            chosen_id = rng.choice(ids)

            # Each draw is a new bootstrap interaction
            new_id = f"bootstrap:{domain}:{draw_index}"

            for row in interactions[chosen_id]:
                resampled.append({
                    **row,
                    "trajectory_id": new_id
                })

    return resampled
