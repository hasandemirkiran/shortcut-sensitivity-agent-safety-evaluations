# AgentDojo interaction weighting

The [original paper](https://openreview.net/forum?id=IDTgteFueV) reports F1 per step for its AgentDojo tool-name lookup baseline. I checked what happens when each interaction counts equally instead. The predictions are unchanged.

## Run

From the repository root, run:

```bash
python extensions/agentdojo_lookup_weighting/analyze_agentdojo_weighting.py
```

The script downloads the pinned AgentDojo data, checks the original result and writes `agentdojo_interaction_weighting.json` in this folder.

## Results

The evaluation contains 1 220 steps from 589 interactions across four domains.

| Weighting | F1 |
| --- | ---: |
| Per step (original) | 0.495479 |
| Per interaction | 0.537346 |
| Per interaction, preserving domain shares | 0.519678 |

With 2 000 paired, domain-stratified bootstrap samples (with fixed seed), the F1 differences from the original are:

| Comparison | Difference | 95% interval |
| --- | ---: | --- |
| Per interaction | +0.04187 | [0.01446, 0.06772] |
| Preserving domain shares | +0.02420 | [-0.00055, 0.04768] |

Each step gets weight `1 / number_of_steps_in_its_interaction`. The domain-preserving version rescales these weights to keep each domain's original total step weight.

The bootstrap resamples interactions within each domain without retraining the lookup. These intervals do not describe uncertainty across new domains. This is an evaluation of the lookup baseline, not of a safety guard.

The code builds on the authors' released analysis.
