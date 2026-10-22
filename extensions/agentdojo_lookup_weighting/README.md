# AgentDojo lookup evaluation-unit sensitivity

This is a contributed post-publication extension of [Shortcut Sensitivity in Agent-Safety Evaluations](https://openreview.net/forum?id=IDTgteFueV).

This paper reports interaction-weighting sensitivity for TS-Guard on AgentHarm in Table `weighting`. This extension applies the same three evaluation units to the AgentDojo exact-tool-name lookup diagnostic, under both interaction-grouped five-fold cross-validation (grouped-CV) and leave-one-domain-out (LODO) evaluation.

The original lookup predictions are unchanged. This analysis evaluates a label-lookup diagnostic.

## Reproduction

From the repository root, run:

```bash
python extensions/agentdojo_lookup_weighting/analyze_agentdojo_weighting.py
```

The script requires scikit-learn for `GroupKFold`. It downloads and verifies the pinned AgentDojo data, reproduces the original lookup results and writes `agentdojo_interaction_weighting.json` in this directory.

The evaluation contains 1 220 steps from 589 interactions across four domains.

## Evaluation units

The three units follow the paper AgentHarm weighting comparison:

- **Rows:** Each step receives equal weight.
- **Equal interaction weight:** Each step receives weight `1 / number_of_steps_in_its_interaction`.
- **Interaction majority:** Each interaction contributes one majority-voted label and prediction. Ties resolve to unsafe (`1`).

All confusion counts treat unsafe (`1`) as the positive class. Counts under equal-interaction weighting are fractional.

### Leave-one-domain-out

| Unit | Accuracy | F1 | Balanced accuracy | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rows | 0.771311 | 0.495479 | 0.657736 | 137 | 804 | 64 | 215 |
| Equal interaction weight | 0.767611 | 0.537346 | 0.676172 | 79.487 | 372.636 | 25.917 | 110.960 |
| Interaction majority | 0.750424 | 0.561194 | 0.683626 | 94 | 348 | 27 | 120 |

### Interaction-grouped five-fold CV

| Unit | Accuracy | F1 | Balanced accuracy | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rows | 0.944262 | 0.900875 | 0.924520 | 309 | 843 | 25 | 43 |
| Equal interaction weight | 0.951634 | 0.923437 | 0.938690 | 171.794 | 388.719 | 9.833 | 18.654 |
| Interaction majority | 0.937182 | 0.911271 | 0.926592 | 190 | 362 | 13 | 24 |

The grouped-CV versus LODO F1 contrast remains large under equal-interaction weighting: 0.923437 versus 0.537346, compared with 0.900875 versus 0.495479 at the step level.

## Domain composition

| Domain | Steps | Step share | Interactions | Interaction share | Step F1 | Interaction F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Banking | 87 | 7.13% | 45 | 7.64% | 0.000000 | 0.000000 |
| Slack | 101 | 8.28% | 28 | 4.75% | 0.000000 | 0.000000 |
| Travel | 177 | 14.51% | 42 | 7.13% | 0.344828 | 0.333378 |
| Workspace | 855 | 70.08% | 474 | 80.48% | 0.564103 | 0.590744 |

Workspace contributes approximately 70% of steps but 80% of interactions and has the highest held-out F1. Consequently, part of the global interaction-weighting increase reflects the change in domain composition.

To separate this concern from weighting differences within domains, an additional analysis rescales interaction weights to preserve each domain's original total step weight. This produces LODO F1 of 0.519678, compared with 0.495479 under row weighting.

Because F1 is nonlinear, these comparisons are not an exact additive decomposition of domain-composition and within-domain effects.

## Uncertainty

We used 10,000 paired, domain-stratified interaction bootstrap replicates with seed `260828`.

| Comparison with row-weighted LODO F1 | Difference | 95% percentile interval |
| --- | ---: | --- |
| Equal interaction weight | +0.041867 | [0.014272, 0.067827] |
| Domain-preserving interaction weight | +0.024198 | [-0.001074, 0.048210] |

The global interval is positive, while the domain-preserving interval includes zero.

The bootstrap resamples interactions within domains without retraining the lookup classifier. These intervals are conditional on the existing domains and fixed predictions, not covering unseen domains.

## Task-level sensitivity

The 589 interactions correspond to 58 distinct `(domain, instruction)` pairs, used here as task identifiers.

Giving each such task equal total weight produces LODO F1 of **0.499316**, rather than 0.537346 under equal-interaction weighting.

This is task weighting of step predictions, not task-majority voting. Identical instructions within the same domain are treated as the same task.
