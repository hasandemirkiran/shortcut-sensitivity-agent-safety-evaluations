# Shortcut Sensitivity in Agent-Safety Evaluations

Research artifact for the paper by **Hasan Demirkıran and Jens Ernstberger**, both at **[Kontext](https://kontext.security)**.

[Paper and TAE reviews](https://openreview.net/forum?id=IDTgteFueV)

This repository contains the approved camera-ready draft source, analysis and
transformation code, frozen protocols, and aggregate results. The camera-ready
draft is awaiting the authors' final PDF and consistency check. No new experiments
were run for this release.

## Verify the committed results offline

From the repository root, using Python 3.10 or newer:

```bash
python3 scripts/verify_numbers.py
```

This standard-library-only check verifies headline values against the committed
aggregates, checks the seen/unseen lookup confusion counts and metrics, and checks
selected corresponding manuscript values. It requires no API key, model download,
network access, or paid inference. It verifies the released artifacts; it does not
independently regenerate excluded model responses or their bootstrap estimates.

## Recompute the seen/unseen tool-name analysis

```bash
python3 scripts/verify_lookup_coverage.py
```

This command downloads four checksum-pinned public ToolSafe evaluation files
(about 11 MB), repeats the original leave-one-domain-out lookup rule, and writes
`results/agentdojo_lookup_coverage.json`. It uses Python's standard library, makes
no guard calls, and checks that the pooled predictions reproduce the original
lookup metrics. It accesses benchmark text locally without executing tool calls.

## Reproduce the other analyses

See [artifact documentation](artifact/README.md), the
[frozen audit protocol](artifact/protocol/BENCHMARK_SHORTCUT_AUDIT_PROTOCOL.md), and
[repeated hosted-run protocol](artifact/protocol/OPENROUTER_HAIKU_AUDIT.md).
The [ToolSafe-Lab snapshot](artifact/code/toolsafe_lab/README.md) includes Python
dependency metadata, source, and focused tests. The LinuxArena analysis script is
`artifact/code/authz_bench/control/audit_linuxarena_shortcuts.py`; it requires
NumPy, requests, and scikit-learn and can fetch its pinned public sample (about
704 MB). Its inputs are analyzed as inert data.

Fresh local model runs require the pinned checkpoint and documented runtime.
Fresh hosted runs require provider access and incur API charges. Raw provider
responses from the paper are excluded, so exact recomputation of those original
response-based statistics is unavailable from this repository alone. Aggregates,
whole-file hashes, prompts, schedules, parsing code, and analysis code document
what was run. The planned larger Q8 marker census was incomplete and is not a
reported experiment.

## Build the paper

`main.tex` is the TAE camera-ready draft, using the unmodified NeurIPS 2026 style
and `dblblindworkshop,final`. Import the source into Overleaf or use an existing
Tectonic installation:

```bash
make paper
```

TAE permits up to 10 content pages, excluding references and appendices.

## Contents and rights

Only code, protocols, manuscript source, and aggregate results are distributed.
Raw upstream benchmark rows, model weights, provider responses, credentials, and
private review files are excluded. Downloaded inputs and generated raw artifacts
should remain outside version control. See [LICENSES.md](LICENSES.md) for the
rights boundary and [artifact/README.md](artifact/README.md) for provenance.
