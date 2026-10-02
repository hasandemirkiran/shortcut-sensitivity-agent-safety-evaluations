# Repeated OpenRouter Haiku shortcut audit

This post-hoc robustness run uses the already frozen 64-base AgentDojo shortcut cohort.
It sends only text classification requests and exposes no tools to the model. Benchmark
tool calls are inert text and are never executed.

## Design

- Model: `anthropic/claude-haiku-4.5` through OpenRouter.
- Provider: pinned to `anthropic`; fallbacks disabled.
- Prompt/parser: frozen `authors_v2_structured` schema and TS-Guard composite parser.
- Conditions: original, marker rewrite (artifact key `marker_paraphrase`),
  neutral outer-JSON formatting placebo,
  tool alias, and combined marker-plus-alias.
- Repetitions: three per condition and base, for 960 calls.
- Scheduling: seed-`260829` round/base blocks with conditions shuffled inside every block.
- Sampling: temperature zero, 256 output-token cap, concurrency two, at most three
  attempts on transient HTTP failures.
- Safety: no `tools`, no tool or web plugins, and no execution path for benchmark actions.

The primary effect is the transformation-minus-original difference in empirical block
probability. Each base is weighted equally after averaging its three calls. The aggregate
uses a domain-stratified source-interaction bootstrap; label strata are not used because
one source interaction can contain base steps with different labels. Repeated-call
disagreement is a stability diagnostic only; same-round one-draw flips are secondary.

## Completed run

All 960 calls parsed on the first attempt and resolved to
`anthropic/claude-haiku-4.5` at Anthropic. The marker, tool-alias, combined, and
formatting-placebo block-probability contrasts were -8.33, +11.46, -2.60, and
-1.56 percentage points. Marker-minus-placebo was -6.77 points and
alias-minus-placebo was +13.02 points. Repeated originals had 2.08% mean pairwise
disagreement. The provider-reported aggregate cost was $3.407663. The cohort hash is
`bdfe8848090aab7f6efd0d575d6547ad88aae84831c8d0792055bb0076c638a2`; the schedule
hash is `ecb2b0cdae6b9a9f5f4365d116b2c310397df3537c006f57651e6041a40da621`.

Every unsafe-step contrast was zero. Safe-only marker and alias contrasts were -16.67
points (exploratory clustered 95% interval [-30.00, -5.88]) and +22.92 points
([+5.56, +39.22]). Marker changes occurred in six non-tied source interactions, all
negative; the marker direction persisted in all three repetitions. Exact interaction-sign
diagnostics, repetition/schedule-third summaries, and label-stratified intervals are stored
in the aggregate. They are post-hoc robustness diagnostics without multiplicity adjustment.

The ignored raw JSONL contains 960 parsed records, is 1,608,985 bytes, and has SHA-256
`f21a34ca9bb49a5c112213dbe0d563055737fd114dbb0db6ba490338c063b188`.
The aggregate embeds this provenance record but no raw benchmark text or model response.

## Commands

Prepare and inspect the manifest without reading the key or making an API request:

```bash
.venv/bin/python -m toolsafe_lab.openrouter_shortcut_runner
```

Run a ten-call API/schema pilot. It resumes into the final output file:

```bash
.venv/bin/python -m toolsafe_lab.openrouter_shortcut_runner \
  --execute --max-calls 10 --max-total-cost-usd 0.10
```

The ten-call pilot is projected at about $0.019 expected list price and $0.031 if every
call reaches the 256-token output cap. After checking parsed coverage, resolved provider,
and spend, resume the remaining calls with:

```bash
.venv/bin/python -m toolsafe_lab.openrouter_shortcut_runner --execute
```

The complete 960-call run is projected at about $3.40 using prior Haiku token counts for
the same original inputs, or $4.46 under the single-attempt maximum-output projection.
The runner stops scheduling new work at $6 of observed or token-estimated spend by default.

Once all 960 calls are valid, write the text-free aggregate:

```bash
.venv/bin/python -m toolsafe_lab.openrouter_shortcut_analysis
```

The raw/resumable JSONL and its manifest remain under ignored
`artifacts/shortcut_audit/openrouter/`. The aggregate is written to
`results/benchmark_shortcut_audit.openrouter_haiku_k3.json`. The analyzer refuses to write
an incomplete headline aggregate unless `--allow-partial` is explicitly supplied.

The OpenRouter key is read at execution time from a local file supplied with
`--keys-file`. It is never printed or persisted.
