# Artifact boundary and provenance

The repository ships aggregate outputs needed to verify the manuscript, the frozen
audit protocol, and source snapshots of the transformation, parsing, and analysis
code. Upstream benchmark rows and raw model responses are intentionally excluded.

| File | Role |
|---|---|
| **results/linuxarena_shortcut_audit.json** | Corrected LinuxArena deduplication, proxy, monitor, and metric-unit audit |
| **results/tsbench_cached_shortcut_audit.json** | Cached AgentDojo-Traj and AgentHarm-Traj label and weighting diagnostics |
| **results/agentdojo_lookup_coverage.json** | Seen/unseen tool-name LODO confusion counts and metrics |
| **results/counterfactual_cohort_manifest.json** | Frozen AgentDojo cohort counts and hashes |
| **results/agentdojo_counterfactual_tsguard_q8.json** | Local Q8 TS-Guard reproduction gate and paired results |
| **results/agentdojo_counterfactual_gpt55.json** | Hosted GPT-5.5 paired results |
| **results/gpt55_followup_provenance.json** | Hosted run provenance, accuracy deltas, and prior-repeat comparison |
| **results/shortcut_audit_cached_reanalysis.json** | Corrected interaction weighting, rule strata, component changes, and AgentHarm execution-matched analysis |
| **results/benchmark_shortcut_audit.openrouter_haiku_k3.json** | Text-free 960-call repeated Haiku aggregate, placebo, stability, accuracy, and provider/cost provenance |
| **results/benchmark_shortcut_audit.openrouter_gpt_5_6_luna_k3.json** | Text-free 960-call repeated GPT-5.6 Luna aggregate using the same cohort, conditions, and estimands |
| **results/stepguard_feasibility.json** | Text-free one-row StepGuard feasibility gate, runtime boundary, and raw-record hash |
| **artifact/protocol/BENCHMARK_SHORTCUT_AUDIT_PROTOCOL.md** | Original protocol plus dated, pre-follow-up corrective amendment |
| **artifact/protocol/OPENROUTER_HAIKU_AUDIT.md** | Repeated-condition Haiku execution and verification note |
| **artifact/protocol/STEPGUARD_FEASIBILITY.md** | Pinned StepGuard one-row feasibility gate and boundary |
| **artifact/code/authz_bench/** | LinuxArena deduplication and shortcut audit source snapshot |
| **artifact/code/toolsafe_lab/** | TS-Bench, intervention, TS-Guard, and paired-analysis source snapshots and tests |

The protocol amendment also records a planned 673-base Q8 marker census. Its cohort
was prepared and a resumable local run was initiated, but full inference was not
completed or used in this draft; no partial census outcomes are shipped or claimed.

The repeated-Haiku aggregate is linked to the excluded raw 960-record JSONL: the raw file
is 1,608,985 bytes with SHA-256
**f21a34ca9bb49a5c112213dbe0d563055737fd114dbb0db6ba490338c063b188**. The aggregate
contains this provenance but no benchmark text or provider response.

The repeated-Luna aggregate is linked to a separate excluded 960-record JSONL of
1,762,198 bytes with SHA-256
**b39f08ac98dfbc3d8c8d1aaebb04860ec7491b5c77a2a1032b2163b3cbfd1fcf**. It likewise
contains no benchmark text or provider response. The normalized base, condition,
repetition, and schedule-index order is identical between the Haiku and Luna runs;
model-specific request and job hashes remain distinct.

Run **python3 scripts/verify_numbers.py** from the repository root. The script
asserts the headline table, weighting, observational, repeated-intervention,
placebo, stability, accuracy, provider, and resource values directly against these
files, including the new seen/unseen table, and fails if a covered value changes.

Run **python3 scripts/verify_lookup_coverage.py** to regenerate the seen/unseen
analysis from four checksum-pinned public AgentDojo-Traj files. This downloads
about 11 MB, makes no model calls, and writes only aggregate counts and metrics.

The source snapshots preserve the analysis implementations used for this
draft; the offline verification script also covers the accepted seen/unseen addition. The ToolSafe-Lab snapshot now includes its complete Python package and dependency
metadata; focused installation and test commands are in its README. Pinned upstream
benchmark inputs, model weights, and provider responses remain outside the artifact and
are needed for a fresh end-to-end inference run. The aggregate verification command above
is self-contained apart from Python's standard library.
The public runner snapshot expresses the local credential-file default through
`Path.home()` rather than retaining the executing machine's username; this
double-blind redaction does not change the resolved path or experimental behavior.

## Pinned upstream inputs

- LinuxArena public sample revision:
  **2fd29f9c7bc190fb873bd76c6bd648f22b9ab0a7**
- ToolSafe evaluation commit: **46358fa**
- TS-Guard checkpoint revision:
  **ad2f82df3ae3e0dcdaed7b7c71e8b7f0b16ead7a**
- Local TS-Guard Q8 artifact SHA-256:
  **913afe8cc195fc5d56189ac2ccc012903d2bbe137d69b254253e532abf638957**

No command or tool call contained in upstream data is executed by the audit.
