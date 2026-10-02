# ToolSafe-Lab audit source snapshot

This directory contains the complete Python package needed by the shortcut-audit modules,
the focused audit tests, and dependency metadata. It is an anonymized source snapshot of
the code used for the paper, not a redistribution of upstream benchmark rows, model
weights, raw provider responses, or credentials.

## Install and test

Use Python 3.10 or newer. From this directory (replace `python3` with an explicit
newer interpreter such as `python3.12` if the system default is older):

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest \
  tests/test_cached_shortcut_audit.py \
  tests/test_openrouter_luna_shortcut_analysis.py \
  tests/test_openrouter_luna_shortcut_runner.py \
  tests/test_openrouter_shortcut_analysis.py \
  tests/test_openrouter_shortcut_runner.py \
  tests/test_shortcut_audit.py \
  tests/test_shortcut_audit_reanalysis.py \
  tests/test_ts_guard_local.py \
  tests/test_ts_guard_shortcut_analysis.py
```

The package's checksum-pinned public TS-Bench inputs can be fetched with:

```bash
.venv/bin/toolsafe-lab --project-root . fetch
```

The exact frozen cohort hashes, prompts, transformations, schedules, provider settings,
and analysis commands are documented under `../../protocol/`. A fresh hosted rerun also
requires an API key supplied at runtime; a fresh local Q8 rerun requires the pinned model
artifact documented there. The excluded raw response JSONL is cryptographically linked to
the committed text-free aggregate, but is not needed for `scripts/verify_numbers.py` at the
paper repository root.
