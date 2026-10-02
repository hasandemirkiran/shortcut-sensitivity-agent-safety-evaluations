"""Text-free aggregate wrapper for the repeated GPT-5.6 Luna audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import toolsafe_lab.openrouter_shortcut_analysis as shared_analysis
from toolsafe_lab.openrouter_luna_shortcut_runner import (
    CONDITIONS,
    INPUT_USD_PER_MTOK,
    MODEL_ID,
    OUTPUT_USD_PER_MTOK,
    PINNED_PROVIDER,
    PROMPT_VERSION,
    PROTOCOL_VERSION,
    REPETITIONS,
    SCHEDULE_SEED,
    PlannedCall,
    _repair_and_read_jsonl,
    _sha256_path,
    build_interleaved_plan,
    load_frozen_agentdojo_cohort,
)
from toolsafe_lab.openrouter_shortcut_analysis import (
    OpenRouterAnalysisSpec,
    _raw_records_artifact_provenance,
    _write_json,
    analyze_openrouter_records,
    load_agentdojo_source_maps,
)


ANALYSIS_VERSION = "OPENROUTER_GPT_5_6_LUNA_AGENTDOJO_K3_ANALYSIS_v1.0"
LUNA_ANALYSIS_SPEC = OpenRouterAnalysisSpec(
    analysis_version=ANALYSIS_VERSION,
    protocol_version=PROTOCOL_VERSION,
    model_id=MODEL_ID,
    pinned_provider=PINNED_PROVIDER,
    prompt_version=PROMPT_VERSION,
    schedule_seed=SCHEDULE_SEED,
    repetitions=REPETITIONS,
    conditions=CONDITIONS,
    input_usd_per_mtok=INPUT_USD_PER_MTOK,
    output_usd_per_mtok=OUTPUT_USD_PER_MTOK,
    resolved_model_fragment="gpt-5.6-luna",
)

DEFAULT_RECORDS_PATH = Path(
    "artifacts/shortcut_audit/openrouter/gpt-5.6-luna-authors-v2-k3.jsonl"
)
DEFAULT_AGGREGATE_PATH = Path(
    "results/benchmark_shortcut_audit.openrouter_gpt_5_6_luna_k3.json"
)


def analyze_luna_records(
    *,
    calls: Sequence[PlannedCall],
    records: Sequence[Mapping[str, object]],
    cohort_sha256: str,
    source_interaction_by_base: Mapping[str, str],
    fixed_rule_by_base: Mapping[str, int],
    bootstrap_replicates: int = 10_000,
    raw_records_artifact: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run the shared Haiku estimands and checks with Luna provenance."""
    result = analyze_openrouter_records(
        calls=calls,
        records=records,
        cohort_sha256=cohort_sha256,
        source_interaction_by_base=source_interaction_by_base,
        fixed_rule_by_base=fixed_rule_by_base,
        bootstrap_replicates=bootstrap_replicates,
        raw_records_artifact=raw_records_artifact,
        spec=LUNA_ANALYSIS_SPEC,
    )
    result["analysis_implementation_sha256"] = {
        "luna_wrapper": _sha256_path(Path(__file__).resolve()),
        "shared_estimands": _sha256_path(Path(shared_analysis.__file__).resolve()),
    }
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Aggregate the repeated OpenRouter GPT-5.6 Luna shortcut audit"
    )
    value.add_argument(
        "--cohort-file",
        type=Path,
        default=Path("artifacts/shortcut_audit/cohorts/agentdojo.jsonl"),
    )
    value.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Raw TS-Bench root used to recover upstream AgentDojo interaction IDs",
    )
    value.add_argument("--records", type=Path, default=DEFAULT_RECORDS_PATH)
    value.add_argument("--output", type=Path, default=DEFAULT_AGGREGATE_PATH)
    value.add_argument("--bootstrap-replicates", type=int, default=10_000)
    value.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write an explicitly partial aggregate even if calls failed or are missing",
    )
    return value


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    examples, cohort_sha256 = load_frozen_agentdojo_cohort(args.cohort_file.resolve())
    calls = build_interleaved_plan(examples)
    source_interaction_by_base, fixed_rule_by_base = load_agentdojo_source_maps(
        args.data_root.resolve(),
        (call.base_sample_id for call in calls),
    )
    records_path = args.records.resolve()
    records = _repair_and_read_jsonl(records_path)
    raw_records_artifact = _raw_records_artifact_provenance(
        records_path,
        artifact_path=args.records.as_posix(),
        parsed_records=len(records),
    )
    result = analyze_luna_records(
        calls=calls,
        records=records,
        cohort_sha256=cohort_sha256,
        source_interaction_by_base=source_interaction_by_base,
        fixed_rule_by_base=fixed_rule_by_base,
        bootstrap_replicates=args.bootstrap_replicates,
        raw_records_artifact=raw_records_artifact,
    )
    if result["analysis_status"] != "complete" and not args.allow_partial:
        coverage = result["overall_coverage"]
        assert isinstance(coverage, dict)
        raise ValueError(
            "OpenRouter Luna audit is incomplete: "
            f"{coverage['valid']}/{coverage['expected']} valid; pass --allow-partial "
            "only for a clearly labeled diagnostic aggregate"
        )
    _write_json(args.output.resolve(), result)
    print(f"Wrote text-free OpenRouter Luna aggregate to {args.output.resolve()}")


if __name__ == "__main__":
    main()
