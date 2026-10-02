from __future__ import annotations

import argparse
from pathlib import Path

from toolsafe_lab.experiment import (
    reproduce_ts_guard,
    run_baselines,
    run_semantic_baseline,
)
from toolsafe_lab.cascade_experiment import (
    run_repeated_cascades,
    run_ts_guard_cascades,
)
from toolsafe_lab.fetch import fetch_all
from toolsafe_lab.llm_api import (
    DEFAULT_HOSTED_MODEL_IDS,
    MODEL_SPECS,
    run_llm_evaluation,
)
from toolsafe_lab.llm_batch import (
    collect_batches,
    refresh_batch_states,
    submit_batches,
)
from toolsafe_lab.llm_prompt import DEFAULT_PROMPT_VERSION, PROMPTS
from toolsafe_lab.hosted_cascade import run_hosted_cascades
from toolsafe_lab.hosted_systems import summarize_direct_latency
from toolsafe_lab.paper_artifacts import generate_paper_artifacts
from toolsafe_lab.secondary_analysis import run_secondary_analysis
from toolsafe_lab.error_audit import generate_error_audit
from toolsafe_lab.representation_audit import run_representation_audit
from toolsafe_lab.react_parser import run_react_parser
from toolsafe_lab.representation_experiment import run_representation_ablation
from toolsafe_lab.models import model_factories
from toolsafe_lab.rationale_analysis import run_rationale_analysis
from toolsafe_lab.field_sparse import run_field_sparse_experiment
from toolsafe_lab.standalone_encoder import run_standalone_encoder_experiment
from toolsafe_lab.thought_encoder import run_thought_encoder_ablation
from toolsafe_lab.source_composition_audit import run_source_composition_audit
from toolsafe_lab.synthetic_data import write_synthetic_audit
from toolsafe_lab.synthetic_batch import (
    collect_synthetic_batch,
    refresh_synthetic_batch,
    submit_synthetic_batch,
)
from toolsafe_lab.shortcut_audit import (
    analyze_shortcut_audit,
    prepare_shortcut_audit,
    run_shortcut_audit_inference,
)
from toolsafe_lab.ts_guard_local import (
    TS_GUARD_DEFAULT_SEED,
    TS_GUARD_TRANSFORMERS_MAX_NEW_TOKENS,
    run_ts_guard_shortcut_inference,
)
from toolsafe_lab.ts_guard_shortcut_analysis import analyze_ts_guard_shortcut_audit
from toolsafe_lab.local_qwen import (
    GENERIC_SYSTEM_PROMPTS,
    run_local_qwen_pilot,
)
from toolsafe_lab.qwen_lora import VARIANTS as QWEN_LORA_VARIANTS
from toolsafe_lab.qwen_lora import (
    run_qwen_lora_evaluation,
    run_qwen_lora_experiment,
)
from toolsafe_lab.report import render_report


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    value = argparse.ArgumentParser(
        prog="toolsafe-lab",
        description="Reproduce and extend TS-Bench guardrail experiments.",
    )
    value.add_argument("--project-root", type=Path, default=root)
    subcommands = value.add_subparsers(dest="command", required=True)
    fetch = subcommands.add_parser("fetch", help="Fetch checksum-pinned upstream data")
    fetch.add_argument("--force", action="store_true")
    fetch.add_argument("--workers", type=int, default=4)
    subcommands.add_parser(
        "reproduce", help="Recompute TS-Guard metrics from released predictions"
    )
    subcommands.add_parser("train", help="Train/evaluate/benchmark compact ML baselines")
    subcommands.add_parser(
        "train-semantic",
        help="Train and benchmark the optional frozen-embedding relation model",
    )
    llm = subcommands.add_parser(
        "llm-eval",
        help="Evaluate hosted LLMs with the TS-Guard-compatible structured prompt",
    )
    llm.add_argument(
        "--keys-file",
        type=Path,
        default=Path.home() / "Desktop" / "api_keys.txt",
    )
    llm.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=list(DEFAULT_HOSTED_MODEL_IDS),
    )
    llm.add_argument("--split", choices=("validation", "eval"), default="validation")
    selection = llm.add_mutually_exclusive_group(required=True)
    selection.add_argument("--limit", type=int)
    selection.add_argument("--full", action="store_true")
    llm.add_argument("--seed", type=int, default=260110156)
    llm.add_argument("--concurrency", type=int, default=2)
    llm.add_argument("--max-attempts", type=int, default=5)
    llm.add_argument(
        "--run-label",
        default="quality",
        help="Separate resumable cache namespace, e.g. latency-c1-n10",
    )
    llm.add_argument(
        "--prompt-version",
        choices=sorted(PROMPTS),
        default=DEFAULT_PROMPT_VERSION,
    )
    llm.add_argument(
        "--publish",
        action="store_true",
        help="Merge a complete eval run into the two main result tables",
    )
    batch = subcommands.add_parser(
        "llm-batch",
        help="Submit, inspect, or collect provider batch evaluations",
    )
    batch.add_argument("action", choices=("submit", "status", "collect"))
    batch.add_argument(
        "--keys-file",
        type=Path,
        default=Path.home() / "Desktop" / "api_keys.txt",
    )
    batch.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=list(DEFAULT_HOSTED_MODEL_IDS),
    )
    batch.add_argument(
        "--prompt-version",
        choices=sorted(PROMPTS),
        default=DEFAULT_PROMPT_VERSION,
    )
    batch.add_argument(
        "--publish",
        action="store_true",
        help="Publish collected full-evaluation summaries to the main tables",
    )
    cascade = subcommands.add_parser(
        "cascade",
        help="Calibrate local guards and evaluate validation-frozen TS-Guard cascades",
    )
    cascade.add_argument("--seed", type=int, default=260110156)
    cascade.add_argument("--bootstrap-replicates", type=int, default=10_000)
    cascade_seeds = subcommands.add_parser(
        "cascade-seeds",
        help="Repeat validation-frozen conservative cascades across training seeds",
    )
    cascade_seeds.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[260110156, 260110157, 260110158, 260110159, 260110160],
    )
    cascade_seeds.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip the frozen MiniLM relation baseline",
    )
    hosted_cascade = subcommands.add_parser(
        "hosted-cascade",
        help="Evaluate every calibrated local gate in front of collected hosted monitors",
    )
    hosted_cascade.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
    )
    hosted_cascade.add_argument("--bootstrap-replicates", type=int, default=10_000)
    hosted_cascade.add_argument(
        "--prompt-version",
        choices=sorted(PROMPTS),
        default=DEFAULT_PROMPT_VERSION,
    )
    hosted_systems = subcommands.add_parser(
        "hosted-systems",
        help="Summarize cached direct hosted latency and composed cascade latency",
    )
    hosted_systems.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=list(DEFAULT_HOSTED_MODEL_IDS),
    )
    hosted_systems.add_argument(
        "--prompt-version",
        choices=sorted(PROMPTS),
        default=DEFAULT_PROMPT_VERSION,
    )
    hosted_systems.add_argument("--run-label", default="latency-c1-n256")
    hosted_systems.add_argument("--limit", type=int, default=256)
    subcommands.add_parser("report", help="Render the two result tables as Markdown")
    subcommands.add_parser(
        "paper-artifacts",
        help="Generate LaTeX tables and vector figures from frozen results",
    )
    subcommands.add_parser(
        "secondary",
        help="Run label-mode, subgroup, field, and training-source sensitivity analyses",
    )
    subcommands.add_parser(
        "representation-audit",
        help="Audit field schemas and ReAct markers without emitting benchmark text",
    )
    subcommands.add_parser(
        "parse-actions",
        help="Derive deterministic execution-only rows and aggregate parser coverage",
    )
    representation_ablation = subcommands.add_parser(
        "representation-ablation",
        help="Compare full-ReAct, Thought-removed, and execution-only local guards",
    )
    representation_ablation.add_argument(
        "--models",
        nargs="+",
        choices=sorted(model_factories()),
        default=list(model_factories()),
    )
    representation_ablation.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
    )
    rationale = subcommands.add_parser(
        "rationale-analysis",
        help="Measure Thought-only, severity, subset, and route-overlap effects",
    )
    rationale.add_argument("--bootstrap-replicates", type=int, default=10_000)
    field_sparse = subcommands.add_parser(
        "field-sparse",
        help="Compare concatenated and field-namespaced execution-only sparse guards",
    )
    field_sparse.add_argument("--bootstrap-replicates", type=int, default=10_000)
    standalone = subcommands.add_parser(
        "train-standalone-encoders",
        help="Fine-tune compact standalone execution-context safety encoders",
    )
    standalone.add_argument(
        "--models",
        default="minilm_l12,deberta_v3_xsmall",
        help="Comma-separated frozen encoder keys",
    )
    standalone.add_argument("--device", default="auto")
    standalone.add_argument("--epochs", type=int, default=5)
    standalone.add_argument(
        "--resume",
        action="store_true",
        help="Resume every requested encoder from its last completed epoch",
    )
    thought_encoder = subcommands.add_parser(
        "train-thought-encoders",
        help="Fine-tune and symmetrically test compact encoders with ReAct Thought",
    )
    thought_encoder.add_argument(
        "--models",
        default="minilm_l12,deberta_v3_xsmall",
        help="Comma-separated frozen encoder keys",
    )
    thought_encoder.add_argument("--device", default="auto")
    thought_encoder.add_argument("--epochs", type=int, default=5)
    thought_encoder.add_argument(
        "--resume",
        action="store_true",
        help="Resume every requested Thought encoder from its last completed epoch",
    )
    multitask = subcommands.add_parser(
        "train-multitask-minilm",
        help="Train gold auxiliary-task and relation-fusion MiniLM guards",
    )
    multitask.add_argument(
        "--variants",
        default="packed_multitask,relation_multitask",
        help="Comma-separated frozen follow-up MiniLM variants",
    )
    multitask.add_argument("--device", default="auto")
    multitask.add_argument("--epochs", type=int, default=5)
    multitask.add_argument(
        "--teacher-targets",
        type=Path,
        help="Optional training-only JSONL with sample_id and unsafe_probability",
    )
    multitask.add_argument(
        "--teacher-alpha",
        type=float,
        default=0.0,
        help="Weight on teacher soft targets; requires --teacher-targets",
    )
    subcommands.add_parser(
        "standalone-source-audit",
        help="Audit aggregate label/source geometry for the standalone experiment",
    )
    local_qwen = subcommands.add_parser(
        "local-qwen-pilot",
        help="Run the frozen sub-1B local Qwen validation pilot",
    )
    local_qwen.add_argument(
        "--models",
        default="qwen3guard_gen_0_6b,qwen3_0_6b",
        help="Comma-separated frozen local Qwen model keys",
    )
    local_qwen.add_argument("--limit", type=int, default=90)
    local_qwen.add_argument("--seed", type=int, default=260110156)
    local_qwen.add_argument("--device", default="auto")
    local_qwen.add_argument(
        "--prompt-version",
        choices=sorted(GENERIC_SYSTEM_PROMPTS),
        default="causal_binary_v1",
    )
    qwen_lora = subcommands.add_parser(
        "train-local-qwen",
        help="Train frozen Qwen3-0.6B classification-head/LoRA candidates",
    )
    qwen_lora.add_argument(
        "--variants",
        default=",".join(QWEN_LORA_VARIANTS),
        help="Comma-separated frozen sampler variants",
    )
    qwen_lora.add_argument("--device", default="auto")
    qwen_lora.add_argument("--epochs", type=int, default=3)
    qwen_lora.add_argument("--resume", action="store_true")
    qwen_lora_eval = subcommands.add_parser(
        "evaluate-local-qwen",
        help="Evaluate the frozen, validation-selected Qwen LoRA guard once",
    )
    qwen_lora_eval.add_argument("--device", default="auto")
    synthetic_audit = subcommands.add_parser(
        "synthetic-audit",
        help="Validate and benchmark-deduplicate an unreviewed synthetic JSONL file",
    )
    synthetic_audit.add_argument("input", type=Path)
    synthetic_batch = subcommands.add_parser(
        "synthetic-batch",
        help="Submit, inspect, or collect the frozen synthetic-triad pilot",
    )
    synthetic_batch.add_argument("action", choices=("submit", "status", "collect"))
    synthetic_batch.add_argument(
        "--keys-file",
        type=Path,
        default=Path.home() / "Desktop" / "api_keys.txt",
    )
    shortcut = subcommands.add_parser(
        "shortcut-audit",
        help="Prepare, score, or analyze the frozen benchmark shortcut audit",
    )
    shortcut.add_argument(
        "action",
        choices=(
            "prepare",
            "infer",
            "infer-ts-guard",
            "analyze",
            "analyze-ts-guard",
        ),
    )
    shortcut.add_argument(
        "--cohort",
        choices=("agentdojo", "authorization", "all"),
        default="all",
        help="Cohort for inference/analysis; prepare always materializes both",
    )
    shortcut.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        help=(
            "Override the protocol-default models for every selected cohort; "
            "gpt-5.6-sol is an explicit post-hoc replication"
        ),
    )
    shortcut.add_argument(
        "--synthetic-input",
        type=Path,
        default=Path("artifacts/synthetic/raw/pilot.jsonl"),
    )
    shortcut.add_argument(
        "--keys-file",
        type=Path,
        default=Path.home() / "Desktop" / "api_keys.txt",
    )
    shortcut.add_argument("--concurrency", type=int, default=2)
    shortcut.add_argument("--max-attempts", type=int, default=5)
    shortcut.add_argument("--bootstrap-replicates", type=int, default=10_000)
    shortcut.add_argument(
        "--ts-guard-backend",
        choices=("transformers", "vllm", "mlx_bf16", "mlx_q8"),
        default="transformers",
        help="Local TS-Guard runtime; MLX choices require an explicit local model path",
    )
    shortcut.add_argument(
        "--ts-guard-device",
        default="auto",
        help="Transformers device: auto, mps, cpu, cuda, or cuda:N",
    )
    shortcut.add_argument(
        "--ts-guard-max-new-tokens",
        type=int,
        default=TS_GUARD_TRANSFORMERS_MAX_NEW_TOKENS,
        help="Transformers continuation budget; inputs are never truncated",
    )
    shortcut.add_argument(
        "--ts-guard-decoding",
        choices=("greedy", "authors_sampling"),
        default="greedy",
        help="Greedy is the deterministic paired-audit primary",
    )
    shortcut.add_argument(
        "--ts-guard-seed",
        type=int,
        default=TS_GUARD_DEFAULT_SEED,
        help="Base seed for deterministic per-row authors-style sampling",
    )
    shortcut.add_argument(
        "--ts-guard-mlx-model-path",
        type=Path,
        help="Pinned HF snapshot or converted MLX artifact for an MLX backend",
    )
    shortcut.add_argument(
        "--ts-guard-max-records",
        type=int,
        help="Score at most this many missing records per cohort, then resume later",
    )
    audit = subcommands.add_parser(
        "error-audit",
        help="Generate the frozen, text-free error-audit sampling manifest",
    )
    audit.add_argument(
        "--prompt-version",
        choices=sorted(PROMPTS),
        default=DEFAULT_PROMPT_VERSION,
    )
    audit.add_argument("--limit-per-category", type=int, default=50)
    return value


def main() -> None:
    args = parser().parse_args()
    project_root = args.project_root.resolve()
    data_root = project_root / "data" / "raw"
    results_root = project_root / "results"
    artifacts_root = project_root / "artifacts"

    if args.command == "fetch":
        rows = fetch_all(data_root, force=args.force, workers=args.workers)
        downloaded = sum(int(row["size"]) for row in rows)
        print(f"Verified {len(rows)} assets ({downloaded / (1024**2):.1f} MiB manifest size)")
    elif args.command == "reproduce":
        reproduce_ts_guard(data_root, results_root)
    elif args.command == "train":
        run_baselines(data_root, results_root, artifacts_root)
    elif args.command == "train-semantic":
        run_semantic_baseline(data_root, results_root, artifacts_root)
    elif args.command == "llm-eval":
        run_llm_evaluation(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
            keys_file=args.keys_file.resolve(),
            model_ids=args.models,
            split=args.split,
            limit=None if args.full else args.limit,
            seed=args.seed,
            concurrency=args.concurrency,
            max_attempts=args.max_attempts,
            prompt_version=args.prompt_version,
            publish=args.publish,
            run_label=args.run_label,
        )
    elif args.command == "cascade":
        run_ts_guard_cascades(
            project_root=project_root,
            data_root=data_root,
            results_root=results_root,
            seed=args.seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    elif args.command == "llm-batch":
        common = {
            "artifacts_root": artifacts_root,
            "keys_file": args.keys_file.resolve(),
            "model_ids": args.models,
            "prompt_version": args.prompt_version,
        }
        if args.action == "submit":
            submit_batches(data_root=data_root, **common)
        elif args.action == "status":
            refresh_batch_states(**common)
        else:
            collect_batches(
                data_root=data_root,
                results_root=results_root,
                publish=args.publish,
                **common,
            )
    elif args.command == "cascade-seeds":
        run_repeated_cascades(
            data_root=data_root,
            results_root=results_root,
            seeds=args.seeds,
            include_semantic=not args.no_semantic,
        )
    elif args.command == "hosted-cascade":
        run_hosted_cascades(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
            prompt_version=args.prompt_version,
            model_ids=args.models,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    elif args.command == "hosted-systems":
        summarize_direct_latency(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
            model_ids=args.models,
            prompt_version=args.prompt_version,
            run_label=args.run_label,
            limit=args.limit,
        )
    elif args.command == "report":
        output = render_report(results_root)
        print(output)
    elif args.command == "paper-artifacts":
        generate_paper_artifacts(
            results_root=results_root,
            paper_root=project_root / "paper",
        )
    elif args.command == "secondary":
        run_secondary_analysis(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
        )
    elif args.command == "representation-audit":
        run_representation_audit(
            data_root=data_root,
            results_root=results_root,
        )
    elif args.command == "parse-actions":
        run_react_parser(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
        )
    elif args.command == "representation-ablation":
        run_representation_ablation(
            project_root=project_root,
            data_root=data_root,
            results_root=results_root,
            model_names=args.models,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    elif args.command == "rationale-analysis":
        run_rationale_analysis(
            data_root=data_root,
            results_root=results_root,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    elif args.command == "field-sparse":
        run_field_sparse_experiment(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    elif args.command == "train-standalone-encoders":
        run_standalone_encoder_experiment(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
            model_keys=tuple(
                value.strip() for value in args.models.split(",") if value.strip()
            ),
            requested_device=args.device,
            epochs=args.epochs,
            resume=args.resume,
        )
    elif args.command == "train-thought-encoders":
        run_thought_encoder_ablation(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
            model_keys=tuple(
                value.strip() for value in args.models.split(",") if value.strip()
            ),
            requested_device=args.device,
            epochs=args.epochs,
            resume=args.resume,
        )
    elif args.command == "standalone-source-audit":
        run_source_composition_audit(
            data_root=data_root,
            results_root=results_root,
        )
    elif args.command == "train-multitask-minilm":
        from toolsafe_lab.multitask_encoder import run_multitask_encoder_experiment

        run_multitask_encoder_experiment(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
            variants=tuple(
                value.strip() for value in args.variants.split(",") if value.strip()
            ),
            requested_device=args.device,
            epochs=args.epochs,
            teacher_targets_path=args.teacher_targets,
            teacher_alpha=args.teacher_alpha,
        )
    elif args.command == "local-qwen-pilot":
        run_local_qwen_pilot(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
            model_keys=tuple(
                value.strip() for value in args.models.split(",") if value.strip()
            ),
            limit=args.limit,
            seed=args.seed,
            requested_device=args.device,
            prompt_version=args.prompt_version,
        )
    elif args.command == "train-local-qwen":
        run_qwen_lora_experiment(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
            variants=tuple(
                value.strip() for value in args.variants.split(",") if value.strip()
            ),
            requested_device=args.device,
            epochs=args.epochs,
            resume=args.resume,
        )
    elif args.command == "evaluate-local-qwen":
        run_qwen_lora_evaluation(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
            requested_device=args.device,
        )
    elif args.command == "synthetic-audit":
        write_synthetic_audit(
            input_path=args.input,
            data_root=data_root,
            results_root=results_root,
        )
    elif args.command == "synthetic-batch":
        if args.action == "submit":
            submit_synthetic_batch(
                artifacts_root=artifacts_root,
                keys_file=args.keys_file.resolve(),
            )
        elif args.action == "status":
            refresh_synthetic_batch(
                artifacts_root=artifacts_root,
                keys_file=args.keys_file.resolve(),
            )
        else:
            collect_synthetic_batch(
                data_root=data_root,
                results_root=results_root,
                artifacts_root=artifacts_root,
                keys_file=args.keys_file.resolve(),
            )
    elif args.command == "shortcut-audit":
        synthetic_input = args.synthetic_input
        if not synthetic_input.is_absolute():
            synthetic_input = project_root / synthetic_input
        common = {
            "project_root": project_root,
            "data_root": data_root,
            "results_root": results_root,
            "artifacts_root": artifacts_root,
            "synthetic_input": synthetic_input.resolve(),
        }
        if args.action == "prepare":
            prepare_shortcut_audit(**common)
        elif args.action == "infer":
            run_shortcut_audit_inference(
                **common,
                keys_file=args.keys_file.resolve(),
                cohort=args.cohort,
                model_ids=args.models,
                concurrency=args.concurrency,
                max_attempts=args.max_attempts,
            )
        elif args.action == "infer-ts-guard":
            cohorts, _ = prepare_shortcut_audit(**common)
            run_ts_guard_shortcut_inference(
                cohorts=cohorts,
                artifacts_root=artifacts_root,
                cohort=args.cohort,
                backend=args.ts_guard_backend,
                requested_device=args.ts_guard_device,
                max_new_tokens=args.ts_guard_max_new_tokens,
                decoding=args.ts_guard_decoding,
                seed=args.ts_guard_seed,
                mlx_model_path=(
                    args.ts_guard_mlx_model_path.resolve()
                    if args.ts_guard_mlx_model_path is not None
                    else None
                ),
                max_records=args.ts_guard_max_records,
            )
        elif args.action == "analyze-ts-guard":
            analyze_ts_guard_shortcut_audit(
                **common,
                cohort=args.cohort,
                bootstrap_replicates=args.bootstrap_replicates,
                backend=args.ts_guard_backend,
                decoding=args.ts_guard_decoding,
            )
        else:
            analyze_shortcut_audit(
                **common,
                cohort=args.cohort,
                model_ids=args.models,
                bootstrap_replicates=args.bootstrap_replicates,
            )
    elif args.command == "error-audit":
        generate_error_audit(
            data_root=data_root,
            results_root=results_root,
            artifacts_root=artifacts_root,
            prompt_version=args.prompt_version,
            limit_per_category=args.limit_per_category,
        )


if __name__ == "__main__":
    main()
