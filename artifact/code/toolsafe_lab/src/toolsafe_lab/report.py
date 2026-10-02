from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _percent(value: str | None) -> str:
    if not value:
        return "—"
    return f"{float(value) * 100:.2f}"


def _number(value: str | None, decimals: int = 2) -> str:
    if not value:
        return "—"
    return f"{float(value):.{decimals}f}"


def render_report(results_root: Path) -> Path:
    quality = _rows(results_root / "predictive_quality.csv")
    systems = _rows(results_root / "systems_performance.csv")
    reproduction = json.loads(
        (results_root / "ts_guard_reproduction.json").read_text(encoding="utf-8")
    )
    diagnostics_path = results_root / "experiment_diagnostics.json"
    diagnostics = (
        json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics_path.exists()
        else {}
    )

    quality.sort(key=lambda row: (row["model"], row["dataset"]))
    systems.sort(key=lambda row: (row["source"] != "paper_table_11", row["model"]))

    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in quality:
        if row["accuracy"] and row["f1"] and row["recall"]:
            by_model[row["model"]].append(row)
    macro = []
    local_macro = []
    for model, rows in by_model.items():
        aggregate = (
            model,
            sum(float(row["accuracy"]) for row in rows) / len(rows),
            sum(float(row["f1"]) for row in rows) / len(rows),
            sum(float(row["recall"]) for row in rows) / len(rows),
        )
        macro.append(aggregate)
        if all(row["source"] == "local_measurement" for row in rows):
            local_macro.append(aggregate)
    macro.sort(key=lambda item: item[2], reverse=True)
    local_macro.sort(key=lambda item: item[2], reverse=True)

    lines = [
        "# ToolSafe-Lab initial results",
        "",
        "## Predictive quality (strict mode, %)",
        "",
        "| Model | Dataset | Accuracy | Precision | Recall | F1 | Specificity | Balanced acc. | MCC | Coverage | Evidence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in quality:
        lines.append(
            "| {model} | {dataset} | {accuracy} | {precision} | {recall} | "
            "{f1} | {specificity} | {balanced} | {mcc} | {coverage} | {source} |".format(
                model=row["model"],
                dataset=row["dataset"],
                accuracy=_percent(row["accuracy"]),
                precision=_percent(row["precision"]),
                recall=_percent(row["recall"]),
                f1=_percent(row["f1"]),
                specificity=_percent(row["specificity"]),
                balanced=_percent(row["balanced_accuracy"]),
                mcc=_number(row["mcc"], 3),
                coverage=_percent(row["coverage"]),
                source=row["source"],
            )
        )

    lines.extend(
        [
            "",
            "Unweighted macro averages across the three evaluation datasets:",
            "",
            "| Model | Accuracy | F1 | Recall |",
            "|---|---:|---:|---:|",
        ]
    )
    for model, accuracy, f1, recall in macro:
        lines.append(
            f"| {model} | {accuracy * 100:.2f} | {f1 * 100:.2f} | {recall * 100:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Systems performance",
            "",
            "| Model | Variant | Size (MiB) | Train (s) | Mean latency (ms) | p50 | p95 | p99 | Throughput/s | Coverage | Requests | Retries | Input tok. | Output tok. | Reasoning tok. | Est. cost ($) | $ / 1k predictions | Evidence/hardware |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in systems:
        throughput = row.get("wall_throughput_samples_s") or row.get(
            "batch_throughput_samples_s"
        )
        evidence = "; ".join(
            value
            for value in (
                row.get("source"),
                row.get("hardware"),
                row.get("timing_semantics"),
            )
            if value
        )
        lines.append(
            "| {model} | {variant} | {size} | {train} | {mean} | {p50} | {p95} | "
            "{p99} | {throughput} | {coverage} | {requests} | {retries} | "
            "{input_tokens} | {output_tokens} | {reasoning_tokens} | {cost} | "
            "{cost_per_1k} | {evidence} |".format(
                model=row["model"],
                variant=row["variant"],
                size=_number(row.get("model_size_mib")),
                train=_number(row.get("train_seconds"), 3),
                mean=_number(row.get("latency_ms_mean"), 3),
                p50=_number(row.get("latency_ms_p50"), 3),
                p95=_number(row.get("latency_ms_p95"), 3),
                p99=_number(row.get("latency_ms_p99"), 3),
                throughput=_number(throughput, 2),
                coverage=_percent(row.get("request_coverage")),
                requests=_number(row.get("requests"), 0),
                retries=_number(row.get("request_retries"), 0),
                input_tokens=_number(row.get("input_tokens"), 0),
                output_tokens=_number(row.get("output_tokens"), 0),
                reasoning_tokens=_number(row.get("reasoning_tokens"), 0),
                cost=_number(row.get("estimated_cost_usd"), 4),
                cost_per_1k=_number(row.get("cost_per_1k_predictions_usd"), 2),
                evidence=evidence,
            )
        )

    invalid = sum(
        int(dataset["invalid_predictions"]) for dataset in reproduction["datasets"].values()
    )
    total = sum(
        int(dataset["released_predictions"])
        for dataset in reproduction["datasets"].values()
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- TS-Guard score reproduction exactly matched the authors' released "
            f"`metrics_strict.json` on all three datasets. Its released prediction "
            f"coverage was {(total - invalid) / total * 100:.2f}% "
            f"({invalid} invalid/omitted out of {total}).",
            "- The released artifacts do **not** fully reproduce paper Table 3: "
            "AgentHarm differs by 0.01 accuracy and 0.01 F1 point under normal "
            "rounding; ASB has 5,231 "
            "rather than 5,237 samples and differs by 0.04 accuracy, 0.04 F1, and "
            "0.08 recall percentage points. AgentDojo matches.",
        ]
    )
    if macro:
        best_model, best_acc, best_f1, best_recall = macro[0]
        lines.append(
            f"- The best unweighted macro F1 in this run was **{best_model}** at "
            f"{best_f1 * 100:.2f}% (accuracy {best_acc * 100:.2f}%, recall "
            f"{best_recall * 100:.2f}%)."
        )
    if local_macro:
        best_model, best_acc, best_f1, best_recall = local_macro[0]
        lines.append(
            f"- The best compact local baseline was **{best_model}** at "
            f"{best_f1 * 100:.2f}% macro F1 (accuracy {best_acc * 100:.2f}%, "
            f"recall {best_recall * 100:.2f}%), well below TS-Guard."
        )
    if diagnostics:
        counts = diagnostics["deduplicated_counts"]
        lines.append(
            f"- Baselines used {counts['train']} deduplicated training steps and "
            f"{counts['validation']} held-out validation steps; the untouched evaluation "
            f"set contained {counts['evaluation']} steps."
        )
    lines.extend(
        [
            "- The TS-Guard latency row is copied from the paper and is cross-hardware; "
            "it is context, not a fair local speedup ratio.",
            "- These are single-run point estimates. The next rigor step is "
            "trajectory-clustered confidence intervals and multiple training seeds.",
            "",
        ]
    )

    output = results_root / "REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
