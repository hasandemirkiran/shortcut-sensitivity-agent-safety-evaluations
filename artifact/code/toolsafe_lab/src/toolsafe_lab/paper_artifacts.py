from __future__ import annotations

import csv
import json
import math
from pathlib import Path


MODEL_LABELS = {
    "word_tfidf_logreg": "Word TF--IDF",
    "char_tfidf_linearsvc": "Character TF--IDF",
    "hybrid_tfidf_linearsvc": "Hybrid TF--IDF",
    "hash_sgd_compact": "Hashing SGD",
    "minilm_relation_logreg": "MiniLM relations",
}
COLORS = {
    "word_tfidf_logreg": "#0072B2",
    "char_tfidf_linearsvc": "#E69F00",
    "hybrid_tfidf_linearsvc": "#009E73",
    "hash_sgd_compact": "#CC79A7",
    "minilm_relation_logreg": "#56B4E9",
}


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object in {path}")
    return value


def _escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def _percent(value: object, digits: int = 2) -> str:
    return f"{float(value) * 100:.{digits}f}"


def _metrics_from_counts(
    *,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> dict[str, float]:
    n = tn + fp + fn + tp
    accuracy = (tn + tp) / n
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _cascade_overall(cascade: dict[str, object]) -> str:
    monitor = cascade["monitor"]
    monitor_micro = monitor["micro"]  # type: ignore[index]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"System & Saved & Precision & Recall & Specificity & F1 \\",
        r"\midrule",
        (
            "TS-Guard only & 0.00 & "
            f"{_percent(monitor_micro['precision'])} & "  # type: ignore[index]
            f"{_percent(monitor_micro['recall'])} & "  # type: ignore[index]
            f"{_percent(monitor_micro['specificity'])} & "  # type: ignore[index]
            f"{_percent(monitor_micro['f1'])} \\\\"  # type: ignore[index]
        ),
    ]
    for model_name, model in cascade["models"].items():  # type: ignore[union-attr]
        result = model["policies"]["0.99"]["micro"]
        label = MODEL_LABELS.get(model_name, model_name)
        prefix = (
            r"\textbf{" + label + "}"
            if model_name == "hybrid_tfidf_linearsvc"
            else label
        )
        lines.append(
            f"{prefix} $\\rightarrow$ TS-Guard & "
            f"{_percent(result['monitor_call_reduction'])} & "
            f"{_percent(result['precision'])} & "
            f"{_percent(result['recall'])} & "
            f"{_percent(result['specificity'])} & "
            f"{_percent(result['f1'])} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _cascade_domains(cascade: dict[str, object]) -> str:
    best = cascade["models"]["hybrid_tfidf_linearsvc"]["policies"]["0.99"]  # type: ignore[index]
    monitor = cascade["monitor"]["by_dataset"]  # type: ignore[index]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Domain & $n$ & Saved & Recall & $\Delta$R & Specificity & $\Delta$S \\",
        r"\midrule",
    ]
    for dataset, result in best["by_dataset"].items():
        baseline = monitor[dataset]
        lines.append(
            f"{_escape(dataset.replace('-Traj', ''))} & {int(result['n'])} & "
            f"{_percent(result['monitor_call_reduction'])} & "
            f"{_percent(result['recall'])} & "
            f"{(float(result['recall']) - float(baseline['recall'])) * 100:+.2f} & "
            f"{_percent(result['specificity'])} & "
            f"{(float(result['specificity']) - float(baseline['specificity'])) * 100:+.2f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _robustness_table(cascade: dict[str, object]) -> str:
    robustness = cascade["models"]["hybrid_tfidf_linearsvc"]["robustness"]  # type: ignore[index]
    names = {
        "tool_names": "Tool names",
        "argument_keys": "Argument keys",
        "field_order": "Field order",
        "irrelevant_tool": "Irrelevant tool",
        "untrusted_injection": "Untrusted injection",
        "format_noise": "Format noise",
        "combined": "Combined",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Perturbation & Saved & Gate recall & Safe among allowed & New leaks \\",
        r"\midrule",
    ]
    for key, result in robustness.items():
        lines.append(
            f"{names[key]} & {_percent(result['allow_rate'])} & "
            f"{_percent(result['ideal_gate_recall'])} & "
            f"{_percent(result['safe_fraction_among_allowed'])} & "
            f"{int(result['newly_allowed_harmful'])} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _quality_table(results_root: Path) -> str:
    with (results_root / "predictive_quality.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    selected_sources = {"official_cached_predictions", "local_measurement"}
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["source"] not in selected_sources:
            continue
        model = row["model"]
        bucket = counts.setdefault(
            model, {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        )
        for field in bucket:
            bucket[field] += int(row[field])
    labels = {
        "TS-Guard (released artifacts)": "TS-Guard",
        **MODEL_LABELS,
    }
    order = [
        "TS-Guard (released artifacts)",
        "word_tfidf_logreg",
        "char_tfidf_linearsvc",
        "hybrid_tfidf_linearsvc",
        "hash_sgd_compact",
        "minilm_relation_logreg",
    ]
    table_rows: list[tuple[str, dict[str, float], float]] = []
    for model in order:
        metrics = _metrics_from_counts(**counts[model])
        table_rows.append((labels[model], metrics, 1.0))

    hosted_path = results_root / "cascade" / "hosted_cascades.json"
    if hosted_path.exists():
        hosted = _load_json(hosted_path)
        for monitor in hosted["monitors"].values():  # type: ignore[union-attr]
            table_rows.append(
                (
                    str(monitor["model"]),
                    monitor["micro"],
                    float(monitor["request_coverage"]),
                )
            )

    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & Accuracy & Precision & Recall & Specificity & F1 & MCC & Coverage \\",
        r"\midrule",
    ]
    for label, metrics, coverage in table_rows:
        lines.append(
            f"{_escape(label)} & {_percent(metrics['accuracy'])} & "
            f"{_percent(metrics['precision'])} & {_percent(metrics['recall'])} & "
            f"{_percent(metrics['specificity'])} & {_percent(metrics['f1'])} & "
            f"{float(metrics['mcc']):.3f} & {_percent(coverage)} \\\\"
        )
        if label == "TS-Guard":
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _systems_table(results_root: Path) -> str:
    with (results_root / "systems_performance.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        "TS-Guard": "TS-Guard (source paper)",
        "word_tfidf_logreg": "Word TF--IDF",
        "hybrid_tfidf_linearsvc": "Hybrid TF--IDF",
        "minilm_relation_logreg": "MiniLM relations",
    }
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"System & Size (MiB) & Mean (ms) & p50 (ms) & p95 (ms) & p99 (ms) & Throughput/s & Cost/1k \\",
        r"\midrule",
    ]
    for row in rows:
        model = row["model"]
        if model not in selected:
            continue
        lines.append(
            f"{selected[model]} & "
            f"{float(row['model_size_mib']):.2f} & "
            f"{float(row['latency_ms_mean']):.2f} & "
            f"{float(row['latency_ms_p50']):.2f} & "
            f"{float(row['latency_ms_p95']):.2f} & "
            f"{float(row['latency_ms_p99']):.2f} & "
            f"{float(row['batch_throughput_samples_s']):.1f} & -- \\\\"
            if row["latency_ms_p50"]
            else (
                f"{selected[model]} & {float(row['model_size_mib']):.2f} & "
                f"{float(row['latency_ms_mean']):.1f} & -- & -- & -- & -- & -- \\\\"
            )
        )
    direct_path = results_root / "llm_eval" / "direct_latency_n256.json"
    if direct_path.exists():
        direct = _load_json(direct_path)
        for summary in direct["summaries"]:  # type: ignore[union-attr]
            system = summary["system_row"]
            lines.append(
                f"{_escape(summary['model'])} (direct) & -- & "
                f"{float(system['latency_ms_mean']):.1f} & "
                f"{float(system['latency_ms_p50']):.1f} & "
                f"{float(system['latency_ms_p95']):.1f} & "
                f"{float(system['latency_ms_p99']):.1f} & "
                f"{float(system['wall_throughput_samples_s']):.3f} & "
                f"\\${float(system['cost_per_1k_predictions_usd']):.2f} \\\\"
            )
            cascade = summary["hybrid_cascade_simulation"]
            lines.append(
                f"\\quad + hybrid gate (sim.) & 1.61 & "
                f"{float(cascade['latency_ms_mean']):.1f} & "
                f"{float(cascade['latency_ms_p50']):.1f} & "
                f"{float(cascade['latency_ms_p95']):.1f} & "
                f"{float(cascade['latency_ms_p99']):.1f} & -- & "
                f"\\${float(cascade['cost_per_1k_successful_composed_decisions_usd']):.2f} \\\\"
            )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _hosted_table(results_root: Path) -> str:
    path = results_root / "llm_eval" / "eval-full-batch.json"
    if not path.exists():
        return "% Hosted full-evaluation batches are still processing."
    result = _load_json(path)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Monitor & Accuracy & Precision & Recall & F1 & Coverage & Cost (\$) \\",
        r"\midrule",
    ]
    for summary in result["summaries"]:  # type: ignore[union-attr]
        macro = summary["macro"]
        system = summary["system_row"]
        lines.append(
            f"{_escape(summary['model'])} & "
            f"{_percent(macro['accuracy'])} & "
            f"{_percent(macro['precision'])} & "
            f"{_percent(macro['recall'])} & "
            f"{_percent(macro['f1'])} & "
            f"{_percent(system['request_coverage'])} & "
            f"{float(system['estimated_cost_usd']):.2f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _hosted_cascade_table(results_root: Path) -> str:
    path = results_root / "cascade" / "hosted_cascades.json"
    if not path.exists():
        return (
            r"\begin{tabular}{l}\toprule Hosted batches are still processing. "
            r"\\ \bottomrule\end{tabular}"
        )
    result = _load_json(path)
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Monitor & Mode & Saved & Cost saved & Precision & Recall & Specificity & F1 & Coverage \\",
        r"\midrule",
    ]
    for monitor in result["monitors"].values():  # type: ignore[union-attr]
        name = _escape(str(monitor["model"]))
        baseline = monitor["micro"]
        coverage = float(monitor["request_coverage"])
        lines.append(
            f"{name} & Monitor & 0.00 & 0.00 & "
            f"{_percent(baseline['precision'])} & {_percent(baseline['recall'])} & "
            f"{_percent(baseline['specificity'])} & {_percent(baseline['f1'])} & "
            f"{_percent(coverage)} \\\\"
        )
        cascade = monitor["cascades"]["hybrid_tfidf_linearsvc"]
        metrics = cascade["micro"]
        lines.append(
            f" & Cascade & {_percent(cascade['monitor_request_reduction'])} & "
            f"{_percent(cascade['observed_cost_reduction'])} & "
            f"{_percent(metrics['precision'])} & {_percent(metrics['recall'])} & "
            f"{_percent(metrics['specificity'])} & {_percent(metrics['f1'])} & "
            f"{_percent(coverage)} \\\\"
        )
        lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _secondary_table(results_root: Path) -> str:
    path = results_root / "secondary_analysis.json"
    if not path.exists():
        return r"\begin{tabular}{l}\toprule Not run.\\\bottomrule\end{tabular}"
    result = _load_json(path)
    fields = result["field_ablations"]
    sources = result["training_source_ablations"]
    labels = {
        "none": "All fields",
        "drop_instruction": "Drop user request",
        "drop_history": "Drop history",
        "drop_current_action": "Drop current action",
        "drop_env_info": "Drop tool descriptions",
    }
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Training/input condition & Saved & Recall & Specificity \\",
        r"\midrule",
    ]
    for key, label in labels.items():
        metrics = fields[key]["metrics"]
        lines.append(
            f"{label} & {_percent(metrics['monitor_call_reduction'])} & "
            f"{_percent(metrics['recall'])} & {_percent(metrics['specificity'])} \\\\"
        )
    lines.append(r"\midrule")
    for source, label in (
        ("ASB-Traj", "Train ASB only"),
        ("AgentAlign-Traj", "Train AgentAlign only"),
    ):
        metrics = sources[source]["metrics"]
        lines.append(
            f"{label} & {_percent(metrics['monitor_call_reduction'])} & "
            f"{_percent(metrics['recall'])} & {_percent(metrics['specificity'])} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _tradeoff_figure(cascade: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    baseline_recall = float(cascade["monitor"]["micro"]["recall"]) * 100  # type: ignore[index]
    figure, axis = plt.subplots(figsize=(3.35, 2.35))
    for model_name, model in cascade["models"].items():  # type: ignore[union-attr]
        points = sorted(
            (
                float(result["micro"]["monitor_call_reduction"]) * 100,
                (float(result["micro"]["recall"]) * 100) - baseline_recall,
            )
            for result in model["policies"].values()
        )
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )
    axis.axhline(
        -1.0, color="#666666", linestyle="--", linewidth=0.9, label="-1 pp bound"
    )
    axis.axhline(0.0, color="#999999", linestyle=":", linewidth=0.7)
    axis.set_xlabel("TS-Guard calls avoided (%)")
    axis.set_ylabel("Recall change (percentage points)")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.legend(frameon=False, ncol=2, loc="lower left")
    figure.tight_layout(pad=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _robustness_figure(cascade: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    robustness = cascade["models"]["hybrid_tfidf_linearsvc"]["robustness"]  # type: ignore[index]
    labels = {
        "tool_names": "tool",
        "argument_keys": "args",
        "field_order": "order",
        "irrelevant_tool": "extra",
        "untrusted_injection": "inject",
        "format_noise": "format",
        "combined": "combined",
    }
    offsets = {
        "tool_names": (3, 3),
        "argument_keys": (-18, -10),
        "field_order": (3, 8),
        "irrelevant_tool": (3, 3),
        "untrusted_injection": (5, -10),
        "format_noise": (4, -7),
        "combined": (3, 3),
    }
    figure, axis = plt.subplots(figsize=(3.35, 2.2))
    for key, result in robustness.items():
        x = float(result["allow_rate"]) * 100
        y = float(result["ideal_gate_recall"]) * 100
        axis.scatter(x, y, s=20, color="#009E73")
        axis.annotate(
            labels[key],
            (x, y),
            xytext=offsets[key],
            textcoords="offset points",
            fontsize=6.2,
        )
    axis.axhline(99.0, color="#666666", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Calls avoided under perturbation (%)")
    axis.set_ylabel("Ideal gate recall (%)")
    axis.set_ylim(99.45, 100.03)
    axis.grid(alpha=0.2, linewidth=0.5)
    figure.tight_layout(pad=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _hosted_cost_figure(results_root: Path, path: Path) -> bool:
    import matplotlib.pyplot as plt

    results_path = results_root / "cascade" / "hosted_cascades.json"
    if not results_path.exists():
        return False
    results = _load_json(results_path)
    colors = (
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#D55E00",
    )
    figure, axis = plt.subplots(figsize=(3.35, 2.35))
    for color, monitor in zip(colors, results["monitors"].values()):  # type: ignore[union-attr]
        baseline_cost = float(monitor["observed_batch_cost_usd"]) * 1000 / 7182
        baseline_f1 = float(monitor["micro"]["f1"]) * 100
        cascade = monitor["cascades"]["hybrid_tfidf_linearsvc"]
        cascade_cost = float(cascade["observed_batch_cost_usd"]) * 1000 / 7182
        cascade_f1 = float(cascade["micro"]["f1"]) * 100
        axis.annotate(
            "",
            xy=(cascade_cost, cascade_f1),
            xytext=(baseline_cost, baseline_f1),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
        )
        axis.scatter(
            [baseline_cost],
            [baseline_f1],
            marker="o",
            facecolors="none",
            edgecolors=color,
            s=27,
        )
        axis.scatter([cascade_cost], [cascade_f1], marker="o", color=color, s=27)
        short_name = (
            str(monitor["model"])
            .replace("Claude ", "")
            .replace("GPT-", "GPT ")
        )
        axis.annotate(
            short_name,
            (cascade_cost, cascade_f1),
            xytext=((6, 7) if short_name == "GPT 5.6 Luna" else (3, 4)),
            textcoords="offset points",
            fontsize=5.8,
        )
    axis.set_xlabel("Observed batch cost per 1,000 decisions (USD)")
    axis.set_ylabel("Strict pooled F1 (%)")
    axis.grid(alpha=0.2, linewidth=0.5)
    figure.tight_layout(pad=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)
    return True


def generate_paper_artifacts(
    *,
    results_root: Path,
    paper_root: Path,
) -> list[Path]:
    cascade = _load_json(results_root / "cascade" / "ts_guard_summary.json")
    generated = paper_root / "generated"
    figures = paper_root / "figures"
    outputs = {
        generated / "quality.tex": _quality_table(results_root),
        generated / "cascade_overall.tex": _cascade_overall(cascade),
        generated / "cascade_domains.tex": _cascade_domains(cascade),
        generated / "robustness.tex": _robustness_table(cascade),
        generated / "systems.tex": _systems_table(results_root),
        generated / "hosted.tex": _hosted_table(results_root),
        generated / "hosted_cascades.tex": _hosted_cascade_table(results_root),
        generated / "secondary.tex": _secondary_table(results_root),
    }
    for path, content in outputs.items():
        _write(path, content)
    _tradeoff_figure(cascade, figures / "security_efficiency.pdf")
    _robustness_figure(cascade, figures / "robustness_tradeoff.pdf")
    hosted_figure = figures / "hosted_cost_quality.pdf"
    has_hosted_figure = _hosted_cost_figure(results_root, hosted_figure)
    created = [
        *outputs,
        figures / "security_efficiency.pdf",
        figures / "robustness_tradeoff.pdf",
        *([hosted_figure] if has_hosted_figure else []),
    ]
    for path in created:
        print(path)
    return created
