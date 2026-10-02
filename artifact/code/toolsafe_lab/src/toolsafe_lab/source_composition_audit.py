from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from toolsafe_lab.data import Sample, load_eval, load_training


def _label_counts(samples: Iterable[Sample]) -> dict[str, int]:
    counts = Counter(str(sample.label) for sample in samples)
    return {label: counts.get(label, 0) for label in ("0.0", "0.5", "1.0")}


def _strict_counts(samples: Iterable[Sample]) -> dict[str, int]:
    counts = Counter(sample.strict_label for sample in samples)
    return {"safe": counts.get(0, 0), "unsafe": counts.get(1, 0)}


def _profile(samples: Sequence[Sample]) -> dict[str, object]:
    by_source: dict[str, object] = {}
    for source in sorted({sample.source for sample in samples}):
        source_samples = [sample for sample in samples if sample.source == source]
        subsets: dict[str, object] = {}
        pure_subsets = 0
        for subset in sorted({sample.subset for sample in source_samples}):
            subset_samples = [
                sample for sample in source_samples if sample.subset == subset
            ]
            strict = _strict_counts(subset_samples)
            pure = strict["safe"] == 0 or strict["unsafe"] == 0
            pure_subsets += int(pure)
            subsets[subset] = {
                "n": len(subset_samples),
                "labels": _label_counts(subset_samples),
                "strict_binary": strict,
                "strict_label_pure": pure,
            }
        by_source[source] = {
            "n": len(source_samples),
            "labels": _label_counts(source_samples),
            "strict_binary": _strict_counts(source_samples),
            "subset_count": len(subsets),
            "strict_label_pure_subset_count": pure_subsets,
            "subsets": subsets,
        }
    return {
        "n": len(samples),
        "labels": _label_counts(samples),
        "strict_binary": _strict_counts(samples),
        "sources": by_source,
    }


def _overlap(left: Sequence[Sample], right: Sequence[Sample]) -> dict[str, int]:
    left_by_id = {sample.sample_id: sample for sample in left}
    right_by_id = {sample.sample_id: sample for sample in right}
    shared = set(left_by_id) & set(right_by_id)
    conflicts = sum(
        left_by_id[sample_id].label != right_by_id[sample_id].label
        for sample_id in shared
    )
    return {"exact_sample_ids": len(shared), "label_conflicts": conflicts}


def build_source_composition_audit(
    train: Sequence[Sample],
    validation: Sequence[Sample],
    evaluation: Sequence[Sample],
) -> dict[str, object]:
    evaluation_ids = {sample.sample_id for sample in evaluation}
    fitted_train = [
        sample for sample in train if sample.sample_id not in evaluation_ids
    ]
    fitted_validation = [
        sample for sample in validation if sample.sample_id not in evaluation_ids
    ]
    return {
        "schema_version": 1,
        "privacy": {
            "aggregate_only": True,
            "raw_benchmark_text_included": False,
        },
        "splits": {
            "train_loaded": _profile(train),
            "train_fitted": _profile(fitted_train),
            "validation_loaded": _profile(validation),
            "validation_fitted": _profile(fitted_validation),
            "evaluation": _profile(evaluation),
        },
        "exact_sample_id_overlap": {
            "train_validation": _overlap(train, validation),
            "train_evaluation": _overlap(train, evaluation),
            "validation_evaluation": _overlap(validation, evaluation),
        },
        "exclusions": {
            "train_removed_for_evaluation_overlap": len(train) - len(fitted_train),
            "validation_removed_for_evaluation_overlap": (
                len(validation) - len(fitted_validation)
            ),
        },
    }


def run_source_composition_audit(
    data_root: Path,
    results_root: Path,
) -> dict[str, object]:
    train = load_training(data_root, "train")
    validation = load_training(data_root, "validation")
    evaluation_by_source = load_eval(data_root)
    evaluation = [
        sample
        for source_samples in evaluation_by_source.values()
        for sample in source_samples
    ]
    result = build_source_composition_audit(train, validation, evaluation)
    results_root.mkdir(parents=True, exist_ok=True)
    output_path = results_root / "standalone_source_composition.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return result
