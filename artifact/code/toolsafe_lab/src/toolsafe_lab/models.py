from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


RANDOM_STATE = 260110156


def model_factories(random_state: int = RANDOM_STATE) -> dict[str, callable]:
    """Return fresh, CPU-friendly binary classifiers.

    Configurations are fixed before looking at TS-Bench-eval. The authors'
    validation split may be used for analysis, but no test-set tuning occurs.
    """

    return {
        "word_tfidf_logreg": lambda: Pipeline(
            (
                (
                    "vectorizer",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.995,
                        max_features=120_000,
                        sublinear_tf=True,
                        strip_accents="unicode",
                        dtype=np.float32,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=random_state,
                        solver="liblinear",
                    ),
                ),
            )
        ),
        "char_tfidf_linearsvc": lambda: Pipeline(
            (
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=150_000,
                        sublinear_tf=True,
                        dtype=np.float32,
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(C=1.0, class_weight="balanced", random_state=random_state),
                ),
            )
        ),
        "hybrid_tfidf_linearsvc": lambda: Pipeline(
            (
                (
                    "vectorizer",
                    FeatureUnion(
                        (
                            (
                                "word",
                                TfidfVectorizer(
                                    ngram_range=(1, 2),
                                    min_df=2,
                                    max_features=80_000,
                                    sublinear_tf=True,
                                    strip_accents="unicode",
                                    dtype=np.float32,
                                ),
                            ),
                            (
                                "char",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    min_df=2,
                                    max_features=100_000,
                                    sublinear_tf=True,
                                    dtype=np.float32,
                                ),
                            ),
                        )
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(C=1.0, class_weight="balanced", random_state=random_state),
                ),
            )
        ),
        "hash_sgd_compact": lambda: Pipeline(
            (
                (
                    "vectorizer",
                    HashingVectorizer(
                        n_features=2**18,
                        ngram_range=(1, 2),
                        alternate_sign=False,
                        norm="l2",
                        dtype=np.float32,
                    ),
                ),
                (
                    "classifier",
                    SGDClassifier(
                        loss="modified_huber",
                        alpha=1e-5,
                        max_iter=2_000,
                        class_weight="balanced",
                        average=True,
                        early_stopping=False,
                        random_state=random_state,
                    ),
                ),
            )
        ),
    }
