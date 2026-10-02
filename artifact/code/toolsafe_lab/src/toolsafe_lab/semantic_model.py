from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FIELD_MARKERS = (
    "[USER_REQUEST]\n",
    "\n[INTERACTION_HISTORY]\n",
    "\n[CURRENT_ACTION]\n",
    "\n[TOOL_DESCRIPTIONS]\n",
)


def parse_fields(text: str) -> tuple[str, str, str, str]:
    if not text.startswith(FIELD_MARKERS[0]):
        raise ValueError("Missing user-request field marker")
    instruction_and_rest = text[len(FIELD_MARKERS[0]) :]
    instruction, rest = instruction_and_rest.split(FIELD_MARKERS[1], 1)
    history, rest = rest.split(FIELD_MARKERS[2], 1)
    current_action, env_info = rest.split(FIELD_MARKERS[3], 1)
    return instruction, history, current_action, env_info


class SemanticRelationGuard:
    """Frozen MiniLM field embeddings plus a conventional linear classifier.

    Relationship features explicitly represent request/action and
    history/action alignment. The encoder remains frozen; only logistic
    regression is trained on TS-Bench.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        device: str = "cpu",
        batch_size: int = 64,
        random_state: int = 260110156,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "Install the embedding extra: pip install -e '.[embeddings]'"
            ) from error
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.batch_size = batch_size
        self.random_state = random_state
        self.encoder = SentenceTransformer(
            model_name,
            revision=model_revision,
            device=device,
        )
        self.classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.25,
                max_iter=3_000,
                random_state=random_state,
                solver="liblinear",
            ),
        )

    def _features(self, texts: Sequence[str]) -> np.ndarray:
        parsed = [parse_fields(text) for text in texts]
        flat_fields = [field for sample_fields in parsed for field in sample_fields]
        embeddings = self.encoder.encode(
            flat_fields,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        by_field = embeddings.reshape(len(texts), 4, -1)
        relationships = []
        # User↔action alignment, history↔action correlation, and whether the
        # action matches the advertised tool capabilities.
        for left, right in ((0, 2), (1, 2), (2, 3)):
            relationships.extend(
                (
                    np.abs(by_field[:, left] - by_field[:, right]),
                    by_field[:, left] * by_field[:, right],
                )
            )
        cosine = np.stack(
            [
                np.sum(by_field[:, left] * by_field[:, right], axis=1)
                for left, right in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            ],
            axis=1,
        )
        lengths = np.log1p(
            np.asarray(
                [[len(field) for field in sample_fields] for sample_fields in parsed],
                dtype=np.float32,
            )
        )
        return np.concatenate(
            (by_field.reshape(len(texts), -1), *relationships, cosine, lengths),
            axis=1,
        ).astype(np.float32)

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "SemanticRelationGuard":
        self.classifier.fit(self._features(texts), labels)
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.classifier.predict(self._features(texts))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.classifier.predict_proba(self._features(texts))
