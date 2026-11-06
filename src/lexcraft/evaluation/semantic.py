"""Optional neural meaning checks (install `lexcraft[eval]`).

- Embedding cosine similarity between source and rewrite.
- Bidirectional entailment with a small NLI cross-encoder: the rewrite should
  entail the source and the source should entail the rewrite.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


def available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


@dataclass
class SemanticScores:
    similarity: list[float]
    entail_fwd: list[float]  # P(source -> output)
    entail_bwd: list[float]  # P(output -> source)


class SemanticScorer:
    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.nli = CrossEncoder(NLI_MODEL)
        labels = getattr(self.nli.model.config, "id2label", {}) or {}
        self.entail_idx = next(
            (int(i) for i, lab in labels.items() if str(lab).lower().startswith("entail")), 1
        )

    def score(self, sources: list[str], outputs: list[str]) -> SemanticScores:
        import numpy as np

        a = self.embedder.encode(sources, normalize_embeddings=True, convert_to_numpy=True)
        b = self.embedder.encode(outputs, normalize_embeddings=True, convert_to_numpy=True)
        sim = (a * b).sum(axis=1).tolist()

        def entail(pairs: list[tuple[str, str]]) -> list[float]:
            logits = np.asarray(self.nli.predict(pairs, apply_softmax=True))
            return logits[:, self.entail_idx].tolist()

        fwd = entail(list(zip(sources, outputs, strict=True)))
        bwd = entail(list(zip(outputs, sources, strict=True)))
        return SemanticScores(sim, fwd, bwd)
