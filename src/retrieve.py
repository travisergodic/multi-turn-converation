from collections import Counter
from typing import Any

import numpy as np
import yaml

from langfuse import get_client, observe

from src.logging_utils import get_logger

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:  # pragma: no cover - exercised only in stripped-down environments
    CrossEncoder = None
    SentenceTransformer = None

with open("configs/config.yaml", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    compact = "".join(text.split()).lower()
    if not compact:
        return []
    if len(compact) == 1:
        return [compact]
    return [compact[idx : idx + 2] for idx in range(len(compact) - 1)]


def _counter_similarity(left: str, right: str) -> float:
    left_counts = Counter(_tokenize(left))
    right_counts = Counter(_tokenize(right))
    if not left_counts or not right_counts:
        return 0.0

    overlap = sum(min(left_counts[token], right_counts[token]) for token in left_counts.keys() & right_counts.keys())
    left_norm = sum(value * value for value in left_counts.values()) ** 0.5
    right_norm = sum(value * value for value in right_counts.values()) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return overlap / (left_norm * right_norm)


class Retriever:
    def __init__(
        self,
        embed_topk: int = _cfg["retrieval"]["embed_topk"],
        rerank_topk: int = _cfg["retrieval"]["rerank_topk"],
        embed_model_name: str | None = None,
        rerank_model_name: str | None = None,
        embedder: Any | None = None,
        reranker: Any | None = None,
    ):
        self.embed_topk = embed_topk
        self.rerank_topk = rerank_topk
        self.embed_model_name = embed_model_name or _cfg["retrieval"]["embed_model"]
        self.rerank_model_name = rerank_model_name or _cfg["retrieval"]["rerank_model"]
        self._embedder = embedder
        self._reranker = reranker
        self._embedding_disabled = embedder is None
        self._rerank_disabled = reranker is None

    def _get_embedder(self) -> Any | None:
        if self._embedder is not None:
            return self._embedder
        if self._embedding_disabled:
            if SentenceTransformer is None:
                return None
            try:
                self._embedder = SentenceTransformer(self.embed_model_name)
            except Exception:
                return None
            self._embedding_disabled = False
        return self._embedder

    def _get_reranker(self) -> Any | None:
        if self._reranker is not None:
            return self._reranker
        if self._rerank_disabled:
            if CrossEncoder is None:
                return None
            try:
                self._reranker = CrossEncoder(self.rerank_model_name)
            except Exception:
                return None
            self._rerank_disabled = False
        return self._reranker

    def _embedding_scores(self, query: str, memories: list[dict]) -> np.ndarray:
        embedder = self._get_embedder()
        contents = [memory["content"] for memory in memories]
        if embedder is None:
            return np.array([_counter_similarity(query, content) for content in contents], dtype=float)

        query_vec = np.asarray(embedder.encode(query, normalize_embeddings=True))
        mem_vecs = np.asarray(embedder.encode(contents, normalize_embeddings=True))
        return np.dot(mem_vecs, query_vec)

    def _rerank_scores(self, query: str, memories: list[dict]) -> np.ndarray:
        reranker = self._get_reranker()
        if reranker is None:
            return np.array([_counter_similarity(query, memory["content"]) for memory in memories], dtype=float)

        pairs = [[query, memory["content"]] for memory in memories]
        return np.asarray(reranker.predict(pairs), dtype=float)

    @observe(name="retrieve")
    def retrieve(self, query: str, memories: list[dict]) -> list[dict]:
        if not memories:
            logger.info("Retriever received no memories for query_chars=%s", len(query))
            return []

        topk = min(self.embed_topk, len(memories))
        scores = self._embedding_scores(query, memories)
        top_indices = np.argsort(scores)[::-1][:topk].tolist()
        candidates = [memories[index] for index in top_indices]

        rerank_topk = min(self.rerank_topk, len(candidates))
        rerank_scores = self._rerank_scores(query, candidates)
        ranked = sorted(zip(rerank_scores, candidates), key=lambda item: item[0], reverse=True)
        results = [memory for _, memory in ranked[:rerank_topk]]
        get_client().update_current_span(
            metadata={
                "candidate_count": len(candidates),
                "final_count": len(results),
                "total_memories": len(memories),
            }
        )
        logger.info(
            "Retriever completed query_chars=%s total_memories=%s embed_topk=%s rerank_topk=%s returned=%s",
            len(query),
            len(memories),
            topk,
            rerank_topk,
            len(results),
        )
        return results
