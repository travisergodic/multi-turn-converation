import numpy as np
import yaml
from sentence_transformers import SentenceTransformer, CrossEncoder

with open("configs/config.yaml") as f:
    _cfg = yaml.safe_load(f)


class Retriever:
    def __init__(
        self,
        embed_topk: int = _cfg["retrieval"]["embed_topk"],
        rerank_topk: int = _cfg["retrieval"]["rerank_topk"],
    ):
        self.embed_topk = embed_topk
        self.rerank_topk = rerank_topk
        self._embedder = SentenceTransformer(_cfg["retrieval"]["embed_model"])
        self._reranker = CrossEncoder(_cfg["retrieval"]["rerank_model"])

    def retrieve(self, query: str, memories: list[dict]) -> list[dict]:
        if not memories:
            return []

        contents = [m["content"] for m in memories]
        query_vec = self._embedder.encode(query, normalize_embeddings=True)
        mem_vecs = self._embedder.encode(contents, normalize_embeddings=True)

        scores = np.dot(mem_vecs, query_vec)
        topk = min(self.embed_topk, len(memories))
        top_indices = np.argsort(scores)[::-1][:topk].tolist()
        candidates = [memories[i] for i in top_indices]

        if len(candidates) <= self.rerank_topk:
            return candidates

        pairs = [[query, m["content"]] for m in candidates]
        rerank_scores = self._reranker.predict(pairs)
        rerank_topk = min(self.rerank_topk, len(candidates))
        ranked = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
        return [m for _, m in ranked[:rerank_topk]]
