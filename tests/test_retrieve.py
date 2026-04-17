import numpy as np

from src.retrieve import Retriever

MEMORIES = [
    {"id": "1", "content": "用户叫 Alice，是一名工程师"},
    {"id": "2", "content": "用户喜欢喝咖啡"},
    {"id": "3", "content": "用户在北京工作"},
    {"id": "4", "content": "用户有一只猫叫 Mimi"},
    {"id": "5", "content": "用户偏好简洁的 UI 设计"},
]


class FakeEmbedder:
    def encode(self, payload, normalize_embeddings=True):
        if isinstance(payload, str):
            return np.array([1.0, 0.0, 0.0])

        vectors = []
        for item in payload:
            if "工程师" in item or "工作" in item:
                vectors.append(np.array([0.9, 0.1, 0.0]))
            elif "咖啡" in item:
                vectors.append(np.array([0.2, 0.8, 0.0]))
            else:
                vectors.append(np.array([0.1, 0.1, 0.8]))
        return np.vstack(vectors)


class FakeReranker:
    def predict(self, pairs):
        scores = []
        for _, content in pairs:
            if "北京工作" in content:
                scores.append(1.0)
            elif "工程师" in content:
                scores.append(0.9)
            else:
                scores.append(0.1)
        return np.array(scores)


def test_retrieve_returns_topk():
    retriever = Retriever(
        embed_topk=3,
        rerank_topk=2,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
    )
    results = retriever.retrieve("工程师的工作地点", MEMORIES)
    assert len(results) == 2
    assert results[0]["id"] == "3"


def test_retrieve_empty_memories():
    retriever = Retriever(embed_topk=5, rerank_topk=3)
    results = retriever.retrieve("任意问题", [])
    assert results == []


def test_retrieve_fewer_than_topk():
    retriever = Retriever(embed_topk=10, rerank_topk=5)
    results = retriever.retrieve("用户信息", MEMORIES[:2])
    assert len(results) <= 2
