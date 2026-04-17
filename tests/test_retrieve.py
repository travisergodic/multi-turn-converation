from src.retrieve import Retriever

MEMORIES = [
    {"id": "1", "content": "用户叫 Alice，是一名工程师"},
    {"id": "2", "content": "用户喜欢喝咖啡"},
    {"id": "3", "content": "用户在北京工作"},
    {"id": "4", "content": "用户有一只猫叫 Mimi"},
    {"id": "5", "content": "用户偏好简洁的 UI 设计"},
]

def test_retrieve_returns_topk():
    retriever = Retriever(embed_topk=3, rerank_topk=2)
    results = retriever.retrieve("工程师的工作地点", MEMORIES)
    assert len(results) == 2

def test_retrieve_empty_memories():
    retriever = Retriever(embed_topk=5, rerank_topk=3)
    results = retriever.retrieve("任意问题", [])
    assert results == []

def test_retrieve_fewer_than_topk():
    retriever = Retriever(embed_topk=10, rerank_topk=5)
    results = retriever.retrieve("用户信息", MEMORIES[:2])
    assert len(results) <= 2
