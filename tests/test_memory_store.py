from langgraph.store.memory import InMemoryStore
from src.memory_store import add_memory, append_memory, delete_memory, list_memories

def test_add_and_list():
    store = InMemoryStore()
    add_memory(store, "u1", "用户叫 Alice")
    memories = list_memories(store, "u1")
    assert len(memories) == 1
    assert memories[0]["content"] == "用户叫 Alice"

def test_append_memory():
    store = InMemoryStore()
    add_memory(store, "u1", "喜欢深色主题")
    mid = list_memories(store, "u1")[0]["id"]
    append_memory(store, "u1", mid, "，也喜欢简洁界面")
    updated = list_memories(store, "u1")[0]["content"]
    assert "简洁界面" in updated

def test_delete_memory():
    store = InMemoryStore()
    add_memory(store, "u1", "临时记忆")
    mid = list_memories(store, "u1")[0]["id"]
    delete_memory(store, "u1", mid)
    assert list_memories(store, "u1") == []
