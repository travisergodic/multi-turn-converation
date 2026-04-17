import uuid
from datetime import datetime
from langgraph.store.base import BaseStore

NAMESPACE = ("long_term_memory",)
INDEX_KEY = "__index__"


def _load_index(store: BaseStore, user_id: str) -> dict[str, dict]:
    item = store.get(NAMESPACE, f"{user_id}:{INDEX_KEY}")
    return item.value if item else {}


def _save_index(store: BaseStore, user_id: str, index: dict[str, dict]) -> None:
    store.put(NAMESPACE, f"{user_id}:{INDEX_KEY}", index)


def list_memories(store: BaseStore, user_id: str) -> list[dict]:
    return list(_load_index(store, user_id).values())


def add_memory(store: BaseStore, user_id: str, content: str) -> str:
    index = _load_index(store, user_id)
    mid = str(uuid.uuid4())
    index[mid] = {"id": mid, "content": content, "created_at": datetime.utcnow().isoformat()}
    _save_index(store, user_id, index)
    return mid


def append_memory(store: BaseStore, user_id: str, memory_id: str, extra: str) -> None:
    index = _load_index(store, user_id)
    if memory_id in index:
        index[memory_id]["content"] += extra
        index[memory_id]["updated_at"] = datetime.utcnow().isoformat()
        _save_index(store, user_id, index)


def delete_memory(store: BaseStore, user_id: str, memory_id: str) -> None:
    index = _load_index(store, user_id)
    index.pop(memory_id, None)
    _save_index(store, user_id, index)
