import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

from src.logging_utils import get_logger

with open("configs/config.yaml", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

logger = get_logger(__name__)

NAMESPACE = (_cfg["memory"].get("namespace") or "long_term_memory",)
MEMORY_STORE_PATH = Path(_cfg["memory"].get("store_path", "data/long_term_memory.json"))


@dataclass
class MemoryEntry:
    id: str
    content: str
    created_at: str
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at,
        }
        if self.updated_at:
            payload["updated_at"] = self.updated_at
        return payload


@dataclass
class StoreItem:
    key: str
    value: dict[str, Any]


class FileMemoryStore:
    def __init__(self, path: str | Path = MEMORY_STORE_PATH):
        self.path = Path(path)
        self._lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict[str, dict[str, Any]]]:
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as f:
            return json.load(f)

    def _save(self, payload: dict[str, dict[str, dict[str, Any]]]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    def _namespace_key(self, namespace: tuple[str, ...]) -> str:
        return "::".join(namespace)

    def get(self, namespace: tuple[str, ...], key: str) -> StoreItem | None:
        with self._lock:
            payload = self._load()
            namespace_payload = payload.get(self._namespace_key(namespace), {})
            if key not in namespace_payload:
                return None
            return StoreItem(key=key, value=namespace_payload[key])

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        with self._lock:
            payload = self._load()
            namespace_key = self._namespace_key(namespace)
            namespace_payload = payload.setdefault(namespace_key, {})
            namespace_payload[key] = value
            self._save(payload)

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        with self._lock:
            payload = self._load()
            namespace_key = self._namespace_key(namespace)
            namespace_payload = payload.get(namespace_key, {})
            namespace_payload.pop(key, None)
            if namespace_payload:
                payload[namespace_key] = namespace_payload
            else:
                payload.pop(namespace_key, None)
            self._save(payload)

    def list_prefix(self, namespace: tuple[str, ...], prefix: str) -> list[StoreItem]:
        with self._lock:
            payload = self._load()
            namespace_payload = payload.get(self._namespace_key(namespace), {})
            matches = [
                StoreItem(key=key, value=value)
                for key, value in namespace_payload.items()
                if key.startswith(prefix)
            ]
        return sorted(matches, key=lambda item: item.value.get("created_at", ""))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_key(user_id: str, memory_id: str) -> str:
    return f"{user_id}:{memory_id}"


def _store_supports_prefix_listing(store: Any) -> bool:
    return hasattr(store, "list_prefix")


def _store_supports_search(store: Any) -> bool:
    return hasattr(store, "search")


def _list_store_items(store: Any, user_id: str) -> list[StoreItem]:
    prefix = f"{user_id}:"
    if _store_supports_prefix_listing(store):
        return store.list_prefix(NAMESPACE, prefix)

    if _store_supports_search(store):
        try:
            return [
                StoreItem(key=item.key, value=item.value)
                for item in store.search(NAMESPACE, query=prefix)
                if getattr(item, "key", "").startswith(prefix)
            ]
        except TypeError:
            try:
                return [
                    StoreItem(key=item.key, value=item.value)
                    for item in store.search(NAMESPACE)
                    if getattr(item, "key", "").startswith(prefix)
                ]
            except Exception:
                return []
        except Exception:
            return []

    return []


def list_memories(store: Any, user_id: str) -> list[dict[str, Any]]:
    memories = [item.value for item in _list_store_items(store, user_id)]
    return sorted(memories, key=lambda memory: memory.get("created_at", ""))


def add_memory(store: Any, user_id: str, content: str) -> str:
    mid = str(uuid.uuid4())
    entry = MemoryEntry(id=mid, content=content, created_at=_utc_now())
    store.put(NAMESPACE, _memory_key(user_id, mid), entry.to_dict())
    logger.info("Added memory user_id=%s memory_id=%s content_chars=%s", user_id, mid, len(content))
    return mid


def update_memory(store: Any, user_id: str, memory_id: str, content: str) -> None:
    key = _memory_key(user_id, memory_id)
    item = store.get(NAMESPACE, key)
    if not item:
        logger.warning("Skipped update for missing memory user_id=%s memory_id=%s", user_id, memory_id)
        return

    updated = dict(item.value)
    updated["content"] = content
    updated["updated_at"] = _utc_now()
    store.put(NAMESPACE, key, updated)
    logger.info("Updated memory user_id=%s memory_id=%s content_chars=%s", user_id, memory_id, len(content))


def delete_memory(store: Any, user_id: str, memory_id: str) -> None:
    store.delete(NAMESPACE, _memory_key(user_id, memory_id))
    logger.info("Deleted memory user_id=%s memory_id=%s", user_id, memory_id)
