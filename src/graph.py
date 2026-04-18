import sqlite3
from pathlib import Path

import yaml
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.logging_utils import get_logger
from src.memory_store import FileMemoryStore
from src.nodes import llm_node, memory_update_node, profile_update_node, summary_update_node
from src.state import ConversationState

with open("configs/config.yaml") as f:
    _cfg = yaml.safe_load(f)

logger = get_logger(__name__)


def _resolve_store_path(store_path: str | None = None) -> str:
    return store_path or _cfg["memory"].get("store_path", "data/long_term_memory.json")


def _build_checkpointer(db_path: str | None = None) -> SqliteSaver:
    path = db_path or _cfg["memory"]["db_path"]
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)


def build_response_graph(db_path: str | None = None, store_path: str | None = None):
    path = db_path or _cfg["memory"]["db_path"]
    resolved_store_path = _resolve_store_path(store_path)
    checkpointer = _build_checkpointer(path)

    builder = StateGraph(ConversationState)
    builder.add_node("llm", llm_node)
    builder.set_entry_point("llm")
    builder.add_edge("llm", END)

    store = FileMemoryStore(resolved_store_path)
    logger.info("Built response graph db_path=%s store_path=%s", path, resolved_store_path)
    return builder.compile(checkpointer=checkpointer, store=store)


def build_maintenance_graph(db_path: str | None = None, store_path: str | None = None):
    path = db_path or _cfg["memory"]["db_path"]
    resolved_store_path = _resolve_store_path(store_path)
    checkpointer = _build_checkpointer(path)

    builder = StateGraph(ConversationState)
    builder.add_node("memory_update", memory_update_node)
    builder.add_node("profile_update", profile_update_node)
    builder.add_node("summary_update", summary_update_node)

    builder.set_entry_point("memory_update")
    builder.add_edge("memory_update", "profile_update")
    builder.add_edge("profile_update", "summary_update")
    builder.add_edge("summary_update", END)

    store = FileMemoryStore(resolved_store_path)
    logger.info("Built maintenance graph db_path=%s store_path=%s", path, resolved_store_path)
    return builder.compile(checkpointer=checkpointer, store=store)


def build_graph(db_path: str | None = None, store_path: str | None = None):
    path = db_path or _cfg["memory"]["db_path"]
    resolved_store_path = _resolve_store_path(store_path)
    checkpointer = _build_checkpointer(path)

    builder = StateGraph(ConversationState)
    builder.add_node("llm", llm_node)
    builder.add_node("memory_update", memory_update_node)
    builder.add_node("profile_update", profile_update_node)
    builder.add_node("summary_update", summary_update_node)

    builder.set_entry_point("llm")
    builder.add_edge("llm", "memory_update")
    builder.add_edge("memory_update", "profile_update")
    builder.add_edge("profile_update", "summary_update")
    builder.add_edge("summary_update", END)

    store = FileMemoryStore(resolved_store_path)
    logger.info("Built graph db_path=%s store_path=%s", path, resolved_store_path)
    return builder.compile(checkpointer=checkpointer, store=store)
