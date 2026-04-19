import json
from time import perf_counter
from typing import Any

import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langfuse import get_client, observe
from langgraph.store.base import BaseStore

from src.llm_client import chat_completion
from src.logging_utils import get_logger
from src.memory_store import add_memory, delete_memory, list_memories, update_memory
from src.prompt import (
    MEMORY_SECTION_TEMPLATE,
    MEMORY_UPDATE_PROMPT,
    PROFILE_SECTION_TEMPLATE,
    PROFILE_UPDATE_PROMPT,
    SUMMARY_SECTION_TEMPLATE,
    SUMMARY_UPDATE_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
)
from src.retrieve import Retriever
from src.state import ConversationState

with open("configs/config.yaml") as f:
    _cfg = yaml.safe_load(f)

logger = get_logger(__name__)
WINDOW_MESSAGES = int(_cfg.get("context", {}).get("window_messages", 16))
SUMMARY_TRIGGER_MESSAGES = int(_cfg.get("context", {}).get("summary_trigger_messages", 24))
SUMMARY_BATCH_MESSAGES = int(_cfg.get("context", {}).get("summary_batch_messages", 8))

retriever = Retriever(
    embed_topk=_cfg["retrieval"]["embed_topk"],
    rerank_topk=_cfg["retrieval"]["rerank_topk"],
)


def _message_role(msg: Any) -> str:
    if isinstance(msg, HumanMessage):
        return "用户"
    if isinstance(msg, AIMessage):
        return "助手"
    return "消息"


def _windowed_messages(messages: list[Any]) -> list[Any]:
    if len(messages) <= WINDOW_MESSAGES:
        return messages
    return messages[-WINDOW_MESSAGES:]


def _format_messages_for_summary(messages: list[Any]) -> str:
    lines = [f"{_message_role(msg)}：{msg.content}" for msg in messages]
    return "\n".join(lines)


def _build_system_prompt(profile: dict, memories: list[dict], summary: str) -> str:
    profile_text = "\n".join(f"- {k}：{v}" for k, v in profile.items() if v)
    profile_section = PROFILE_SECTION_TEMPLATE.format(profile_text=profile_text) if profile_text else ""

    memory_text = "\n".join(f"- [{m['id'][:8]}] {m['content']}" for m in memories)
    memory_section = MEMORY_SECTION_TEMPLATE.format(memory_text=memory_text) if memory_text else ""
    summary_section = SUMMARY_SECTION_TEMPLATE.format(summary_text=summary) if summary else ""

    return SYSTEM_PROMPT_TEMPLATE.format(
        profile_section=profile_section,
        memory_section=memory_section,
        summary_section=summary_section,
    )


@observe(name="llm_node")
def llm_node(state: ConversationState, store: BaseStore) -> dict:
    started_at = perf_counter()
    user_id = state["user_id"]
    user_message = state["messages"][-1].content

    all_memories = list_memories(store, user_id)
    retrieve_started_at = perf_counter()
    retrieved = retriever.retrieve(user_message, all_memories)
    retrieve_elapsed_ms = (perf_counter() - retrieve_started_at) * 1000
    logger.info(
        "llm_node thread_user=%s total_memories=%s retrieved_memories=%s message_chars=%s retrieve_ms=%.2f",
        user_id,
        len(all_memories),
        len(retrieved),
        len(user_message),
        retrieve_elapsed_ms,
    )

    summary_text = state.get("conversation_summary", "").strip()
    system_content = _build_system_prompt(state.get("profile_snapshot", {}), retrieved, summary_text)
    messages = [{"role": "system", "content": system_content}]
    windowed_messages = _windowed_messages(state["messages"])
    for msg in windowed_messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        messages.append({"role": role, "content": msg.content})

    reply = chat_completion(messages)
    total_elapsed_ms = (perf_counter() - started_at) * 1000
    logger.info(
        "llm_node produced reply_chars=%s total_elapsed_ms=%.2f total_messages=%s windowed_messages=%s summary_chars=%s",
        len(reply or ""),
        total_elapsed_ms,
        len(state["messages"]),
        len(windowed_messages),
        len(summary_text),
    )
    get_client().update_current_span(
        input=user_message,
        output=reply,
        metadata={
            "retrieved_count": len(retrieved),
            "total_memories": len(all_memories),
            "reply_chars": len(reply or ""),
            "windowed_messages": len(windowed_messages),
        },
    )
    return {
        "messages": [AIMessage(content=reply)],
        "retrieved_memories": retrieved,
    }


@observe(name="memory_update_node")
def memory_update_node(state: ConversationState, store: BaseStore) -> dict:
    started_at = perf_counter()
    msgs = state["messages"]
    if len(msgs) < 2:
        logger.info("memory_update_node skipped because message_count=%s", len(msgs))
        get_client().update_current_span(metadata={"skipped": "insufficient_messages"})
        return {}

    user_message = msgs[-2].content
    assistant_message = msgs[-1].content
    existing = state.get("retrieved_memories", [])
    existing_text = "\n".join(f"[{m['id']}] {m['content']}" for m in existing) or "（无）"

    prompt_messages = [
        {
            "role": "user",
            "content": MEMORY_UPDATE_PROMPT.format(
                existing_memories=existing_text,
                user_message=user_message,
                assistant_message=assistant_message,
            ),
        }
    ]

    raw = chat_completion(prompt_messages)
    try:
        operations = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("memory_update_node received invalid JSON: %s", raw)
        get_client().score_current_span(name="memory_update.parse_success", value=0.0)
        get_client().update_current_span(metadata={"operation_count": 0, "parse_failed": True})
        return {}

    get_client().score_current_span(name="memory_update.parse_success", value=1.0)

    user_id = state["user_id"]
    logger.info("memory_update_node executing operation_count=%s for user_id=%s", len(operations), user_id)
    action_counts: dict[str, int] = {}
    for op in operations:
        action = op.get("action")
        mid = op.get("memory_id")
        content = op.get("content")
        action_counts[action or "unknown"] = action_counts.get(action or "unknown", 0) + 1
        if action == "add" and content:
            add_memory(store, user_id, content)
        elif action == "update" and mid and content:
            update_memory(store, user_id, mid, content)
        elif action == "append" and mid and content:
            # Backward compatibility: treat old "append" action as full update.
            update_memory(store, user_id, mid, content)
        elif action == "delete" and mid:
            delete_memory(store, user_id, mid)
        else:
            logger.info("memory_update_node ignored operation action=%s memory_id=%s", action, mid)

    get_client().update_current_span(
        metadata={
            "action_distribution": action_counts,
            "operation_count": len(operations),
        }
    )

    total_elapsed_ms = (perf_counter() - started_at) * 1000
    logger.info("memory_update_node completed user_id=%s total_elapsed_ms=%.2f", user_id, total_elapsed_ms)
    return {}


@observe(name="profile_update_node")
def profile_update_node(state: ConversationState, store: BaseStore) -> dict:
    started_at = perf_counter()
    msgs = state["messages"]
    if len(msgs) < 2:
        logger.info("profile_update_node skipped because message_count=%s", len(msgs))
        get_client().update_current_span(metadata={"skipped": "insufficient_messages"})
        return {}

    user_message = msgs[-2].content
    assistant_message = msgs[-1].content
    current_profile = state.get("profile_snapshot", {})

    prompt_messages = [
        {
            "role": "user",
            "content": PROFILE_UPDATE_PROMPT.format(
                current_profile=json.dumps(current_profile, ensure_ascii=False, indent=2),
                user_message=user_message,
                assistant_message=assistant_message,
            ),
        }
    ]

    raw = chat_completion(prompt_messages).strip()
    if raw == "null":
        total_elapsed_ms = (perf_counter() - started_at) * 1000
        logger.info("profile_update_node decided no profile update is needed total_elapsed_ms=%.2f", total_elapsed_ms)
        get_client().score_current_span(name="profile_update.parse_success", value=1.0)
        get_client().update_current_span(metadata={"updated": False})
        return {}

    try:
        updated_profile = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("profile_update_node received invalid JSON: %s", raw)
        get_client().score_current_span(name="profile_update.parse_success", value=0.0)
        return {}

    profile_path = _cfg["profile"]["path"]
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(updated_profile, f, ensure_ascii=False, indent=2)

    total_elapsed_ms = (perf_counter() - started_at) * 1000
    logger.info("profile_update_node wrote updated profile to %s total_elapsed_ms=%.2f", profile_path, total_elapsed_ms)
    get_client().score_current_span(name="profile_update.parse_success", value=1.0)
    get_client().update_current_span(metadata={"updated": True})
    return {"profile_snapshot": updated_profile}


def summary_update_node(state: ConversationState, store: BaseStore) -> dict:
    del store
    started_at = perf_counter()
    messages = state.get("messages", [])
    summarized_count = int(state.get("summarized_message_count", 0))
    unsummarized = messages[summarized_count:]

    if len(unsummarized) <= SUMMARY_TRIGGER_MESSAGES:
        logger.info(
            "summary_update_node skipped unsummarized_messages=%s trigger=%s",
            len(unsummarized),
            SUMMARY_TRIGGER_MESSAGES,
        )
        return {}

    overflow_count = max(0, len(unsummarized) - WINDOW_MESSAGES)
    summarize_count = min(SUMMARY_BATCH_MESSAGES, overflow_count)
    if summarize_count <= 0:
        logger.info(
            "summary_update_node skipped because summarize_count=%s overflow_count=%s",
            summarize_count,
            overflow_count,
        )
        return {}

    to_summarize = unsummarized[:summarize_count]
    existing_summary = state.get("conversation_summary", "").strip() or "（暂无摘要）"
    old_messages_text = _format_messages_for_summary(to_summarize)

    prompt_messages = [
        {
            "role": "user",
            "content": SUMMARY_UPDATE_PROMPT.format(
                existing_summary=existing_summary,
                old_messages=old_messages_text,
            ),
        }
    ]

    raw_summary = chat_completion(prompt_messages).strip()
    if not raw_summary:
        logger.warning("summary_update_node received empty summary response")
        return {}

    total_elapsed_ms = (perf_counter() - started_at) * 1000
    new_summarized_count = summarized_count + summarize_count
    logger.info(
        "summary_update_node completed summarized_count=%s->%s summarize_count=%s total_elapsed_ms=%.2f summary_chars=%s",
        summarized_count,
        new_summarized_count,
        summarize_count,
        total_elapsed_ms,
        len(raw_summary),
    )
    return {
        "conversation_summary": raw_summary,
        "summarized_message_count": new_summarized_count,
    }
