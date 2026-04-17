from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    retrieved_memories: list[dict]  # 检索出的记忆条目，注入 system prompt
    profile_snapshot: dict          # 当前 user profile 快照
    conversation_summary: str       # 历史对话摘要（用于控制上下文长度）
    summarized_message_count: int   # 已经纳入 conversation_summary 的消息数
