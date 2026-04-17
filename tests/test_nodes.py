from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.memory_store import FileMemoryStore, list_memories
from src.nodes import llm_node, memory_update_node, profile_update_node
from src.state import ConversationState

MOCK_STATE: ConversationState = {
    "messages": [HumanMessage(content="我叫 Bob，是一名设计师")],
    "user_id": "u1",
    "retrieved_memories": [],
    "profile_snapshot": {"name": None, "occupation": None, "interests": []},
}


def test_llm_node_appends_ai_message(tmp_path):
    store = FileMemoryStore(tmp_path / "memory.json")
    with patch("src.nodes.chat_completion", return_value="你好 Bob！"):
        with patch("src.nodes.retriever.retrieve", return_value=[]):
            result = llm_node(MOCK_STATE, store=store)
    assert result["messages"][-1].content == "你好 Bob！"


def test_memory_update_node_add(tmp_path):
    store = FileMemoryStore(tmp_path / "memory.json")
    state = {
        **MOCK_STATE,
        "messages": [
            HumanMessage(content="我叫 Bob"),
            AIMessage(content="你好 Bob！"),
        ],
        "retrieved_memories": [],
    }
    mock_ops = '[{"action": "add", "memory_id": null, "content": "用户叫 Bob"}]'
    with patch("src.nodes.chat_completion", return_value=mock_ops):
        memory_update_node(state, store=store)

    assert len(list_memories(store, "u1")) == 1


def test_profile_update_node_noop(tmp_path):
    store = FileMemoryStore(tmp_path / "memory.json")
    state = {
        **MOCK_STATE,
        "messages": [
            HumanMessage(content="今天天气真好"),
            AIMessage(content="确实！"),
        ],
    }
    with patch("src.nodes.chat_completion", return_value="null"):
        result = profile_update_node(state, store=store)
    assert result == {}
