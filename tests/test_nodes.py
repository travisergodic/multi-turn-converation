from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.memory_store import FileMemoryStore, add_memory, list_memories
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


def test_memory_update_node_update_and_append_compat(tmp_path):
    store = FileMemoryStore(tmp_path / "memory.json")
    mid = add_memory(store, "u1", "喜欢深色主题")
    state = {
        **MOCK_STATE,
        "messages": [
            HumanMessage(content="我其实更喜欢浅色主题"),
            AIMessage(content="明白了，我记下你的新偏好"),
        ],
        "retrieved_memories": [{"id": mid, "content": "喜欢深色主题"}],
    }
    mock_ops = f'[{{"action": "append", "memory_id": "{mid}", "content": "偏好浅色主题"}}]'
    with patch("src.nodes.chat_completion", return_value=mock_ops):
        memory_update_node(state, store=store)

    memories = list_memories(store, "u1")
    assert memories[0]["content"] == "偏好浅色主题"


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


def test_memory_update_node_reports_parse_success(tmp_path):
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
        with patch("src.nodes.get_client") as mock_get_client:
            mock_lf = MagicMock()
            mock_get_client.return_value = mock_lf
            memory_update_node(state, store=store)
    score_calls = [
        call for call in mock_lf.score_current_span.call_args_list
        if call.kwargs.get("name") == "memory_update.parse_success"
    ]
    assert score_calls, "Expected parse_success score to be emitted"
    assert score_calls[0].kwargs["value"] == 1.0
    metadata_calls = mock_lf.update_current_span.call_args_list
    assert any(
        call.kwargs.get("metadata", {}).get("action_distribution") == {"add": 1}
        for call in metadata_calls
    ), "Expected action_distribution metadata to be emitted"


def test_memory_update_node_reports_parse_failure(tmp_path):
    store = FileMemoryStore(tmp_path / "memory.json")
    state = {
        **MOCK_STATE,
        "messages": [
            HumanMessage(content="我叫 Bob"),
            AIMessage(content="你好 Bob！"),
        ],
        "retrieved_memories": [],
    }
    with patch("src.nodes.chat_completion", return_value="not valid json"):
        with patch("src.nodes.get_client") as mock_get_client:
            mock_lf = MagicMock()
            mock_get_client.return_value = mock_lf
            memory_update_node(state, store=store)
    score_calls = [
        call for call in mock_lf.score_current_span.call_args_list
        if call.kwargs.get("name") == "memory_update.parse_success"
    ]
    assert score_calls, "Expected parse_success score to be emitted"
    assert score_calls[0].kwargs["value"] == 0.0


def test_profile_update_node_reports_parse_success_on_noop(tmp_path):
    store = FileMemoryStore(tmp_path / "memory.json")
    state = {
        **MOCK_STATE,
        "messages": [
            HumanMessage(content="今天天气真好"),
            AIMessage(content="确实！"),
        ],
    }
    with patch("src.nodes.chat_completion", return_value="null"):
        with patch("src.nodes.get_client") as mock_get_client:
            mock_lf = MagicMock()
            mock_get_client.return_value = mock_lf
            profile_update_node(state, store=store)
    score_calls = [
        call for call in mock_lf.score_current_span.call_args_list
        if call.kwargs.get("name") == "profile_update.parse_success"
    ]
    assert score_calls
    assert score_calls[0].kwargs["value"] == 1.0


def test_profile_update_node_reports_parse_success_on_update(tmp_path):
    import json as _json
    import src.nodes as _nodes_mod

    store = FileMemoryStore(tmp_path / "memory.json")
    profile_path = tmp_path / "profile.json"
    updated_profile = {"name": "Bob", "occupation": "设计师", "interests": [], "other": {}}
    state = {
        **MOCK_STATE,
        "messages": [
            HumanMessage(content="我叫 Bob，是一名设计师"),
            AIMessage(content="好的，我记下来了"),
        ],
    }
    patched_cfg = {**_nodes_mod._cfg, "profile": {"path": str(profile_path)}}
    with patch("src.nodes.chat_completion", return_value=_json.dumps(updated_profile)):
        with patch.object(_nodes_mod, "_cfg", patched_cfg):
            with patch("src.nodes.get_client") as mock_get_client:
                mock_lf = MagicMock()
                mock_get_client.return_value = mock_lf
                profile_update_node(state, store=store)
    score_calls = [
        call for call in mock_lf.score_current_span.call_args_list
        if call.kwargs.get("name") == "profile_update.parse_success"
    ]
    assert score_calls
    assert score_calls[0].kwargs["value"] == 1.0
    metadata_calls = mock_lf.update_current_span.call_args_list
    assert any(call.kwargs.get("metadata", {}).get("updated") is True for call in metadata_calls)


def test_profile_update_node_reports_parse_failure(tmp_path):
    store = FileMemoryStore(tmp_path / "memory.json")
    state = {
        **MOCK_STATE,
        "messages": [
            HumanMessage(content="我叫 Bob"),
            AIMessage(content="好的"),
        ],
    }
    with patch("src.nodes.chat_completion", return_value="not valid json"):
        with patch("src.nodes.get_client") as mock_get_client:
            mock_lf = MagicMock()
            mock_get_client.return_value = mock_lf
            profile_update_node(state, store=store)
    score_calls = [
        call for call in mock_lf.score_current_span.call_args_list
        if call.kwargs.get("name") == "profile_update.parse_success"
    ]
    assert score_calls
    assert score_calls[0].kwargs["value"] == 0.0


def test_llm_node_updates_span(tmp_path):
    store = FileMemoryStore(tmp_path / "memory.json")
    with patch("src.nodes.chat_completion", return_value="你好 Bob！"):
        with patch("src.nodes.retriever.retrieve", return_value=[]):
            with patch("src.nodes.get_client") as mock_get_client:
                mock_lf = MagicMock()
                mock_get_client.return_value = mock_lf
                llm_node(MOCK_STATE, store=store)
    span_calls = mock_lf.update_current_span.call_args_list
    assert span_calls, "Expected update_current_span to be called"
    call_kwargs = span_calls[0].kwargs
    assert call_kwargs.get("input") == "我叫 Bob，是一名设计师"
    assert call_kwargs.get("output") == "你好 Bob！"
    assert "retrieved_count" in call_kwargs.get("metadata", {})
