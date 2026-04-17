from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.graph import build_graph


def test_graph_returns_ai_response(tmp_path):
    graph = build_graph(db_path=":memory:", store_path=str(tmp_path / "memory.json"))
    config = {"configurable": {"thread_id": "test-1"}}
    state = {
        "messages": [HumanMessage(content="你好")],
        "user_id": "u1",
        "retrieved_memories": [],
        "profile_snapshot": {},
    }
    with patch("src.nodes.chat_completion", return_value="你好！"):
        with patch("src.nodes.retriever.retrieve", return_value=[]):
            result = graph.invoke(state, config=config)
    assert any(m.__class__.__name__ == "AIMessage" for m in result["messages"])
