from unittest.mock import patch

from src.eval.judge import score_memory_utilization


def test_score_memory_utilization_valid():
    memories = [{"id": "abc", "content": "用户叫 Bob，是设计师"}]
    reply = "好的，Bob，作为设计师你应该..."
    with patch("src.eval.judge.chat_completion", return_value='{"score": 0.9, "reason": "提到了名字和职业"}'):
        result = score_memory_utilization(memories, reply)
    assert 0.0 <= result <= 1.0
    assert abs(result - 0.9) < 1e-6


def test_score_memory_utilization_no_memories():
    result = score_memory_utilization([], "你好！")
    assert result is None


def test_score_memory_utilization_bad_json():
    memories = [{"id": "abc", "content": "用户叫 Bob"}]
    with patch("src.eval.judge.chat_completion", return_value="not json"):
        result = score_memory_utilization(memories, "你好！")
    assert result is None
