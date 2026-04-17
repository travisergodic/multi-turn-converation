from unittest.mock import patch, MagicMock
from src.llm_client import chat_completion

def test_chat_completion_returns_string():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "你好！"
    with patch("src.llm_client.client.chat.completions.create", return_value=mock_response):
        result = chat_completion([{"role": "user", "content": "你好"}])
    assert result == "你好！"
