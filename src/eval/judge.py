import json

from src.llm_client import chat_completion

JUDGE_MODEL = "anthropic/claude-3.5-sonnet"

MEMORY_UTILIZATION_PROMPT = """你是一个评测助手。判断"助手回复"是否实际利用了"检索到的记忆"中的信息。

## 检索到的记忆
{memories_text}

## 助手回复
{reply}

输出格式（只输出 JSON，不要其他内容）：
{{"score": <0.0到1.0之间的浮点数>, "reason": "<一句话说明>"}}

评分标准：
- 1.0：回复明确使用了记忆中的具体信息（如称呼用户名字、引用已知偏好）
- 0.5：回复与记忆相关但未明确引用
- 0.0：回复完全忽略了可用的记忆信息"""


def score_memory_utilization(retrieved_memories: list[dict], reply: str) -> float | None:
    if not retrieved_memories:
        return None

    memories_text = "\n".join(f"- [{m['id'][:8]}] {m['content']}" for m in retrieved_memories)
    prompt = MEMORY_UTILIZATION_PROMPT.format(memories_text=memories_text, reply=reply)

    try:
        raw = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=JUDGE_MODEL,
        )
        result = json.loads(raw)
        score = float(result["score"])
        return max(0.0, min(1.0, score))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
