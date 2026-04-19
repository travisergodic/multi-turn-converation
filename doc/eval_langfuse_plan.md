# Langfuse Eval Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument the agentic memory system with Langfuse Cloud tracing and emit P0/P1 scores (parse_success_rate, action_distribution, memory_utilization, latency) on every conversation turn.

**Architecture:** Wrap each turn in a root `@observe` span in `chat.py`; node functions each carry their own `@observe` span (child spans via contextvars). The maintenance graph runs in a background thread — we copy the context before spawning so Langfuse sees all spans under the same trace. LLM-as-judge for `memory_utilization` fires asynchronously after the assistant reply, also under the same trace. If `LANGFUSE_PUBLIC_KEY` is absent, a disabled client is used so the code works without credentials.

**Tech Stack:** `langfuse>=3.0.0`, existing `openai` SDK for judge calls, `contextvars.copy_context()` for thread propagation.

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/eval/__init__.py` | package marker |
| Create | `src/eval/tracing.py` | disabled-safe Langfuse singleton + score helpers |
| Create | `src/eval/judge.py` | LLM-as-judge prompt + `score_memory_utilization()` |
| Modify | `requirements.txt` | add `langfuse>=3.0.0` |
| Modify | `.env.example` | add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` |
| Modify | `src/nodes.py` | `@observe` on `llm_node`, `memory_update_node`, `profile_update_node`; emit scores |
| Modify | `src/retrieve.py` | `@observe` on `Retriever.retrieve()`; emit `candidate_count`, `final_count` |
| Modify | `tools/chat.py` | root `@observe` per turn; context copy for background thread; flush on exit |

---

## Task 1: Install langfuse + create `src/eval/tracing.py`

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Create: `src/eval/__init__.py`
- Create: `src/eval/tracing.py`

- [ ] **Step 1: Add langfuse to requirements.txt**

Open `requirements.txt` and add at the end:
```
langfuse>=3.0.0
```

- [ ] **Step 2: Add env vars to .env.example**

Append to `.env.example`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

- [ ] **Step 3: Install**

```bash
pip install langfuse>=3.0.0
```

- [ ] **Step 4: Write the failing test**

Create `tests/test_tracing.py`:
```python
import os
import importlib

def test_get_langfuse_returns_instance_when_disabled():
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
    import src.eval.tracing as tracing
    lf = tracing.get_langfuse()
    assert lf is not None

def test_get_langfuse_singleton():
    import src.eval.tracing as tracing
    assert tracing.get_langfuse() is tracing.get_langfuse()
```

Run:
```bash
pytest tests/test_tracing.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 5: Create `src/eval/__init__.py`**

```python
```
(empty file)

- [ ] **Step 6: Create `src/eval/tracing.py`**

```python
import os
from functools import lru_cache

from langfuse import Langfuse


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    disabled = not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    return Langfuse(disabled=disabled)
```

- [ ] **Step 7: Run test to verify it passes**

```bash
pytest tests/test_tracing.py -v
```
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add requirements.txt .env.example src/eval/__init__.py src/eval/tracing.py tests/test_tracing.py
git commit -m "feat: langfuse setup with disabled-safe singleton"
```

---

## Task 2: Instrument `Retriever.retrieve()` with `@observe`

**Files:**
- Modify: `src/retrieve.py`

The `retrieve()` method should become a child span that logs `candidate_count` and `final_count` as metadata, and emits `retrieval.latency_ms` via the span duration (automatic from `@observe`).

- [ ] **Step 1: Run existing retrieve tests to confirm baseline**

```bash
pytest tests/test_retrieve.py -v
```
Expected: all PASS

- [ ] **Step 2: Add `LANGFUSE_DISABLED=true` to conftest so tests never need real creds**

Open `tests/conftest.py` and add at the top (or create the file if absent):
```python
import os
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
```

- [ ] **Step 3: Add `@observe` import and decorate `retrieve()`**

In `src/retrieve.py`, add the import after existing imports:
```python
from langfuse.decorators import langfuse_context, observe
```

Replace the `def retrieve(self, query: str, memories: list[dict]) -> list[dict]:` method signature and its first two lines with:

```python
@observe(name="retrieve")
def retrieve(self, query: str, memories: list[dict]) -> list[dict]:
    if not memories:
        logger.info("Retriever received no memories for query_chars=%s", len(query))
        return []

    topk = min(self.embed_topk, len(memories))
    scores = self._embedding_scores(query, memories)
    top_indices = np.argsort(scores)[::-1][:topk].tolist()
    candidates = [memories[index] for index in top_indices]

    rerank_topk = min(self.rerank_topk, len(candidates))
    rerank_scores = self._rerank_scores(query, candidates)
    ranked = sorted(zip(rerank_scores, candidates), key=lambda item: item[0], reverse=True)
    results = [memory for _, memory in ranked[:rerank_topk]]

    langfuse_context.update_current_observation(
        metadata={
            "candidate_count": len(candidates),
            "final_count": len(results),
            "total_memories": len(memories),
        }
    )
    logger.info(
        "Retriever completed query_chars=%s total_memories=%s embed_topk=%s rerank_topk=%s returned=%s",
        len(query),
        len(memories),
        topk,
        rerank_topk,
        len(results),
    )
    return results
```

- [ ] **Step 4: Run retrieve tests**

```bash
pytest tests/test_retrieve.py -v
```
Expected: all PASS (decorator is a no-op when disabled)

- [ ] **Step 5: Commit**

```bash
git add src/retrieve.py tests/conftest.py
git commit -m "feat: instrument Retriever.retrieve with @observe"
```

---

## Task 3: Instrument `llm_node` with `@observe`

**Files:**
- Modify: `src/nodes.py`

The `llm_node` span should log `retrieved_count`, `total_memories`, `reply_chars` as metadata, and emit `response.latency_ms` automatically via span duration.

- [ ] **Step 1: Run existing node tests to confirm baseline**

```bash
pytest tests/test_nodes.py -v
```
Expected: all PASS

- [ ] **Step 2: Add `@observe` import to `src/nodes.py`**

After the existing imports in `src/nodes.py`, add:
```python
from langfuse.decorators import langfuse_context, observe
```

- [ ] **Step 3: Decorate `llm_node`**

Replace the function signature `def llm_node(state: ConversationState, store: BaseStore) -> dict:` with:

```python
@observe(name="llm_node")
def llm_node(state: ConversationState, store: BaseStore) -> dict:
```

Then, just before `return`, add:
```python
    langfuse_context.update_current_observation(
        input=user_message,
        output=reply,
        metadata={
            "retrieved_count": len(retrieved),
            "total_memories": len(all_memories),
            "reply_chars": len(reply or ""),
            "windowed_messages": len(windowed_messages),
        },
    )
```

- [ ] **Step 4: Run node tests**

```bash
pytest tests/test_nodes.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/nodes.py
git commit -m "feat: instrument llm_node with @observe"
```

---

## Task 4: Instrument `memory_update_node` — P0 scores

**Files:**
- Modify: `src/nodes.py`

Emit `memory_update.parse_success` (0 or 1) and `memory_update.action_distribution` (as metadata dict) immediately after JSON parsing.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_nodes.py`:
```python
from unittest.mock import MagicMock, patch

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
        with patch("src.nodes.langfuse_context") as mock_ctx:
            memory_update_node(state, store=store)
    score_calls = [
        call for call in mock_ctx.score_current_observation.call_args_list
        if call.kwargs.get("name") == "memory_update.parse_success"
    ]
    assert score_calls, "Expected parse_success score to be emitted"
    assert score_calls[0].kwargs["value"] == 1.0


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
        with patch("src.nodes.langfuse_context") as mock_ctx:
            memory_update_node(state, store=store)
    score_calls = [
        call for call in mock_ctx.score_current_observation.call_args_list
        if call.kwargs.get("name") == "memory_update.parse_success"
    ]
    assert score_calls, "Expected parse_success score to be emitted"
    assert score_calls[0].kwargs["value"] == 0.0
```

Run:
```bash
pytest tests/test_nodes.py::test_memory_update_node_reports_parse_success tests/test_nodes.py::test_memory_update_node_reports_parse_failure -v
```
Expected: FAIL

- [ ] **Step 2: Decorate `memory_update_node` and emit scores**

Replace `def memory_update_node(state: ConversationState, store: BaseStore) -> dict:` with:

```python
@observe(name="memory_update_node")
def memory_update_node(state: ConversationState, store: BaseStore) -> dict:
```

Find the `try/except json.JSONDecodeError` block and replace it with:

```python
    try:
        operations = json.loads(raw)
        langfuse_context.score_current_observation(name="memory_update.parse_success", value=1.0)
    except json.JSONDecodeError:
        logger.warning("memory_update_node received invalid JSON: %s", raw)
        langfuse_context.score_current_observation(name="memory_update.parse_success", value=0.0)
        return {}
```

Then after the `for op in operations:` loop (before `total_elapsed_ms`), add:

```python
    action_counts: dict[str, int] = {}
    for op in operations:
        action_counts[op.get("action", "unknown")] = action_counts.get(op.get("action", "unknown"), 0) + 1
    langfuse_context.update_current_observation(
        metadata={
            "action_distribution": action_counts,
            "operation_count": len(operations),
        }
    )
```

Wait — the original loop already iterates `operations`. Replace the existing loop body with a combined version:

```python
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
            update_memory(store, user_id, mid, content)
        elif action == "delete" and mid:
            delete_memory(store, user_id, mid)
        else:
            logger.info("memory_update_node ignored operation action=%s memory_id=%s", action, mid)

    langfuse_context.update_current_observation(
        metadata={
            "action_distribution": action_counts,
            "operation_count": len(operations),
        }
    )
```

- [ ] **Step 3: Run all node tests**

```bash
pytest tests/test_nodes.py -v
```
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add src/nodes.py tests/test_nodes.py
git commit -m "feat: memory_update_node @observe + P0 scores (parse_success, action_distribution)"
```

---

## Task 5: Instrument `profile_update_node` with `@observe` + scores

**Files:**
- Modify: `src/nodes.py`

Emit `profile_update.parse_success` and record whether an update happened.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_nodes.py`:
```python
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
        with patch("src.nodes.langfuse_context") as mock_ctx:
            profile_update_node(state, store=store)
    score_calls = [
        call for call in mock_ctx.score_current_observation.call_args_list
        if call.kwargs.get("name") == "profile_update.parse_success"
    ]
    assert score_calls
    assert score_calls[0].kwargs["value"] == 1.0
```

Run:
```bash
pytest tests/test_nodes.py::test_profile_update_node_reports_parse_success_on_noop -v
```
Expected: FAIL

- [ ] **Step 2: Decorate `profile_update_node` and emit scores**

Replace `def profile_update_node(state: ConversationState, store: BaseStore) -> dict:` with:

```python
@observe(name="profile_update_node")
def profile_update_node(state: ConversationState, store: BaseStore) -> dict:
```

After `raw = chat_completion(prompt_messages).strip()`, in the `if raw == "null":` branch add:
```python
    if raw == "null":
        langfuse_context.score_current_observation(name="profile_update.parse_success", value=1.0)
        langfuse_context.update_current_observation(metadata={"updated": False})
        # ... existing log and return
```

In the `except json.JSONDecodeError:` branch add:
```python
        langfuse_context.score_current_observation(name="profile_update.parse_success", value=0.0)
```

In the success path (after writing the file), add:
```python
        langfuse_context.score_current_observation(name="profile_update.parse_success", value=1.0)
        langfuse_context.update_current_observation(metadata={"updated": True})
```

Full updated `profile_update_node` body after the prompt call:

```python
    raw = chat_completion(prompt_messages).strip()
    if raw == "null":
        total_elapsed_ms = (perf_counter() - started_at) * 1000
        logger.info("profile_update_node decided no profile update is needed total_elapsed_ms=%.2f", total_elapsed_ms)
        langfuse_context.score_current_observation(name="profile_update.parse_success", value=1.0)
        langfuse_context.update_current_observation(metadata={"updated": False})
        return {}

    try:
        updated_profile = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("profile_update_node received invalid JSON: %s", raw)
        langfuse_context.score_current_observation(name="profile_update.parse_success", value=0.0)
        return {}

    profile_path = _cfg["profile"]["path"]
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(updated_profile, f, ensure_ascii=False, indent=2)

    total_elapsed_ms = (perf_counter() - started_at) * 1000
    logger.info("profile_update_node wrote updated profile to %s total_elapsed_ms=%.2f", profile_path, total_elapsed_ms)
    langfuse_context.score_current_observation(name="profile_update.parse_success", value=1.0)
    langfuse_context.update_current_observation(metadata={"updated": True})
    return {"profile_snapshot": updated_profile}
```

- [ ] **Step 3: Run all node tests**

```bash
pytest tests/test_nodes.py -v
```
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add src/nodes.py tests/test_nodes.py
git commit -m "feat: profile_update_node @observe + parse_success score"
```

---

## Task 6: Wire root trace in `tools/chat.py` + thread context propagation

**Files:**
- Modify: `tools/chat.py`

Each conversation turn becomes one Langfuse trace. The `@observe(name="conversation_turn")` on the turn handler creates the root span. Before spawning the background thread, we copy the current context so all maintenance-graph node spans appear as children of the same trace.

- [ ] **Step 1: Add imports to `tools/chat.py`**

After the existing `from langfuse import Langfuse` (add it if missing) imports, add:
```python
import contextvars
from langfuse.decorators import langfuse_context, observe
from src.eval.tracing import get_langfuse
```

- [ ] **Step 2: Wrap the turn logic in an `@observe` function**

Extract the per-turn logic out of the `while True:` loop into a dedicated function:

```python
@observe(name="conversation_turn")
def handle_turn(
    user_input: str,
    thread_id: str,
    user_id: str,
    response_graph,
    maintenance_graph,
    known_threads: list[str],
) -> tuple[str | None, dict]:
    """Returns (reply_text, full_result_state)."""
    langfuse_context.update_current_trace(
        user_id=user_id,
        session_id=thread_id,
        input=user_input,
    )
    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "messages": [HumanMessage(content=user_input)],
        "user_id": user_id,
        "retrieved_memories": [],
        "profile_snapshot": load_profile(),
    }
    result = response_graph.invoke(state, config=config)
    ai_messages = [m for m in result["messages"] if m.__class__.__name__ == "AIMessage"]
    reply = ai_messages[-1].content if ai_messages else None
    if reply:
        langfuse_context.update_current_trace(output=reply)
    return reply, result
```

- [ ] **Step 3: Update the `while True:` loop to call `handle_turn` and propagate context to background thread**

Replace the section inside the loop (from `config = ...` through `worker.start()`) with:

```python
        ctx = contextvars.copy_context()
        reply, result = handle_turn(
            user_input, thread_id, user_id, graph, maintenance_graph, known_threads
        )
        if reply:
            logger.info("CLI received assistant reply thread_id=%s reply_chars=%s", thread_id, len(reply))
            print(f"助手：{reply}\n")
            worker = threading.Thread(
                target=ctx.run,
                args=(run_background_updates, maintenance_graph, dict(result), config, thread_id),
                daemon=True,
                name=f"memory-maintenance-{thread_id[:8]}",
            )
            worker.start()
            logger.info("CLI scheduled background updates thread_id=%s", thread_id)
```

Note: `config` must still be defined before this block:
```python
        config = {"configurable": {"thread_id": thread_id}}
```

- [ ] **Step 4: Flush Langfuse on exit**

At the end of `main()`, after the `while True:` loop, add:
```python
    get_langfuse().flush()
```

- [ ] **Step 5: Smoke test**

```bash
python tools/chat.py
```
Type one message, type `/quit`. Verify no errors. If `LANGFUSE_PUBLIC_KEY` is set, check Langfuse Cloud for a trace. If not set, verify it exits cleanly.

- [ ] **Step 6: Commit**

```bash
git add tools/chat.py
git commit -m "feat: root Langfuse trace per turn with background thread context propagation"
```

---

## Task 7: LLM-as-judge for `response.memory_utilization`

**Files:**
- Create: `src/eval/judge.py`
- Modify: `tools/chat.py` (add async judge call in background)

The judge fires after the reply is shown. It uses `claude-3-5-sonnet` (different from the main model) and emits a 0–1 score on the trace.

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge.py`:
```python
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
```

Run:
```bash
pytest tests/test_judge.py -v
```
Expected: FAIL

- [ ] **Step 2: Create `src/eval/judge.py`**

```python
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
```

- [ ] **Step 3: Run judge tests**

```bash
pytest tests/test_judge.py -v
```
Expected: all PASS

- [ ] **Step 4: Add judge call to `run_background_updates` in `tools/chat.py`**

Modify `run_background_updates` to accept and score memory utilization:

```python
def run_background_updates(
    maintenance_graph,
    state: dict,
    config: dict,
    thread_id: str,
    retrieved_memories: list[dict] | None = None,
    reply: str | None = None,
) -> None:
    from src.eval.judge import score_memory_utilization
    started_at = perf_counter()
    logger.info("Background updates started thread_id=%s", thread_id)

    if retrieved_memories and reply:
        score = score_memory_utilization(retrieved_memories, reply)
        if score is not None:
            langfuse_context.score_current_trace(
                name="response.memory_utilization",
                value=score,
                comment="LLM-as-judge: did reply use retrieved memories?",
            )
            logger.info("Scored memory_utilization=%.2f thread_id=%s", score, thread_id)

    try:
        maintenance_graph.invoke(state, config=config)
        elapsed_ms = (perf_counter() - started_at) * 1000
        logger.info("Background updates finished thread_id=%s total_elapsed_ms=%.2f", thread_id, elapsed_ms)
    except Exception:
        logger.exception("Background updates failed thread_id=%s", thread_id)
```

Update the call site in `main()` to pass `retrieved_memories` and `reply`:

```python
            worker = threading.Thread(
                target=ctx.run,
                args=(
                    run_background_updates,
                    maintenance_graph,
                    dict(result),
                    config,
                    thread_id,
                ),
                kwargs={
                    "retrieved_memories": result.get("retrieved_memories", []),
                    "reply": reply,
                },
                daemon=True,
                name=f"memory-maintenance-{thread_id[:8]}",
            )
```

- [ ] **Step 5: Add `langfuse_context` import to `tools/chat.py`** (if not already present from Task 6)

Verify `from langfuse.decorators import langfuse_context, observe` is at the top of `tools/chat.py`.

- [ ] **Step 6: Run all tests**

```bash
pytest -v
```
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/eval/judge.py tests/test_judge.py tools/chat.py
git commit -m "feat: LLM-as-judge for response.memory_utilization"
```

---

## Self-Review

### Spec coverage check

| Requirement | Covered by |
|-------------|-----------|
| P0: `memory_update.parse_success_rate` | Task 4 — `score_current_observation` on JSON parse |
| P0: `memory_update.action_distribution` | Task 4 — metadata dict in observation |
| P1: `response.memory_utilization` | Task 7 — judge in background thread |
| P1: `turn.total_latency_ms` | Automatic from `@observe` span duration on `conversation_turn` |
| Trace per turn with session_id | Task 6 — `update_current_trace(session_id=thread_id)` |
| Child spans for each node | Tasks 2–5 — `@observe` on each node, nested under turn trace |
| No-creds graceful degradation | Task 1 — `disabled=True` when env vars absent |
| Background thread context | Task 6 — `contextvars.copy_context()` before thread spawn |
| `retrieval.candidate_count` / `final_count` | Task 2 — metadata on retrieve span |
| `profile_update.parse_success` | Task 5 |
| Flush on exit | Task 6 — `get_langfuse().flush()` |

### Type consistency check

- `score_memory_utilization(retrieved_memories: list[dict], reply: str) -> float | None` — used in Task 7 with `result.get("retrieved_memories", [])` (list[dict]) and `reply` (str). ✓
- `handle_turn(...)` returns `tuple[str | None, dict]` — caller unpacks as `reply, result`. ✓
- `langfuse_context.score_current_observation(name=..., value=...)` — consistent across Tasks 4, 5. ✓
- `langfuse_context.score_current_trace(name=..., value=..., comment=...)` — used in Task 7 for judge score (trace-level, not span-level, because it represents the full turn quality). ✓
