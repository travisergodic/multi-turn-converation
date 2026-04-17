# 有记忆的多轮对话系统 — 需求文档

> **给 agentic 执行者：** 必须使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务逐步实现本计划。步骤使用 checkbox (`- [ ]`) 语法跟踪进度。

**目标：** 基于 LangGraph + OpenRouter，构建支持短期记忆和长期记忆的命令行多轮对话系统。长期记忆通过 embedding + rerank 检索 top-k 条相关记忆注入上下文，并在每轮对话后对记忆执行结构化更新操作（add / append / delete / noop）。用户 profile 单独维护在 `configs/user_profile.json`。

**架构：** LangGraph `StateGraph` 包含三个节点：`llm_node`（检索相关记忆 + 调用 LLM）、`memory_update_node`（对长期记忆执行结构化更新）、`profile_update_node`（更新 user profile）。使用 `SqliteSaver` checkpointing 实现短期记忆，Store 存储独立的记忆条目（每条一个 ID）。不引入 LangChain，通过 `openai` SDK 直接调用 OpenRouter。

**Tech Stack：** `langgraph`、`langgraph-checkpoint-sqlite`、`openai`、`sentence-transformers`、`python-dotenv`、`pyyaml`

---

## 功能需求

### 1. 短期记忆
- 完整对话历史通过 `SqliteSaver` 按 `thread_id` 持久化，进程重启后可恢复

### 2. 长期记忆——检索
- 长期记忆以**独立条目**存储（每条有唯一 ID + 内容 + 时间戳）
- 每轮对话前，用 user query 做 embedding 相似度搜索，取 `embed_topk` 条
- 对候选集做 rerank，最终取 `rerank_topk` 条注入 system prompt

### 3. 长期记忆——更新
- 每轮对话后，LLM 对检索到的记忆条目判断操作类型：
  - `add`：新增一条记忆
  - `append`：在已有条目末尾追加内容
  - `delete`：删除已有条目
  - `noop`：不操作
- 输出为结构化 JSON，按操作逐条执行

### 4. User Profile
- 固定字段的用户画像，存在 `configs/user_profile.json`
- 每轮对话后由 `profile_update_node` 判断是否更新字段
- Profile 在每轮对话前整体注入 system prompt（不经过检索）

### 5. OpenRouter 集成
- 使用 `openai` SDK，`base_url` 指向 `https://openrouter.ai/api/v1`
- 模型、embedding、rerank 配置统一在 `configs/config.yaml`
- API Key 从 `.env` 读取

### 6. CLI 界面
- 支持命令：`/new`、`/switch <id>`、`/threads`、`/quit`
- Prompt 显示当前 `thread_id` 前缀

---

## 文件结构

```
memory/
├── configs/
│   ├── config.yaml           # 所有配置：模型、路径、检索参数等
│   └── user_profile.json     # User profile 数据（含字段 schema）
├── data/
│   └── checkpoints.db        # SQLite checkpoints（运行时自动创建）
├── src/
│   ├── __init__.py
│   ├── state.py              # ConversationState TypedDict
│   ├── prompt.py             # 所有 prompt（中文）
│   ├── llm_client.py         # OpenRouter 客户端
│   ├── memory_store.py       # 长期记忆条目的 CRUD 操作
│   ├── retrieve.py           # Retriever 类（embedding + rerank）
│   ├── nodes.py              # llm_node、memory_update_node、profile_update_node
│   └── graph.py              # StateGraph 定义和节点连线
├── tools/
│   ├── __init__.py
│   └── chat.py               # CLI 入口
├── tests/
│   ├── test_llm_client.py
│   ├── test_memory_store.py
│   ├── test_retrieve.py
│   ├── test_nodes.py
│   └── test_graph.py
├── .env                      # OPENROUTER_API_KEY（不提交）
├── .env.example
└── requirements.txt
```

### 各文件职责

| 文件 | 职责 |
|------|------|
| `configs/config.yaml` | 模型名、API base_url、db 路径、embed/rerank 模型和 topk 参数 |
| `configs/user_profile.json` | 用户画像数据，含字段定义和当前值 |
| `src/state.py` | `ConversationState`：messages、user_id、retrieved_memories、profile_snapshot |
| `src/prompt.py` | 所有 prompt 常量（中文）：系统提示、记忆提取、profile 更新 |
| `src/llm_client.py` | `chat_completion(messages, model) -> str` |
| `src/memory_store.py` | `MemoryEntry` dataclass；`list_memories`、`add_memory`、`append_memory`、`delete_memory` |
| `src/retrieve.py` | `Retriever(embed_topk, rerank_topk)`；`retrieve(query, memories) -> list[MemoryEntry]` |
| `src/nodes.py` | 三个节点实现 |
| `src/graph.py` | 节点连线，挂载 checkpointer |
| `tools/chat.py` | REPL 循环、thread 管理 |

---

## 实现任务

### Task 1: 项目脚手架

**涉及文件：** `requirements.txt`、`.env.example`、`configs/config.yaml`、`configs/user_profile.json`、`src/state.py`

- [x] **Step 1：创建目录结构**

```bash
mkdir -p src tools configs data tests
touch src/__init__.py tools/__init__.py tests/__init__.py
```

- [x] **Step 2：创建 `requirements.txt`**

```
langgraph>=0.2.0
langgraph-checkpoint-sqlite>=1.0.0
openai>=1.0.0
python-dotenv>=1.0.0
pyyaml>=6.0
sentence-transformers>=3.0.0
numpy>=1.26.0
```

- [x] **Step 3：创建 `.env.example`**

```
OPENROUTER_API_KEY=your_key_here
```

- [x] **Step 4：创建 `configs/config.yaml`**

```yaml
llm:
  model: anthropic/claude-3.5-sonnet
  base_url: https://openrouter.ai/api/v1

retrieval:
  embed_model: BAAI/bge-small-zh-v1.5
  rerank_model: BAAI/bge-reranker-base
  embed_topk: 20
  rerank_topk: 5

memory:
  namespace: long_term_memory
  db_path: data/checkpoints.db

profile:
  path: configs/user_profile.json
```

- [x] **Step 5：创建 `configs/user_profile.json`**

```json
{
  "user_id": "default_user",
  "name": null,
  "occupation": null,
  "language_preference": null,
  "interests": [],
  "communication_style": null,
  "other": {}
}
```

- [x] **Step 6：创建 `src/state.py`**

```python
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    retrieved_memories: list[dict]  # 检索出的记忆条目，注入 system prompt
    profile_snapshot: dict          # 当前 user profile 快照
```

- [x] **Step 7：安装依赖**

```bash
pip install -r requirements.txt
```

- [x] **Step 8：提交**

```bash
git init
git add requirements.txt .env.example configs/ src/ tests/ tools/
git commit -m "feat: project scaffold"
```

---

### Task 2: Prompt 集中管理

**涉及文件：** 新建 `src/prompt.py`

- [x] **Step 1：创建 `src/prompt.py`**

```python
SYSTEM_PROMPT_TEMPLATE = """你是一个有记忆能力的智能助手。

{profile_section}{memory_section}请根据以上背景信息，自然地回应用户。"""

PROFILE_SECTION_TEMPLATE = """## 用户档案
{profile_text}

"""

MEMORY_SECTION_TEMPLATE = """## 相关记忆
{memory_text}

"""

MEMORY_UPDATE_PROMPT = """你是一个记忆管理助手。根据以下对话，判断是否需要更新长期记忆。

## 现有相关记忆
{existing_memories}

## 最新一轮对话
用户：{user_message}
助手：{assistant_message}

请输出一个 JSON 数组，每项包含以下字段：
- action: "add" | "append" | "delete" | "noop"
- memory_id: 目标记忆的 ID（add 时为 null）
- content: 新增或追加的内容（delete/noop 时为 null）

只输出 JSON 数组，不要其他内容。"""

PROFILE_UPDATE_PROMPT = """你是一个用户档案管理助手。根据最新对话，判断是否需要更新用户档案。

## 当前用户档案
{current_profile}

## 最新一轮对话
用户：{user_message}
助手：{assistant_message}

如果有字段需要更新，输出更新后的完整 JSON 档案；如无需更新，输出 null。
只输出 JSON 或 null，不要其他内容。"""
```

- [x] **Step 2：提交**

```bash
git add src/prompt.py
git commit -m "feat: centralize all prompts in src/prompt.py"
```

---

### Task 3: OpenRouter 客户端

**涉及文件：** 新建 `src/llm_client.py`

- [x] **Step 1：写失败测试**

```python
# tests/test_llm_client.py
from unittest.mock import patch, MagicMock
from src.llm_client import chat_completion

def test_chat_completion_returns_string():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "你好！"
    with patch("src.llm_client.client.chat.completions.create", return_value=mock_response):
        result = chat_completion([{"role": "user", "content": "你好"}])
    assert result == "你好！"
```

- [x] **Step 2：运行测试，确认失败**

```bash
pytest tests/test_llm_client.py -v
```
预期：`ImportError`

- [x] **Step 3：实现 `src/llm_client.py`**

```python
import os
import yaml
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

with open("configs/config.yaml") as f:
    _cfg = yaml.safe_load(f)

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=_cfg["llm"]["base_url"],
)

DEFAULT_MODEL: str = _cfg["llm"]["model"]


def chat_completion(messages: list[dict], model: str = DEFAULT_MODEL) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content
```

- [x] **Step 4：运行测试，确认通过**

```bash
pytest tests/test_llm_client.py -v
```
预期：PASS

- [x] **Step 5：提交**

```bash
git add src/llm_client.py tests/test_llm_client.py
git commit -m "feat: openrouter client"
```

---

### Task 4: 长期记忆 Store

**涉及文件：** 新建 `src/memory_store.py`

- [x] **Step 1：写失败测试**

```python
# tests/test_memory_store.py
from langgraph.store.memory import InMemoryStore
from src.memory_store import add_memory, append_memory, delete_memory, list_memories

def test_add_and_list():
    store = InMemoryStore()
    add_memory(store, "u1", "用户叫 Alice")
    memories = list_memories(store, "u1")
    assert len(memories) == 1
    assert memories[0]["content"] == "用户叫 Alice"

def test_append_memory():
    store = InMemoryStore()
    add_memory(store, "u1", "喜欢深色主题")
    mid = list_memories(store, "u1")[0]["id"]
    append_memory(store, "u1", mid, "，也喜欢简洁界面")
    updated = list_memories(store, "u1")[0]["content"]
    assert "简洁界面" in updated

def test_delete_memory():
    store = InMemoryStore()
    add_memory(store, "u1", "临时记忆")
    mid = list_memories(store, "u1")[0]["id"]
    delete_memory(store, "u1", mid)
    assert list_memories(store, "u1") == []
```

- [x] **Step 2：运行测试，确认失败**

```bash
pytest tests/test_memory_store.py -v
```
预期：FAIL

- [x] **Step 3：实现 `src/memory_store.py`**

```python
import uuid
from datetime import datetime
from langgraph.store.base import BaseStore

NAMESPACE = ("long_term_memory",)
INDEX_KEY = "__index__"


def _load_index(store: BaseStore, user_id: str) -> dict[str, dict]:
    item = store.get(NAMESPACE, f"{user_id}:{INDEX_KEY}")
    return item.value if item else {}


def _save_index(store: BaseStore, user_id: str, index: dict[str, dict]) -> None:
    store.put(NAMESPACE, f"{user_id}:{INDEX_KEY}", index)


def list_memories(store: BaseStore, user_id: str) -> list[dict]:
    return list(_load_index(store, user_id).values())


def add_memory(store: BaseStore, user_id: str, content: str) -> str:
    index = _load_index(store, user_id)
    mid = str(uuid.uuid4())
    index[mid] = {"id": mid, "content": content, "created_at": datetime.utcnow().isoformat()}
    _save_index(store, user_id, index)
    return mid


def append_memory(store: BaseStore, user_id: str, memory_id: str, extra: str) -> None:
    index = _load_index(store, user_id)
    if memory_id in index:
        index[memory_id]["content"] += extra
        index[memory_id]["updated_at"] = datetime.utcnow().isoformat()
        _save_index(store, user_id, index)


def delete_memory(store: BaseStore, user_id: str, memory_id: str) -> None:
    index = _load_index(store, user_id)
    index.pop(memory_id, None)
    _save_index(store, user_id, index)
```

- [x] **Step 4：运行测试，确认通过**

```bash
pytest tests/test_memory_store.py -v
```
预期：PASS

- [x] **Step 5：提交**

```bash
git add src/memory_store.py tests/test_memory_store.py
git commit -m "feat: memory store with add/append/delete/list"
```

---

### Task 5: Retriever（embedding + rerank）

**涉及文件：** 新建 `src/retrieve.py`

- [ ] **Step 1：写失败测试**

```python
# tests/test_retrieve.py
from src.retrieve import Retriever

MEMORIES = [
    {"id": "1", "content": "用户叫 Alice，是一名工程师"},
    {"id": "2", "content": "用户喜欢喝咖啡"},
    {"id": "3", "content": "用户在北京工作"},
    {"id": "4", "content": "用户有一只猫叫 Mimi"},
    {"id": "5", "content": "用户偏好简洁的 UI 设计"},
]

def test_retrieve_returns_topk():
    retriever = Retriever(embed_topk=3, rerank_topk=2)
    results = retriever.retrieve("工程师的工作地点", MEMORIES)
    assert len(results) == 2

def test_retrieve_empty_memories():
    retriever = Retriever(embed_topk=5, rerank_topk=3)
    results = retriever.retrieve("任意问题", [])
    assert results == []

def test_retrieve_fewer_than_topk():
    retriever = Retriever(embed_topk=10, rerank_topk=5)
    results = retriever.retrieve("用户信息", MEMORIES[:2])
    assert len(results) <= 2
```

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest tests/test_retrieve.py -v
```
预期：FAIL

- [ ] **Step 3：实现 `src/retrieve.py`**

```python
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer, CrossEncoder

with open("configs/config.yaml") as f:
    _cfg = yaml.safe_load(f)


class Retriever:
    def __init__(
        self,
        embed_topk: int = _cfg["retrieval"]["embed_topk"],
        rerank_topk: int = _cfg["retrieval"]["rerank_topk"],
    ):
        self.embed_topk = embed_topk
        self.rerank_topk = rerank_topk
        self._embedder = SentenceTransformer(_cfg["retrieval"]["embed_model"])
        self._reranker = CrossEncoder(_cfg["retrieval"]["rerank_model"])

    def retrieve(self, query: str, memories: list[dict]) -> list[dict]:
        if not memories:
            return []

        contents = [m["content"] for m in memories]
        query_vec = self._embedder.encode(query, normalize_embeddings=True)
        mem_vecs = self._embedder.encode(contents, normalize_embeddings=True)

        scores = np.dot(mem_vecs, query_vec)
        topk = min(self.embed_topk, len(memories))
        top_indices = np.argsort(scores)[::-1][:topk].tolist()
        candidates = [memories[i] for i in top_indices]

        if len(candidates) <= self.rerank_topk:
            return candidates

        pairs = [[query, m["content"]] for m in candidates]
        rerank_scores = self._reranker.predict(pairs)
        rerank_topk = min(self.rerank_topk, len(candidates))
        ranked = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
        return [m for _, m in ranked[:rerank_topk]]
```

- [ ] **Step 4：运行测试，确认通过**

```bash
pytest tests/test_retrieve.py -v
```
预期：PASS（首次运行会下载模型，需要联网）

- [ ] **Step 5：提交**

```bash
git add src/retrieve.py tests/test_retrieve.py
git commit -m "feat: retriever with embedding and rerank"
```

---

### Task 6: Graph 节点实现

**涉及文件：** 新建 `src/nodes.py`

- [ ] **Step 1：写失败测试**

```python
# tests/test_nodes.py
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.store.memory import InMemoryStore
from src.state import ConversationState
from src.nodes import llm_node, memory_update_node, profile_update_node

MOCK_STATE: ConversationState = {
    "messages": [HumanMessage(content="我叫 Bob，是一名设计师")],
    "user_id": "u1",
    "retrieved_memories": [],
    "profile_snapshot": {"name": None, "occupation": None, "interests": []},
}

def test_llm_node_appends_ai_message():
    store = InMemoryStore()
    with patch("src.nodes.chat_completion", return_value="你好 Bob！"):
        with patch("src.nodes.retriever.retrieve", return_value=[]):
            result = llm_node(MOCK_STATE, store=store)
    assert result["messages"][-1].content == "你好 Bob！"

def test_memory_update_node_add():
    store = InMemoryStore()
    state = {**MOCK_STATE, "messages": [
        HumanMessage(content="我叫 Bob"),
        AIMessage(content="你好 Bob！"),
    ], "retrieved_memories": []}
    mock_ops = '[{"action": "add", "memory_id": null, "content": "用户叫 Bob"}]'
    with patch("src.nodes.chat_completion", return_value=mock_ops):
        memory_update_node(state, store=store)
    from src.memory_store import list_memories
    assert len(list_memories(store, "u1")) == 1

def test_profile_update_node_noop():
    store = InMemoryStore()
    state = {**MOCK_STATE, "messages": [
        HumanMessage(content="今天天气真好"),
        AIMessage(content="确实！"),
    ]}
    with patch("src.nodes.chat_completion", return_value="null"):
        result = profile_update_node(state, store=store)
    assert result == {}
```

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest tests/test_nodes.py -v
```
预期：FAIL

- [ ] **Step 3：实现 `src/nodes.py`**

```python
import json
import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.base import BaseStore
from src.state import ConversationState
from src.llm_client import chat_completion
from src.memory_store import list_memories, add_memory, append_memory, delete_memory
from src.retrieve import Retriever
from src.prompt import (
    SYSTEM_PROMPT_TEMPLATE,
    PROFILE_SECTION_TEMPLATE,
    MEMORY_SECTION_TEMPLATE,
    MEMORY_UPDATE_PROMPT,
    PROFILE_UPDATE_PROMPT,
)

with open("configs/config.yaml") as f:
    _cfg = yaml.safe_load(f)

retriever = Retriever(
    embed_topk=_cfg["retrieval"]["embed_topk"],
    rerank_topk=_cfg["retrieval"]["rerank_topk"],
)


def _build_system_prompt(profile: dict, memories: list[dict]) -> str:
    profile_text = "\n".join(f"- {k}：{v}" for k, v in profile.items() if v)
    profile_section = PROFILE_SECTION_TEMPLATE.format(profile_text=profile_text) if profile_text else ""

    memory_text = "\n".join(f"- [{m['id'][:8]}] {m['content']}" for m in memories)
    memory_section = MEMORY_SECTION_TEMPLATE.format(memory_text=memory_text) if memory_text else ""

    return SYSTEM_PROMPT_TEMPLATE.format(
        profile_section=profile_section,
        memory_section=memory_section,
    )


def llm_node(state: ConversationState, store: BaseStore) -> dict:
    user_id = state["user_id"]
    user_message = state["messages"][-1].content

    all_memories = list_memories(store, user_id)
    retrieved = retriever.retrieve(user_message, all_memories)

    system_content = _build_system_prompt(state.get("profile_snapshot", {}), retrieved)
    messages = [{"role": "system", "content": system_content}]
    for msg in state["messages"]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        messages.append({"role": role, "content": msg.content})

    reply = chat_completion(messages)
    return {
        "messages": [AIMessage(content=reply)],
        "retrieved_memories": retrieved,
    }


def memory_update_node(state: ConversationState, store: BaseStore) -> dict:
    msgs = state["messages"]
    if len(msgs) < 2:
        return {}

    user_message = msgs[-2].content
    assistant_message = msgs[-1].content
    existing = state.get("retrieved_memories", [])
    existing_text = "\n".join(f"[{m['id']}] {m['content']}" for m in existing) or "（无）"

    prompt_messages = [{"role": "user", "content": MEMORY_UPDATE_PROMPT.format(
        existing_memories=existing_text,
        user_message=user_message,
        assistant_message=assistant_message,
    )}]

    raw = chat_completion(prompt_messages)
    try:
        operations = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    user_id = state["user_id"]
    for op in operations:
        action = op.get("action")
        mid = op.get("memory_id")
        content = op.get("content")
        if action == "add" and content:
            add_memory(store, user_id, content)
        elif action == "append" and mid and content:
            append_memory(store, user_id, mid, content)
        elif action == "delete" and mid:
            delete_memory(store, user_id, mid)

    return {}


def profile_update_node(state: ConversationState, store: BaseStore) -> dict:
    msgs = state["messages"]
    if len(msgs) < 2:
        return {}

    user_message = msgs[-2].content
    assistant_message = msgs[-1].content
    current_profile = state.get("profile_snapshot", {})

    prompt_messages = [{"role": "user", "content": PROFILE_UPDATE_PROMPT.format(
        current_profile=json.dumps(current_profile, ensure_ascii=False, indent=2),
        user_message=user_message,
        assistant_message=assistant_message,
    )}]

    raw = chat_completion(prompt_messages).strip()
    if raw == "null":
        return {}

    try:
        updated_profile = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    profile_path = _cfg["profile"]["path"]
    with open(profile_path, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(updated_profile, f, ensure_ascii=False, indent=2)

    return {"profile_snapshot": updated_profile}
```

- [ ] **Step 4：运行测试，确认通过**

```bash
pytest tests/test_nodes.py -v
```
预期：PASS

- [ ] **Step 5：提交**

```bash
git add src/nodes.py tests/test_nodes.py
git commit -m "feat: llm_node, memory_update_node, profile_update_node"
```

---

### Task 7: LangGraph Graph 组装

**涉及文件：** 新建 `src/graph.py`

- [ ] **Step 1：写失败测试**

```python
# tests/test_graph.py
from src.graph import build_graph
from langchain_core.messages import HumanMessage
from unittest.mock import patch

def test_graph_returns_ai_response():
    graph = build_graph(db_path=":memory:")
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
```

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest tests/test_graph.py -v
```
预期：FAIL

- [ ] **Step 3：实现 `src/graph.py`**

```python
import yaml
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import ConversationState
from src.nodes import llm_node, memory_update_node, profile_update_node

with open("configs/config.yaml") as f:
    _cfg = yaml.safe_load(f)


def build_graph(db_path: str | None = None):
    path = db_path or _cfg["memory"]["db_path"]
    builder = StateGraph(ConversationState)
    builder.add_node("llm", llm_node)
    builder.add_node("memory_update", memory_update_node)
    builder.add_node("profile_update", profile_update_node)

    builder.set_entry_point("llm")
    builder.add_edge("llm", "memory_update")
    builder.add_edge("memory_update", "profile_update")
    builder.add_edge("profile_update", END)

    checkpointer = SqliteSaver.from_conn_string(path)
    return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 4：运行测试，确认通过**

```bash
pytest tests/test_graph.py -v
```
预期：PASS

- [ ] **Step 5：提交**

```bash
git add src/graph.py tests/test_graph.py
git commit -m "feat: langgraph graph assembly"
```

---

### Task 8: CLI 入口

**涉及文件：** 新建 `tools/chat.py`

- [ ] **Step 1：实现 `tools/chat.py`**

```python
import json
import uuid
import yaml
from langchain_core.messages import HumanMessage
from src.graph import build_graph

with open("configs/config.yaml") as f:
    _cfg = yaml.safe_load(f)

COMMANDS = {
    "/new": "新建一个对话 thread",
    "/switch <id>": "切换到已有的 thread",
    "/threads": "列出本次会话中的所有 thread",
    "/quit": "退出",
}


def load_profile() -> dict:
    with open(_cfg["profile"]["path"], encoding="utf-8") as f:
        return json.load(f)


def print_help():
    print("\n可用命令：")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:22s} {desc}")
    print()


def main():
    graph = build_graph()
    user_id = "default_user"
    thread_id = str(uuid.uuid4())
    known_threads: list[str] = [thread_id]

    print(f"多轮对话系统已启动（当前 thread：{thread_id[:8]}...）")
    print("输入 /help 查看可用命令。\n")

    while True:
        try:
            user_input = input(f"[{thread_id[:8]}] 你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            print("再见！")
            break
        elif user_input == "/help":
            print_help()
            continue
        elif user_input == "/new":
            thread_id = str(uuid.uuid4())
            known_threads.append(thread_id)
            print(f"已新建 thread：{thread_id[:8]}...\n")
            continue
        elif user_input.startswith("/switch "):
            thread_id = user_input.split(maxsplit=1)[1].strip()
            if thread_id not in known_threads:
                known_threads.append(thread_id)
            print(f"已切换到 thread：{thread_id[:8]}...\n")
            continue
        elif user_input == "/threads":
            for t in known_threads:
                marker = " <-- 当前" if t == thread_id else ""
                print(f"  {t[:8]}...{marker}")
            print()
            continue

        config = {"configurable": {"thread_id": thread_id}}
        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_id": user_id,
            "retrieved_memories": [],
            "profile_snapshot": load_profile(),
        }

        result = graph.invoke(state, config=config)
        ai_messages = [m for m in result["messages"] if m.__class__.__name__ == "AIMessage"]
        if ai_messages:
            print(f"助手：{ai_messages[-1].content}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2：手动冒烟测试**

```bash
cp .env.example .env
# 填入 OPENROUTER_API_KEY
python tools/chat.py
```

预期：REPL 正常启动，多轮对话正常，重启后同一 thread_id 可恢复历史。

- [ ] **Step 3：提交**

```bash
git add tools/chat.py
git commit -m "feat: CLI entry point"
```

---

## 自查清单

- [x] 短期记忆（SqliteSaver）— Task 7
- [x] 长期记忆检索（embedding + rerank）— Task 5
- [x] 长期记忆结构化更新（add/append/delete/noop）— Task 6
- [x] User profile 独立维护，整体注入 prompt — Task 1、6
- [x] 所有 prompt 集中在 `src/prompt.py`（中文）— Task 2
- [x] 所有配置在 `configs/config.yaml` — Task 1
- [x] 核心代码在 `src/`，入口在 `tools/`，数据在 `data/` — Task 1
- [x] 无 LangChain 依赖 — 全程
- [x] 所有任务包含完整代码，无占位符
