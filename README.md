# Memory

一个基于 LangGraph + OpenRouter 的命令行多轮对话系统，支持短期记忆、长期记忆和用户画像。

它的目标不是做一个单轮问答脚本，而是让助手在多轮对话里逐步“记住你”：

- 短期记忆：同一个 `thread_id` 下的对话历史会持久化到 SQLite，重启后可以继续接着聊。
- 长期记忆：系统会把值得保留的信息保存成独立记忆条目，在后续对话前检索并注入上下文。
- 用户画像：姓名、职业、语言偏好、兴趣等字段保存在单独的 profile 文件中，每轮对话后可更新。

## 功能概览

- 使用 OpenRouter 作为大模型调用入口
- 使用 embedding + rerank 检索相关长期记忆
- 每轮对话后自动执行记忆更新操作：`add`、`append`、`delete`、`noop`
- 支持多线程会话切换
- 支持会话索引持久化，重启后仍可查看历史 thread 列表
- 可选接入 [Langfuse](https://langfuse.com) Cloud：每轮 CLI 对话一条 trace，检索 / LLM / 记忆与画像更新等子步骤为子 span，并上报解析成功率、记忆利用率（LLM judge）等评分

## 目录说明

```text
memory/
├── configs/
│   ├── config.yaml
│   └── user_profile.json
├── data/
│   ├── checkpoints.db
│   ├── long_term_memory.json
│   └── threads.json
├── src/
│   └── eval/          # Langfuse 辅助：tracing 单例、记忆利用率 judge
├── tools/
│   └── chat.py
└── tests/
```

几个关键文件：

- `configs/config.yaml`：模型、检索参数、数据路径
- `configs/user_profile.json`：用户画像
- `data/checkpoints.db`：短期记忆 checkpoint
- `data/long_term_memory.json`：长期记忆条目
- `data/threads.json`：本地记录过的 thread 列表
- `tools/chat.py`：CLI 入口（每轮用户消息会创建根 trace，退出前 flush）
- `src/eval/tracing.py`：`get_langfuse()` 等（测试与可选工具）
- `src/eval/judge.py`：对「回复是否用上检索记忆」的 LLM-as-judge
- `doc/eval_langfuse_plan.md`：Langfuse 接入与任务的实现说明（与代码可能略有 API 版本差异时以代码为准）

## 安装

建议先创建虚拟环境，再安装依赖。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你只是想运行聊天功能，核心依赖已经包含在 `requirements.txt` 中。

## 配置

### 1. 配置 OpenRouter API Key

在项目根目录创建 `.env` 文件：

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

也可以从示例文件复制：

```bash
cp .env.example .env
```

然后把 `your_openrouter_api_key` 改成你自己的 key。

### 2. Langfuse（可选）

不配 Langfuse 时，对话与测试照常运行；配置 `LANGFUSE_PUBLIC_KEY` 与 `LANGFUSE_SECRET_KEY` 后，运行 `tools/chat.py` 会把每轮用户输入到助手回复记为一条 trace，并在 Langfuse 控制台查看。

在 `.env` 中增加（也可参考根目录 `.env.example`）：

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

行为摘要：

- **Trace**：每轮用户消息对应根 span `conversation_turn`；`user_id` / `session_id`（thread）通过 SDK 的 `propagate_attributes` 写入；trace 级 input/output 为当轮用户原文与助手回复。
- **子 span**：检索 `retrieve`、`llm_node`、记忆更新、画像更新等节点上的 `@observe` 会挂在同一条 trace 下。
- **后台线程**：维护图在独立线程里执行；启动线程前会 `contextvars.copy_context()`，以便子 span 仍归属当前 trace。
- **评分**：记忆更新 / 画像解析会打 span 级分数；若有检索记忆且本轮有回复，后台会异步调用 judge，并写入 trace 级分数 `response.memory_utilization`（失败或未配置 key 时静默跳过）。
- **退出**：正常退出 CLI 时会 `flush`，尽量把未发送的观测数据送出。

更细的字段与任务拆解见 `doc/eval_langfuse_plan.md`。

### 3. 检查配置文件

默认配置在 [configs/config.yaml](configs/config.yaml)：

```yaml
llm:
  model: qwen/qwen3.6-plus
  base_url: https://openrouter.ai/api/v1

retrieval:
  embed_model: BAAI/bge-small-zh-v1.5
  rerank_model: BAAI/bge-reranker-base
  embed_topk: 20
  rerank_topk: 5

memory:
  namespace: long_term_memory
  db_path: data/checkpoints.db
  store_path: data/long_term_memory.json

profile:
  path: configs/user_profile.json
```

你通常只需要关心这些字段：

- `llm.model`：主对话模型
- `retrieval.embed_topk`：embedding 召回条数
- `retrieval.rerank_topk`：最终注入上下文的记忆条数
- `memory.db_path`：短期记忆 SQLite 路径
- `memory.store_path`：长期记忆 JSON 存储路径

### 4. 用户画像

默认画像文件是 [configs/user_profile.json](configs/user_profile.json)：

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

系统会在对话过程中自动更新这些字段。你也可以在启动前手动填写初始值。

## 启动

```bash
python3 tools/chat.py
```

启动后你会看到类似提示：

```text
多轮对话系统已启动（当前 thread：a1b2c3d4...）
输入 /help 查看可用命令。
```

输入普通文本即可开始对话。

## CLI 命令

支持以下命令：

- `/new`：新建一个 thread
- `/switch <id>`：切换到指定 thread
- `/threads`：列出本机记录过的 thread
- `/quit`：退出程序
- `/help`：显示帮助

示例：

```text
[a1b2c3d4] 你：我叫 Alice，是一名设计师
助手：你好 Alice，很高兴认识你。

[a1b2c3d4] 你：/new
已新建 thread：e5f6g7h8...

[e5f6g7h8] 你：/threads
  a1b2c3d4...
  e5f6g7h8... <-- 当前
```

## 数据持久化说明

系统运行后会在 `data/` 下生成或更新这些文件：

- `checkpoints.db`：按 `thread_id` 保存短期对话历史
- `long_term_memory.json`：保存长期记忆条目
- `threads.json`：保存你本机上用过的 thread 列表

这意味着：

- 重启程序后，之前的 thread 仍然可以通过 `/switch <id>` 切回
- 长期记忆不会因为进程退出而丢失
- 用户画像会保存在 `configs/user_profile.json`

## 典型使用流程

1. 第一次启动程序。
2. 告诉助手你的基本信息，例如姓名、职业、偏好。
3. 连续进行多轮对话，让系统逐步积累长期记忆和 profile。
4. 使用 `/new` 创建新会话，或用 `/switch <id>` 回到旧会话。
5. 重启程序后，用 `/threads` 查看已记录的 thread。

## 开发与测试

如果你想跑测试：

```bash
python3 -m pytest -q
```

Langfuse 相关逻辑有 `tests/test_tracing.py`、`tests/test_judge.py` 等；`tests/conftest.py` 会为测试环境设置空的 Langfuse key，避免依赖真实凭证。

如果本地缺少依赖，可以先重新安装：

```bash
pip install -r requirements.txt
```

## 常见问题

### 启动时报 `OPENROUTER_API_KEY` 相关错误

通常说明 `.env` 没有配置，或者没有成功加载。确认项目根目录存在 `.env`，并且包含：

```env
OPENROUTER_API_KEY=你的_key
```

### 第一次运行检索相关组件比较慢

如果本地需要下载或初始化 embedding / rerank 模型，首次启动可能会更慢一些。这通常是正常现象。

### 为什么换了 thread 后对话历史不一样？

短期记忆是按 `thread_id` 隔离的。不同 thread 有各自的对话上下文；但长期记忆和用户画像是跨 thread 共享的。

## 适合谁用

这个项目适合：

- 想体验“有记忆”的命令行助手
- 想学习 LangGraph 状态图式对话流程
- 想做长期记忆、profile、检索增强原型
- 想基于 OpenRouter 快速搭一个可持续对话的实验项目
