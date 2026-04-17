# 多轮对话技术档案（Memory 项目）

本文档用于说明本项目多轮对话系统中的关键技术模块，以及它们在运行流程中的定位与协作关系。重点覆盖：

- `sliding window`（短期上下文窗口）
- `summary`（历史增量摘要）
- `semantic memory`（语义记忆检索）
- `profile memory`（结构化用户画像）
- 前台快速回复与后台异步维护的职责分离

---

## 1. 总体设计目标

项目目标不是“每轮都把完整历史丢给模型”，而是通过多层记忆分工，在保证上下文质量的同时控制延迟与成本。

核心策略：

1. 前台只做“生成本轮回答”这件最关键的事。
2. 后台异步维护长期记忆、画像和历史摘要。
3. 短期上下文用窗口控制长度，远期上下文用摘要压缩。

---

## 2. 记忆分层与定位

### 2.1 Sliding Window（短期会话窗口）

- 定位：保留最近轮次的原始对话细节，提供高保真近场上下文。
- 作用位置：`llm_node` 组 prompt 前（[src/nodes.py](/Users/travishu/Documents/Projects/memory/src/nodes.py)）。
- 当前参数：`context.window_messages`（默认 `16`，见 [configs/config.yaml](/Users/travishu/Documents/Projects/memory/configs/config.yaml)）。
- 行为：只把最近 `window_messages` 条消息送给模型，而非 thread 全量历史。

适合承载的信息：

- 最近几轮明确指令
- 最新约束和修正
- 正在进行的局部任务细节

### 2.2 Summary（历史增量摘要）

- 定位：压缩窗口外的旧对话，保留 thread 级长期上下文。
- 作用位置：后台 `summary_update_node`（[src/nodes.py](/Users/travishu/Documents/Projects/memory/src/nodes.py)）。
- 状态字段：
  - `conversation_summary`：当前历史摘要文本
  - `summarized_message_count`：已经被摘要覆盖的消息计数（见 [src/state.py](/Users/travishu/Documents/Projects/memory/src/state.py)）
- 触发条件：
  - `unsummarized_messages > context.summary_trigger_messages`
  - 默认阈值 `24`
- 每次折叠批量：
  - `context.summary_batch_messages`，默认 `8`

关键特性：增量摘要，不重跑全量摘要。每次只处理“新溢出的旧消息片段”。

### 2.3 Semantic Memory（语义长期记忆）

- 定位：跨轮次、跨窗口保留“事实与偏好”，通过语义召回补充当前问题。
- 作用位置：`llm_node` 调用 `Retriever` 前半段流程（[src/retrieve.py](/Users/travishu/Documents/Projects/memory/src/retrieve.py)）。
- 机制：
  1. 对用户 query 与记忆做 embedding 相似度召回（`embed_topk`）
  2. 对候选做 rerank（`rerank_topk`）
  3. 将结果注入 system prompt 的“相关记忆”区块
- 数据落地：`data/long_term_memory.json`（[src/memory_store.py](/Users/travishu/Documents/Projects/memory/src/memory_store.py)）。

### 2.4 Profile Memory（结构化用户画像）

- 定位：持久化“用户是谁”的结构化背景，不走检索。
- 作用位置：
  - 前台：每轮在 system prompt 中整体注入 profile 快照
  - 后台：`profile_update_node` 判断并更新
- 数据落地：`configs/user_profile.json`

---

## 3. 端到端流程（一次用户输入）

### 3.1 前台（阻塞用户感知）

入口：`tools/chat.py` -> `build_response_graph()`（[src/graph.py](/Users/travishu/Documents/Projects/memory/src/graph.py)）

前台只运行一个节点：

1. `llm_node`
2. 构建 system prompt：
   - profile section
   - semantic memory section（检索结果）
   - summary section（历史摘要）
3. 拼接最近窗口消息（sliding window）
4. 调用主模型生成回复
5. 立即返回给用户

这条链路保证“先响应，后维护”。

### 3.2 后台（异步维护）

在用户看到回复后，后台线程串行执行：

1. `memory_update_node`：对长期记忆执行 `add/append/delete/noop`
2. `profile_update_node`：更新结构化画像
3. `summary_update_node`：超过阈值时增量摘要旧消息

入口：`run_background_updates()`（[tools/chat.py](/Users/travishu/Documents/Projects/memory/tools/chat.py)）

---

## 4. Prompt 拼装结构

当前主模型输入可抽象为：

1. `system`：角色指令 + profile + semantic memory + summary
2. 最近 `N` 条对话消息（`sliding window`）

其中：

- `N = context.window_messages`
- `summary` 来自 `conversation_summary`
- `semantic memory` 来自 `Retriever.retrieve(...)`

这意味着模型看到的是“近场原文 + 远场压缩 + 事实记忆 + 画像背景”。

---

## 5. 为什么这四类机制要同时存在

仅用一种机制会出现明显缺陷：

- 只用全量 history：延迟和 token 成本随轮次线性增长。
- 只用 sliding window：旧上下文容易遗失。
- 只用 summary：近期细节不够精确。
- 只用 semantic memory：缺少 thread 级任务状态连续性。

组合后：

- `sliding window` 负责“近期精确细节”
- `summary` 负责“历史语义连续性”
- `semantic memory` 负责“跨轮事实与偏好召回”
- `profile` 负责“稳定用户背景”

---

## 6. 参数与调优建议

当前默认值：

- `window_messages: 16`
- `summary_trigger_messages: 24`
- `summary_batch_messages: 8`
- `embed_topk: 20`
- `rerank_topk: 5`

调优方向：

1. 如果响应仍偏慢：优先下调 `window_messages`。
2. 如果模型“忘记较早决策”：适度上调 `summary_trigger_messages` 或优化摘要提示词。
3. 如果记忆注入噪声大：下调 `embed_topk` 或 `rerank_topk`。
4. 如果摘要过频繁：增大 `summary_batch_messages` 或提升触发阈值。

---

## 7. 可观测性（日志）

项目日志记录到 `log/session_YYYYMMDD_HHMMSS.log`，包含：

- 前台响应耗时 `response_elapsed_ms`
- 检索耗时 `retrieve_ms`
- 节点耗时 `total_elapsed_ms`
- 摘要节点推进信息：
  - `summarized_count old -> new`
  - `summarize_count`
  - `summary_chars`

这些指标可直接用于判断瓶颈在：

- 主模型回答
- 检索阶段
- 后台维护（记忆/画像/摘要）

---

## 8. 已知边界

1. `summary_update_node` 当前更新摘要与计数，不物理删除 checkpoint 中旧消息；前台依靠窗口裁剪控制输入长度。
2. 后台线程与用户连续高速输入时，存在轻微并发时序差异，但不影响主回复链路可用性。
3. 摘要质量取决于模型与提示词，建议结合日志周期性抽检摘要内容。

---

## 9. 代码映射索引

- 状态定义：[src/state.py](/Users/travishu/Documents/Projects/memory/src/state.py)
- Prompt 模板：[src/prompt.py](/Users/travishu/Documents/Projects/memory/src/prompt.py)
- 核心节点：[src/nodes.py](/Users/travishu/Documents/Projects/memory/src/nodes.py)
- 图编排：[src/graph.py](/Users/travishu/Documents/Projects/memory/src/graph.py)
- 语义检索：[src/retrieve.py](/Users/travishu/Documents/Projects/memory/src/retrieve.py)
- 长期记忆存储：[src/memory_store.py](/Users/travishu/Documents/Projects/memory/src/memory_store.py)
- CLI 与异步后台：[tools/chat.py](/Users/travishu/Documents/Projects/memory/tools/chat.py)

