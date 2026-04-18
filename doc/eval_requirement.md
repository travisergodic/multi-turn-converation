# 评测体系需求文档

> 基于 Langfuse Cloud，面向本项目 agentic memory 系统的评测设计。

---

## 1. 评测目标

本系统有三条核心链路，每条都可能出错，需要分别追踪：

| 链路 | 可能出错的地方 |
|------|--------------|
| 检索（retrieve） | 没找到相关记忆，或召回了不相关的 |
| 记忆更新（memory_update） | 该记的没记，不该记的记了，或操作类型判断错 |
| 对话回复（llm_node） | 忽略了已检索的记忆，或 profile 信息没用上 |

---

## 2. Langfuse 数据模型映射

```
每轮对话 → 1 个 Trace
  ├── Span: retrieve         （embedding + rerank）
  ├── Span: llm_node         （system prompt 构建 + LLM call）
  ├── Span: memory_update    （LLM 判断操作 + 执行）
  └── Span: profile_update   （LLM 判断 + 写文件）
```

每个 Span 挂载 input/output，Score 则在 Trace 层或 Span 层打分。

---

## 3. 追踪指标

### 3.1 检索质量

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| `retrieval.recall_at_k` | 相关记忆是否被召回 | 需要构建带标注的 golden 测试集，比对 retrieved IDs |
| `retrieval.precision_at_k` | 召回的记忆中有多少是真正相关的 | 同上 |
| `retrieval.candidate_count` | embed 阶段候选数量 | 直接记录 |
| `retrieval.final_count` | rerank 后最终数量 | 直接记录 |
| `retrieval.latency_ms` | 检索耗时 | Span duration |

> **为什么重要**：检索是 llm_node 的上游，检索错了后面全错。

### 3.2 记忆更新质量

| 指标 | 含义 | 评测方式 |
|------|------|---------|
| `memory_update.action_distribution` | add/append/delete/noop 各占比 | 统计，用于发现异常（如 noop 占 99% 说明没在学习） |
| `memory_update.parse_success_rate` | LLM 输出合法 JSON 的比例 | 直接检测 |
| `memory_update.relevance_score` | 新增/追加的内容是否与对话相关 | LLM-as-judge |
| `memory_update.redundancy_score` | 新增内容是否与已有记忆重复 | LLM-as-judge（输入：新内容 + 现有记忆列表） |
| `memory_update.latency_ms` | 节点耗时 | Span duration |

### 3.3 回复质量

| 指标 | 含义 | 评测方式 |
|------|------|---------|
| `response.memory_utilization` | 回复中是否实际用到了检索到的记忆 | LLM-as-judge（输入：retrieved_memories + response） |
| `response.profile_utilization` | 回复中是否用到了 profile 信息（如称呼用户名字） | LLM-as-judge |
| `response.coherence` | 跨轮对话是否连贯一致 | LLM-as-judge，需要多轮 trace 联合评测 |
| `response.latency_ms` | llm_node 耗时 | Span duration |

### 3.4 Profile 更新质量

| 指标 | 含义 | 评测方式 |
|------|------|---------|
| `profile_update.update_rate` | 每轮更新 profile 的频率 | 统计（过高 = 过度更新，过低 = 没在学习） |
| `profile_update.parse_success_rate` | LLM 输出合法 JSON 的比例 | 直接检测 |
| `profile_update.field_accuracy` | 更新的字段是否正确（需 golden 测试集） | 对比期望值 |

### 3.5 系统级指标

| 指标 | 含义 |
|------|------|
| `turn.total_latency_ms` | 每轮总耗时 |
| `turn.llm_call_count` | 每轮 LLM 调用次数（当前固定 3 次） |
| `memory_store.size` | 当前用户记忆条目数量（追踪记忆膨胀） |

---

## 4. 评测策略

### 4.1 在线监控（生产 trace）

所有正式对话都上报 Langfuse。关注：
- `memory_update.parse_success_rate` < 0.95 → prompt 有问题
- `memory_update.action_distribution` 中 noop 占比 > 80% → 系统没在积累记忆
- `turn.total_latency_ms` 异常升高 → rerank 模型或 LLM 响应慢

### 4.2 离线回归（Dataset 测试）

在 Langfuse 中创建 Dataset，手工构造典型对话场景，定义期望输出：

| Dataset 名称 | 场景 |
|------------|------|
| `retrieval_golden` | 多条记忆 + 特定 query，标注哪些应该被召回 |
| `memory_update_golden` | 给定对话，标注正确的 add/append/delete/noop 操作 |
| `profile_update_golden` | 给定对话，标注 profile 哪些字段应被更新及其值 |
| `coherence_golden` | 多轮对话序列，验证回复的连贯性 |

每次修改 prompt 或检索参数后，跑一遍 Dataset，对比 Scores 不能下降。

### 4.3 LLM-as-judge 设计原则

- **输入要明确**：把 retrieved_memories、response、profile 都塞进 prompt，不让 judge 凭空猜
- **输出要结构化**：打分 0-1，并要求给出 1 句理由
- **judge 模型**：用 claude-3.5-sonnet（与主模型不同，避免自评偏差）

---

## 5. Langfuse 实现要点

### Trace 结构示例

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()

@observe(name="conversation_turn")
def run_turn(user_input, thread_id, user_id):
    langfuse_context.update_current_trace(
        user_id=user_id,
        session_id=thread_id,
        metadata={"turn_index": ...}
    )
    # ... 调用 graph.invoke(...)
```

每个节点用 `@observe` 单独包裹，自动产生子 Span。

### Score 上报时机

| Score | 上报时机 |
|-------|---------|
| `parse_success_rate` 类 | 节点执行完立即上报（deterministic） |
| `memory_utilization` 类 | 回复生成后，异步调用 LLM-as-judge 上报 |
| `retrieval.recall` 类 | 仅在 Dataset 跑 eval 时上报，不在在线链路上报 |

---

## 6. 优先级

| 优先级 | 指标 | 理由 |
|--------|------|------|
| P0 | `memory_update.parse_success_rate` | 解析失败 = 系统静默出错，用户无感知 |
| P0 | `memory_update.action_distribution` | 最直观反映记忆系统是否在工作 |
| P1 | `response.memory_utilization` | 核心价值：记忆有没有被用上 |
| P1 | `turn.total_latency_ms` | 用户体验直接相关 |
| P2 | `retrieval.recall_at_k` | 需要 golden 数据集，成本高，后期补 |
| P2 | `response.coherence` | 多轮联合评测复杂，后期补 |
