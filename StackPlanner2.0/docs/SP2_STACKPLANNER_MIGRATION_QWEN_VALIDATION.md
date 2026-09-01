# StackPlanner 1.0 → SP2.0 机制迁移与 Qwen 验证

验证日期：2026-07-26
本地模型：`Qwen3-32B`（vLLM，`http://127.0.0.1:8000/v1`）

## 审计结论

SP2 原有的并行委派、短期记忆栈、版本化 Artifact、失败元数据、回退/修订/压缩和工具隔离已经强于 1.0。尚未完整迁移的优势集中在报告之前的任务收敛与跨阶段交接，而不是报告 Markdown 模板本身。

| StackPlanner 1.0 优势 | 迁移前缺口 | SP2.0 本次实现 |
| --- | --- | --- |
| 结构化需求简报 | perception 只做薄层文件观察 | 生成目标、交付物、受众、范围、约束、验收标准、假设和关键缺口；最多返回一批 1–5 个正交问题 |
| 大纲确认与依赖拆解 | outline 只有通用“列大纲”提示 | 支持文档大纲或最小节点研究/执行 DAG，每个节点包含依赖、证据需求和可观察验收标准 |
| 人类反馈最高优先级 | 普通 specialist 只能看到被截断的 900 字记忆项 | 所有 specialist 获得独立、全局有界的 `mandatory_requirements.human_feedback` |
| 上阶段完整结果进入下阶段 | 除 Reporter 外主要只传 ref/摘要 | Outline 获得需求简报正文，Researcher 获得大纲正文，Reporter 获得完整有界证据包 |
| 报告修订闭环 | 模型必须把原因放进通用 metadata，容易漏填 | `revision_reason` 成为显式动作参数；传输边界可从 revision 动作的 reason 做确定性保底 |
| 最终产物验收 | 任意非中间 Artifact 可能让 FINISH 通过 | `final_artifact_ref` 与 `required_artifact_type` 同时校验，报告和生成文件不能互相冒充 |
| 阶段状态可见 | 中枢要从大量 refs 自行推断 | `workflow_status` 提供去重后的阶段产物计数、反馈数和 partial/blocked 数 |

没有照搬 1.0 的固定“感知→大纲→研究→报告”流水线。SP2 使用三种自适应 profile：简单请求直接回答，单一明确执行采用 bounded execution，模糊/高影响/多阶段/长报告才进入 deliberate workflow。中枢仍只调用动作；Tool 和 Skill 仍由被委派的子 Agent 执行。

## Qwen 兼容修复

当前 vLLM 在未启用匹配的自动工具解析器时，可能把 Qwen 的 `<tool_call>` JSON 放在 `reasoning` 或普通 `content` 中，导致 LangGraph 得到“空内容、无工具调用”。`VllmChatModel` 现在会把这些块恢复成标准 LangChain tool calls；工具调用的流式路径使用一次非流式 provider 请求再输出标准 chunk，避免文本工具调用在流中丢失。

此外，delegate 传输层会把 specialist 的记账阶段规范化：Perception→`perception`、Outline→`planning/revision`、Researcher→`research`、Reporter→`reporting/revision`。这只修正阶段标签和修订保底，不限制中枢选择哪一种动作。

## 真实 Qwen 行为结果

命令：

```bash
cd backend
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  uv run python scripts/evaluate_sp_qwen_actions.py --async-mode
```

结果：Gateway 同形的异步流式路径 5/5 通过；同步流式路径也为 5/5。

| 场景 | 有效动作 | 结果 |
| --- | --- | --- |
| 今日黄金价格 | `sp_delegate(researcher, stage=research)` | 通过 |
| 信息不足的行业报告 | `sp_delegate(perception, stage=perception)` | 通过 |
| 对现有报告提出结构反馈 | `sp_delegate(reporter, stage=revision, input_refs=[report-1])`，补全 revision lineage | 通过 |
| 搜索已连续失败 | `sp_ask_human`，没有再次研究或提前结束 | 通过 |
| 报告已完整生成 | `sp_finish(final_artifact_ref=report-1, required_artifact_type=report)` | 通过 |

思考模式额外验证：

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  uv run python scripts/evaluate_sp_qwen_actions.py --thinking \
  --scenario current_information_routes_to_research
```

结果：1/1 通过，Qwen 在 thinking 模式下仍生成标准 `researcher + research` 动作。

## 验证边界

- 这是对真实 Qwen 流式动作决策和 SP 传输归一化的在线验证，并由 handler/unit tests 覆盖后续执行约束。
- 行为测试不伪造搜索结果，也不把外部搜索服务可用性当作中枢策略正确性的前提。
- 模型输出存在非确定性，因此阶段标签、修订原因和最终产物类型仍有确定性代码保底；动作本身保持连续推理与离散控制并存。
