# SP2.0 迁移框架组会 PPT 提纲

参考 `SP-0617技术汇报-v4.pdf` 的技术汇报语言与版式组织。建议 13 页、16:9，采用“问题 - 定义 - 现状 - 具体设计 - 优势”的叙事方式。

## 整体视觉与语言

- 白色背景，深红色页标题，右上角放单位标识或 SP2.0 Logo。
- 每页顶部给出一个核心问题，正文中用红色强调关键词。
- 结论或优势使用页底浅蓝色横条，不使用宣传式口号。
- 框架页优先使用流程图，机制页优先使用对比表或分区图。
- 页脚统一显示页号，如 `< 6 >`。

## P01｜Stack Planner 2.0 框架迁移技术汇报

**标题**

Stack Planner 2.0

**副标题**

基于 DeerFlow 2.0 的 Multi-Agent Runtime 迁移

**内容**

- 汇报人、团队、日期。
- 一句话说明工作：在保留 SP 中枢规划语义的基础上，引入 DeerFlow 2.0 的 Runtime、Memory、Artifact 与 Subagent 能力。

**布局**

- 标题居中偏左，SP2.0 Logo 放在右侧。
- 下方放一条简化架构链：`SP Planning → DR2 Runtime → Long-horizon Task`。

## P02｜背景：为什么需要更新 Stack Planner

**核心问题**

SP1.0 已经具备中枢调度、记忆栈、子 Agent、反思和总结能力，为什么还需要更新框架？

**SP1.0 已有基础**

- 中枢 Agent 负责多步推理与任务调度。
- 通过 MemoryStack 管理推理过程。
- 通过子 Agent 完成检索、任务拆解和报告生成。
- 具备 Think、Reflect、Summarize 和 Backtrack 等控制语义。

**存在的问题**

1. Agent、动作、任务分发和记忆管理仍有较强耦合。
2. 长任务中上下文、中间结果和最终产物边界不够清晰。
3. 子 Agent 调用、失败处理和结果回传缺少统一协议。
4. 缺乏完整的中断恢复、运行记录和个性化能力。

**页底结论条**

> 本次迁移不是重新设计 SP，而是为 SP 的规划能力补充更成熟的运行时基础。

**布局**

- 左侧放“SP1.0 已有能力”，右侧放“长任务中的问题”。
- 问题编号使用红色，结论使用浅蓝色横条。

## P03｜为什么选择 DeerFlow 2.0

**DeerFlow 2.0 简介**

DeerFlow 2.0 是面向长时序任务的 Agent Runtime，提供从任务入口、状态管理到子 Agent、文件和长期记忆的完整运行能力。

**与 SP 的需求对应关系**

| SP 长任务需求      | DeerFlow 2.0 能力          |
| ------------------ | -------------------------- |
| 统一任务入口       | Gateway                    |
| 状态持久化与恢复   | ThreadState + Checkpointer |
| 中间结果与文件管理 | Workspace + Artifact       |
| 子 Agent 调度      | SubagentExecutor           |
| 执行过程追踪       | Run/Event                  |
| 跨任务经验复用     | Memory                     |
| 个性化与能力扩展   | SOUL + Skills              |

**我们的选择**

- 复用 DeerFlow 2.0 Runtime，不新建 Gateway、Checkpoint、Workspace、Event Store 或 Memory Store。
- 将 CentralAgent、ActionLoop、TaskMemoryStack 和 Handler 作为 SP Extension 接入。

**页底结论条**

> DR2 提供运行时基础，SP 保留规划调度与记忆控制能力。

**布局**

- 上半部分放需求映射表。
- 下半部分用两层结构展示 `DR2 Runtime + SP Extension`。

## P04｜SP2.0 整体框架

**核心设计**

中枢 Agent 只决定下一步行动，不直接调用搜索、文件、Shell 或 Memory 工具。

**主流程**

```text
用户请求
   ↓
Gateway / ThreadState / Checkpointer
   ↓
prepare_context → central_decide → validate_action
   ↓
Action(JSON) → ActionRouter → Handler
   ↓
SubagentExecutor / Memory / Workspace / Artifact
   ↓
route_next → 下一步动作 / HITL / FINISH
```

**动作空间**

`THINK / DELEGATE / RECALL_MEMORY / REFLECT / BACKTRACK / REPLAN / SUMMARIZE / ASK_HUMAN / FINISH`

**关键约束**

- 每个 Action 包含 `action_id`、`idempotency_key` 和 schema 校验。
- 每个 Handler 统一返回 state、memory、artifact、event 和 error。
- 每一步均写入既有 Run/Event，支持追踪与复现。

**布局**

- 中央放纵向流程图。
- 右侧放动作空间，左下角放“CentralAgent 不直接调用工具”的红色提示框。

## P05｜SP2.0 三层记忆结构

**记忆分层的目的**

不同生命周期的信息应该采用不同的管理机制，避免把所有内容都放入中枢 Agent 上下文。

| 层级     | 管理对象                           | 生命周期        | SP2.0 实现                             |
| -------- | ---------------------------------- | --------------- | -------------------------------------- |
| 短期记忆 | 当前任务中的决策、观察、反馈       | 当前 Thread/Run | TaskMemoryStack + PromptContextBuilder |
| 中期记忆 | 子任务状态、中间产物、版本与恢复点 | 完整任务过程    | Delegation + Artifact + Checkpointer   |
| 长期记忆 | 跨任务经验、用户偏好与个性化能力   | 跨 Thread       | DR2 Memory + SOUL + Skills             |

**概念边界**

- 短期记忆用于中枢 Agent 当前决策。
- 中期记忆是 SP2.0 的任务级信息组织方式，不新增独立存储系统。
- SOUL 和 Skills 不属于 Memory Store 中的记忆条目，但与长期 Memory 一起构成跨任务个性化能力。

**布局**

- 使用三层阶梯图：短期记忆在上，中期记忆居中，长期记忆在下。
- 每层只保留“对象、机制、生命周期”三个标签。

## P06｜核心问题一：短期记忆管理

**定义**

短期记忆是中枢 Agent 在执行当前任务时，用于决定下一步 Action 的任务控制信息。

**SP1.0 的实现**

- MemoryStack 主要记录时间、动作、Agent、内容和执行结果。
- 主要通过 push、pop、recent history 和完整历史进行管理。
- 超出上限时按时间删除，缺少优先级和状态语义。
- 子 Agent 大结果可能随 result 进入中枢上下文。

**SP2.0 的优化**

1. **结构化记忆**：增加 thread、run、stage、priority、status、result ref、parent 和 failure 信息。
2. **语义化管理**：支持 append、summarize、condense、prune 和 backtrack。
3. **反馈优先**：Human Feedback 固定为 `critical + pinned`，不被普通记忆覆盖。
4. **多层限长**：同时限制活跃条目、字符数、单条内容和最终 Prompt 大小。
5. **运行时恢复**：由 TaskMemoryMiddleware 从 ThreadState 自动恢复，并兼容 SP1.0 MemoryStack。

**页底优势条**

> 优势：中枢 Agent 只看到与当前决策相关的高价值信息，长任务中的上下文更稳定、更可控。

**布局**

- 左侧放 SP1.0 简单栈，右侧放 SP2.0 带 priority/status 的结构化栈。
- 页底使用浅蓝色优势横条。

## P07｜从 SP1.0 MemoryStack 到 SP2.0 中间件/上下文处理链

**为什么需要中间件链**

SP1.0 中，CentralAgent 同时负责读取 MemoryStack、执行 push/pop、拼接 Prompt 和决定下一步动作。随着任务变长，记忆管理逻辑会与规划逻辑逐渐耦合。

SP2.0 保留 MemoryStack 的核心思想，但把记忆处理拆分为一条独立的任务上下文链路。其中，ThreadState 和 Checkpointer 属于 DR2 Runtime；TaskMemoryMiddleware、PromptContextBuilder 和 Handler 属于 SP Extension。

**CentralAgent 决策前**

```text
DR2 Checkpointer
  → 恢复 ThreadState
  → TaskMemoryMiddleware
      ① 兼容并迁移 SP1.0 memory_stack
      ② 补充 thread_id / run_id
      ③ 按条目数和字符数 prune
  → prepare_context 反馈处理
      ④ 用户反馈写为 critical + pinned
  → PromptContextBuilder
      ⑤ critical feedback 优先
      ⑥ 加入当前 stage、delegate 和 artifact refs
      ⑦ 压缩为有界上下文
  → CentralAgent 决定下一步 Action
```

**CentralAgent 决策后**

```text
Action
  → ActionRouter / Handler
  → 追加 Think、Observe、Reflect、Replan 等记忆
  → 更新 ThreadState 与 Artifact refs
  → 写入 Run/Event
  → DR2 Checkpointer 持久化
```

**SP1.0 与 SP2.0 的职责变化**

| 记忆操作   | SP1.0                         | SP2.0                                       |
| ---------- | ----------------------------- | ------------------------------------------- |
| 恢复记忆   | CentralAgent/业务代码手工加载 | Checkpointer + TaskMemoryMiddleware         |
| 选择上下文 | recent 或 full history        | priority + status + stage + bounded context |
| 修改记忆   | CentralAgent 直接 push/pop    | Handler 产生语义化 memory entries           |
| 保存状态   | 业务状态手工序列化            | ThreadState 统一 checkpoint                 |

**页底优势条**

> 优势：记忆栈仍然存在，但 CentralAgent 不再管理存储细节，只消费整理后的决策上下文。

**讲解口径**

- 当前 SP Graph 在 `prepare_context` 节点显式调用 TaskMemoryMiddleware，并在 `central_decide` 前使用 PromptContextBuilder。
- 这里的“链”表示职责处理顺序，不代表新建了第二套 Middleware Runtime。

**布局**

- 左侧放“决策前”链路，右侧放“决策后”链路。
- 页底放 SP1.0/SP2.0 对比表，突出 CentralAgent 职责的减少。

## P08｜核心问题二：中期记忆管理

**定义**

中期记忆是一个任务从规划到完成过程中，需要跨多个 Action 持续保留的执行信息，包括子任务状态、中间产物、版本关系和恢复位置。

**SP2.0 的四项主要优势**

### 1. 更完善的中枢 Agent 调用子 Agent 机制

- 中枢 Agent 只生成 `DELEGATE Action`，不直接调用工具。
- Router 与 DelegateHandler 完成任务分发。
- SubagentExecutor 提供隔离的执行上下文、工具和状态。
- 子 Agent 结果统一归一化为 summary、artifact、status 和 error。

### 2. 更完善的中间产物管理机制

- 报告、提纲、研究结果和生成文件进入 Workspace/Artifact。
- ThreadState 只保存 Artifact refs，不保存大文本正文。
- 支持 version、parent、source、stage 和 feedback lineage。

### 3. 具备中断恢复能力

- ThreadState 与 Checkpointer 保存当前 Action、任务阶段和 pending HITL。
- `action_id + idempotency_key` 防止恢复后重复副作用。
- 服务中断或 ASK_HUMAN 后可以从已有状态继续执行。

### 4. 功能更丰富

- 预设 researcher、coder、reporter、outline 和 perception 子 Agent。
- perception 提供图片等多模态信息理解入口。
- memory_recaller 作为内部子 Agent 负责长期记忆召回。

**布局**

- 使用 2×2 分区图展示四项优势。
- 每个分区只放“机制图 + 3 个关键词”，详细解释由汇报人口述。

## P09｜中期记忆：消息传递与产物恢复流程

**中枢 Agent 调用子 Agent**

```text
CentralAgent
  → DELEGATE Action
  → ActionRouter
  → DelegateHandler
  → SubagentExecutor
  → 专业 Subagent
  → Tool / Sandbox
```

**子 Agent 返回结果**

```text
完整结果
  ├─ compact summary → TaskMemoryStack
  ├─ large content   → Workspace / Artifact
  ├─ artifact refs   → ThreadState
  └─ status/error    → Run/Event
```

**中断恢复流程**

```text
ASK_HUMAN / 服务中断
  → Checkpoint
  → 用户反馈或服务恢复
  → restore ThreadState
  → 幂等校验
  → 从下一 Action 继续
```

**页底优势条**

> 优势：任务描述、执行结果、大文本产物和恢复状态分别管理，避免中枢上下文与业务产物相互污染。

**布局**

- 左侧放调用流程，右上放结果分流，右下放中断恢复流程。
- 使用相同颜色标识 summary、Artifact、state 和 event。

## P10｜核心问题三：长期记忆与个性化

**定义**

长期层负责跨任务复用稳定信息，并让不同用户或不同 Agent 形成持续一致的行为与专业能力。

| 组成   | 解决的问题                           | SP2.0 能力                           |
| ------ | ------------------------------------ | ------------------------------------ |
| Memory | “过去发生过什么、用户有哪些稳定偏好” | 跨 Thread 召回、来源归一化、候选提取 |
| SOUL   | “这个 Agent 是谁、应当如何行为”      | 人格、价值、角色和行为边界持久化     |
| Skills | “这个 Agent 会怎样完成某类任务”      | 工作流、最佳实践、领域知识和工具配置 |

**Memory 召回流程**

```text
RECALL_MEMORY
  → memory_recaller
  → DR2 Memory
  → normalize
  → TaskMemoryStack
```

**个性化优势**

- SOUL 以用户和自定义 Agent 为边界持久化，可维护稳定角色与行为方式。
- Skills 支持 public/custom、按需加载和 Agent 级白名单。
- 用户可以组合不同 SOUL 与 Skills，构造研究、医疗、数据分析等专业 Agent。
- 长期 Memory 写入默认 dry-run，避免未经确认的内容自动固化。

**当前落地边界**

- DR2 Memory snapshot 已作为只读决策上下文进入 SP CentralAgent；显式 RECALL 结果会以摘要进入 TaskMemoryStack。
- SOUL 已进入中枢系统约束，Skills 由中枢选择、执行器校验、子 Agent 按需加载。
- 长期 Memory 写入仍保持 dry-run，避免未经确认的信息自动固化。

**页底优势条**

> 优势：SP2.0 不仅能记住任务历史，还具备形成稳定人格和复用专业能力的基础。

**布局**

- 使用 Memory、SOUL、Skills 三列结构。
- 每列放一个问题句和三个能力关键词。

## P11｜SP2.0 相比 SP1.0 的提升

**对比表**

| 维度          | SP1.0                         | SP2.0                                       |
| ------------- | ----------------------------- | ------------------------------------------- |
| 短期记忆      | 简单 push/pop，主要按时间管理 | 结构化、分优先级、可压缩、受限长            |
| 人类反馈      | 容易被后续信息稀释            | `critical + pinned`，优先级最高             |
| 子 Agent 调用 | 调用与业务逻辑耦合            | Action → Router → Handler → Executor        |
| 中间产物      | 结果与状态边界不清晰          | Workspace/Artifact + refs + version lineage |
| 中断恢复      | 依赖重新执行或人工恢复        | Checkpoint + idempotency + HITL restore     |
| 预设能力      | 以检索和报告为主              | research/coding/report/outline/perception   |
| 长期记忆      | 跨任务个性化能力有限          | DR2 Memory + SOUL + Skills                  |
| 可观测性      | 难以定位每一步决策            | Action ID + Run/Event + Artifact lineage    |

**页底结论条**

> SP2.0 的提升主要体现在记忆分层、执行边界、任务恢复和个性化能力，而不是简单增加更多工具。

**布局**

- 全页对比表，SP2.0 列使用浅蓝色底色。
- 重点行使用红色关键词，不增加额外大段说明。

## P12｜演示与 Case 测试

**演示任务**

以一个需要多轮检索、报告生成和人工修改的研究任务为例。

**执行流程**

1. CentralAgent 调用 RECALL_MEMORY，并生成任务规划。
2. Outline 与 Researcher 分别完成提纲和资料收集。
3. Researcher 超时后，CentralAgent 执行 REFLECT 与 REPLAN。
4. Reporter 生成报告 v1，并触发 ASK_HUMAN。
5. 用户反馈以 `critical + pinned` 写入短期记忆。
6. Reporter 基于反馈生成报告 v2，CentralAgent FINISH。

**测试结果**

- 长任务回归覆盖 2 个 run、12 个 action 和报告 v1/v2。
- 验证短期记忆限长、反馈优先级和 checkpoint restore。
- 验证子 Agent 结果归一化、Artifact lineage 和大文本不进入 ThreadState。
- SP 测试组 `120 passed`；DR2 指定回归 `321 passed`。
- 前端 unit `588 passed`，format、lint、typecheck 和 production build 已通过。
- 真实 Qwen3-32B 已完成一次 SP CentralAgent smoke test。

**暂未完成的验证**

- 真实模型下 15 至 30 分钟长任务的质量与 token 成本。
- 登录后的 browser/thread/Artifact/HITL 端到端流程。
- 视觉模型下的 perception 多模态质量和生产跨进程恢复。

**布局**

- 上半部分放 6 步执行时间线。
- 下半部分左侧放测试结果，右侧放“已验证/待验证”两栏。

## P13｜SP2.0 在后续研究中的应用

**研究方向一：中枢 Agent 能力提升**

- Action 选择、任务分发、反思与重规划能力训练。
- 研究 Action-level Reward 与子 Agent 调用 Reward。

**研究方向二：分层记忆机制**

- 短期、中期、长期记忆的独立消融实验。
- 研究 condense/prune 对任务质量和 token 成本的影响。

**研究方向三：个性化 Agent**

- 基于 Memory + SOUL + Skills 构建用户级专业 Agent。
- 研究用户反馈如何影响长期偏好、角色和能力组合。

**研究方向四：Multi-Agent 与多模态**

- 研究不同任务分发策略、Agent 组合和协作成本。
- 引入图片、文档和结构化数据等多模态任务。

**评估指标**

`任务完成质量 / Token 成本 / 恢复成功率 / 人工介入次数 / 产物可追踪性 / 个性化一致性`

**页底结论条**

> SP2.0 为中枢规划、分层记忆、个性化和多模态 Multi-Agent 研究提供了统一实验平台。

**布局**

- 使用四象限展示四类研究方向。
- 页底横向放置六个评估指标。
