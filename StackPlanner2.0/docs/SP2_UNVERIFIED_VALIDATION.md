# SP2.0 暂未完成验证项

> 更新日期：2026-07-11
> 本地实现基线：`codex/sp-memory-migration` 本轮审计提交
> 完整迁移审计与演示 Case：`docs/SP2_MIGRATION_FINAL_AUDIT_AND_DEMO_CASE.md`

## 当前结论

- SP Action Loop、三层记忆、Subagent、Artifact、HITL、SOUL、Skills 和 Run/Event 已有单元与回归覆盖。
- Gateway 可启动，`/health` 返回 200。
- `http://123.59.6.244:8000/v1` 可访问，真实 Qwen3-32B 已完成一次 SP CentralAgent smoke test。
- 本地账号认证、工作区加载和 SP Runtime 标识已完成浏览器验证。
- 当前仍未完成的是认证状态下的真实 Artifact/HITL 全流程、真实外部工具长任务、多模态和生产跨进程恢复。
- 长期 Memory 写入默认 dry-run 是设计约束，不属于验证失败。
- 真实 Qwen 调用的主要剩余问题是远端响应延迟，不是 SP Action Loop 的本地计算耗时。

## 部署前安全检查

上线服务器前逐项确认：

- [ ] 保持鉴权开启；不要设置 `DEER_FLOW_AUTH_DISABLED=1`。
- [ ] 设置稳定且高强度的 `AUTH_JWT_SECRET`，多进程和重启期间保持不变。
- [ ] 生产环境设置 `GATEWAY_ENABLE_DOCS=false`，避免暴露 Swagger/ReDoc。
- [ ] `GATEWAY_CORS_ORIGINS` 只填写实际前端 Origin，不使用 `*`。
- [ ] 前端通过 HTTPS 访问，Gateway 只对内网或反向代理开放。
- [ ] `DEER_FLOW_INTERNAL_GATEWAY_BASE_URL` 指向内网 Gateway，不把内部地址暴露给浏览器。
- [ ] 生产启动不要使用 `--reload`；开发入口脚本只用于本地调试。
- [ ] 模型 API Key 只放在服务器环境变量或密钥服务中，不提交 Git，不写入前端。
- [ ] 已在聊天记录中暴露过的模型 API Key 应在部署前轮换。
- [ ] 反向代理限制请求体大小、连接超时和并发，日志中不要记录 Authorization、Cookie 或 API Key。

本轮修复还增加了 Artifact 下载链接的协议校验、认证下载、路径归属校验和安全响应头；这些是代码防线，不能替代服务器侧的 HTTPS、鉴权和网络隔离。

## 1. GitHub Lint 历史失败

[Lint Check #29086580243](https://github.com/chichuxwx/StackPlanner2/actions/runs/29086580243) 的失败点是 5 个 TSX 文件未通过 Prettier。

本轮已经格式化对应文件，并在本地通过：

- Prettier。
- ESLint。
- TypeScript。
- Frontend unit：`593 passed`。
- Next.js production build。

当前修改已推送到 `origin/codex/sp-memory-migration`。仓库工作流只在 Pull Request 上触发，原 PR #2 已关闭，因此本轮新 SHA 没有自动创建 Actions run。

仍需完成：

- [ ] 为当前迁移分支创建新 PR，触发 GitHub Actions。
- [ ] 确认 CI 使用仓库声明的 `pnpm 10.26.2` 后仍通过。

本机 `pnpm 10.11.0` 自动下载 10.26.2 时受 npm 证书链影响；本地检查使用已安装在 `node_modules/.bin` 的锁定工具完成。

## 2. Authenticated API 与浏览器 E2E

本地管理员账号已经创建，本轮已验证：

- `/health`：200。
- 本地账号登录：200。
- 登录后 `/api/v1/auth/me`、thread search、models 和 skills：200。
- 新对话页显示 `SP Runtime`，DOM 明确携带 `data-assistant-id="stackplanner"`。
- 前端 E2E 已断言 `/runs/stream` 请求体使用 `assistant_id=stackplanner`。
- 浏览器控制台未再出现 Dialog description、通知权限和 controlled/uncontrolled 警告。

仍需完成：

- [ ] 登录后完成真实 Artifact 打开和 HITL resume 全流程。
- [ ] 验证桌面端与移动端关键页面。

## 3. 真实 Subagent 与外部服务

协议和 fake executor 已覆盖，以下组合仍待真实服务验证：

- [ ] Researcher 使用真实搜索服务并生成带引用的 research Artifact。
- [ ] Coder 在真实 Workspace/Sandbox 中修改文件并运行测试。
- [ ] Reporter 基于多个 Artifact 生成 report v1/v2。
- [ ] Memory recaller 从已写入的真实 DR2 Memory 召回偏好。
- [ ] Skill 白名单、用户级 custom Skill 和远程 Sandbox 组合验证。

## 4. 多模态 Perception

当前 `Qwen3-32B` 配置为 `supports_vision: false`，因此只能验证 perception 的注册、路由、上下文和 Artifact 协议，不能证明真实图片理解质量。

仍需完成：

- [ ] 配置一个 `supports_vision: true` 的模型。
- [ ] 用 PDF 页面截图、架构图和表格图片验证 perception observation。
- [ ] 检查图片正文不进入 ThreadState。

## 5. 真实长任务质量

确定性长任务回归已覆盖 recall、委派失败、reflect、replan、artifact v1/v2、HITL 和 finish。模型质量仍需实测：

- [ ] 至少运行 3 个 15 至 30 分钟 Case。
- [ ] 记录完成率、重复 Action、人工介入、token、耗时和 Artifact 质量。
- [ ] 验证 Summary/Condense/Prune 对质量与成本的影响。
- [ ] 将失败 Case 固化为可重复测试。

## 6. 持久化与生产恢复

- [ ] 使用目标生产数据库做跨进程 checkpoint restore。
- [ ] Gateway 重启后恢复 pending HITL，且不重复副作用。
- [ ] 多用户、多线程并发下验证 Artifact、Memory、SOUL 和 custom Skill 隔离。
- [ ] 在目标容器、反向代理、鉴权和日志环境中完成 smoke test。

## 7. 长期 Memory 写入边界

当前已实现：

- 自动只读 Memory snapshot。
- `RECALL_MEMORY` 主动召回并写入 TaskMemoryStack 摘要。
- CandidateExtractor、Judge 和 dry-run event。

当前刻意不做：

- 自动写入真实长期 Memory。
- 自动把单次成功流程升级为 Skill。

未来若启用，需要：

- [ ] 明确人工审批或策略门控。
- [ ] 通过 DR2 MemoryUpdater 写入，不新增 Memory Store。
- [ ] 验证去重、撤销、来源追踪、用户隔离和隐私策略。

## 已有验证证据

测试组存在重叠，数字不能直接相加。

| 范围                           | 结果                                                            |
| ------------------------------ | --------------------------------------------------------------- |
| SP/Artifact/Path/RunJournal 回归 | `218 passed`                                                    |
| CentralAgent 动作与长任务矩阵    | 已包含在上项，不能重复相加                                      |
| DR2 指定回归                   | `321 passed`，1 warning                                         |
| Frontend unit                  | `593 passed`                                                    |
| Prettier / ESLint / TypeScript | 通过                                                            |
| Next production build          | 通过，1 Turbopack warning                                       |
| Gateway health                 | 200                                                             |
| Authenticated browser smoke    | 登录成功，`SP Runtime` 可见，控制台 0 warning/error             |
| 真实 Qwen3-32B SP smoke        | run `ffa32e98...`，1 次 LLM、1 Action FINISH、2602 tokens，成功 |

## 如何确认运行的是 SP2，而不是仅替换界面

按可信度从界面到后端检查：

1. 聊天页显示 `SP Runtime`，元素的 `data-assistant-id` 为 `stackplanner`。
2. `/runs/stream` 请求体中的 `assistant_id` 为 `stackplanner`。
3. `runs` 表记录 `assistant_id=stackplanner`。
4. 同一 run 存在 `sp.action.created`、`sp.central.decided`、`sp.handler.*`、`sp.loop.*` 事件。
5. 最终 checkpoint/state 包含 `sp_task_memory`、`sp_current_stage`、`sp_last_action_id` 等 SP 字段。

本轮真实 smoke 满足第 2 至第 5 项。修复后 CentralAgent 从 Qwen/vLLM 的 `reasoning_content` 正确解析动作 JSON，避免空 `content` 被误判为无动作并重复调用模型。
