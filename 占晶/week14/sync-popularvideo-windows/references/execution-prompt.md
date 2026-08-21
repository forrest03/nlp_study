# 中文执行 Prompt 模板

从此模板生成一个自包含的 goal objective。替换所有尖括号占位符，删除不适用段落，不保留模糊概括。

```text
把已审计的 PopularVideo 上游行为同步到 PopularVideoWindows。

仓库与不可变范围：
- 只读上游：<上游绝对路径>
- 可写 Windows：<Windows 绝对路径>
- 审计时上游分支：<branch>
- 上次成功基线：<完整 base hash>（<subject>）
- 固定目标：<完整 target hash>（<subject>）
- 范围：<完整 base hash>..<完整 target hash>
- 远程新鲜度：<已验证一致 | 本地落后详情 | 未验证原因>
- Windows 起始 HEAD：<完整 Windows hash>
- Windows 既有未提交改动：<无 | 精确路径和用户授权>

编辑前完整读取 <Windows 路径>/docs/WINDOWS_CODE_RANGES.md 和适用的 AGENTS.md。绝不修改 PopularVideo 上游工作树或 Git 元数据；遇到服务端问题只做只读定位并用中文报告。固定目标后出现的新 commit 留待下次。

上游细粒度改动账本：

CHANGE-001：<可独立理解和测试的功能/契约>
- 上游依据：<commit、路径、测试>
- 改动前：<具体旧行为>
- 改动后：<具体新行为>
- Mac/共享变化：<具体变化或无及原因>
- 服务端变化：<具体变化或无及原因>
- 用户影响：<入口、状态、产物或错误的可观察变化>
- Windows 当前证据：<代码/测试/缺口>
- Windows 结论：<已存在 | 需同步 | 不适用 | 仅服务端 | 阻塞>
- Windows 计划落点：<文件/模块/测试>
- 手测映射：<MANUAL-001... | 无（具体原因）>

<继续列出所有 CHANGE；不要把不相关改动合并>

提交覆盖索引：
- <commit> → <CHANGE 编号或明确排除原因>
- 未解释 commit：0

Windows 实施顺序：
1. <shared 契约、策略、i18n>
2. <Main runtime、持久化、adapter>
3. <preload/IPC>
4. <renderer 编排>
5. <Windows 原生 UI>
6. <contract/regression 测试>

计划手动测试：

MANUAL-001：<单一场景名称>
- 关联改动：<CHANGE 编号>
- 优先级：<P0/P1/P2>
- 入口：<具体页面/按钮/操作>
- 前置条件：<环境、账号、文件、设置>
- 测试数据：<精确输入和格式>
- 是否需要真实凭据：<否 | 是，说明哪类且不由 Codex 擅自使用>
- 预计耗时：<分钟>
- 操作步骤：<编号步骤>
- 预期结果：<逐步可观察结果>
- 异常/边界：<至少一个受影响的失败或边界场景>
- 失败时记录：<conversation/request/run/task/tool-call/job ID 或日志>
- 自动覆盖：<对应命令/测试或无>
- 状态：待用户手测

<按 P0 → P1 → P2 列出全部 MANUAL>

约束：
- 保持 Windows 分层、现有组件、主题变量和 UTF-8 i18n。
- 结构化 server intent/workflow/slot/task parameter/tool call/execution plan/task frame 和显式 UI 操作具有权威性。
- 不新增基于 prompt、关键词、同义词、regex、文件名或自然语言的客户端路由兜底。
- 未经单独授权，不 commit、push、发布、使用真实凭据、调用付费 provider、修改发布配置或服务端。
- 保留并单独报告所有已授权的 Windows 既有改动。

自动验证：
- <逐个 CHANGE 对应的聚焦 contract 命令>
- npm run typecheck <适用时>
- npm run build:app <适用时>
- git diff --check
- 对每个 CHANGE 重新执行 Windows parity 搜索。

完成条件：
- 每个审计改动都有 CHANGE 编号和明确结论；每个上游 commit 都映射到 CHANGE 或明确排除理由，未解释数为 0。
- 每个影响 Windows 用户行为的 CHANGE 都映射到具体 MANUAL；无需手测项写明具体原因。
- 报告逐项列出改动前后、Mac/服务端变化、Windows 实现、自动验证和用户待手测步骤，不能只给概述或报告链接。
- 运行 sync_state.py validate-report 并通过；失败则修订报告且不推进基线。
- 只有实现和验证成功后，才对固定 target 调用 sync_state.py complete，并确认状态中的 last-successful commit 等于目标。
- 状态落盘后才把 Codex goal 标记完成。
```
