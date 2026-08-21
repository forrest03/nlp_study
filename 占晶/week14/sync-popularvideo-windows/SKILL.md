---
name: sync-popularvideo-windows
description: "审计本地 PopularVideo 上游仓库中 Mac、共享契约和服务端的增量改动，生成固定 Git 范围的中文执行 Prompt，在用户要求完整同步时用该 Prompt 设置 Codex goal，将适用能力同步到 PopularVideoWindows，完成自动验证，输出逐项改动清单和可直接执行的手动测试清单，并记录最后一次成功同步的上游 commit。适用于比较或同步最新 PopularVideo 到 Windows、从上次基线继续、生成 Windows 移植 Prompt/goal、查询 Mac/服务端改动与 Windows 对齐情况，以及制定同步后的手测计划。"
---

# 同步 PopularVideo 到 Windows

把 `PopularVideo` 视为只读的 Mac/服务端上游，把 `PopularVideoWindows` 视为唯一可写的产品仓库。所有范围都固定到完整 commit；只有实现、验证、逐项报告都完成后才能推进成功基线。

## 默认位置

- 上游仓库：`D:\工作\Video Agent\code\PopularVideo`
- Windows 仓库：`D:\工作\Video Agent\code\PopularVideoWindows`
- Windows 必读约束：`docs/WINDOWS_CODE_RANGES.md`
- 同步状态：`<windows>/.codex/popularvideo-sync/state.json`
- 同步报告：`<windows>/.codex/popularvideo-sync/reports/`

接受用户显式指定的路径。执行本 skill 时始终只读上游，包括工作树、索引、refs、分支和 Git 配置。不要在上游运行 `git pull`、`git fetch`、checkout、reset、clean、stash、格式化、迁移、生成、提交或任何写操作。只用 `git ls-remote` 做远程新鲜度检查。即使定位到服务端问题，也只用中文报告证据、根因、影响、推荐方案和回归建议，不修改服务端。

## 按需读取资源

- 分类上游改动前，读取 [references/path-map.md](references/path-map.md)。
- 生成执行目标前，读取 [references/execution-prompt.md](references/execution-prompt.md)。
- 编写最终报告和手测清单前，读取 [references/report-template.md](references/report-template.md)。
- 每次运行都完整读取 Windows 仓库里的实时约束，不把它复制进 skill。

使用 `scripts/sync_state.py` 收集 commit 范围、校验报告并维护状态。脚本只依赖 Python 标准库。

## 工作流程

### 1. 只读预检

1. 用 UTF-8 完整读取实时 `WINDOWS_CODE_RANGES.md` 和适用的 `AGENTS.md`。
2. 解析两个 Git 根目录，检查分支、HEAD、origin、跟踪状态和工作树状态。
3. 要求上游工作树干净；完全忽略未提交的上游内容，不替用户清理或提交。
4. 运行 `inspect --check-remote`。无法访问远程时标注“新鲜度未验证”；远程 HEAD 与本地不同则停止，请用户自行拉取上游。
5. 把 Windows 脏工作树视为用户已有工作。未得到明确授权前不要编辑，也不要 stash、丢弃或覆盖。
6. 检查当前 Codex goal，不替换无关的活动 goal。

```powershell
python scripts/sync_state.py inspect --check-remote
```

### 2. 确定成功基线

状态存在时，自动读取 `last_successful_upstream_commit`，并验证它是固定目标的祖先。

状态不存在时，只做一次初始化：

1. 优先搜索非归档的专用同步报告，如 `docs/WINDOWS_MAC_SYNC*.md`。
2. 只接受明确写成“已完成/已同步”的 commit；拒绝仅用于检查、复现、规划或对比的 HEAD。
3. 验证候选 commit 存在且属于目标历史。
4. 候选不唯一或证据不足时，请用户确认。
5. 只初始化确认后的 commit；初始化不代表本轮产生了新同步。

```powershell
python scripts/sync_state.py bootstrap --commit <确认后的完整哈希> --note "<确认依据>"
```

不要为了缩小差异而把当前 HEAD 猜成已同步基线。

### 3. 审计固定范围

在审计开始时把本地上游 HEAD 固定为完整哈希：

```powershell
python scripts/sync_state.py audit --target <完整目标哈希> --output <windows>/.codex/popularvideo-sync/audit.json
```

把 audit JSON 当作索引，不当作语义结论。逐个阅读 `<base>..<target>` 内每个 commit 的说明、文件状态、相关 diff、测试、模型、API 契约和调用方。

建立细粒度改动账本：

- 用 `CHANGE-001`、`CHANGE-002` 连续编号。
- 按一个可独立理解、实现或测试的行为/契约划分；同一 commit 可以拆成多个 `CHANGE`，不要把不相关用户流程合并成“大量优化”。
- 每项分别写清：上游依据、改动前、改动后、Mac/共享变化、服务端变化、用户影响、Windows 现状、Windows 结论、预期落点、自动测试和手测映射。
- Windows 结论只能是：`已同步`、`已存在`、`需同步`、`不适用`、`仅服务端`、`阻塞`。
- 用 `rg` 和当前代码/测试为每个结论提供证据。
- 对每个上游 commit 建立“提交 → CHANGE 编号/明确排除原因”的覆盖索引，最终未解释数必须为 0。

检查契约两端。即使没有 Mac 文件，服务端的请求、响应、事件、认证、错误、任务状态和 artifact 改动也可能要求 Windows 适配。路由只能使用结构化 intent、workflow、plan、slot、tool call、binding、task frame 或显式 UI 操作，禁止从 prompt 文本猜测。

### 4. 同时设计手动测试

审计每个 `CHANGE` 时就生成手测映射，实施后再校正：

- 用 `MANUAL-001`、`MANUAL-002` 连续编号；一个场景一个测试项，不把多个独立流程塞进“测试相关功能”。
- 标明关联 `CHANGE`、P0/P1/P2、测试入口、前置条件、测试数据、是否需要真实凭据、预计耗时、逐步操作、逐步可观察结果、异常/边界、失败时需要记录的 ID/日志、自动覆盖和当前状态。
- 覆盖正常路径、失败路径、取消/重试、重启/持久化、多语言和空状态中实际受影响的部分。
- Agent/服务端流程要写清应记录的 conversation/message/request/run/task/tool-call/job ID。
- 媒体流程要写清输入类型、附件绑定、输出文件、artifact lineage、进度和失败展示。
- 某项确实无需手测时，必须写 `无（具体原因）`，不能只写“无需测试”。
- 按 P0 → P1 → P2 排序，明确哪些是用户必须手测，哪些自动测试已充分覆盖。

禁止使用“测试相关功能”“验证一下”“确保正常”“按需测试”“基本没问题”等含糊表述。

### 5. 生成中文执行 Prompt

按 [references/execution-prompt.md](references/execution-prompt.md) 生成自包含的中文 Prompt。必须填入完整 base/target、细粒度 `CHANGE` 账本、计划的 `MANUAL` 清单、Windows 已有脏改动、验证命令和明确排除项；不能只写“同步最新版”。

创建 goal 前先向用户展示完整 Prompt。用户只要求审计或 Prompt 时，到此停止，不创建 goal、不改代码。

### 6. 设置 goal 并开始记录

用户明确要求端到端同步时，把刚生成的完整中文 Prompt 原样作为 `create_goal` 的 objective。相同范围已有活动 goal 时继续它；goal 工具不可用时直接执行并说明未创建持久 goal。

在首次 Windows 编辑前记录活动同步：

```powershell
python scripts/sync_state.py begin --target <完整目标哈希>
```

默认拒绝脏 Windows 工作树。只有用户明确同意保留现有改动继续时才加 `--allow-dirty-windows`，并在 Prompt、报告和最终回复里单独列出现有改动。

### 7. 按依赖顺序移植

只实现标记为 `需同步` 的项，通常按以下顺序：

1. shared 类型、归一化契约、纯策略和 i18n。
2. Main runtime、持久化、认证、上传下载和服务端 adapter。
3. Preload/IPC typed bridge。
4. Renderer feature 编排。
5. 复用现有组件和主题变量的 Windows 原生 UI。
6. contract/regression 测试。

翻译行为，不复制 Swift 布局。不要修改上游，不新增 prompt/关键词/regex 路由兜底。若只能靠自然语言猜测完成某项，停止该项并报告缺失的结构化服务端信号。

固定范围后出现的新上游 commit 留到下次，不静默扩展目标。

### 8. 自动验证与对账

按实时 Windows 规则和 `package.json` 选择验证：

- 为每个高风险 `CHANGE` 运行最小相关 contract 测试。
- Renderer/shared/i18n 至少运行 `npm run typecheck`。
- Electron Main、preload、bundle 或大范围 UI 再运行 `npm run build:app`。
- 所有代码/文档改动运行 `git diff --check`。
- 未经授权不运行真实凭据、付费 provider、发布、破坏性或线上写入测试。

逐项回查所有 `CHANGE`。允许 `已存在`、`不适用`、`仅服务端`，但不能遗留无解释的 `需同步`。自动测试通过不等于手测完成；报告中把实际执行状态写清。

### 9. 输出详细报告并推进状态

按 [references/report-template.md](references/report-template.md) 写入忽略目录，并把全部 `CHANGE` 摘要和全部待执行 `MANUAL` 清单直接放进最终回复，不能只给报告链接。

报告完成后先运行：

```powershell
python scripts/sync_state.py validate-report --report-file <绝对报告路径> --base <完整基线> --target <完整目标>
```

校验器会检查：必需章节、每个 CHANGE/MANUAL 的字段、手测映射、commit 覆盖、占位符、含糊措辞和覆盖计数。校验失败时修订报告，不能推进基线。

实现、自动验证、覆盖对账和报告校验全部通过后，才运行：

```powershell
python scripts/sync_state.py complete --target <完整目标哈希> --report-file <绝对报告路径> --summary "<具体结果>" --verification "npm run typecheck: passed" --windows-file "src/..."
```

`complete` 会再次校验报告。随后用 `inspect` 确认 `last_successful_upstream_commit` 等于目标，再把 Codex goal 标记完成并报告 goal 的最终 token 用量。

## 完成红线

- 上游始终只读；Windows 是唯一可写产品仓库。
- 状态表示“已成功同步并验证到该 commit”，不表示“只审计到该 commit”。
- 报告不完整、手测不具体、commit 未覆盖、验证失败或实现未完成时，不推进状态。
- 缺失状态文件不代表可以猜基线。
- 不用日期、分支名或短哈希替代固定范围。
- 本 skill 不隐含 commit、push、PR、发布、付费调用、真实凭据或服务端修改授权。
