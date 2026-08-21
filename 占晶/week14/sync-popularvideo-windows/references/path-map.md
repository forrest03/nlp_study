# 上游路径分类与核对方法

路径只用于定位调查范围。行为结论必须来自 diff、测试、模型和调用链。

| 上游范围 | 默认分类 | Windows 需要回答的问题 |
| --- | --- | --- |
| `macos-app/Sources/**` | Mac 客户端 | 哪个行为、状态、UI、本地工具、持久化或错误处理发生变化？ |
| `macos-app/Tests/**` | Mac 契约证据 | 哪条不变量或回归需要 Windows contract test？ |
| `shared-swift/Sources/**` | Mac/共享契约 | Windows 是否需要对应模型、枚举、默认值、解析器、事件、binding 或 i18n key？ |
| `shared-swift/Tests/**` | 共享契约证据 | Windows 必须归一化或拒绝哪些边界输入？ |
| `src/addsubtitle/cloud/**` | Hosted API/runtime | 路由、SSE、job、artifact、失败、billing、auth 或编排是否变化？ |
| `src/addsubtitle/core/**` | 服务端领域契约 | intent、plan、tool、任务状态、模型、策略、媒体 lineage 或错误是否变化？ |
| `tests/**` | 服务端行为证据 | 哪些外部可观察契约影响 Windows？不要照搬 Python 实现细节。 |
| `deploy/**`、`Dockerfile*`、`.github/**` | 部署/运维 | 通常不移植客户端；检查 endpoint、auth、环境、发布和兼容性影响。 |
| `docs/**` | 契约/文档证据 | 移植前用当前代码和测试再次确认。 |
| `web-app/**`、`ios-app/**` | 其他客户端 | 通常不适用；若暴露共享服务端契约或可复用行为则单独记录。 |
| `scripts/**`、`skill/**`、`.codex/**` | 工具/Agent 支持 | 检查 runtime 协议或开发验证是否变化，否则明确排除。 |

## 跨范围核对

- Mac 改动：同时检查共享 Swift 模型和服务端 endpoint，再设计 Windows 行为。
- 服务端改动：即使 commit 没有 Mac 文件，也检查 Mac 消费方和 Windows 请求链。
- 字段新增、重命名或删除：搜索 Windows 全链路 `shared model → main runtime → IPC/preload → renderer feature → UI/test`。
- artifact/媒体：检查 producer binding、cloud ID、remote path、run/step lineage、本地化和 final output 选择。
- auth/billing：区分 session token 与手动 API Key，默认不使用真实 provider 验证。
- Agent 路由：只接受结构化 intent、workflow、plan、slot、tool call、task frame 或显式 UI 操作，不用 prompt 启发式。

## Windows 常见落点

实时 `WINDOWS_CODE_RANGES.md` 始终优先。

| 契约 | Windows 范围 |
| --- | --- |
| shared model/policy/i18n | `src/shared/**` |
| HTTP/SSE/job/媒体 runtime | `src/main/runtime.ts` 与聚焦的 `src/main/**` 模块 |
| 进程边界 | `src/main/ipc.ts`、`src/preload/preload.ts` |
| Agent 编排 | `src/renderer/features/agent/**`、`src/shared/models/agent*.ts` |
| Composer/context/附件 | `features/composer/**`、`features/media/**`、`features/sessions/**` |
| UI | 现有 `src/renderer/components/**` 和主题样式 |
| 本地 Python runtime | 仅任务明确涉及随附 runtime 时修改 `runtime/**` |
| 回归证据 | `scripts/*contract_test*` 和对应 package scripts |

## 拆分 CHANGE 的粒度

以下情况必须拆成不同 `CHANGE`：

- 用户入口、前置条件或最终产物不同。
- 成功路径与独立的错误/重试策略发生不同变化。
- 请求契约、任务状态、artifact 交付和 UI 呈现可以独立回归。
- 同一 commit 同时修改认证、Agent、字幕、媒体或项目管理等不同领域。

只有同一用户行为的模型、runtime、UI 和测试是同一条端到端链路时，才合并到一个 `CHANGE`。
