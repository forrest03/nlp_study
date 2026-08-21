---
name: production-db-query
description: 在 DMS 平台查询生产数据库时使用：建立不落盘的短期登录会话，只允许只读 SQL，强制先执行并人工审阅 EXPLAIN，再凭审批文件执行正式查询。适用于生产数据排查、报表核对和性能风险评估。
---

# 生产数据库安全查询

使用本 Skill 查询 `https://dms.yzw.cn/` 上的 `pbets`、`pbid` 生产数据库。仅支持只读 `SELECT` 或只读 `WITH` 查询。

## 安全边界

- 不读取浏览器 Cookie 库、浏览器配置文件、密码库或环境变量中的 Cookie。
- 不在 Skill、脚本、查询文件、审批文件、日志或对话中记录 Cookie、Token、CSRF Token。
- 只在当前进程的环境变量中保存临时凭据；结束后立即清除。
- 拒绝写操作、多语句、锁定读和常见资源消耗函数。无法确定为只读时拒绝执行。
- 不使用用户提供的完整 cURL；其中可能含已泄露的短期生产凭据。

## 使用本地 Cookie 文件

- 不直接访问 `dms.yzw.cn` 登录，不启动浏览器，也不读取浏览器 Cookie 库。
- 只读取当前用户的 `~/Documents/.dms_cookie`；内容必须是完整 Cookie 请求头值，并包含 `csrftoken`。
- macOS/Linux 必须将文件权限设为 `600`；Windows 应仅授予当前用户读取权限。
- Cookie 不得写入命令参数、日志或聊天；脚本不会回显文件内容。
- 查询前仅校验文件，不接收交互式粘贴输入：

```bash
# macOS/Linux：python3；Windows PowerShell：py -3
<python-command> /Users/ranpengcheng/.codex/skills/production-db-query/scripts/dms_session.py
```

文件缺失、权限不安全、Cookie 失效或接口认证失败时直接停止；应由文件维护方刷新 Cookie。查询结束后不会改写该文件。

## 强制查询流程

1. 先根据问题选择数据库，并从工程源码确认结构，不能凭表名猜测：
   - `pbets`：优先查看 `/Users/ranpengcheng/projects/pbets_agg/pbets/pbets-common-dal/src/main/java/**/mysql/**/model/`、对应 `mapper/`，以及 `/Users/ranpengcheng/projects/pbets_agg/pbets/ddl/`。
   - `pbid`：优先查看 `/Users/ranpengcheng/projects/pbets_agg/pbid/**/model/`、对应 `mapper/` 和 Mapper XML，以及 `/Users/ranpengcheng/projects/pbets_agg/pbid/sql/init/`。
   - 必须确认表名、列名、逻辑删除条件、关联键及业务范围；DDL 与当前 Mapper 不一致时以当前 Mapper/实体为准，并将差异标记为待验证。
2. 将候选 SQL 保存为本地文件；不含 `EXPLAIN` 前缀和分号。
3. 只执行 EXPLAIN，并保存原始响应：

```bash
<python-command> /Users/ranpengcheng/.codex/skills/production-db-query/scripts/dms_query.py \
  --phase explain --database pbets --sql-file /absolute/path/query.sql --output /absolute/path/explain.json
```

4. 审阅 `explain.json`：确认没有全表扫描或大范围扫描、无笛卡尔积、关联字段命中合适索引、预估扫描行数与业务范围匹配。若结果格式或风险无法判断，停止并请数据库负责人确认索引或改写 SQL。
5. 仅在审阅结论为“通过”后，创建审批文件。审批文件必须是 JSON，且 `sql_sha256`、`explain_response_sha256` 分别等于脚本报出的 SQL 摘要、EXPLAIN 响应摘要：

```json
{"sql_sha256":"<SQL 摘要>","explain_response_sha256":"<EXPLAIN 响应摘要>","verdict":"approved","reason":"已审阅 EXPLAIN：索引和扫描范围符合预期"}
```

6. 使用该审批文件执行正式查询：

```bash
<python-command> /Users/ranpengcheng/.codex/skills/production-db-query/scripts/dms_query.py \
  --phase query --database pbets --sql-file /absolute/path/query.sql \
  --approval-file /absolute/path/explain-approval.json \
  --output /absolute/path/result.json
```

正式查询会先再次执行 EXPLAIN；审批文件与 SQL 摘要或该次 EXPLAIN 响应摘要不一致、结论非 `approved` 或任何 EXPLAIN 风险未消除时，脚本拒绝执行正式 SQL。

## 处理结果

- 生产数据仅输出到用户指定的本地文件；汇报时默认给出聚合、脱敏或最小必要字段。
- 接口错误、超时、认证失败或 JSON 格式异常均应停止；不得绕过 EXPLAIN 或降低校验。
- 查询结束后清除会话环境变量。
