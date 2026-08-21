# Progressive Skill Harness

一个可运行的最小 Harness：把本地 skill 的加载拆成四个阶段，避免启动时把所有 `SKILL.md`、参考资料和脚本内容一次性加入模型上下文。

项目已复制课程目录中的 `skills/baoyu-diagram`（包含 skill 说明、四个参考文档和 SVG 转 PNG 脚本）。复制时刻意不保留源目录中的 `node_modules`：其中包含 Windows 平台二进制；如需运行转换脚本，请在 `skills/baoyu-diagram/scripts` 内按本机平台安装 Bun 依赖。

## 渐进式加载流程

```text
skills/*/SKILL.md
        │
        ├─ Stage 1：只读取 YAML front matter（name / description / version）
        │
用户请求 ─┼─ Stage 2：以中英文词法匹配筛选候选 skill
        │
        ├─ Stage 3：只读取得分最高 skill 的完整 SKILL.md
        │
        └─ Stage 4：只读取显式要求的 references/*.md
```

`SkillLoader` 对引用文件实施 basename、`.md` 后缀和真实路径边界校验，拒绝 `../` 等路径穿越输入。CLI 日志只记录请求长度，不记录用户原始文本。

## 目录与职责

```text
src/progressive_harness/
  catalog.py       # 仅解析元数据（Stage 1）
  matcher.py       # 元数据候选匹配（Stage 2）
  loader.py        # 按需加载说明/引用并校验路径（Stage 3-4）
  harness.py       # 协调服务
  cli.py           # 可观察的命令行入口
skills/baoyu-diagram/  # 复制的本地 diagram skill
tests/                 # 正常、边界与异常路径单元测试
```

## 使用

无需第三方 Python 依赖。使用 `PYTHONPATH=src` 直接运行：

```bash
PYTHONPATH=src python3 -m progressive_harness.cli --list
PYTHONPATH=src python3 -m progressive_harness.cli "画一个订单系统架构图" --verbose
PYTHONPATH=src python3 -m progressive_harness.cli "画一个订单系统架构图" --reference architecture.md --json
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

也可安装为本地可编辑包：

```bash
python3 -m pip install -e .
skill-harness "画一个登录流程图" --reference flowchart.md
```

## 接入 LLM Harness 的方式

把 `ProgressiveHarness` 挂到 Agent 的请求组装阶段：先调用 `discover()` 和 `select()`，只将候选名称/描述交给路由层；确认选择后调用 `load_skill()`；仅当完整说明要求某一参考类型时再调用 `load_reference()`。本示例不绑定任何模型或 API Key，因此可以安全地接入任意模型提供商。

## LLM 与 skill 工具调用

`SkillEnabledChat` 提供 OpenAI 兼容的函数调用接入：首轮只将 skill 的名称、说明和基于请求的路由建议发送给模型。模型必须调用 `load_skill` 才能得到完整 `SKILL.md`；仅在已加载对应 skill 后，才可调用 `load_reference` 读取某一份引用文件。每次回答最多允许 6 次工具调用，防止模型工具循环失控。

设置环境变量（不要将密钥写入代码或提交到仓库）：

```bash
export LLM_API_KEY='你的密钥'
export LLM_MODEL='你的模型名称'
# 可选；默认 https://api.openai.com/v1
export LLM_BASE_URL='https://你的兼容服务/v1'
```

调用示例：

```python
from pathlib import Path
from progressive_harness import (
    LLMConfiguration,
    OpenAICompatibleLLM,
    ProgressiveHarness,
    SkillEnabledChat,
)

harness = ProgressiveHarness(Path("skills"))
client = OpenAICompatibleLLM(LLMConfiguration.from_environment())
result = SkillEnabledChat(harness, client).answer("画一个订单系统架构图")
print(result.content)
```

该模块使用 Python 标准库发送 HTTPS 请求，不新增运行时依赖。HTTP 失败、超时、响应过大、无效 JSON 和非法工具参数都会显式报错或以工具错误结果返回给模型；日志中不会记录 API Key 或用户消息原文。
