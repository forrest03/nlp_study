---
name: image-to-markdown
description: 图片识别（大模型视觉理解，非 OCR）：把图片内容识别为 Markdown 文本。当用户提供图片（截图、照片、文档、表格、图表等）并要求识别、读取、转写其内容，或对图片内容做后续处理时使用。
---

# 图片识别 → Markdown

通过本地 Ollama 视觉模型（默认 minicpm-v）识别图片内容，输出 Markdown 文本。
注意：这是大模型图像理解，**不是 OCR**——能理解表格结构、图表、排版与语义，并按原排版输出 Markdown。

## 触发方式

用户提供图片（给出路径 / 拖入图片 / 消息附带图片）并希望识别内容时，自动执行下面步骤。

## 执行步骤

1. **确定图片路径**
   - 用户给了路径 → 直接用；
   - 图片以附件形式提供 → 使用附件保存的本地路径；
   - 无法确定 → 询问用户。

2. **调用识别脚本**（位于本技能目录 `scripts/image_to_markdown.py`，与 SKILL.md 同目录；当前工作目录不对时用绝对路径调用）

   ```bash
   python <技能目录>/scripts/image_to_markdown.py <图片路径或URL> [更多图片...]
   ```

   - 多张图片可一次传入，一次调用批量识别；
   - 默认提示词：`识别图片里所有信息，使用 markdown 输出全部内容，并保持排版的一致`；
   - 需要自定义识别要求时加 `--prompt "..."`；
   - 模型可用 `--model` 切换，默认 `minicpm-v`；接口可用 `--api-style openai|native` 切换。

3. **读取结果**：脚本把 Markdown 输出到 stdout，直接取用。

4. **交给用户**：原样展示 Markdown；若用户需要翻译、总结、提取表格、生成文档等后续操作，基于该 Markdown 继续处理。

## 后端分流（重要）

`--backend` 三档：`auto`（默认）/ `ollama` / `dashscope`。**执行时按下表预判并显式传参**：

| 情况 | backend | 理由 |
|---|---|---|
| 能确认内容：终端/代码/小字报表/密集文字 | `--backend dashscope` | 本地 7B 模型小字识别差、易幻觉 |
| 能确认内容：PPT/文档/大字海报/讲义/照片 | `--backend auto` | 本地够用；漏识时自动云端兜底 |
| 无法预览图片内容 | `--backend auto` | 本地优先；**结果出现幻觉特征（乱码、整段重复、专有名词错乱）→ 云端重跑** |
| 上下文强信号（连续几张同一终端会话截图） | `--backend dashscope` | 大概率仍是终端内容 |
| 用户明确要求纯本地/离线/省钱 | `--backend ollama` | 强制本地 |

- 判断依据是**图片内容类型**，不是截图工具（Snipaste 只是工具，内容千差万别）。
- `auto` 机制：先跑本地 minicpm-v，**输出 <30 字符**（疑似没读出内容）且配置了 `DASHSCOPE_API_KEY` 时，自动用云端 qwen-vl-max 兜底重跑一次。
- **已知局限**：minicpm-v 会产生「长幻觉」（乱码复读），超 30 字符不触发脚本兜底 → 幻觉检测由执行者（Claude）在结果上判断。
- 未配置 `DASHSCOPE_API_KEY` 时 auto 退化为纯本地。

## 环境要求

- 本地 **Ollama** 已启动，且已拉取视觉模型（`ollama pull minicpm-v`），默认地址 `http://127.0.0.1:11434`。
- Python 3.8+，需要 Pillow：`pip install Pillow`。
- 脚本内部自动处理：最长边缩放到 ≤1280px → 统一转 JPEG → base64 → 调用后端。
- `DASHSCOPE_API_KEY`：仅 `--backend dashscope` 或 auto 兜底时需要。
- 可选环境变量：`OLLAMA_BASE_URL`（地址）、`OLLAMA_MODEL`（模型）、`OLLAMA_API_STYLE`（openai/native）。

## 常见问题

- 报「无法连接本地 Ollama」→ 启动 Ollama 应用或 `ollama serve`，并确认 `ollama pull minicpm-v` 已完成。
- 提示 `缺少 Pillow` → 先安装依赖再重试。
- 图片格式不支持（如 HEIC）→ 先转换格式再试。
