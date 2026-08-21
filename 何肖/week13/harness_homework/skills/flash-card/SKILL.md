---
name: flash-card
description: "英语单词闪卡生成 Skill：输入一个英语单词，由大模型自动补全音标、词性、释义、3条中英例句、近义词，生成静态HTML学习闪卡页面。Invoke when user asks for a flash card / 闪卡 for an English word, e.g. '给我做张 crazy 词的闪卡'、'给我做 crazy 的 flash card'、'做一个 resilient 的单词卡'。"
---

# Flash Card 单词闪卡生成

为英语单词生成一张静态 HTML 学习卡片。卡片版面顺序：单词+音标 → 释义 → 近义词 → 3 条中英对照例句。

## 功能

输入一个英语单词生成一张静态 HTML 学习卡片，包含：
  - 单词、音标、词性、释义
  - 固定 3 条中英对照例句
  - 近义词标签（位于例句上方）

## 触发场景

当用户说出类似下面的话时触发本 skill：
- "给我做张 crazy 词的闪卡"
- "给我做 crazy 的 flash card"
- "做一个 resilient 的单词卡"
- "帮我生成 meticulous 的闪卡"

## 渐进式执行步骤（Agent Side）

1. **Step 1 · 单词识别与信息补全**：从用户输入提取目标英语单词（小写），由大模型利用自身语言知识填写完整的学习数据（音标、词性、中文释义、3 条地道中英对照例句、4-6 个近义词）。
2. **Step 2 · 保存 JSON**：调用 `save_flashcard_data` 将完整数据保存到 `json_data/<word>.json`。
3. **Step 3 · 生成 HTML**：调用 `generate_flashcard` 运行 `scripts/make_flashcard.py` 从 JSON 生成 HTML 闪卡到 `html_data/<word>.html`。
4. **Step 4 · 查看与分享**：通过 `/files/skills/flash-card/html_data/<word>.html` 访问生成的闪卡页面，或直接双击 HTML 文件离线查看。

## 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| 单词 | 目标英语单词（小写） | `crazy`、`resilient`、`meticulous` |

## 执行流程

1. **识别单词**：从用户话语中提取目标英语单词（小写化作为文件名）。

2. **由大模型补全信息**：调用 `save_flashcard_data` 工具时，大模型需利用自身语言知识
   为该单词填写完整的学习数据（工具参数均为必填）：
   - `word`：单词（小写）
   - `phonetic`：音标（如 `/rɪˈzɪliənt/`）
   - `pos`：词性（如 `adj.`）
   - `definition`：中文释义
   - `examples`：**恰好 3 条**，每条含 `en`（英文例句）和 `zh`（中文翻译）
   - `synonyms`：近义词列表（4-6 个为宜）

   例句要求：地道、长度适中、能体现该词典型用法；近义词要尽量贴近该词在释义下的核心含义。
   JSON 数据保存到 skill 的 `json_data/<word>.json`。

3. **生成 HTML**：调用 `generate_flashcard` 工具，运行 `scripts/make_flashcard.py`
   从 JSON 生成 HTML 页面，输出到 `html_data/<word>.html`，可通过服务路由直接访问。

4. **总结与跳转**：两个工具执行完毕后，大模型总结执行耗时与完成效果，
   并以 `📄 [文件名](URL)` 格式给出 JSON 与 HTML 文件的可点击链接。
   链接必须使用工具返回的 `generated_urls`，例如：
   - `/files/skills/flash-card/json_data/<word>.json`
   - `/files/skills/flash-card/html_data/<word>.html`

## 目录结构

```
flash-card/
├── json_data/     ← 单词 JSON 数据（json_data/<word>.json）
├── html_data/     ← 生成的 HTML 闪卡（html_data/<word>.html）
├── scripts/       ← make_flashcard.py 生成脚本
└── SKILL.md
```

## 数据 JSON 示例

```json
{
  "word": "resilient",
  "phonetic": "/rɪˈzɪliənt/",
  "pos": "adj.",
  "definition": "能迅速从困难、挫折中恢复过来的；有韧性的，适应力强的",
  "examples": [
    {"en": "She is a resilient child who bounces back quickly from setbacks.", "zh": "她是个有韧性的孩子，遇到挫折能很快恢复过来。"},
    {"en": "The economy proved remarkably resilient during the crisis.", "zh": "在危机期间，经济表现出了惊人的韧性。"},
    {"en": "A resilient mindset helps you cope with life's challenges.", "zh": "一种有韧性的心态能帮你应对生活中的挑战。"}
  ],
  "synonyms": ["tough", "flexible", "strong", "hardy", "buoyant", "springy"]
}
```

## 注意事项

- 例句固定 3 条，脚本会自动截断或补占位，但生成数据时应直接给齐 3 条。
- 原始 JSON 数据集中存放在 skill 的 `json_data/` 目录，方便复用与回顾。
- HTML 文件统一输出到 `html_data/` 目录，可通过 `/files/skills/flash-card/html_data/<word>.html` 访问。
