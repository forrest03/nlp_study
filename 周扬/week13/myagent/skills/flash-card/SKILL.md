---
name: flash-card
description: >-
  为一个英语单词生成静态 HTML 学习闪卡（含音标、词性、释义、3 条中英对照例句、近义词）。
  Use when the user asks to make a flash card / 闪卡 for an English word,
  e.g. "给我做张 crazy 词的闪卡"、"给我做 crazy 的 flash card"、"做一个 resilient 的单词卡"。
---

# Flash Card 单词闪卡生成

为英语单词生成一张静态 HTML 学习卡片。卡片版面顺序：单词+音标 → 释义 → 近义词 → 3 条中英对照例句。

上方系统会注入当前 skill 的目录路径 `skill_dir`。这个 skill 目录下有：
- `data/`：放单词 json 数据
- `scripts/make_flashcard.py`：根据 json 生成 html

处理这类任务时，优先使用这些工具：
- `write_text_file`：写入 `<skill_dir>/data/<word>.json`
- `run_local_script`：执行 `<skill_dir>/scripts/make_flashcard.py`
- `open_file`：打开生成好的 html

注意：
- 不要把 shell 命令整串塞进 `run_local_script`
- `run_local_script` 的 `script_path` 必须是一个真实存在的脚本文件路径
- 不需要先用 `/bin/bash`、`ls -la`、`/usr/bin/python3 -c` 这类方式试探

## 触发场景

当用户说出类似下面的话时触发本 skill：
- "给我做张 crazy 词的闪卡"
- "给我做 crazy 的 flash card"
- "做一个 resilient 的单词卡"
- "帮我生成 meticulous 的闪卡"

## 执行流程

1. **识别单词**：从用户话语中提取目标英语单词（小写化作为文件名）。

2. **生成 JSON 数据**：自己写出该单词的学习数据，字段如下，保存到当前 skill 的 `data/` 目录：
   - 路径：`<skill_dir>/data/<word>.json`
   - `word`：单词
   - `phonetic`：音标（如 `/rɪˈzɪliənt/`）
   - `pos`：词性（如 `adj.`）
   - `definition`：中文释义
   - `examples`：**恰好 3 条**，每条含 `en`（英文例句）和 `zh`（中文翻译）
   - `synonyms`：近义词列表（4-6 个为宜）

   例句要求：地道、长度适中、能体现该词典型用法；近义词要尽量贴近该词在释义下的核心含义。

3. **生成 HTML**：运行脚本，HTML 输出到**当前工作目录**（不是 skill 目录）：
   - 脚本路径：`<skill_dir>/scripts/make_flashcard.py`
   - 数据路径：`<skill_dir>/data/<word>.json`
   - 推荐输出：`./<word>.html`
   - 调用方式：把脚本路径作为 `run_local_script.script_path`，把 json 路径和 `-o` 输出路径放进 `args`

4. **打开预览**：用 `open_file` 打开生成的 HTML 文件，让用户立即看到效果。

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
- HTML 文件始终输出到当前工作目录，便于用户在任意项目下使用。
- 原始 JSON 数据集中存放在 skill 的 `data/` 目录，方便复用与回顾。
