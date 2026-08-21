---
name: flash-card
description: >-
  为一个英语单词生成静态 HTML 学习闪卡（含音标、词性、释义、3 条中英对照例句、近义词）。
  Use when the user asks to make a flash card / 闪卡 for an English word,
  e.g. "给我做张 crazy 词的闪卡"、"给我做 crazy 的 flash card"、"做一个 resilient 的单词卡"。
triggers:
  - 闪卡
  - flash card
  - flashcard
  - 单词卡
---

# Flash Card 单词闪卡生成

为英语单词生成一张静态 HTML 学习卡片。卡片版面顺序：单词+音标 → 释义 → 近义词 → 3 条中英对照例句。

先确定本 SKILL.md 所在目录为 `{baseDir}`。工作产物写到 harness 的 `workspace/` 目录。

## 触发场景

- "给我做张 crazy 词的闪卡"
- "给我做 crazy 的 flash card"
- "做一个 resilient 的单词卡"

## 执行流程

1. **识别单词**：从用户话语中提取目标英语单词（小写化作为文件名）。

2. **生成 JSON 数据**：写出该单词的学习数据，用 `write_file` 保存到：
   - `{baseDir}/data/<word>.json`
   - 字段：`word` / `phonetic` / `pos` / `definition` / `examples`（恰好 3 条，含 `en`/`zh`）/ `synonyms`（4-6 个）

3. **生成 HTML**：用 `run_skill_script` 执行：
   ```bash
   python {baseDir}/scripts/make_flashcard.py {baseDir}/data/<word>.json -o workspace/<word>.html
   ```

4. **汇报结果**：告知用户 HTML 路径，并简要展示单词释义。

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

- 例句固定 3 条。
- HTML 输出到 `workspace/`，不要写到 skill 目录外的随意位置。
- 原始 JSON 放在 skill 的 `data/` 目录便于复用。
