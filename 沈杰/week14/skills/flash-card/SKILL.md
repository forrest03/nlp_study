---
name: flash-card
description: >-
  为英语单词生成含音标、词性、释义、3条中英例句、近义词的静态HTML闪卡。
  Use when user asks for flash card / 闪卡 / word card for an English word,
  e.g. "做张resilient的闪卡"、"crazy flash card"。
---

# Flash Card 闪卡生成

为英语单词生成静态HTML学习卡片。版面：单词+音标 → 释义 → 近义词 → 3条中英例句。

## 触发场景
- "给我做张 X 的闪卡" / "X flash card" / "X 单词卡"

## 执行流程

1. **识别单词**：提取目标词，小写化作为文件名。

2. **生成数据**：编写 JSON 保存到 `skills/flash-card/data/<word>.json`，字段：

   | 字段 | 说明 |
   |------|------|
   | `word` | 单词 |
   | `phonetic` | 音标，如 `/rɪˈzɪliənt/` |
   | `pos` | 词性，如 `adj.` |
   | `definition` | 中文释义 |
   | `examples` | 恰好3条，每条 `{en: 英文, zh: 中文}` |
   | `synonyms` | 近义词数组(4-6个) |

   例句要求地道、长度适中、体现典型用法；近义词贴近核心含义。

3. **生成HTML**：
   ```bash
   python skills/flash-card/scripts/make_flashcard.py skills/flash-card/data/<word>.json -o output/<word>.html
   ```

4. **打开预览**：浏览器打开生成的HTML。

## 注意
- 例句固定3条，生成时给齐
- 产物统一到 output/ 目录