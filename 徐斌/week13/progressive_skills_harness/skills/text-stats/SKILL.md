---
name: text-stats
description: >-
  统计一段文本的字数、词数、行数、中英字符占比，并输出 JSON / Markdown 报告。
  Use when the user asks for word count / 字数统计 / 文本统计 / text stats.
triggers:
  - 字数
  - 词数
  - 文本统计
  - word count
  - text stats
---

# Text Stats 文本统计

对用户提供的文本做基础统计，适合演示「代码工具型 Skill」：Skill 描述行为，真正计算由脚本完成。

先确定本 SKILL.md 所在目录为 `{baseDir}`。

## 执行流程

1. 从用户消息提取待统计文本；若文本较长，先用 `write_file` 写入 `workspace/input.txt`。
2. 调用脚本：
   ```bash
   python {baseDir}/scripts/analyze.py --text "..." -o workspace/text_stats.json
   # 或
   python {baseDir}/scripts/analyze.py --file workspace/input.txt -o workspace/text_stats.json
   ```
3. 读取 JSON 结果，用中文向用户汇报关键指标。

## 输出字段

- `chars`：总字符数（含空白）
- `chars_no_space`：去空白字符数
- `words`：英文词数（空白分词）
- `cjk_chars`：中日韩统一表意文字数量
- `lines`：行数
- `summary`：一句话摘要

## 自检

- [ ] 空文本时脚本返回明确错误而非崩溃
- [ ] 报告文件已写入 `workspace/`
