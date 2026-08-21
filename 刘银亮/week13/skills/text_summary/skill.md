---
name: text_summary
description: 文本摘要工具，对输入文本生成简洁摘要
executor: script
script: text_summary.py
parameters:
  - name: text
    type: string
    description: 待摘要的文本内容
    required: true
  - name: max_words
    type: integer
    description: 摘要最大字数
    required: false
    default: 100
---

# 文本摘要技能

## 使用说明
当用户需要对一段文本进行摘要、提取要点时使用此技能。

## 示例
- "帮我总结一下这段文字：..."
- "提取这段内容的关键信息"
