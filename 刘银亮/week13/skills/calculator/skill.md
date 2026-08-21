---
name: calculator
description: 简单计算器，支持四则运算、幂运算和 math 模块函数
executor: script
script: calculator.py
parameters:
  - name: expr
    type: string
    description: 数学表达式，如 "(747 - 524) / 524 * 100" 或 "sqrt(144)"
    required: true
---

# 计算器技能

## 使用说明
当用户需要进行数学计算时使用此技能。
支持加减乘除、幂运算（**）、math 模块函数（sqrt/log/pow/sin/cos 等）。

## 示例
- "帮我算一下 123 * 456"
- "(100 - 30) / 7 是多少"
- "sqrt(144) 等于几"
- "2 的 10 次方是多少"
