# Skill 优化实验

## 目标

让模型将一份冗长的物流客服 Skill 优化为更短、决策顺序更明确的版本，同时不丢失政策事实。

## 版本

- `before/SKILL.md`：模型生成的完整说明版，包含重复解释和示例。
- `after/SKILL.md`：模型优化版，用一张配送表和三个有序决策分支表达相同政策。
- `requirements.json`：8 组不可丢失的原子政策要求，以及8道物流评测题。

优化只允许改变表达和结构，不允许改变政策数值。静态检查要求两个版本的原子规则覆盖率都为100%。

## 运行

离线比较：

```powershell
python experiments/skill_optimization/compare.py
```

真实模型 A/B：

```powershell
$env:DEEPSEEK_API_KEY = "..."
python experiments/skill_optimization/compare.py --llm
```

真实 A/B 对两个版本使用相同系统提示、模型参数和8道题，并记录规则准确率、服务端返回的 prompt/completion tokens 和总延迟。结果写入 `outputs/skill_optimization_comparison.json`。


## 本次结果

在8/8组原子政策要求均通过的前提下：

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| 字符数 | 1541 | 579 | -62.4% |
| 估算token | 1313 | 392 | -70.1% |
| 规则覆盖率 | 100% | 100% | 0 |
| 每千token覆盖规则数 | 6.09 | 20.41 | 约3.35倍 |

