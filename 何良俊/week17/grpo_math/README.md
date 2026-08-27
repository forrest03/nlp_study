# grpo_math — 基于 GRPO 强化学习提升模型数学能力

一个精简但功能完整的 GRPO（Group Relative Policy Optimization）训练示例：用强化学习让
**Qwen3.5-0.8B** 学会做多位数乘法（三位数×两位数、三位数×三位数，基座模型实测准确率
仅 ~70% / ~20%），训练后可通过 `--test` 一键自动化评估，结果自动保存为 JSON。

参考项目 `grpo_arithmetic`（Qwen2-0.5B，6 个难度级别、基线摸底、对比分析、曲线绘图），
本项目做了精简：**4 个难度级别、单入口命令、无基线摸底/绘图**，但训练与评估的核心链路完整保留。

## 目录结构

```
grpo_math/
├── main.py              # 统一入口：--train 训练 / --test 测试
├── requirements.txt
├── src/
│   ├── data.py          # 题目生成、输出解析、Prompt 构造（训练/测试共用）
│   ├── train.py         # GRPO 训练（TRL 实现，默认 LoRA）
│   └── test.py          # 自动化测试，结果保存为 JSON
└── outputs/             # 运行时生成：checkpoint、train_log、test_result
```

## 原理简述

GRPO 是 PPO 的轻量变体，专为 LLM 强化学习设计：

1. **无价值网络、无奖励模型**：对每个 prompt 采样 K 条（本工程 K=8）回答，用组内奖励的
   均值/标准差归一化得到 advantage，奖励高的回答被强化、低的被抑制。
2. **KL 系数 β=0**：不加载参考模型（参考模型用于约束策略不漂移，β=0 表示完全信任奖励信号），
   省 1GB 显存。
3. **复合奖励塑形**：正确分 1.0 + 格式分 0.2。格式分让模型先学会输出 `<answer>` 标签
   的结构，正确分再驱动算对结果 —— 训练日志中能看到"格式先收敛、正确率后爬坡"的典型动态。
4. **甜区选题**：训练集只含两位数/三位数加减法。太简单（全对）或太难（全错）的组
   advantage 恒为 0，纯浪费算力；只有"组内有对有错"的组才产生梯度（informative group）。

### 为什么用强化学习而不是 SFT？

SFT 只能教模型"模仿"正确答案的写法；RL 则让模型在**自己的采样分布**上收到**正确/错误的
直接信号**，通过试错学会泛化 —— 这正是"做对题"而非"背答案"的差别。

## 环境准备

```bash
pip install -r requirements.txt
```

已验证环境：Python 3.13 + torch 2.9 (CUDA) + transformers 5.9 + trl 1.12，RTX 4060 Laptop (8GB)。

## 使用方式

### 1. 训练（GRPO）

```bash
python main.py --train                          # 默认 LoRA 训练 100 步
python main.py --train --max_steps 3            # 冒烟测试：验证流程与显存
python main.py --train --n_prompts 1000         # 更大训练集
python main.py --train --full                   # 全量微调（需 >12GB 显存）
python main.py --train --tag exp1               # 输出目录加后缀，区分实验
```

**为什么默认 LoRA？** Qwen3.5-0.8B 全量微调的 AdamW 状态约 9GB，8GB 显存必 OOM。
LoRA(r=16, α=32) 只训练注意力投影的一小部分参数，8GB 显存轻松运行。

**训练时长估算**：`总时长 ≈ max_steps × 每步耗时（约 4~15 秒）`。每步耗时受
`generation_batch_size` 影响 —— 该参数限制采样生成的并行条数，默认 32（8 prompt × 8 采样
分两批生成），避免峰值显存超过 8GB 物理显存（溢出到共享显存会导致速度暴跌）。

**学习率注意**：LoRA 默认 `lr=5e-5`。实测 `lr=2e-4` 会把模型已会的简单题训崩
（正确率 100%→20%，典型 RL 灾难性遗忘），因此刻意调低、小步慢走。

### 2. 自动化测试

```bash
python main.py --test                           # 自动选最新 checkpoint（无则测基座）
python main.py --test --quick                   # 快速模式（每难度 5 题）
python main.py --test --model outputs/grpo_lora_ckpt
python main.py --test --n 50 --seed 99          # 更大评估集，换种子避免与训练集重叠
```

测试结果自动保存到 `outputs/test_result.json`，包含每个难度的完整指标与样例输出。

## 指标说明

| 指标 | 含义 |
|---|---|
| `greedy_format_rate` | 确定性解码下输出包含 `<answer>` 标签的比例 |
| `greedy_strict_acc` | greedy 下 `<answer>` 标签内数字正确的比例 |
| `greedy_loose_acc` | greedy 下"输出最后一个数字正确"的比例（宽松口径） |
| `sample_loose_acc` | 温度采样 K 条后逐条统计的宽松正确率 |
| `pass@k` | 每组 K 条中至少一条宽松正确的比例 |
| `informative_group_rate` | 0<正确数<K 的组占比，衡量 GRPO 训练燃料 |

> 正确分奖励使用**宽松口径**（最后一个数字正确即得分），避免格式冷启动阶段正确信号也为 0。

## 难度与训练/测试划分

难度基于 **基座模型实测摸底** 设计（Qwen3.5-0.8B greedy 准确率）：

| 级别 | 内容 | 基座准确率 | 是否在训练集 |
|---|---|---|---|
| L1_add_2digit | 两位数加减 | ~100% | 否（sanity check） |
| L2_mul_2x1digit | 两位数×一位数 | ~100% | 否 |
| L3_mul_2x2digit | 两位数×两位数 | ~90% | 否 |
| L4_mul_3x2digit | 三位数×两位数 | ~70% | **是**（甜区） |
| L5_mul_3x3digit | 三位数×三位数 | ~20% | **是**（甜区，提升空间最大） |

> GRPO 只对"组内有对有错"的 prompt 产生梯度：太简单（全对）或太难（全错）的组
> advantage 恒为 0，纯浪费算力。训练集只保留基座 20%~70% 正确率的甜区难度；
> 简单题（L1~L3）留作测试，用于观察 RL 训练是否导致能力退化。

## 实测效果（本机 RTX 4060 Laptop 8GB）

训练配置：LoRA r=16 / lr=5e-5 / 50 步（约 17 分钟），评估为 greedy 宽松正确率。

| 级别 | 基座（5 题样本） | 训练后（30 题样本） | 说明 |
|---|---|---|---|
| L1_add_2digit | 100% | 100% | 简单题不退化 ✓ |
| L2_mul_2x1digit | 100% | 100% | 简单题不退化 ✓ |
| L3_mul_2x2digit | ~100% | 73% | 训练集外，样本波动（5题 vs 30题） |
| L4_mul_3x2digit | 60~70% | **87%** | **训练甜区，提升明显** ✓ |
| L5_mul_3x3digit | 20~40% | 23% | 最难题，50 步训练提升有限 |

格式遵循率（输出 `<answer>` 标签）训练前后均为 100%。训练日志显示正确分奖励从 0.19
爬升到 0.3~0.5 区间、策略熵从 0.54 收敛到 0.11 —— RL 确实在起作用。

**说明**：基座行是 5 题小样本（波动大），训练后行是 30 题。步数越多（如 `--max_steps 200`）、
每步训练量越大，L5 这类难题的提升会越明显。

## 训练日志解读

`outputs/train_log_lora.json` 中每个 step 记录：
- `rewards/reward_correct/mean`、`rewards/reward_format/mean`：两条奖励曲线
- `entropy`：策略熵（下降说明策略收敛）
- `frac_reward_zero_std`：组内奖励标准差为 0 的退化组比例

## 常见问题

- **训练要多久？有进度条吗？** 训练终端会显示 tqdm 进度条（步数/总步数 + 每步耗时），
  每 5 步打印一条奖励指标。总时长 ≈ `max_steps × 每步耗时`，本机 50 步约 17 分钟，
  每步 10~20 秒。想快一点就调小 `--max_steps`（如 25 步约 8 分钟）。
- **训练好的模型保存在哪？** `outputs/grpo_lora_ckpt/`（LoRA adapter + tokenizer + 训练配置），
  约 22MB；`--test` 会自动优先加载它。全量模式保存在 `outputs/grpo_ckpt/`。
- **显存不足**：默认 LoRA 已是最省显存配置；若仍不足可减小 `per_device_train_batch_size` 或
  `gradient_accumulation_steps`（在 `src/train.py` 中调整），或把 `generation_batch_size`
  调小到 16。
- **模型输出带 `<think>` 标签**：本工程已在训练与测试中统一设置 `enable_thinking=False`
  关闭思考模式，输出更短更直接。
- **训练后简单题反而变差（灾难性遗忘）**：把学习率调低（`--lr 5e-5` 以下）。实测
  `lr=2e-4` 会把已会的题训崩（100%→20%）。
