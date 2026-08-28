# 实验方案

## 研究问题

在模型、提示模板和解码策略固定的条件下，使用可验证最终答案作为奖励进行 GRPO LoRA 训练，能否提升 GSM8K 未见测试题的 exact match？

主要假设：GRPO 后测试集 exact match 高于基础模型。零假设：逐题正确率差值为 0。

## 数据边界

- `GSM8K train`：GRPO 训练。
- 从 train 固定划出的 256 题：观察训练期间的泛化奖励、选择训练步数。
- `GSM8K test`：只在实验配置冻结后运行完整评测；不能据此调奖励、提示或步数。

数据准备在划分后才限量，测试题带稳定的 `row_id`，因此两次评测可以逐题配对。

## 变量与控制

自变量是是否加载 GRPO LoRA adapter。两组必须使用同一基础模型 revision、同一系统提示、同一 tokenizer、相同 `max_new_tokens` 和 greedy decoding。

优先做以下实验矩阵：

| 组别 | 正确性奖励 | 格式奖励 | 用途 |
|---|---:|---:|---|
| Base | 无 | 无 | 未训练基线 |
| GRPO-main | 1.0 | 0.1 | 主实验 |
| GRPO-answer-only | 1.0 | 0.0 | 判断格式奖励的影响 |
| GRPO-format-only | 0.0 | 0.1 | 奖励投机负对照，不作为候选模型 |

使用 `--correctness-weight` 和 `--format-weight` 运行消融；每次运行的实际参数都会随模型保存。

## 指标

主要指标：

- test exact match
- 相对 Base 的逐题绝对提升
- 配对 bootstrap 95% 置信区间
- McNemar 精确检验 p 值

诊断指标：

- boxed-answer rate
- invalid-answer rate
- mean completion characters
- 训练日志中的 reward、reward standard deviation、KL、completion length

若 reward 持续上升但验证 exact match 不升，首先检查格式奖励占比、答案解析漏洞和生成截断，而不是继续增加训练步数。

## 判定标准

把“有提升”预注册为：

1. 三个训练随机种子的平均 test exact match 高于 Base；
2. 每个种子的方向至少不显著为负；
3. 合并的逐题 paired bootstrap 95% 区间下界大于 0；
4. invalid-answer rate 不增加超过 1 个百分点；
5. 人工检查至少 30 个由错变对和全部由对变错样本，没有明显解析器误判。

如果未达到标准，也应保留结果。负结果可能来自基础模型在 GSM8K 上已接近饱和、组内奖励方差过低、生成长度不足或训练样本规模过小。

## 复现实验记录

每次训练目录自动写入 `run_config.json`，包含参数、数据条数、平台及核心包版本。另需记录：

- GPU 型号、数量和驱动版本
- 基础模型的精确 revision/commit
- 完整命令行
- wall-clock 时间与峰值显存
- baseline、adapter 评测目录和 comparison JSON
