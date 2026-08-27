# 用 GRPO 提升模型数学解题能力

这是一个可复现的最小实验仓库：用 TRL 的 GRPOTrainer 在 GSM8K 训练集上对数学指令模型进行 LoRA 强化学习，并在官方测试集上与**同一个基础模型**做逐题配对比较。

默认组合：

- 基础模型：`Qwen/Qwen2.5-Math-1.5B-Instruct`
- 数据集：`openai/gsm8k`（训练集再固定划出 256 题作为验证集）
- 奖励：答案精确正确 `1.0` + 严格 `\boxed{}` 格式 `0.1`
- 训练：LoRA + GRPO，默认 4 个候选回答、200 个优化步
- 主指标：GSM8K test exact match；同时报告配对 bootstrap 95% CI 和 McNemar 精确检验

## 1. 环境

推荐 Python 3.11 或 3.12，并使用带 CUDA 的 NVIDIA GPU。先按 [PyTorch 官方安装器](https://pytorch.org/get-started/locally/)安装与机器 CUDA 匹配的 PyTorch，再安装本项目：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

首次运行会从 Hugging Face 下载模型和数据。GSM8K 官方卡片显示训练集 7,473 条、测试集 1,319 条，许可证为 MIT；本项目不把测试集用于训练或超参数选择。

## 2. 先跑小规模冒烟实验

先用 128 条训练样本和 50 个测试题验证环境、显存与数据链路：

```powershell
math-grpo-eval `
  --model Qwen/Qwen2.5-Math-1.5B-Instruct `
  --output-dir results/smoke-base `
  --limit 50

math-grpo-train `
  --model Qwen/Qwen2.5-Math-1.5B-Instruct `
  --output-dir outputs/smoke-grpo `
  --train-limit 128 `
  --validation-limit 16 `
  --max-steps 5

math-grpo-eval `
  --model Qwen/Qwen2.5-Math-1.5B-Instruct `
  --adapter outputs/smoke-grpo `
  --output-dir results/smoke-grpo `
  --limit 50

math-grpo-compare `
  --baseline results/smoke-base/predictions.jsonl `
  --candidate results/smoke-grpo/predictions.jsonl `
  --output results/smoke-comparison.json
```

冒烟结果不能用于判断是否有效；它只证明代码能够完整运行。

### 本机 4GB 显存配置

当前机器检测到的是 RTX 3050 Ti Laptop（4GB）。先安装 QLoRA 依赖：

```powershell
pip install -e ".[dev,qlora]"
```

然后用更小模型和更短输出做链路验证：

```powershell
math-grpo-train `
  --model Qwen/Qwen2.5-Math-0.5B-Instruct `
  --output-dir outputs/4gb-smoke-grpo `
  --load-in-4bit `
  --num-generations 2 `
  --gradient-accumulation-steps 2 `
  --max-completion-length 192 `
  --train-limit 64 `
  --validation-limit 8 `
  --max-steps 3
```

评测 Base 与 adapter 时都使用同一个 `--load-in-4bit` 选项，保证量化条件一致。4GB 是否能稳定运行还取决于 Windows CUDA、bitsandbytes 版本和其他显存占用；若仍 OOM，应转到显存更大的 Linux/云端 GPU，而不是把正式实验缩小到失去统计意义。

## 3. 正式实验

先对完整测试集记录训练前基线：

```powershell
math-grpo-eval `
  --model Qwen/Qwen2.5-Math-1.5B-Instruct `
  --output-dir results/seed-42/base
```

训练并评测 LoRA adapter：

```powershell
math-grpo-train `
  --model Qwen/Qwen2.5-Math-1.5B-Instruct `
  --output-dir outputs/seed-42-grpo `
  --seed 42 `
  --max-steps 200

math-grpo-eval `
  --model Qwen/Qwen2.5-Math-1.5B-Instruct `
  --adapter outputs/seed-42-grpo `
  --output-dir results/seed-42/grpo

math-grpo-compare `
  --baseline results/seed-42/base/predictions.jsonl `
  --candidate results/seed-42/grpo/predictions.jsonl `
  --output results/seed-42/comparison.json
```

建议再以 `--seed 43`、`--seed 44` 重复训练。正式结论至少应同时满足：平均 exact match 上升、三次种子方向基本一致、配对区间不跨 0，并且无效答案率和回答长度没有异常恶化。

## 4. 显存不足时

按顺序减小：

1. `--max-completion-length 256`
2. 启用 `--load-in-4bit`（需安装 `.[qlora]`）
3. 改用 `Qwen/Qwen2.5-Math-0.5B-Instruct`
4. 保持有效批量可整除：例如把 `--num-generations 2` 与 `--gradient-accumulation-steps 4` 配合
5. 减少验证题：`--validation-limit 16`

`num_generations` 必须整除 `WORLD_SIZE × per_device_batch_size × gradient_accumulation_steps`，训练脚本会在下载模型前检查这一点。
