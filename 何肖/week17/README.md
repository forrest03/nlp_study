# Week17 · 基于 GRPO 的强化学习提升模型做数学题能力

## 作业目标

使用 **GRPO（Group Relative Policy Optimization）** 对 `Qwen2-0.5B-Instruct` 做规则奖励 RL 微调，提升小模型的**算术解题能力**，并学会按指令输出 `<answer>数字</answer>` 格式。

本作业不依赖奖励模型或人工标注，奖励由程序根据标准答案自动判定，完整流程为：

```text
基线摸底 → GRPO 训练 → 训练后评估 → 前后对比
```

---

## 1. 项目结构

```text
grpo_arithmetic/
├── src/
│   ├── probe_baseline.py       # 基线摸底 / 训练后评估
│   ├── train_grpo.py           # GRPO 训练（TRL GRPOTrainer）
│   ├── compare_results.py      # 前后对比 + 训练曲线
│   ├── test_general_dialog.py  # 通用对话能力抽检（副作用观察）
│   ├── trl_compat.py           # trl 0.21 + transformers 5.x 兼容补丁
│   └── model_paths.py          # 自动解析基座模型路径
├── outputs/
│   ├── baseline_probe.json         # 基线评估结果
│   ├── post_train_probe.json       # 全量微调后评估
│   ├── post_train_probe_lora.json  # LoRA 微调后评估
│   ├── train_log.json              # 全量训练日志
│   ├── train_log_lora.json         # LoRA 训练日志
│   └── figures/train_curves.png    # 训练曲线
├── requirements.txt
└── README.md
```

> 本次提交已包含完整实验 JSON 与训练曲线。模型 checkpoint 体积较大，未随作业提交；需要复现训练时在 GPU 环境按下方流程执行 Step 2～3 即可。

---

## 2. 环境准备

```bash
cd grpo_arithmetic
pip install -r requirements.txt

# 默认模型路径（相对八斗根目录）：
#   八斗/pretrain_models/Qwen2-0.5B-Instruct
# 如需覆盖，可设置环境变量：
# export GRPO_MODEL_PATH="/Users/hattyhatty288/Documents/Hatty/八斗/pretrain_models/Qwen2-0.5B-Instruct"
```

| 依赖 | 用途 |
|------|------|
| torch + transformers | 模型加载与推理 |
| trl | GRPOTrainer |
| peft | 可选，`--lora` 省显存 |
| matplotlib | 训练曲线绘图 |

> 训练需要 **GPU**（推荐 8GB+ 显存）。显存不足可使用 `--lora`。基座目录需含 `model.safetensors` 或 `pytorch_model.bin`。

---

## 3. 运行流程

### Step 1：基线摸底

```bash
python src/probe_baseline.py              # 6 难度 × 50 题
python src/probe_baseline.py --quick      # 快速验证（每难度 10 题）
```

输出：`outputs/baseline_probe.json`

关注指标：**greedy 正确率、格式遵循率、pass@8、informative group rate**（组内有对有错的比例，决定 GRPO 能否产生有效梯度）。

### Step 2：GRPO 训练

```bash
python src/train_grpo.py                       # 全量微调，200 步
python src/train_grpo.py --max_steps 3 --tag smoke   # 冒烟测试
python src/train_grpo.py --lora                # LoRA 降级（省显存）
```

输出：`outputs/grpo_ckpt/` + `outputs/train_log.json`（LoRA 对应 `grpo_lora_ckpt/` + `train_log_lora.json`）

### Step 3：训练后评估

```bash
python src/probe_baseline.py --model outputs/grpo_ckpt \
  --out outputs/post_train_probe.json --seed 42

python src/probe_baseline.py --model outputs/grpo_lora_ckpt \
  --out outputs/post_train_probe_lora.json --seed 42
```

> `--seed 42` 必须与基线一致，保证评估题为同一套，前后可配对比较。

### Step 4：对比分析

```bash
python src/compare_results.py
```

打印基线 / 全量 / LoRA 三方对比表，并生成 `outputs/figures/train_curves.png`。

**无需 GPU 即可查看已有实验结果**：本目录已附带 JSON，直接运行 Step 4 即可复现对比表与曲线图。

---

## 4. 核心设计

### 整体流程

```text
程序化生成算术题（6 个难度级别）
    ↓
基线 probe：greedy + 温度采样 K=8，统计正确率 / 格式率 / pass@8
    ↓
按 informative group rate 选取 L2/L3/L5 组成训练集
    ↓
GRPO：每题采样 8 条 completion，组内相对优势更新策略
    ↓
同一评估集（seed=42）复测，对比训练前后
```

### 奖励函数

| 奖励 | 权重 | 判定 |
|------|------|------|
| `reward_correct` | 1.0 | 答案正确（宽松解析：取最后一个数字） |
| `reward_format` | 0.2 | 输出含 `<answer>N</answer>` |

### 训练集难度配比

| 难度 | 占比 | 基线 informative | 说明 |
|------|------|:---:|------|
| L3 三位数加减 | 50% | 0.76 | 主训练区间 |
| L5 两位×一位 | 25% | 0.66 | 乘法入门 |
| L2 两位数加减 | 25% | 0.68 | 基础巩固 |

L1 / L4 / L6 不进训练集，留作**泛化评估**（检验是否背题 vs 真正提升算术能力）。

### 关键超参

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_generations` | 8 | 组内采样数 K |
| `beta` | 0.0 | 不加载参考模型，省显存 |
| `temperature` | 1.0 | 保持组内多样性 |
| `learning_rate` | 2e-6（全量）/ 2e-4（LoRA） | |

---

## 5. 实验结果

**实验环境**：RTX 4060 Laptop（8GB 显存），全量微调峰值约 6GB，LoRA 约 3GB。  
**评估设置**：seed=42，6 难度 × 50 题，指标为 格式率 / greedy 正确率 / pass@8。

| 难度 | 训练集 | 基线 | 全量微调 | LoRA |
|------|:---:|---|---|---|
| L1 个位数加法 | — | 0.00 / 0.98 / 1.00 | 0.96 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| L2 两位数加减 | √ | 0.02 / 0.76 / 0.96 | 1.00 / 0.98 / 1.00 | 1.00 / 0.98 / 0.98 |
| L3 三位数加减 | √ | 0.02 / 0.40 / 0.76 | 0.96 / 0.90 / 0.94 | 1.00 / 0.92 / 0.94 |
| L4 表内乘法 | — | 0.00 / 0.56 / 0.88 | 0.82 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| L5 两位×一位 | √ | 0.00 / 0.20 / 0.66 | 0.98 / 0.88 / 0.98 | 1.00 / 0.94 / 1.00 |
| L6 两位×两位 | — | 0.04 / 0.08 / 0.24 | 0.98 / 0.24 / 0.38 | 1.00 / 0.28 / 0.40 |

### 典型样例（训练前 → 训练后）

```text
902 - 848 = 54
  前: '154'
  后: '<answer>54</answer>'

28 × 8 = 224
  前: '112'
  后: '<answer>224</answer>'
```

### 结论

1. **格式学习效果显著**：基线几乎不遵循 `<answer>` 格式（格式率≈0），训练后达 0.82～1.00，且泛化到未训练难度。
2. **训练集内正确率大幅提升**：L5 从 0.20 → 0.88（全量）/ 0.94（LoRA）。
3. **未训练难度也有提升**：L4 正确率 0.56 → 1.00，说明 RL 在强化算术能力而非单纯背题。
4. **超出能力边界的 L6 提升有限**：0.08 → 0.24～0.28；RL 在能力边界内重排概率，无法凭空创造新能力。
5. **LoRA 显存更省且效果略优**，但策略熵更低，长训练需注意探索枯竭。

训练曲线：`outputs/figures/train_curves.png`

---

## 6. 已提交产物

可直接查看的实验文件：

| 文件 | 说明 |
|------|------|
| `outputs/baseline_probe.json` | 基线 6 难度评估 |
| `outputs/post_train_probe.json` | 全量微调后评估 |
| `outputs/post_train_probe_lora.json` | LoRA 微调后评估 |
| `outputs/train_log.json` | 全量训练过程指标 |
| `outputs/train_log_lora.json` | LoRA 训练过程指标 |
| `outputs/figures/train_curves.png` | 奖励 / 熵 / clip 比例曲线 |

---

## 7. 常见问题

| 问题 | 处理 |
|------|------|
| `import vllm` 报错 | 各脚本首行已 `import trl_compat`，从项目根目录运行 |
| 训练奖励全 0 / 乱码 | 不要开 `gradient_checkpointing`；改用 `--lora` |
| 一步训练后权重 NaN | 确认 `model_init_kwargs={"torch_dtype": "bfloat16"}` |
| CUDA OOM | `python src/train_grpo.py --lora` |
| 找不到基座模型 | 确认 `八斗/pretrain_models/Qwen2-0.5B-Instruct` 存在且含权重文件；或设置 `GRPO_MODEL_PATH` |
