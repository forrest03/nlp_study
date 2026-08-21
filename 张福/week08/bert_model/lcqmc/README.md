# LCQMC BiEncoder文本匹配模型

## 目录结构

```
bert_model/lcqmc/
├── dataset.py          # 数据处理模块
├── model.py            # BiEncoder模型定义
├── train.py            # 训练脚本
├── requirements.txt    # 依赖清单
└── README.md           # 操作说明
```

## 功能概述

针对LCQMC（Large-scale Chinese Question Matching Corpus）数据集的BiEncoder文本匹配模型，具有以下特性：

- **默认采用LoRA高效微调**：大幅减少可训练参数
- **数据随机采样**：支持按比例采样数据，默认使用30%数据
- **完整的训练流程**：训练、验证、测试一体化
- **丰富的输出**：详细的训练报告和预测结果

## 数据集说明

LCQMC是一个大规模中文问句匹配数据集：
- 训练集：238,766条
- 验证集：8,802条
- 测试集：12,500条

数据格式：
```json
{"sentence1": "文本1", "sentence2": "文本2", "label": 0或1}
```

- **label=1**: 句子相似（正确示例）
- **label=0**: 句子不相似（错误示例）

## LoRA高效微调

### 什么是LoRA？

LoRA (Low-Rank Adaptation) 是一种高效的微调方法：
- **大幅减少可训练参数**：从100%降至约0.1-1%
- **加速训练**：训练速度提升2-3倍
- **降低显存占用**：减少50%以上的GPU显存需求
- **保持模型性能**：与传统微调相当的性能

### LoRA工作原理

```
原始权重 W: [768 x 768] = 589,824 参数
LoRA分解: W + ΔW = W + BA
其中: B: [768 x r], A: [r x 768], r=8
LoRA参数: 768×8 + 8×768 = 12,288 参数
参数减少: 589,824 / 12,288 = 48倍！
```

### LoRA参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --lora_r | 8 | LoRA秩rank，越大表达能力越强，但参数越多 |
| --lora_alpha | 16 | 缩放因子，通常设为rank的2倍 |
| --lora_dropout | 0.1 | LoRA层的dropout比例 |

## 模型架构

```
Input1 → BERT+LoRA → Mean Pooling → L2 Norm → Embedding1
Input2 → BERT+LoRA → Mean Pooling → L2 Norm → Embedding2

Similarity = Cosine(Embedding1, Embedding2)
Prediction = 1 if Similarity > 0.5 else 0
```

## 使用方法

### 安装依赖

```bash
pip install torch transformers peft scikit-learn tqdm
```

### 基本训练（默认使用30%数据+LoRA）

```bash
cd bert_model/lcqmc
python train.py
```

### 使用完整数据集

```bash
python train.py --sample_ratio 1.0
```

### 使用50%数据训练

```bash
python train.py --sample_ratio 0.5
```

### 自定义LoRA参数

```bash
python train.py \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1
```

### 禁用LoRA（全参数微调）

```bash
python train.py --use_lora False
```

## 命令行参数

### 数据参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --max_len | int | 64 | 最大序列长度 |
| --sample_ratio | float | 0.3 | 数据集随机采样比例（0.0-1.0），默认0.3即使用30%数据 |
| --seed | int | 42 | 随机种子，用于保证实验可复现性 |

### 模型参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --bert_model | str | bert-base-chinese | 预训练模型 |
| --pool | str | mean | 池化方式（cls/mean/max） |
| --dropout | float | 0.1 | Dropout比例 |
| --num_hidden_layers | int | None | BERT层数 |

### LoRA参数（默认启用）
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --use_lora | bool | True | **启用LoRA微调** |
| --lora_r | int | 8 | LoRA秩rank |
| --lora_alpha | int | 16 | LoRA alpha参数 |
| --lora_dropout | float | 0.1 | LoRA dropout |

### 训练参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --batch_size | int | 32 | 批处理大小 |
| --epochs | int | 1 | 训练轮数 |
| --learning_rate | float | 2e-5 | 学习率 |
| --margin | float | 0.3 | 余弦损失margin |
| --weight_decay | float | 0.01 | 权重衰减 |

## 输出文件

训练完成后在 `outputs/` 目录生成：

| 文件 | 说明 |
|------|------|
| lora_adapter/ | LoRA适配器权重目录 |
| training_metadata.pt | 训练元数据 |
| test_results.json | 测试集预测结果 |
| training_history.json | 训练历史记录 |
| config.json | 训练配置参数 |
| training_report.txt | 详细训练报告 |

## 训练流程

1. **数据加载**: 读取train.jsonl, validation.jsonl, test.jsonl
2. **数据采样**: 根据sample_ratio随机采样指定比例的数据（默认30%）
3. **模型初始化**: 创建BiEncoder模型，应用LoRA配置
4. **参数统计**: 显示可训练参数比例
5. **训练循环**: 只更新LoRA参数，其他参数冻结
6. **模型验证**: 每个epoch后在验证集上评估
7. **模型保存**: 保存LoRA适配器权重
8. **测试评估**: 使用最优模型在测试集上评估
9. **结果保存**: 保存预测结果和报告

## 评估指标

- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1 Score**: F1分数

## 预测结果格式

测试结果保存在 `test_results.json`，每条记录包含：

```json
{
  "similarity": 0.9567,
  "prediction": 1,
  "label": 1,
  "prediction_label": "相似",
  "true_label": "相似",
  "correct": true
}
```

## LoRA优势对比

| 对比项 | 全参数微调 | LoRA微调 |
|--------|-----------|----------|
| 可训练参数 | 100% | ~0.1-1% |
| GPU显存占用 | 高 | 降低50%+ |
| 训练时间 | 长 | 缩短2-3倍 |
| 模型性能 | 好 | 相当 |
| 存储空间 | 大 | 小（几MB） |

## 注意事项

1. **GPU支持**: 代码自动检测CUDA，优先使用GPU
2. **首次运行**: 会自动下载预训练BERT模型
3. **LoRA默认启用**: 默认使用LoRA高效微调
4. **参数调整**: 根据任务复杂度调整lora_r和lora_alpha

## 依赖环境

- Python 3.7+
- PyTorch >= 1.9.0
- Transformers >= 4.20.0
- **PEFT >= 0.4.0** (LoRA支持)
- scikit-learn >= 0.24.0
- tqdm >= 4.62.0

## 快速开始

```bash
# 进入项目目录
cd bert_model/lcqmc

# 安装依赖
pip install -r requirements.txt

# 运行训练（默认使用30%数据+LoRA）
python train.py

# 使用50%数据训练
python train.py --sample_ratio 0.5

# 使用完整数据集
python train.py --sample_ratio 1.0

# 查看结果
cat outputs/training_report.txt
```

## 示例输出

使用采样功能后会显示：

```
加载数据集...
随机采样数据集 (采样比例: 0.3)...
采样后训练集: 71,630条 (原始: 238,766条)
采样后验证集: 2,641条 (原始: 8,802条)
采样后测试集: 3,750条 (原始: 12,500条)

创建BiEncoder模型...
LoRA配置: r=8, alpha=16, dropout=0.1
可训练参数: 2,987,520 / 102,267,648 (0.29%)
```

训练完成后会输出：

```
训练完成！
最优验证F1: 0.8523
测试集F1: 0.8456
所有结果保存在: outputs/
```