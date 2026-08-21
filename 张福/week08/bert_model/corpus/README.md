# BiEncoder文本匹配模型 - 操作说明

## 目录结构

```
bert_model/corpus/
├── dataset.py          # 数据处理模块
├── model.py            # BiEncoder模型定义
├── train.py            # 训练脚本
└── README.md           # 操作说明文档

outputs/                # 训练输出目录
├── best_model.pt       # 最优模型
├── test_results.json   # 测试结果
├── training_history.json # 训练历史
└── config.json         # 配置文件
```

## 功能概述

BiEncoder是一种表示型文本匹配模型，使用Siamese Network架构：
- 两个共享参数的BERT编码器分别对句子对进行编码
- 通过余弦相似度计算匹配程度
- 支持序列标注任务的训练、验证和测试

## 数据格式

输入数据应为JSONL格式，每行包含：
```json
{"sentence1": "文本1", "sentence2": "文本2", "label": 0或1}
```

- `sentence1`: 第一个句子
- `sentence2`: 第二个句子
- `label`: 标签，1表示相似/正确示例，0表示不相似/错误示例

## 使用方法

### 1. 基本训练

```bash
cd bert_model/corpus
python train.py --data_dir ../../data/bq_corpus
```

### 2. 自定义参数训练

```bash
python train.py \
  --data_dir ../../data/bq_corpus \
  --max_len 64 \
  --batch_size 32 \
  --epochs 5 \
  --learning_rate 2e-5 \
  --pool mean \
  --dropout 0.1
```

### 3. 限制BERT层数加速训练

```bash
python train.py \
  --data_dir ../../data/bq_corpus \
  --num_hidden_layers 4 \
  --epochs 3 \
  --batch_size 64
```

### 4. 使用不同池化方式

```bash
# CLS池化
python train.py --data_dir ../../data/bq_corpus --pool cls

# Mean池化
python train.py --data_dir ../../data/bq_corpus --pool mean

# Max池化
python train.py --data_dir ../../data/bq_corpus --pool max
```

## 可配置参数

### 数据参数
- `--data_dir`: 数据目录路径（默认：../../data/bq_corpus）
- `--max_len`: 最大序列长度（默认：64）

### 模型参数
- `--bert_model`: 预训练BERT模型路径（默认：../../pretrain_models/bert-base-chinese）
- `--pool`: 池化方式，可选cls/mean/max（默认：mean）
- `--dropout`: Dropout比例（默认：0.1）
- `--num_hidden_layers`: BERT层数，None表示使用全部12层（默认：None）

### 训练参数
- `--batch_size`: 批处理大小（默认：32）
- `--epochs`: 训练轮数（默认：3）
- `--learning_rate`: 学习率（默认：2e-5）
- `--margin`: 余弦损失margin值（默认：0.3）
- `--weight_decay`: 权重衰减（默认：0.01）

### 输出参数
- `--output_dir`: 输出目录（默认：../../outputs）

## 输出文件说明

### 1. best_model.pt
最优模型检查点，包含：
- 模型权重
- 优化器状态
- 训练epoch
- 验证F1分数
- 训练参数

### 2. test_results.json
测试结果，包含：
- 测试指标（accuracy, precision, recall, f1）
- 所有预测结果
- 相似度分布统计

### 3. training_history.json
训练历史，包含：
- 每个epoch的训练loss和accuracy
- 每个epoch的验证指标
- 最优验证F1分数
- 测试指标

### 4. config.json
训练配置，记录所有命令行参数

## 训练过程

训练过程包括：
1. **数据加载**：从指定目录读取train.jsonl, validation.jsonl, test.jsonl
2. **模型训练**：按epoch进行训练
3. **模型验证**：每个epoch后验证
4. **模型选择**：保存验证集上F1最优的模型
5. **测试评估**：使用最优模型在测试集上评估
6. **结果保存**：保存所有结果和日志

## 评估指标

模型使用以下指标进行评估：
- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1**: F1分数

## 注意事项

1. **GPU支持**：代码会自动检测CUDA，若有GPU则自动使用
2. **数据格式**：确保数据为标准JSONL格式
3. **路径问题**：使用相对路径时注意当前工作目录
4. **内存占用**：较大的batch_size和max_len会占用更多显存
5. **训练时间**：使用全部12层BERT时训练时间较长，可使用num_hidden_layers参数加速

## 快速开始示例

```bash
# 1. 进入项目目录
cd d:\workspace\hub-TroE\张福\week08\bert_model\corpus

# 2. 使用默认参数训练
python train.py --data_dir ../../data/bq_corpus

# 3. 快速训练（4层BERT）
python train.py --data_dir ../../data/bq_corpus --num_hidden_layers 4 --epochs 2

# 4. 查看结果
cat ../../outputs/test_results.json
cat ../../outputs/training_history.json
```

## 模型架构

BiEncoder模型结构：
```
Input1 → BERT Encoder → Pooling → L2 Norm → Embedding1
Input2 → BERT Encoder → Pooling → L2 Norm → Embedding2

Similarity = Cosine(Embedding1, Embedding2)
Prediction = 1 if Similarity > 0.5 else 0
```

## 损失函数

使用CosineEmbeddingLoss：
- 正例对（label=1）：损失 = 1 - similarity
- 负例对（label=0）：损失 = max(0, similarity - margin)

总损失 = 所有样本损失的平均值

## 常见问题

**Q: 训练很慢怎么办？**
A: 使用`--num_hidden_layers 4`限制BERT层数，或增大batch_size

**Q: 如何调整margin值？**
A: margin越大，对负例对的容忍度越高，通过`--margin`参数调整

**Q: 如何选择池化方式？**
A: mean池化在句子相似度任务上通常表现最好，但可根据具体任务尝试不同方式

**Q: 模型预测结果不理想怎么办？**
A: 尝试增加训练轮数、调整学习率、或使用不同的pooling方式