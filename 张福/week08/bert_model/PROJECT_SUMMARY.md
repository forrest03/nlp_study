# BiEncoder文本匹配模型项目总结

## 项目概述

本项目实现了一个基于BERT的BiEncoder文本匹配模型，用于判断两个句子是否相似或匹配。

## 目录结构

```
bert_model/corpus/
├── dataset.py          # 数据处理模块
├── model.py            # BiEncoder模型定义
├── train.py            # 训练脚本
├── test_setup.py       # 代码测试脚本
├── requirements.txt    # 依赖列表
└── README.md           # 操作说明文档

outputs/                # 训练输出目录（训练后生成）
├── best_model.pt       # 最优模型
├── test_results.json   # 测试结果
├── training_history.json # 训练历史
└── config.json         # 配置文件
```

## 功能特点

### 1. BiEncoder模型架构
- Siamese Network结构
- 共享参数的双BERT编码器
- 支持CLS、MEAN、MAX三种池化方式
- L2归一化后的余弦相似度计算

### 2. 完整的训练流程
- 训练、验证、测试三个阶段
- 自动保存最优模型
- 详细的训练日志输出
- 解码后的预测结果保存

### 3. 灵活的参数配置
- 所有训练参数可通过命令行配置
- 支持自定义超参数调优
- 可控制BERT层数加速训练

### 4. 完整的评估指标
- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1 Score（F1分数）

## 数据格式

输入数据为JSONL格式：
```json
{"sentence1": "文本1", "sentence2": "文本2", "label": 0或1}
```

- `sentence1`: 第一个句子
- `sentence2`: 第二个句子  
- `label`: 标签，1表示相似/正确示例，0表示不相似/错误示例

## 快速开始

### 1. 安装依赖

```bash
cd bert_model/corpus
pip install -r requirements.txt
```

### 2. 基本训练

```bash
python train.py --data_dir ../../data/bq_corpus
```

### 3. 自定义参数训练

```bash
python train.py \
  --data_dir ../../data/bq_corpus \
  --max_len 64 \
  --batch_size 32 \
  --epochs 5 \
  --learning_rate 2e-5
```

### 4. 快速训练（使用4层BERT）

```bash
python train.py \
  --data_dir ../../data/bq_corpus \
  --num_hidden_layers 4 \
  --epochs 3
```

## 主要参数说明

### 数据参数
- `--data_dir`: 数据目录路径（默认：../../data/bq_corpus）
- `--max_len`: 最大序列长度（默认：64）

### 模型参数
- `--bert_model`: 预训练BERT模型路径
- `--pool`: 池化方式（cls/mean/max）
- `--dropout`: Dropout比例
- `--num_hidden_layers`: BERT层数

### 训练参数
- `--batch_size`: 批处理大小（默认：32）
- `--epochs`: 训练轮数（默认：3）
- `--learning_rate`: 学习率（默认：2e-5）
- `--margin`: 余弦损失margin值（默认：0.3）

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

## 训练过程说明

### 1. 数据加载
- 从指定目录读取train.jsonl、validation.jsonl、test.jsonl
- 自动统计样本数量

### 2. 模型训练
- 按epoch进行训练
- 每个batch计算损失并反向传播
- 使用梯度裁剪防止梯度爆炸

### 3. 模型验证
- 每个epoch后进行验证
- 计算accuracy、precision、recall、f1指标
- 根据验证F1分数决定是否保存模型

### 4. 模型测试
- 使用最优模型在测试集上评估
- 保存所有预测结果
- 打印预测示例

### 5. 结果保存
- 保存最优模型
- 保存训练历史
- 保存测试结果
- 保存配置信息

## 注意事项

1. **GPU支持**: 代码会自动检测CUDA，若有GPU则自动使用
2. **数据格式**: 确保数据为标准JSONL格式
3. **路径问题**: 使用相对路径时注意当前工作目录
4. **内存占用**: 较大的batch_size和max_len会占用更多显存
5. **训练时间**: 使用全部12层BERT时训练时间较长

## 依赖环境

- Python 3.7+
- PyTorch >= 1.9.0
- Transformers >= 4.20.0
- scikit-learn >= 0.24.0
- tqdm >= 4.62.0

## 代码验证

项目包含完整的测试脚本 `test_setup.py`，可以验证：
- 数据文件是否完整
- 代码模块是否可以正常导入
- Python环境依赖是否满足

## 项目状态

✅ 所有代码文件已创建
✅ 操作说明文档已完成
✅ 数据文件验证通过
✅ 依赖清单已准备

**下一步操作：**
1. 安装依赖：pip install -r requirements.txt
2. 运行训练：python train.py --data_dir ../../data/bq_corpus
3. 查看结果：检查outputs目录下的输出文件