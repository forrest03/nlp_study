# BERT NER 训练模型

## 项目结构

```
bert_model/
├── bert_ner_model.py      # BERT模型定义
├── ner_dataset.py         # 数据集处理类
├── decoder.py             # 实体解码函数
├── train_bert_ner.py      # 主训练脚本（模块化版本）
├── complete_train.py      # 完整训练脚本（单文件版本）
├── run_training.bat       # Windows批处理运行脚本
└── README.md              # 本文件
```

## 功能特点

1. **BIOES标注方案**：支持BIOES（Begin-Inside-Outside-End）标注格式
2. **命令行参数配置**：所有训练参数可通过命令行参数调整
3. **序列标注**：对文本语句进行序列标注训练
4. **解码功能**：将预测结果解码为实体信息
5. **结果保存**：保存训练历史、解码样本和最优模型

## 使用方法

### 基本用法

```bash
# 使用ai_swap环境运行训练
& "D:\Python\anaconda3\envs\ai_swap\python.exe" complete_train.py --max_len 64 --batch_size 32 --epochs 10 --lr 2e-5
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `D:\workspace\hub-TroE\张福\week07\data\peoples_daily` | 数据目录 |
| `--output_dir` | `D:\workspace\hub-TroE\张福\week07\outputs` | 输出目录 |
| `--max_len` | 64 | 最大序列长度 |
| `--batch_size` | 32 | 批次大小 |
| `--epochs` | 10 | 训练轮数 |
| `--lr` | 2e-5 | 学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--bert_path` | `pretrain_models/bert-base-chinese` | BERT模型路径（本地路径或HuggingFace模型名）|
| `--device` | `cpu` | 训练设备 |
| `--use_crf` | False | 是否使用CRF层进行序列标注 |

### 示例命令

```bash
# 快速测试（3个epoch）
& "D:\Python\anaconda3\envs\ai_swap\python.exe" complete_train.py --max_len 64 --batch_size 32 --epochs 3 --lr 2e-5

# 完整训练（10个epoch）
& "D:\Python\anaconda3\envs\ai_swap\python.exe" complete_train.py --max_len 64 --batch_size 32 --epochs 10 --lr 2e-5

# 使用CRF层训练（需要先安装 torchcrf）
& "D:\Python\anaconda3\envs\ai_swap\python.exe" complete_train.py --max_len 64 --batch_size 32 --epochs 10 --lr 2e-5 --use_crf

# 自定义参数
& "D:\Python\anaconda3\envs\ai_swap\python.exe" complete_train.py --max_len 128 --batch_size 16 --epochs 15 --lr 1e-5 --weight_decay 1e-5
```

## 数据格式

### 输入数据格式

```json
[
  {
    "tokens": ["海", "钓", "比", "赛", "地", "点", "在", "厦", "门"],
    "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "B-LOC", "I-LOC"]
  }
]
```

### 标签格式（label_names.json）

```json
[
  "O",
  "B-PER", "I-PER", "E-PER",
  "B-ORG", "I-ORG", "E-ORG",
  "B-LOC", "I-LOC", "E-LOC"
]
```

## 输出结果

训练完成后，在`outputs`目录下会生成以下文件：

1. **best_ner_model.pt** - 最优训练模型
2. **training_results.json** - 完整训练结果，包括：
   - 配置参数
   - 标签映射（label2id, id2label）
   - 训练摘要（最佳epoch和验证F1）
   - 测试结果（loss, F1, precision, recall）
   - 训练历史（每个epoch的详细指标）
3. **validation_decoded_samples.json** - 验证集解码样本
4. **test_decoded_samples.json** - 测试集解码样本

### 解码样本格式

```json
[
  {
    "tokens": ["海", "钓", "比", "赛", "地", "点", "在", "厦", "门"],
    "predicted_entities": [
      {
        "text": "厦门",
        "label": "LOC",
        "start": 7,
        "end": 9
      }
    ],
    "original_tags": ["O", "O", "O", "O", "O", "O", "O", "B-LOC", "I-LOC"]
  }
]
```

## 模型架构

```
输入文本 → BERT Tokenizer → BERT Model → Dropout → Linear Classifier → 预测标签
```

## 评估指标

- **Loss** - 损失值
- **F1 Score** - F1分数（宏平均）
- **Precision** - 精确率（宏平均）
- **Recall** - 召回率（宏平均）

## 注意事项

1. 确保使用ai_swap环境，该环境已安装所需的依赖包
2. 首次运行会下载bert-base-chinese模型，需要网络连接
3. 训练时间取决于数据量、epoch数和硬件配置
4. 建议先使用少量epoch测试，确认无误后再进行完整训练

## 依赖环境

- Python 3.x
- PyTorch 2.11.0
- Transformers 4.55.0
- scikit-learn
- tqdm

## 故障排除

如果训练脚本无法运行，请检查：

1. Python环境是否正确（使用ai_swap环境）
2. 数据文件路径是否正确
3. 输出目录是否有写入权限
4. 网络连接是否正常（首次运行需要下载BERT模型）