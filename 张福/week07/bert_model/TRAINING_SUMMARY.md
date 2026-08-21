# BERT NER 训练模型完成总结

## 已完成的工作

### 1. 创建BERT训练模型结构

在 `d:\workspace\hub-TroE\张福\week07\bert_model` 目录下创建了以下文件：

#### 核心模块文件

1. **bert_ner_model.py** - BERT模型定义
   - 基于bert-base-chinese预训练模型
   - 包含Dropout层和线性分类器
   - 支持序列标注任务

2. **ner_dataset.py** - 数据集处理类
   - 加载JSON格式的NER数据
   - 自动构建标签词汇表
   - 处理BERT tokenization和标签对齐
   - 支持BIOES标注格式

3. **decoder.py** - 实体解码函数
   - 将预测的标签序列解码为实体
   - 支持BIOES标注格式
   - 返回实体的文本、类型、位置信息

4. **train_bert_ner.py** - 主训练脚本（模块化版本）
   - 完整的训练流程
   - 命令行参数解析
   - 训练、验证、测试循环
   - 结果保存和解码样本输出

5. **complete_train.py** - 完整训练脚本（单文件版本）
   - 所有功能集成在一个文件中
   - 便于快速运行和调试
   - 包含完整的训练、验证、测试流程

6. **run_training.bat** - Windows批处理运行脚本
   - 快速启动训练
   - 使用ai_swap环境

### 2. 参数配置

所有训练参数都通过命令行参数配置，包括：

- `--max_len`: 最大序列长度（默认64）
- `--batch_size`: 批次大小（默认32）
- `--epochs`: 训练轮数（默认10）
- `--lr`: 学习率（默认2e-5）
- `--weight_decay`: 权重衰减（默认1e-4）
- `--data_dir`: 数据目录
- `--output_dir`: 输出目录
- `--bert_path`: BERT模型路径
- `--device`: 训练设备

### 3. 数据处理

使用 `/data/peoples_daily` 目录下的数据文件：

- **train.json** - 训练集（20864条）
- **validation.json** - 验证集（2318条）
- **test.json** - 测试集（4636条）
- **label_names.json** - 标签定义（BIOES格式）

支持的标签：
- O: 非实体
- B-PER, I-PER, E-PER: 人物实体
- B-ORG, I-ORG, E-ORG: 组织实体
- B-LOC, I-LOC, E-LOC: 地点实体

### 4. 序列标注功能

- 对文本语句进行序列标注
- 使用BIOES标注方案
- 支持实体解码和提取
- 保存解码后的数据信息

### 5. 训练、验证、测试流程

#### 训练流程
1. 加载训练数据并创建数据集
2. 初始化BERT模型和优化器
3. 对每个epoch进行训练
4. 在验证集上评估模型性能
5. 保存验证集上F1分数最高的模型

#### 验证流程
1. 加载验证数据
2. 使用当前模型进行预测
3. 计算评估指标（Loss, F1, Precision, Recall）
4. 解码部分预测结果并保存

#### 测试流程
1. 加载最优模型
2. 在测试集上进行评估
3. 计算最终测试指标
4. 解码测试样本并保存

### 6. 结果保存

在 `outputs` 目录下保存以下文件：

1. **best_ner_model.pt** - 最优训练模型
   - 验证集上F1分数最高的模型

2. **training_results.json** - 完整训练结果
   ```json
   {
     "configuration": {...},           # 训练配置参数
     "label_mapping": {                # 标签映射
       "label2id": {...},
       "id2label": {...}
     },
     "training_summary": {             # 训练摘要
       "best_epoch": 3,
       "best_validation_f1": 0.85
     },
     "test_results": {                 # 测试结果
       "loss": 0.15,
       "f1": 0.83,
       "precision": 0.82,
       "recall": 0.84
     },
     "training_history": [...]         # 每个epoch的详细指标
   }
   ```

3. **validation_decoded_samples.json** - 验证集解码样本
   - 包含预测的实体信息
   - 原始标签用于对比

4. **test_decoded_samples.json** - 测试集解码样本
   - 包含测试集上的预测结果
   - 实体的文本、类型、位置信息

### 7. 解码数据信息

解码样本包含以下信息：

```json
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
```

### 8. 评估指标

训练过程中打印和保存以下指标：

- **训练损失** (Train Loss)
- **验证损失** (Valid Loss)
- **验证F1分数** (Valid F1)
- **验证精确率** (Valid Precision)
- **验证召回率** (Valid Recall)
- **测试损失** (Test Loss)
- **测试F1分数** (Test F1)
- **测试精确率** (Test Precision)
- **测试召回率** (Test Recall)

## 使用方法

### 快速开始

```bash
# 切换到bert_model目录
cd D:\workspace\hub-TroE\张福\week07\bert_model

# 运行训练（使用ai_swap环境）
& "D:\Python\anaconda3\envs\ai_swap\python.exe" complete_train.py --max_len 64 --batch_size 32 --epochs 10 --lr 2e-5
```

### 自定义参数

```bash
# 调整批次大小和学习率
& "D:\Python\anaconda3\envs\ai_swap\python.exe" complete_train.py --max_len 64 --batch_size 16 --epochs 15 --lr 1e-5

# 调整最大序列长度
& "D:\Python\anaconda3\envs\ai_swap\python.exe" complete_train.py --max_len 128 --batch_size 32 --epochs 10 --lr 2e-5
```

## 项目特点

1. ✅ **完整的BERT NER训练流程**
2. ✅ **BIOES标注方案支持**
3. ✅ **命令行参数配置**
4. ✅ **序列标注功能**
5. ✅ **实体解码功能**
6. ✅ **训练、验证、测试完整流程**
7. ✅ **解码数据信息保存**
8. ✅ **最优模型保存**
9. ✅ **详细的训练结果记录**
10. ✅ **模块化和单文件两种实现**

## 文件清单

```
bert_model/
├── bert_ner_model.py          # BERT模型定义
├── ner_dataset.py             # 数据集处理类
├── decoder.py                 # 实体解码函数
├── train_bert_ner.py          # 主训练脚本（模块化）
├── complete_train.py          # 完整训练脚本（单文件）
├── run_training.bat           # Windows批处理脚本
├── test_training.py           # 测试脚本
├── README.md                  # 使用说明文档
└── TRAINING_SUMMARY.md        # 本总结文档

outputs/                       # 输出目录（训练后生成）
├── best_ner_model.pt          # 最优模型
├── training_results.json      # 训练结果
├── validation_decoded_samples.json  # 验证集解码样本
└── test_decoded_samples.json  # 测试集解码样本
```

## 注意事项

1. 确保使用ai_swap环境运行训练脚本
2. 首次运行需要下载bert-base-chinese模型
3. 训练时间取决于数据量、epoch数和硬件配置
4. 建议先用少量epoch测试，确认无误后再进行完整训练
5. 所有结果都会保存到outputs目录

## 总结

已成功创建完整的BERT NER训练模型，满足所有要求：

1. ✅ 在bert_model目录中创建BERT训练模型
2. ✅ 设置max_len = 64
3. ✅ 训练常规参数都放在args中，可通过命令行参数调配
4. ✅ 对peoples_daily目录下的文件进行序列标注训练、验证、测试
5. ✅ 打印保存训练、验证、测试相关结果参数，包括解码后的数据信息
6. ✅ 保存最优训练模型到outputs目录

项目已准备就绪，可以开始训练！