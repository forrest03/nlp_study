# Week07 NER 实验结果总结

> 数据集：**peoples_daily**（人民日报 NER）  
> 实体类型：**PER**（人名）/ **ORG**（组织机构）/ **LOC**（地名）  
> 数据规模：训练 20,864 / 验证 2,318 / 测试 4,636  
> 基座模型：`bert-base-chinese`  
> 日志目录：`outputs/logs/`

---

## 1. 测试集整体对比

| 方案 | Precision | Recall | F1 | 非法 BIO 序列 |
|------|-----------|--------|-----|---------------|
| BERT + Linear | 0.9295 | 0.9487 | **0.9390** | 129 / 4636（2.8%） |
| BERT + CRF | 0.9421 | 0.9483 | **0.9452** | 93 / 4636（2.0%） |

**结论：** CRF 在测试集 F1 上比 Linear 高 **+0.62%**，Precision 更高；Recall 两者接近（约 94.8%）。

---

## 2. 训练过程（验证集 Entity F1）

### 2.1 BERT + Linear

| Epoch | train_loss | val_loss | val_entity_f1 | 耗时 |
|-------|------------|----------|---------------|------|
| 1 | 0.162 | 0.020 | 0.9284 | ~14 min |
| 2 | 0.017 | 0.016 | 0.9478 | ~14 min |
| 3 | 0.008 | 0.016 | **0.9506** | ~18 min |

- 第 1 epoch 后提升明显，第 3 epoch 验证 F1 最优
- val_loss 在第 2 epoch 最低，第 3 epoch 略升，存在轻微过拟合迹象

日志文件：`outputs/logs/train_linear.json`

### 2.2 BERT + CRF

| Epoch | train_loss | val_loss | val_entity_f1 | 耗时 |
|-------|------------|----------|---------------|------|
| 1 | 6.980 | 0.964 | 0.9345 | ~18 min |
| 2 | 0.801 | 0.835 | 0.9491 | ~21 min |
| 3 | 0.384 | 0.908 | **0.9575** | ~22 min |

- CRF 初期 train_loss 很高（CRF 负对数似然特性），后期快速下降
- 验证 F1 持续上升，第 3 epoch 最优 **0.9575**，高于 Linear 的 0.9506

日志文件：`outputs/logs/train_crf.json`

---

## 3. 测试集详细指标

### BERT + Linear

```json
{
  "model": "BERT+Linear",
  "split": "test",
  "precision": 0.929511,
  "recall": 0.948725,
  "f1": 0.93902,
  "illegal_stats": {
    "illegal_start": 0,
    "illegal_transition": 129,
    "total_seqs": 4636,
    "total_illegal": 129
  }
}
```

日志文件：`outputs/logs/eval_linear_test.json`

### BERT + CRF

```json
{
  "model": "BERT+CRF",
  "split": "test",
  "precision": 0.942126,
  "recall": 0.948322,
  "f1": 0.945214,
  "illegal_stats": {
    "illegal_start": 0,
    "illegal_transition": 93,
    "total_seqs": 4636,
    "total_illegal": 93
  }
}
```

日志文件：`outputs/logs/eval_crf_test.json`

---

## 4. 关键发现

### 4.1 CRF 优于 Linear

| 指标 | Linear | CRF | 差值 |
|------|--------|-----|------|
| 验证集 F1 | 0.9506 | 0.9575 | +0.69% |
| 测试集 F1 | 0.9390 | 0.9452 | +0.62% |

CRF 通过转移矩阵约束标签序列，在 Precision 上优势更明显。

### 4.2 非法 BIO 序列

| 模型 | 非法开头 (I-X) | 非法转移 (B-X→I-Y) | 合计 | 占比 |
|------|----------------|-------------------|------|------|
| Linear | 0 | 129 | 129 | 2.8% |
| CRF | 0 | 93 | 93 | 2.0% |

Linear 逐 token 独立预测，易产生非法标签转移；CRF 非法序列更少，但本次实验未完全归零，可能与训练 epoch 或评估对齐方式有关。

### 4.3 泛化能力

| 模型 | 验证集 F1 | 测试集 F1 | 差距 |
|------|-----------|-----------|------|
| Linear | 0.9506 | 0.9390 | -1.16% |
| CRF | 0.9575 | 0.9452 | -1.23% |

验证/测试差距在正常范围内，CRF 在两侧均略优。

### 4.4 训练成本

- 单 epoch 约 **14–22 分钟**（CPU/MPS）
- 3 epoch 合计约 **45–60 分钟 / 模型**

---
