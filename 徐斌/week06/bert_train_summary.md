# BERT 训练结果汇总

> 数据来源：`outputs/train_log_*.json`（不含 `train_log_sft.json`，该文件为 LLM SFT）  
> 脚本：`src/train.py` | 默认 3 epoch | 验证指标：val_acc、val_macro_f1  
> 说明：实验时 `dataset.py` 对 train/val 各截取 **1000 条**（`[:1000]`），与全量 53K 训练结果不可直接对比。

---

## 1. 最终 Epoch（第 3 轮）对比

| 排名 | 日志文件 | 池化 | 类别加权 | train_loss | train_acc | val_acc | val_macro_f1 | 单轮耗时(s) |
|:---:|----------|:----:|:--------:|-----------:|----------:|--------:|-------------:|------------:|
| 1 | `train_log_cls_weighted.json` | cls | ✓ | 0.812 | 66.9% | **55.93%** | **55.34%** | 339.3 |
| 2 | `train_log_mean_weighted.json` | mean | ✓ | 1.307 | 60.6% | 51.90% | 50.84% | 25.7 |
| 3 | `train_log_mean.json` | mean | ✗ | 1.240 | 63.8% | 51.30% | 45.72% | 44.7 |
| 4 | `train_log_cls.json` | cls | ✗ | 1.293 | 62.3% | 49.10% | 42.78% | 62.6 |
| 5 | `train_log_max.json` | max | ✗ | 1.643 | 56.6% | 48.90% | 42.47% | 25.1 |
| 6 | `train_log_max_weighted.json` | max | ✓ | 1.837 | 51.3% | 48.60% | 48.04% | 41.5 |

**checkpoint 命名**（`outputs/checkpoints/`）：

| 日志 | 对应权重 |
|------|----------|
| `train_log_cls.json` | `best_cls.pt` |
| `train_log_cls_weighted.json` | `best_cls_weighted.pt` |
| `train_log_mean.json` | `best_mean.pt` |
| `train_log_mean_weighted.json` | `best_mean_weighted.pt` |
| `train_log_max.json` | `best_max.pt` |
| `train_log_max_weighted.json` | `best_max_weighted.pt` |

---

## 2. 各实验逐 Epoch 明细

### cls（无加权）— `train_log_cls.json`

| Epoch | train_loss | train_acc | val_acc | val_macro_f1 | 耗时(s) |
|:-----:|-----------:|----------:|--------:|-------------:|--------:|
| 1 | 2.483 | 21.0% | 42.2% | 35.26% | 98.1 |
| 2 | 1.663 | 54.3% | 48.1% | 41.61% | 47.4 |
| 3 | 1.293 | 62.3% | 49.1% | 42.78% | 62.6 |

### cls + 加权 — `train_log_cls_weighted.json`

| Epoch | train_loss | train_acc | val_acc | val_macro_f1 | 耗时(s) |
|:-----:|-----------:|----------:|--------:|-------------:|--------:|
| 1 | 1.454 | 49.9% | 54.2% | 53.36% | 338.9 |
| 2 | 1.038 | 59.9% | 55.2% | 54.35% | 339.2 |
| 3 | 0.812 | 66.9% | **55.9%** | **55.34%** | 339.3 |

### mean（无加权）— `train_log_mean.json`

| Epoch | train_loss | train_acc | val_acc | val_macro_f1 | 耗时(s) |
|:-----:|-----------:|----------:|--------:|-------------:|--------:|
| 1 | 2.483 | 19.7% | 44.0% | 36.67% | 43.5 |
| 2 | 1.613 | 54.8% | 50.6% | 45.22% | 42.5 |
| 3 | 1.240 | 63.8% | 51.3% | 45.72% | 44.7 |

### mean + 加权 — `train_log_mean_weighted.json`

| Epoch | train_loss | train_acc | val_acc | val_macro_f1 | 耗时(s) |
|:-----:|-----------:|----------:|--------:|-------------:|--------:|
| 1 | 2.611 | 17.5% | 35.6% | 31.49% | 26.0 |
| 2 | 1.794 | 49.3% | 51.2% | 48.68% | 25.0 |
| 3 | 1.307 | 60.6% | 51.9% | 50.84% | 25.7 |

### max（无加权）— `train_log_max.json`

| Epoch | train_loss | train_acc | val_acc | val_macro_f1 | 耗时(s) |
|:-----:|-----------:|----------:|--------:|-------------:|--------:|
| 1 | 2.758 | 12.2% | 27.2% | 20.75% | 24.4 |
| 2 | 2.087 | 40.6% | 45.0% | 38.75% | 25.2 |
| 3 | 1.643 | 56.6% | 48.9% | 42.47% | 25.1 |

### max + 加权 — `train_log_max_weighted.json`

| Epoch | train_loss | train_acc | val_acc | val_macro_f1 | 耗时(s) |
|:-----:|-----------:|----------:|--------:|-------------:|--------:|
| 1 | 2.827 | 8.8% | 15.2% | 11.19% | 42.0 |
| 2 | 2.279 | 32.5% | 45.1% | 40.38% | 41.9 |
| 3 | 1.837 | 51.3% | 48.6% | 48.04% | 41.5 |

---

## 3. 简要结论

1. **综合最优**：`cls` + `--use_class_weight`（val_acc **55.9%**，macro F1 **55.3%**）。
2. **类别加权**：6 组实验中，加权在 macro F1 上普遍更好（尤其 cls、max）；mean 加权第 1 epoch 偏低，第 2–3 epoch 回升。
3. **池化策略**（同是否加权）：cls_weighted > mean_weighted ≈ mean > cls ≈ max；**max 池化**在本批 1K 数据上偏弱。
4. **保存策略**：`train.py` 按 **val_acc** 保存最优 checkpoint；若更关注少数类，可同时参考 **val_macro_f1**（二者排序可能不完全一致）。
5. **复现命令示例**：
   ```bash
   cd src
   python train.py --pool cls --use_class_weight
   python train.py --pool mean
   python train.py --pool max
   ```

---

## 4. 未纳入本表

| 文件 | 说明 |
|------|------|
| `train_log_sft.json` | Qwen LoRA SFT，见 [`llm_train_summary.md`](./llm_train_summary.md) |
| `llm_zero_shot_results.json` / `llm_sft_results.json` | LLM 评估结果，见 [`llm_train_summary.md`](./llm_train_summary.md) |
