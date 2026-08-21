# LLM 训练 / 评估结果汇总（src_llm）

> 数据来源：`outputs/llm_*.json` + `outputs/train_log_sft.json`  
> 脚本：`src_llm/classify_llm.py`（zero-shot）、`train_sft.py`（SFT）、`evaluate_sft.py`（SFT 评估）  
> 基座模型：**Qwen2-0.5B-Instruct** | 任务：TNEWS 15 类新闻标题分类（生成式输出类别名）

相关 BERT 结果见 [`bert_train_summary.md`](./bert_train_summary.md)。

---

## 1. 评估结果总览（`llm_*.json`）

在 **验证集前 200 条** 上统计（与 `evaluate_sft.py --num_samples 200` 一致）。

| 方法 | 日志文件 | 脚本 | 是否训练 | accuracy | 正确数 | 无法解析 | 解析后准确率* |
|------|----------|------|:--------:|---------:|-------:|---------:|-------------:|
| Zero-shot | `llm_zero_shot_results.json` | `classify_llm.py` | 否 | **36.0%** | 72/200 | 58 (29.0%) | 50.7% |
| SFT (LoRA) | `llm_sft_results.json` | `evaluate_sft.py` | 是 | **55.0%** | 110/200 | 4 (2.0%) | 56.1% |

\* **解析后准确率** = `correct / (total - unparseable)`，仅统计模型输出能映射到 15 类之一的样本。

**相对 zero-shot，SFT 带来：**

| 指标 | Zero-shot → SFT | 变化 |
|------|-----------------|------|
| accuracy | 36.0% → 55.0% | **+19.0 pp** |
| 无法解析 | 58 → 4 条 | **-54 条** |
| 解析后准确率 | 50.7% → 56.1% | +5.4 pp |

**与 BERT（同 1K 数据实验）对照：**

| 方法 | 指标 | 数值 |
|------|------|------|
| BERT 最优（cls + 加权） | val_acc（1K val） | 55.9% |
| LLM SFT (LoRA) | accuracy（200 条 val） | 55.0% |

> 注意：BERT 与 LLM 评估集大小、指标定义不同，仅作量级参考；严格对比请统一 `num_samples` 与同一 val 切片。

---

## 2. Zero-shot 明细 — `llm_zero_shot_results.json`

| 字段 | 值 |
|------|-----|
| `accuracy` | 0.36 |
| `total` | 200 |
| `correct` | 72 |
| `unparseable` | 58 |
| 分类错误（已解析但类别不对） | 70 |
| 解析后准确率 | 72 / (200 − 58) ≈ **50.7%** |

**典型问题：**

- 输出不在 15 类词表内（如「房地产」「武器」）→ `pred_label: null`
- 语义相近类混淆（如 true=财经，pred=科技）

**复现：**

```bash
cd src_llm
python classify_llm.py --num_samples 200
# → outputs/llm_zero_shot_results.json
```

---

## 3. SFT 评估明细 — `llm_sft_results.json`

| 字段 | 值 |
|------|-----|
| `accuracy` | 0.55 |
| `total` | 200 |
| `correct` | 110 |
| `unparseable` | 4 |
| 分类错误（已解析） | 86 |
| 解析后准确率 | 110 / 196 ≈ **56.1%** |

**复现：**

```bash
cd src_llm
python evaluate_sft.py --num_samples 200
# → outputs/llm_sft_results.json
# 依赖 checkpoint：outputs/sft_adapter/（LoRA adapter）
```

---

## 4. SFT 训练曲线 — `train_log_sft.json`

| Epoch | train_loss | val_loss | 耗时(s) | 备注 |
|:-----:|-----------:|---------:|--------:|------|
| 1 | 0.776 | 0.697 | 3448.2 | |
| 2 | 0.647 | 0.681 | 821.0 | |
| 3 | 0.560 | **0.668** | 672.9 | **最优 val_loss** |

| 汇总项 | 值 |
|--------|-----|
| 最优 epoch | 3 |
| 最优 val_loss | **0.6681** |
| 总训练耗时 | **4942 s（≈82.4 min）** |
| 保存目录 | `outputs/sft_adapter/` |

**训练配置（与当前 `adapter_config.json` 一致）：**

| 项 | 值 |
|----|-----|
| 微调方式 | LoRA（`peft_type: LORA`） |
| base_model | `Qwen2-0.5B-Instruct` |
| LoRA rank `r` | 8 |
| `lora_alpha` | 16 |
| `target_modules` | q_proj, k_proj, v_proj, o_proj |
| 默认 `num_train` | 1000（脚本可调 `-1` 用全量 53K） |
| 默认 `batch_size × grad_accum` | 1 × 8 = **等效 batch 8** |
| 验证集（训练时） | val 前 500 条算 val_loss |

**复现训练：**

```bash
cd src_llm
python train_sft.py
# → outputs/train_log_sft.json
# → outputs/sft_adapter/
```

---

## 5. 输出文件索引

| 文件 | 类型 | 说明 |
|------|------|------|
| `llm_zero_shot_results.json` | 评估 | Zero-shot 逐条预测 + 汇总 |
| `llm_sft_results.json` | 评估 | SFT 逐条预测 + 汇总 |
| `train_log_sft.json` | 训练 | LoRA SFT 每 epoch loss / 耗时 |
| `sft_adapter/` | 权重 | LoRA adapter + tokenizer（非 `llm_` 前缀） |

若使用 `--full_ft` 全量微调，日志为 `train_log_full_ft.json`，权重目录为 `sft_full_ckpt/`。

---

## 6. 简要结论

1. **Zero-shot 不可用生产**：准确率 36%，近三成输出无法映射到合法类别。
2. **LoRA SFT 明显提升**：accuracy 55%，解析失败降至 4 条，与 BERT（cls+加权，1K 数据）同量级。
3. **生成式分类的关键指标**：除 accuracy 外应关注 **unparseable**；SFT 主要改善「格式/词表遵守」。
4. **训练成本**：本次 3 epoch、约 82 分钟（设备为 CPU/MPS 时偏慢）；GPU 上通常更快。
5. **下一步**：全量 `num_train=-1`、调 `lora_r`、或与 BERT 在同一 200 条 val 上跑 `evaluate.py` 做严格对比。
