# 字符级 Transformer 语言模型

基于 `model.py` 与 `corpus.txt` 实现的**字符级**自回归语言模型：使用 `TransformerEncoder` + 因果掩码，在财经报道语料上训练「下一个字」预测，并支持交互式续写。

**学员姓名**：何肖（见 `model.py` 文件头）

## 文件说明

| 文件 | 说明 |
|------|------|
| `model.py` | 训练、验证、保存模型与交互式文本续写（运行入口） |
| `corpus.txt` | UTF-8 训练语料；由 `load_corpus()` 读取，默认与此文件同名 |

训练成功后会在同目录生成 `model.pt`（权重、`char2idx` / `idx2char`、训练超参），由 `model.py` 自动读写，**不在仓库中预置**。

## 语料 `corpus.txt`

- **内容**：财经新闻、期货、基金、银行等报道与评论，每行一段或一句。
- **规模**（当前版本）：约 1000+ 行，约 26 万字符（随你编辑会变化）。
- **编码**：UTF-8；`load_corpus` 使用 `errors="ignore"` 跳过无法解码字符。
- **用法**：
  - 词表 `build_vocab(text)` 对**全文**所有出现过的字建表。
  - 训练/验证按**行**随机划分（默认 90% / 10%），再拼成字符串做滑动窗口。
- **注意**：预测时起始词中的字若不在语料出现过，会被跳过；更换语料后需删除旧 `model.pt` 并重新训练。

## 环境依赖

- Python 3.x
- PyTorch
- 标准库：`argparse`、`glob`、`math`、`os`、`random`

```bash
pip install torch
# 或 conda install pytorch cpuonly -c pytorch
```

## 运行方式

在项目目录下执行：

```bash
python model.py
```

| 情况 | 行为 |
|------|------|
| 不存在 `model.pt` | 调用 `train()` 训练，保存验证 PPL 最低的一轮 |
| 已存在 `model.pt` | 加载模型，循环输入起始词续写；输入 `q` 退出；空行默认 `浦发银行` |

指定语料或超参（训练模式）示例：

```bash
python model.py --corpus corpus.txt --epochs 10 --seq_len 48 --batch_size 32 --d_model 96
```

## 模型结构（`model.py` 中的 `LM`）

```
字符 id
  → Embedding × √d_model
  → PositionalEncoding（正弦位置编码）
  → TransformerEncoder（因果 mask）
  → Linear
  → 每个位置的「下一字」logits
```

- **因果 mask**：`generate_square_subsequent_mask`，位置 \(t\) 只能看见 \(0\ldots t\)，用于语言模型。

## 训练流程

1. `load_corpus(args.corpus)`：默认读取同目录 `corpus.txt`。
2. `build_vocab`：字符级词表。
3. `lines = text.splitlines()` → `shuffle` → 按 `val_ratio`（默认 0.1）分行划分 train / val。
4. `CharDataset`：滑动窗口，`x` 长度 `seq_len`，`y` 为 `x` 右移一位（**teacher forcing**：输入与标签均来自真实语料）。
5. `run_epoch`：交叉熵；Loss / PPL 按**所有 token** 平均，`PPL = exp(loss)`。
6. 仅当 **Val PPL** 创新低时保存 `model.pt`（打印行末 `*`）。

### 命令行参数（`train()` 默认值）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--epochs` | 10 | 训练轮数 |
| `--seq_len` | 48 | 上下文窗口；每个样本 48 字预测下一字 |
| `--batch_size` | 32 | 批大小 |
| `--d_model` | 96 | 嵌入与注意力维度 |
| `--nhead` | 4 | 头数（须整除 `d_model`） |
| `--num_layers` | 2 | Encoder 层数 |
| `--dim_feedforward` | 0 | FFN 隐层；**0 表示 `4 × d_model`**（训练时） |
| `--dropout` | 0.3 | Dropout |
| `--lr` | 2e-4 | AdamW 学习率 |
| `--val_ratio` | 0.1 | 验证集所占行数比例 |
| `--corpus` | corpus.txt | 语料路径或 glob（相对脚本目录） |
| `--save` | model.pt | checkpoint 路径 |

## 预测 / 续写（`predict_sentence`）

- **`max_len`**：在起始词之后**新生成**的字符数（总长度 ≈ `len(起始词) + max_len`）。`__main__` 中默认 `15`。
- **`temperature`**：默认 `0.5`；对 logits 先除以 T，再在 `top_p_sampling` 内 `softmax`（267 行处不必再写 softmax）。
- **`top_p`**：默认 `0.9`，核采样。
- **`context_len`**：默认取训练时的 `seq_len`；序列变长时只将 **最后 `seq_len` 个字** 送入 `model`（与训练窗口对齐）。

```python
if input_seq.size(1) > context_len:
    ctx = input_seq[:, -context_len:]
```

修改 `max_len`、`temperature`、`top_p` **不需要重新训练**；更换 `corpus.txt` 或重新训练后词表变化则必须重训。

## 指标如何理解

- **Val PPL** 比 Train PPL 更值得选模型；保存的是 Val 最低轮。
- 随机猜下一个字，PPL 约等于词表大小（字符种数）。
- Train 很低、Val 持续升高 → 过拟合，仍用带 `*` 的 checkpoint，勿用最后一轮。

## 使用建议

1. 起始词尽量是 `corpus.txt` 里出现过的片段（如 `沪铝`、`浦发银行`、`基金`）。
2. 短续写（`max_len` 10～15）通常比很长一段更通顺。
3. 续写不必与原文逐字相同；字符级模型易在专名处拆字（如「国泰君安」→「泰君安」）。

## 常见问题

**Q: 没有 `model.pt` 时运行会怎样？**  
A: 自动进入 `train()`；训练结束后生成 `model.pt`。

**Q: 改了 `predict_sentence` 的默认 `max_len=20` 但生成仍很短？**  
A: 需同时修改 `__main__` 里调用处的 `max_len=15`，调用实参优先于函数定义默认值。

**Q: `seq_len=48` 是否表示起始词不能超过 48 字？**  
A: 否。起始可以更长，但每次 forward 最多看最近 48 字；过长开头时最前面的字不参与当前步预测。
