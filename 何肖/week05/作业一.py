"""
学员姓名：何肖

字符级 Transformer 语言模型（GPT 式）

- 结构：Embedding + 位置编码 + TransformerEncoder + 因果 mask + 全连接
- 任务：给定前文，预测下一个字符（自回归语言模型）
- 入口：无 model.pt 时训练；有 model.pt 时交互式续写
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os


def load_corpus(pattern="*.txt"):
    """从脚本所在目录加载语料（默认 corpus.txt），拼成一整段文本。"""
    texts = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_files = glob.glob(os.path.join(script_dir, pattern))

    for path in txt_files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())

    return "".join(texts)


def build_vocab(text):
    """字符级词表：语料中出现过的每个不同字符对应一个 id。"""
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """滑动窗口样本：x 为连续 seq_len 个字符，y 为 x 右移一位（预测下一个字）。"""

    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        # 样本数 = 字符数 - 窗口长度（每个起点一个样本）
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


class PositionalEncoding(nn.Module):
    """正弦位置编码（Transformer 原论文），让模型感知字符在序列中的位置。"""

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe 不参与训练，用 register_buffer 随模型保存/搬运设备
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[: x.size(1)]
        else:
            x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LM(nn.Module):
    """
    字符级语言模型（Encoder + 因果 mask，等价 GPT 式只因果自注意力）。

    不用 TransformerDecoder + memory，避免交叉注意力看见未来字符。
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=None,
        batch_first=True,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.batch_first = batch_first
        if dim_feedforward is None:
            dim_feedforward = 3 * d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model, dropout=dropout, batch_first=batch_first
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        # 每个位置输出词表大小的 logits，表示下一个字的分数
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)，元素为字符 id
        # 缩放 embedding，与位置编码量级匹配（Attention Is All You Need）
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        seq_len = x.size(1) if self.batch_first else x.size(0)
        # 上三角为 -inf：位置 t 只能 attend 到 0..t，不能看未来
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.encoder(x, mask=causal_mask)
        return self.fc(x)  # (batch, seq_len, vocab_size)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """
    跑一个 epoch。train=True 时反向传播更新参数；False 时仅评估（验证集）。

    返回按「所有 token」平均的 loss 与困惑度 PPL = exp(loss)。
    """
    model.train(train)
    total_loss = 0
    total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        output = model(x)
        # 把所有位置展平，与 y 中每个「下一个字」算交叉熵
        loss = criterion(output.view(-1, output.size(-1)), y.reshape(-1))
        if train:
            loss.backward()
            optimizer.step()
        # 按 token 数加权累加，得到全数据集平均 loss（非按 batch 平均）
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def train():
    parser = argparse.ArgumentParser(description="训练字符级 Transformer 语言模型")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=48, help="上下文窗口长度")
    parser.add_argument("--nhead", type=int, default=4, help="注意力头数，须整除 d_model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=0,
        help="FFN 隐层维度，0 表示训练时用 4*d_model",
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集行数比例")
    parser.add_argument("--corpus", default="corpus.txt")
    parser.add_argument("--save", default="model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} \n")

    text = load_corpus(args.corpus)
    print(f"语料字符数: {len(text):,}")

    # 词表用全文构建，保证训练和预测时字符 id 一致
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    # 按行划分 train/val，避免相邻行在字符级滑动中混在一起（行级 shuffle）
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))

    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    dim_ff = args.dim_feedforward if args.dim_feedforward > 0 else 4 * args.d_model

    model = LM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=dim_ff,
        batch_first=True,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

        marker = "  *" if va_ppl < best_val_ppl else ""
        # 仅当验证 PPL 创新低时保存，用于预测的是「泛化最好」的一轮
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "char2idx": char2idx,
                    "idx2char": idx2char,
                    "args": vars(args),
                },
                args.save,
            )

        print(
            f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
            f"{va_loss:>10.4f}  {va_ppl:>10.2f}{marker}"
        )

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


@torch.no_grad()
def top_p_sampling(logits, top_p=0.9):
    """
    Top-p（核）采样：在累积概率达到 top_p 的最小字集合里随机抽一个。

    logits: (batch, vocab)，此处应先除以 temperature，再传入本函数；
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False  # 至少保留概率最高的一个 token
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    pick = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(-1, pick)


@torch.no_grad()
def predict_sentence(
    model,
    start_word,
    char2idx,
    idx2char,
    max_len,
    top_p=0.9,
    temperature=0.5,
    device=None,
    context_len=None,
):
    """
    自回归续写：从 start_word 出发，最多再生成 max_len 个字符。

    - max_len：新生成字符数（总长度 ≈ len(start_word) + max_len）
    - context_len：每次 forward 最多看最近多少个字符（应等于训练时的 seq_len）
    - temperature：越小越保守；logits 先除 T，再在 top_p_sampling 里 softmax
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    start_word = start_word.strip()
    skipped = [c for c in start_word if c not in char2idx]
    if skipped:
        print(f"警告: 以下字符不在词表中，已跳过: {''.join(skipped)}")

    input_ids = [char2idx[c] for c in start_word if c in char2idx]
    if not input_ids:
        raise ValueError(
            "起始词在过滤后为空，请至少输入一个语料中出现过的字。"
        )

    input_seq = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_len):
        # 序列变长后只取最后 context_len 个字，与训练时窗口一致（滑动窗口）
        ctx = input_seq
        if context_len is not None and input_seq.size(1) > context_len:
            ctx = input_seq[:, -context_len:]
        output = model(ctx)
        # 只用最后一个位置的 logits 预测「下一个字」
        next_token_logits = output[0, -1, :] / temperature
        next_token = top_p_sampling(next_token_logits.unsqueeze(0), top_p=top_p).item()
        input_seq = torch.cat(
            [input_seq, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1,
        )

    return "".join(idx2char[i] for i in input_seq[0].tolist())


if __name__ == "__main__":
    if os.path.exists("model.pt"):
        print("检测到已保存的模型，加载中...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load("model.pt", map_location=device)
        char2idx = checkpoint["char2idx"]
        idx2char = checkpoint["idx2char"]
        model_args = checkpoint["args"]

        d_model = model_args["d_model"]
        dim_ff = model_args.get("dim_feedforward", 0)
        dim_ff = dim_ff if dim_ff > 0 else 4 * d_model

        # 结构须与训练时一致，再加载权重
        model = LM(
            vocab_size=len(char2idx),
            d_model=d_model,
            nhead=model_args["nhead"],
            num_layers=model_args["num_layers"],
            dim_feedforward=dim_ff,
            batch_first=True,
            dropout=model_args.get("dropout", 0.1),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        while True:
            start_word = input("请输入起始词，按q退出对话: ").strip()
            if not start_word:
                print("未输入起始词，使用默认: 浦发银行")
                start_word = "浦发银行"
            if start_word == "q":
                break
            print("使用模型进行预测...")
            result = predict_sentence(
                model,
                start_word,
                char2idx,
                idx2char,
                max_len=15,
                device=device,
                context_len=model_args.get("seq_len"),
            )
            print(f"预测结果: {result}")
    else:
        print("未发现已保存的模型，开始训练新模型...")
        train()
