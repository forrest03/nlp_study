"""
字符级续写模型：训练 + 根据前缀续写
用法:
    python completion_model.py --epochs 20 --save best_model.pt
    python completion_model.py --checkpoint best_model.pt --prompt "春风" --length 200
    python completion_model.py --checkpoint best_model.pt --prompt "春风" --temperature 0.8 --top_k 40
"""

import argparse
import glob
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.drop(self.embed(x))
        out, _ = self.rnn(e)
        return self.fc(self.drop(out))


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 续写生成 ───────────────────────────

@torch.no_grad()
def complete(
    model,
    char2idx,
    idx2char,
    prompt: str,
    length: int = 200,
    seq_len: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    device: torch.device | None = None,
) -> str:
    """
    自回归续写：用训练好的 LM 根据 prompt 逐字生成后续文本。

    Args:
        prompt: 用户给定的开头文本（仅使用词表内字符）。
        length: 最多新生成的字符数（不含 prompt 本身）。
        seq_len: 每次前向使用的上下文窗口，需与训练时 --seq_len 一致。
        temperature: 采样温度，越大越随机。
        top_k: 只从概率最高的 k 个字符中采样；0 表示不截断。
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        raise ValueError("prompt 中没有任何字符落在词表内，请换一段文本或先训练模型。")

    for _ in range(length):
        ctx = ids[-seq_len:]
        x = torch.LongTensor([ctx]).to(device)
        logits = model(x)[:, -1, :]

        if temperature <= 0:
            next_id = logits.argmax(dim=-1).item()
        else:
            logits = logits / temperature
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k).values[..., -1, None]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        ids.append(next_id)

    return "".join(idx2char[i] for i in ids)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]

    model = LM(
        vocab_size=len(char2idx),
        embed_dim=args["embed_dim"],
        hidden_dim=args["hidden_dim"],
        num_layers=args["num_layers"],
        model_type=args["model"],
        dropout=args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, char2idx, idx2char, args


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}  model: {args.model.upper()}")

    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

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

    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_type=args.model,
        dropout=args.dropout,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

    if args.prompt:
        text_out = complete(
            model,
            char2idx,
            idx2char,
            args.prompt,
            length=args.length,
            seq_len=args.seq_len,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print("\n── 续写示例 ──")
        print(text_out)


def generate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model, char2idx, idx2char, ckpt_args = load_checkpoint(args.checkpoint, device)
    seq_len = args.seq_len or ckpt_args.get("seq_len", 64)

    text_out = complete(
        model,
        char2idx,
        idx2char,
        args.prompt,
        length=args.length,
        seq_len=seq_len,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(text_out)


def main():
    parser = argparse.ArgumentParser(description="字符级续写语言模型")
    parser.add_argument("--checkpoint", default="", help="权重路径；指定则只续写，不训练")
    parser.add_argument("--prompt", default="", help="续写前缀；训练结束后也可附带试生成")
    parser.add_argument("--length", type=int, default=200, help="续写字符数")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50, help="0 表示不截断 top-k")

    parser.add_argument("--model", default="lstm", choices=["rnn", "lstm"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--corpus", default="*.txt")
    parser.add_argument("--save", default="best_model.pt")
    args = parser.parse_args()

    if args.checkpoint:
        if not args.prompt:
            parser.error("续写模式需要 --prompt")
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
