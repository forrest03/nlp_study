"""
基于Transformer的字符级单向语言模型训练与文本生成脚本

核心特性：
1. 使用TransformerEncoder实现单向语言模型
2. 通过Causal Mask确保只能看到历史信息
3. 使用Top-P采样进行文本生成
4. 支持训练和仅生成两种模式

训练时会将模型参数、词表和配置保存到 --save 指定的文件（默认 best_model.pt）
仅生成模式会从该文件加载已训练的模型

用法:
    # 训练模型
    python 作业1.py --epochs 20
    
    # 仅运行文本生成（需先训练并保存模型）
    python 作业1.py --generate --gen_start "从前有座山"
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据处理 ───────────────────────────

def load_corpus(pattern="*.txt"):
    """加载指定模式的文本文件"""
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    """从文本构建字符级词表"""
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """字符级数据集"""
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


# ─────────────────────────── Transformer单向语言模型 ───────────────────────────

class CausalMaskTransformer(nn.Module):
    """
    基于Transformer的单向语言模型
    
    Args:
        vocab_size: 词表大小
        embed_dim: 嵌入维度（需为num_heads的倍数）
        num_heads: 多头注意力头数
        num_layers: Transformer编码器层数
        hidden_dim: 前馈网络隐藏层维度
        dropout: dropout概率
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, dropout):
        super().__init__()
        # 字符嵌入层
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入序列，形状为 (batch_size, seq_len)
        Returns:
            logits: 每个位置的词表概率分布，形状为 (batch_size, seq_len, vocab_size)
        """
        B, T = x.shape
        
        # 计算嵌入（token嵌入 + 位置嵌入）
        x = self.embed(x) + self.pos_embed[:, :T, :]
        x = self.drop(x)
        
        # 创建Causal Mask（下三角矩阵，确保单向性）
        # mask[i][j] = -inf 表示位置i不能看到位置j（j > i时）
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        
        # 通过Transformer编码器
        x = self.encoder(x, mask=causal_mask)
        
        # 映射到词表空间
        logits = self.fc(x)
        
        return logits


# ─────────────────────────── Top-P采样 ───────────────────────────

def top_p_sampling(logits, temperature=0.7, top_p=0.9):
    """
    Top-P (Nucleus) Sampling 采样方法
    
    Args:
        logits: 模型输出的logits，形状为 (vocab_size,)
        temperature: 温度参数，控制分布尖锐程度
                     >1: 增加随机性，<1: 增加确定性
        top_p: 累积概率阈值
        
    Returns:
        next_token: 采样得到的token索引
    """
    # 应用temperature缩放
    scaled_logits = logits / temperature
    
    # 计算概率分布
    probs = torch.softmax(scaled_logits, dim=-1)
    
    # 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 保留累积概率 <= top_p 的token
    mask = cumulative_probs <= top_p
    mask[0] = True  # 至少保留一个token
    
    # 获取候选token集合并归一化
    candidate_indices = sorted_indices[mask]
    candidate_probs = sorted_probs[mask] / sorted_probs[mask].sum()
    
    # 随机采样
    next_token = torch.multinomial(candidate_probs, num_samples=1)
    
    return candidate_indices[next_token]


# ─────────────────────────── 文本生成 ───────────────────────────

def generate_text(model, idx2char, char2idx, start_text, length=100, 
                  device='cpu', temperature=0.7, top_p=0.9):
    """
    自回归文本生成
    
    Args:
        model: 训练好的语言模型
        idx2char: 索引到字符的映射
        char2idx: 字符到索引的映射
        start_text: 起始文本
        length: 生成长度
        temperature: 温度参数
        top_p: top-p采样阈值
        
    Returns:
        generated: 生成的完整文本
    """
    model.eval()
    
    # 转换起始文本为token
    tokens = [char2idx[c] for c in start_text if c in char2idx]
    if not tokens:
        tokens = [random.choice(list(char2idx.values()))]
    
    input_seq = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(length):
            logits = model(input_seq)
            next_token_logits = logits[0, -1, :]
            next_token = top_p_sampling(next_token_logits, temperature, top_p)
            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)
            if input_seq.shape[1] > 500:
                break
    
    return ''.join([idx2char.get(t.item(), '') for t in input_seq[0]])


# ─────────────────────────── 训练循环 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """单轮训练/评估"""
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


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transformer单向语言模型")
    # 模型参数
    parser.add_argument("--embed_dim",   type=int,   default=128,   help="嵌入维度")
    parser.add_argument("--hidden_dim",  type=int,   default=512,   help="前馈网络维度")
    parser.add_argument("--num_layers",  type=int,   default=2,     help="Transformer层数")
    parser.add_argument("--num_heads",   type=int,   default=4,     help="注意力头数")
    parser.add_argument("--dropout",     type=float, default=0.3,   help="dropout概率")
    
    # 训练参数
    parser.add_argument("--epochs",      type=int,   default=20,    help="训练轮数")
    parser.add_argument("--seq_len",     type=int,   default=64,    help="序列长度")
    parser.add_argument("--batch_size",  type=int,   default=128,   help="批次大小")
    parser.add_argument("--lr",          type=float, default=1e-3,  help="学习率")
    parser.add_argument("--val_ratio",   type=float, default=0.05,  help="验证集比例")
    
    # 数据与保存
    parser.add_argument("--corpus",      default="*.txt",          help="语料文件模式")
    parser.add_argument("--save",        default="best_model.pt",  help="模型保存路径")
    
    # 生成参数
    parser.add_argument("--generate",    action="store_true",      help="仅运行文本生成")
    parser.add_argument("--gen_length",  type=int,   default=200,  help="生成文本长度")
    parser.add_argument("--gen_start",   type=str,   default="",   help="生成起始文本")
    parser.add_argument("--temperature", type=float, default=0.7,  help="采样温度")
    parser.add_argument("--top_p",       type=float, default=0.9,  help="top-p阈值")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 加载语料和构建词表
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")
    
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")
    
    # ─────────────────────────── 仅生成模式 ───────────────────────────
    if args.generate:
        # 加载保存的模型
        print(f"\n正在加载模型: {args.save}")
        checkpoint = torch.load(args.save, map_location=device, weights_only=False)
        
        # 创建模型实例（使用保存的配置）
        model_args = checkpoint['args']
        model = CausalMaskTransformer(
            vocab_size=vocab_size,
            embed_dim=model_args['embed_dim'],
            num_heads=model_args['num_heads'],
            num_layers=model_args['num_layers'],
            hidden_dim=model_args['hidden_dim'],
            dropout=model_args['dropout'],
        ).to(device)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state'])
        print("模型加载完成")
        
        # 设置起始文本
        start_text = args.gen_start if args.gen_start else random.choice(["从前有座山", "在一个遥远的地方", "有一天"])
        
        # 生成文本
        print(f"\n生成参数: temperature={args.temperature}, top_p={args.top_p}")
        print(f"起始文本: {start_text}")
        generated = generate_text(
            model, idx2char, char2idx, start_text,
            length=args.gen_length,
            device=device,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(f"\n生成结果:\n{generated}")
        return
    
    # ─────────────────────────── 训练模式 ───────────────────────────
    # 划分训练/验证集
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])
    
    # 创建数据集和数据加载器
    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # 创建模型
    model = CausalMaskTransformer(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练循环
    best_val_ppl = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)
    
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        
        # 保存最佳模型
        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)
        
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")
    
    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")
    
    # 训练后进行文本生成测试
    print("\n" + "="*50)
    print("文本生成测试")
    print("="*50)
    start_options = ["从前有座山", "在一个遥远的地方", "有一天"]
    for start_text in start_options:
        print(f"\n起始: {start_text}")
        generated = generate_text(
            model, idx2char, char2idx, start_text,
            length=100, device=device, temperature=0.7, top_p=0.9
        )
        print(f"生成: {generated}")


if __name__ == "__main__":
    main()