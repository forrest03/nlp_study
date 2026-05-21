import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import numpy as np
from tqdm import tqdm
import requests
import os

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# -------------------- 1. 数据准备（自动下载或使用本地文本）--------------------
def get_text():
    # 方式1：从网络下载（《傲慢与偏见》开头部分，约5万字符）
    # 如果网络不可用，请将任意英文文本文件放在当前目录，并修改这里
    url = "https://www.gutenberg.org/files/1342/1342-0.txt"
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        text = response.text
        # 只取开头部分，加快训练
        text = text[:50000]
        print("成功从网络下载文本，长度:", len(text))
        return text
    except:
        # 备用：使用内置示例文本（几千字符）
        print("网络下载失败，使用内置示例文本")
        text = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters. My dear Mr. Bennet, said his lady to him one day, have you heard that Netherfield Park is let at last? Mr. Bennet replied that he had not. But it is, returned she; for Mrs. Long has just been here, and she told me all about it. Mr. Bennet made no answer. Do not you want to know who has taken it? cried his wife impatiently. You want to tell me, and I have no objection to hearing it. This was invitation enough. Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week. What is his name? Bingley. Is he married or single? Oh! single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls! How so? How can it affect them? My dear Mr. Bennet, replied his wife, how can you be so tiresome! You must know that I am thinking of his marrying one of them. Is that his design in settling here? Design! nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes. I see no occasion for that. You and the girls may go, or you may send them by themselves, which perhaps will be still better, for as you are as handsome as any of them, Mr. Bingley might like you the best of the party. My dear, you flatter me. I certainly have had my share of beauty, but I do not pretend to be anything extraordinary now. When a woman has five grown-up daughters, she ought to give over thinking of her own beauty. In such cases, a woman has not often much beauty to think of. But, my dear, you must indeed go and see Mr. Bingley when he comes into the neighbourhood. It is more than I engage for, I assure you. But consider your daughters. Only think what an establishment it would be for one of them. Sir William and Lady Lucas are determined to go, merely on that account, for in general, you know they visit no newcomers. Indeed you must go, for it will be impossible for us to visit him if you do not. You are over-scrupulous, surely. I dare say Mr. Bingley will be very glad to see you; and I will send a few lines by you to assure him of my hearty consent to his marrying whichever he chooses of the girls; though I must throw in a good word for my little Lizzy. I desire you will do no such thing. Lizzy is not a bit better than the others; and I am sure she is not half so handsome as Jane, nor half so good-humoured as Lydia. But you are always giving her the preference. They have none of them much to recommend them, replied he; they are all silly and ignorant like other girls; but Lizzy has something more of quickness than her sisters. Mr. Bennet, how can you abuse your own children in such a way? You take delight in vexing me. You have no compassion on my poor nerves. You mistake me, my dear. I have a high respect for your nerves. They are my old friends. I have heard you mention them with consideration these twenty years at least."""
        return text

TEXT = get_text()
print(f"文本总长度: {len(TEXT)} 字符")

# 字符级词汇表
chars = sorted(list(set(TEXT)))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
print(f"词汇表大小: {vocab_size}")

data = torch.tensor([char2idx[ch] for ch in TEXT], dtype=torch.long)

# 划分训练/验证
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"训练集字符数: {len(train_data)}, 验证集字符数: {len(val_data)}")

# 数据集
class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

seq_len = 64   # 上下文长度
batch_size = 64
train_ds = CharDataset(train_data, seq_len)
val_ds = CharDataset(val_data, seq_len)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

# -------------------- 2. Transformer --------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1,2); k = k.transpose(1,2); v = v.transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1,1,T,T)
        scores = scores.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1,2).reshape(B,T,C)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight   # 权重共享
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, mask=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.dropout(x)
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1,1,T,T)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=0.7, top_p=0.9, top_k=50):
        self.eval()
        if isinstance(prompt, str):
            idx = [char2idx.get(c, 0) for c in prompt]
        else:
            idx = prompt
        idx = torch.tensor(idx, dtype=torch.long).unsqueeze(0).to(next(self.parameters()).device)
        for _ in range(max_new_tokens):
            if idx.size(1) > self.max_seq_len:
                idx = idx[:, -self.max_seq_len:]
            logits = self(idx)[:, -1, :] / temperature
            # Top-K
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Top-P
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum_probs > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                sorted_logits[mask] = -float('Inf')
                logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return ''.join([idx2char[int(i)] for i in idx[0].tolist()])

# 模型超参数（适中容量，适合 CPU 训练）
d_model = 192
num_heads = 6
num_layers = 4
d_ff = 256
dropout = 0.2
max_seq_len = seq_len

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DecoderOnlyTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# 早停参数
patience = 5
best_val_loss = float('inf')
wait = 0

def train_epoch():
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc="训练", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)

def eval_epoch():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="验证", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item() * x.size(0)
    return total_loss / len(val_loader.dataset)

# 训练循环
epochs = 50
for epoch in range(1, epochs+1):
    train_loss = train_epoch()
    val_loss = eval_epoch()
    print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_lm.pt")
        wait = 0
        print("  -> 保存最佳模型")
    else:
        wait += 1
        if wait >= patience:
            print(f"早停于第 {epoch} 轮")
            break

# 加载最佳模型并生成
model.load_state_dict(torch.load("best_lm.pt", map_location=device))
model.to(device)

print("\n" + "="*60)
print("文本生成示例：")
prompts = ["It is a truth", "Mr. Bennet", "I have", "She said"]
for p in prompts:
    print(f"\nPrompt: {p}")
    out = model.generate(p, max_new_tokens=80, temperature=0.7, top_p=0.9)
    print(f"Generated: {out}")
