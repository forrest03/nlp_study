import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 1. 手写多头注意力 (Multi-Head Attention)
# ---------------------------
class DiyTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim      # 向量总维度
        self.num_heads = num_heads      # 头数
        self.head_dim = embed_dim // num_heads  # 每个头的维度

        # 3个线性层：Q K V
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)

        # 输出线性层
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        N, seq_len, embed_dim = x.shape  # (批次, 序列长度, 向量维度)

        # 1. 线性变换得到 Q K V
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        # 2. 拆分成多头：(N, 头数, 序列长度, 单头维度)
        Q = Q.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 注意力分数：Q * K^T / sqrt(head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 4. 加权求和 V
        out = torch.matmul(attn_weights, V)

        # 5. 拼接多头
        out = out.transpose(1, 2).contiguous().view(N, seq_len, embed_dim)

        # 6. 最后线性层
        out = self.fc_out(out)
        return out


# ---------------------------
# 2. 手写前馈网络 (Feed Forward)
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------
# 3. 手写 Transformer Block (Encoder Layer)
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = DiyTransformer(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 注意力 + 残差 + 归一化
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))

        # 前馈 + 残差 + 归一化
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# ---------------------------
# 测试一下！
# ---------------------------
if __name__ == "__main__":
    # 配置
    embed_dim = 128    # 向量维度
    num_heads = 8      # 8头注意力
    hidden_dim = 256   # 前馈层维度

    # 构建一个 transformer block
    block = TransformerBlock(embed_dim, num_heads, hidden_dim)

    # 构造输入：批次=2, 序列长度=10, 向量维度=128
    x = torch.randn(2, 10, embed_dim)

    # 前向传播
    out = block(x)
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)  # 形状不变！这就是Transformer