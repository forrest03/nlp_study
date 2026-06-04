import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------- 1. 缩放点积注意力 ---------------------
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

    d_k = q.size(-1)  # 每个头的维度

    # Q * K^T / sqrt(d_k)
    attn_scores = torch.matmul(q, k.swapaxes(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))


    # 注意力权重 + 输出
    attn_weights = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v)

    return output, attn_weights


# --------------------- 2. 多头注意力 ---------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "hidden_size 必须能被 num_attention_heads 整除"

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.d_k = hidden_size // num_attention_heads  # 每个头的维度

        # 3个线性层：生成 Q, K, V
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

        # 输出投影
        self.w_o = nn.Linear(hidden_size, hidden_size)

    def split_heads(self, x: torch.Tensor):
        """ [batch, seq_len, hidden_size] -> [batch, num_attention_heads, seq_len, d_k] """
        batch_size, seq_len, hidden_size = x.size()
        return x.view(batch_size, seq_len, self.num_attention_heads, self.d_k).swapaxes(1, 2)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        # 线性变换得到 Q K V
        q = self.w_q(q)
        print(q.shape)
        k = self.w_k(k)
        v = self.w_v(v)

        # 拆分成多头
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 计算注意力
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v)

        # 拼接多头
        attn_output = attn_output.swapaxes(1, 2).reshape(batch_size, -1, self.hidden_size)

        # 最终投影
        output = self.w_o(attn_output)
        return output, attn_weights


# --------------------- 3. 前馈网络 ---------------------
class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, d_ff: int, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, d_ff)  # 升维
        self.linear2 = nn.Linear(d_ff, hidden_size)  # 降维
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# --------------------- 4. 完整 Transformer Encoder 层 ---------------------
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, d_ff: int, dropout=0.2):
        super().__init__()
        self.mha = MultiHeadAttention(hidden_size, num_attention_heads)
        self.ffn = FeedForward(hidden_size, d_ff)

        # 两个 LayerNorm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # 1. 多头注意力 + 残差 + LayerNorm
        attn_output, _ = self.mha(x, x, x)  # 自注意力，qkv来自同一个输入
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. 前馈网络 + 残差 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


# 超参数
hidden_size = 768    # 模型维度
num_attention_heads = 12     # 注意力头数
d_ff = 3072      # 前馈网络中间维度
batch_size = 10  # 批次大小
seq_len = 20    # 序列长度

# 构造随机输入
x = torch.randn(batch_size, seq_len, hidden_size)

# 初始化 Transformer 层
transformer_layer = TransformerLayer(hidden_size, num_attention_heads, d_ff)

# 前向传播
output = transformer_layer(x)

print("输入形状:", x.shape)
print("输出形状:", output.shape)