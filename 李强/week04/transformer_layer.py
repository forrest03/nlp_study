
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v 形状:  (batch, num_heads, seq_len, d_k)
    mask 形状:     (batch, 1, seq_len_q, seq_len_k) 或可广播形状
                   mask 中为 True/1 的位置会被 屏蔽(置为 -inf)
    """
    d_k = q.size(-1)

    # Q · K^T : 衡量每个 query 与所有 key 的相似度
    # scores 形状: (batch, num_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # mask 用于:
    #   - padding mask:把 <pad> 位置的注意力屏蔽,避免模型关注无意义的填充
    #   - causal mask :在 Decoder 中防止当前位置看到未来的 token
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # softmax 得到归一化的注意力权重(每一行加起来等于 1)
    attn = F.softmax(scores, dim=-1)

    # 用注意力权重对 value 做加权求和,得到上下文表示
    output = torch.matmul(attn, v)
    return output, attn


# ---------------------------------------------------------------------------
# 2. 多头注意力 (Multi-Head Attention)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # d_model 必须能被 num_heads 整除,因为每个头分到 d_model / num_heads 维
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 用 4 个独立的线性层分别得到 Q、K、V 以及最终的输出投影 W_O
        # 不用 bias 也常见,这里跟随 PyTorch 默认带上 bias
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query/key/value 形状: (batch, seq_len, d_model)
        在 Self-Attention 中三者相同;在 Cross-Attention 中 query 来自 decoder。
        """
        batch_size = query.size(0)

        # 1) 线性投影,得到 Q、K、V
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # 2) 拆分成多个头: (batch, seq, d_model) -> (batch, num_heads, seq, d_k)
        #    transpose 是为了让后续 matmul 在最后两维上进行
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3) 缩放点积注意力(每个头独立计算)
        out, attn = scaled_dot_product_attention(q, k, v, mask=mask)

        # 4) 把多头拼回去: (batch, num_heads, seq, d_k) -> (batch, seq, d_model)
        #    contiguous() 是因为 transpose 后内存不连续,view 之前需要它
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5) 最终的输出投影 W_O,把多头信息融合
        out = self.W_o(out)
        out = self.dropout(out)
        return out, attn


# ---------------------------------------------------------------------------
# 3. 位置前馈网络 (Position-wise Feed-Forward Network, FFN)
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 这里用 GELU 替代原论文的 ReLU,BERT/GPT 等现代模型的常见选择,
        # 因为 GELU 在 0 附近更平滑,实践中收敛更稳定。
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ---------------------------------------------------------------------------
# 4. Transformer Encoder Layer
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 两个 LayerNorm,分别用在 attention 和 ffn 之前
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout 用于残差路径,防止过拟合
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x    形状: (batch, seq_len, d_model)
        mask 形状: 可广播到 (batch, num_heads, seq_len, seq_len),True 表示屏蔽
        """
        # ---- 子层 1: 多头自注意力 (Pre-LN + 残差) ----
        # 注意:Self-Attention 的 q、k、v 都来自同一个 x,
        # 这样每个位置可以"看到"序列中所有位置,从而建模全局依赖。
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=mask)
        x = x + self.dropout1(attn_out)

        # ---- 子层 2: 前馈网络 (Pre-LN + 残差) ----
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)

        return x


# ---------------------------------------------------------------------------
# 5. 简单的自测:跑一个 forward,看输出形状是否正确
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, seq_len, d_model = 2, 10, 512
    num_heads, d_ff = 8, 2048

    layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout=0.1)

    # 模拟一批已经过 embedding + 位置编码的输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 构造一个 padding mask:假设每个样本最后 2 个位置是 <pad>
    # mask 的 True 位置会被注意力屏蔽
    pad_mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.bool)
    pad_mask[:, :, :, -2:] = True

    y = layer(x, mask=pad_mask)

    print("输入形状:", x.shape)        # (2, 10, 512)
    print("输出形状:", y.shape)        # (2, 10, 512) — 与输入完全一致
    print("参数数量:", sum(p.numel() for p in layer.parameters()))
