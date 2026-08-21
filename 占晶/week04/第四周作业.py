import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力层

    输入:
        x: [batch_size, seq_len, hidden_size]

    输出:
        output: [batch_size, seq_len, hidden_size]
        attention_probs: [batch_size, num_heads, seq_len, seq_len]
    """

    def __init__(self, hidden_size, num_heads, dropout_prob=0.1):
        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Q、K、V 三个线性层
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # 多头拼接后的输出线性层
        self.out_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        """
        把 [batch_size, seq_len, hidden_size]
        变成 [batch_size, num_heads, seq_len, head_dim]
        """

        batch_size, seq_len, hidden_size = x.shape

        # [batch_size, seq_len, hidden_size]
        # -> [batch_size, seq_len, num_heads, head_dim]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # [batch_size, seq_len, num_heads, head_dim]
        # -> [batch_size, num_heads, seq_len, head_dim]
        x = x.transpose(1, 2)

        return x

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, hidden_size = x.shape

        # 1. 生成 Q、K、V
        # q/k/v: [batch_size, seq_len, hidden_size]
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # 2. 拆成多头
        # q/k/v: [batch_size, num_heads, seq_len, head_dim]
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # 3. 计算注意力分数
        # k.transpose(-1, -2): [batch_size, num_heads, head_dim, seq_len]
        # attention_scores: [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        # 4. 缩放，防止点积结果过大
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # 5. 加 attention mask
        if attention_mask is not None:
            # attention_mask 常见形状: [batch_size, seq_len]
            # 其中 1 表示真实 token，0 表示 padding token
            #
            # 扩展成:
            # [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask[:, None, None, :]

            # padding 位置填充为一个很小的负数
            # softmax 后这些位置的概率接近 0
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0,
                -1e9
            )

        # 6. softmax 得到注意力权重
        # attention_probs: [batch_size, num_heads, seq_len, seq_len]
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 7. 用注意力权重加权 V
        # context: [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attention_probs, v)

        # 8. 多头拼接
        # [batch_size, num_heads, seq_len, head_dim]
        # -> [batch_size, seq_len, num_heads, head_dim]
        context = context.transpose(1, 2).contiguous()

        # [batch_size, seq_len, num_heads, head_dim]
        # -> [batch_size, seq_len, hidden_size]
        context = context.view(batch_size, seq_len, hidden_size)

        # 9. 输出线性层
        output = self.out_linear(context)

        return output, attention_probs


class FeedForward(nn.Module):
    """
    Transformer 中的前馈网络 FFN

    结构:
        hidden_size -> intermediate_size -> hidden_size
    """

    def __init__(self, hidden_size, intermediate_size, dropout_prob=0.1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # [batch_size, seq_len, hidden_size]
        x = self.fc1(x)

        # [batch_size, seq_len, intermediate_size]
        x = F.gelu(x)

        x = self.dropout(x)

        # [batch_size, seq_len, hidden_size]
        x = self.fc2(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """
    手写 Transformer Encoder Layer

    结构:
        x
        -> Multi-Head Self-Attention
        -> Add & LayerNorm
        -> Feed Forward
        -> Add & LayerNorm
    """

    def __init__(
        self,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        dropout_prob=0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_prob=dropout_prob
        )

        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_prob=dropout_prob
        )

        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, attention_mask=None):
        """
        x:
            [batch_size, seq_len, hidden_size]

        attention_mask:
            [batch_size, seq_len]
            1 表示真实 token
            0 表示 padding token
        """

        # 1. Self-Attention
        attention_output, attention_probs = self.self_attention(
            x,
            attention_mask=attention_mask
        )

        # 2. 残差连接 + LayerNorm
        x = self.attention_layer_norm(x + self.dropout(attention_output))

        # 3. Feed Forward
        ffn_output = self.feed_forward(x)

        # 4. 残差连接 + LayerNorm
        x = self.ffn_layer_norm(x + self.dropout(ffn_output))

        return x, attention_probs


if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    hidden_size = 768
    num_heads = 12
    intermediate_size = 3072

    # 模拟输入
    x = torch.randn(batch_size, seq_len, hidden_size)

    # 模拟 attention_mask
    # 第一条样本 4 个 token 都有效
    # 第二条样本最后一个 token 是 padding
    attention_mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 0]
    ])

    layer = TransformerEncoderLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        dropout_prob=0.1
    )

    output, attention_probs = layer(x, attention_mask)

    print("output.shape =", output.shape)
    print("attention_probs.shape =", attention_probs.shape)
