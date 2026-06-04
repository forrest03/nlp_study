import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention implemented with PyTorch basic layers.

    Input shape:
        x: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len] or [batch_size, 1, 1, seq_len]
            1 means keep, 0 means mask.
    Output shape:
        [batch_size, seq_len, hidden_size]
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        # 多头注意力: hidden_size 能平均拆分到每个 head 中。
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
         
        # Q/K/V 的线性变换层，以及输出的线性变换层
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_size)
        return x.transpose(1, 2)

    def forward(self, x, attention_mask=None):
        query = self.transpose_for_scores(self.query(x))
        key = self.transpose_for_scores(self.key(x))
        value = self.transpose_for_scores(self.value(x))

        # 计算注意力分数，并进行缩放
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_size)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            # 将 mask 中的 0 替换为一个很大的负数，使得 softmax 后对应位置的概率接近于 0。
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        attention_probs = torch.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(x.size(0), x.size(1), self.hidden_size)
        return self.output(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network used after attention."""

    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return self.dropout(x)


class TransformerLayer(nn.Module):
    """
    One Transformer encoder layer.

    Structure:
        x -> self-attention -> dropout -> residual -> layer norm
          -> feed-forward   -> dropout -> residual -> layer norm
    """

    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_layer_norm = nn.LayerNorm(hidden_size)

        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.feed_forward_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask=None):
        attention_output = self.attention(x, attention_mask)

        # 注意力输出经过 dropout 后与输入 x 做残差连接，再进行 layer norm
        x = self.attention_layer_norm(x + self.attention_dropout(attention_output))

        feed_forward_output = self.feed_forward(x)
        x = self.feed_forward_layer_norm(x + feed_forward_output)
        return x


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    hidden_size = 768

    torch.manual_seed(42)
    inputs = torch.randn(batch_size, seq_len, hidden_size)

    # Example: the last token of the second sample is padding, so it is masked.
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ]
    )

    transformer_layer = TransformerLayer(
        hidden_size=hidden_size,
        num_heads=12,
        intermediate_size=3072,
        dropout=0.1,
    )
    outputs = transformer_layer(inputs, attention_mask)

    print("input shape:", inputs.shape)
    print("output shape:", outputs.shape)
