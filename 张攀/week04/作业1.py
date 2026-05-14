import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.fc2(self.activation(self.fc1(x)))
       

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # Q, K, V linear layers
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        # query, key, value: (batch, seq_len, d_model)
        batch_size = query.size(0)

        # 拆分成多个头
        Q = self.q_linear(query)    # (batch, seq_len, d_model)
        K = self.k_linear(key)      # (batch, seq_len, d_model)
        V = self.v_linear(value)    # (batch, seq_len, d_model)

        # 变形：(batch, seq_len, nhead, d_k) -> (batch, nhead, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # 计算注意力权重: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # softmax
        attn = F.softmax(scores, dim=-1)

        # 计算加权求和: attn @ V
        out = torch.matmul(attn, V) # (batch, nhead, seq_len, d_k)

        # 合并多头：(batch, nhead, seq_len, d_k) -> (batch, seq_len, nhead, d_k) -> (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 输出线性变换
        out = self.out_linear(out)

        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, d_ff)

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-attention: (add & norm)
        x = x + self.self_attn(x, x, x)
        x = self.layer_norm1(x)

        # Feed-forward: (add & norm)
        x = x + self.feed_forward(x)
        x = self.layer_norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff):
        super().__init__()
        
        self.layers = [TransformerEncoderLayer(d_model, nhead, d_ff) for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


if __name__ == '__main__':
    # 超参
    batch_size = 2
    seq_len = 10
    d_modle = 768
    n_head = 12
    d_ff = 3072
    num_layer = 12

    # 输入
    x = torch.randn(batch_size, seq_len, d_modle)

    # 创建编码器
    encoder = TransformerEncoder(num_layer, d_modle, n_head, d_ff)

    # 前向传播
    output = encoder(x)
    print(f"input shape: {x.shape}")
    print(f"output shape: {output.shape}")

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
