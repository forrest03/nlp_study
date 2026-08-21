import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # 每个头的维度
        
        # 线性投影层，用于生成 Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Q, K, V: [batch, num_heads, seq_len, d_k]
        mask: [batch, seq_len] 或 [batch, 1, 1, seq_len] (可选)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # 将 mask 中为 0 的位置设置为极小值（-1e9）
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 1. 线性投影并拆分为多头
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 合并多头并线性变换
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)
        return output


class PositionwiseFeedForward(nn.Module):
    """前馈网络：FFN(x) = max(0, xW1 + b1)W2 + b2"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerLayer(nn.Module):
    """一个完整的 Transformer 编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 子层1：多头注意力 + 残差 + LayerNorm
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)   # 残差连接
        x = self.norm1(x)
        
        # 子层2：前馈网络 + 残差 + LayerNorm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x


# ------------------ 测试 ------------------
if __name__ == "__main__":
    # 超参数
    batch_size = 2
    seq_len = 5
    d_model = 512      # 模型维度
    num_heads = 8      # 注意力头数
    d_ff = 2048        # 前馈网络中间层维度
    
    # 随机输入 (batch, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 实例化一个 Transformer 层
    transformer_layer = TransformerLayer(d_model, num_heads, d_ff, dropout=0.1)
    
    # 前向传播（可选 mask，例如 padding mask，这里不传）
    output = transformer_layer(x)
    
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("Transformer 层参数量: {:,}".format(sum(p.numel() for p in transformer_layer.parameters())))
