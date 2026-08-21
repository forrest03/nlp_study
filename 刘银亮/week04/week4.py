import torch
import torch.nn as nn
import torch.nn.functional as F
import math


"""
基于 pytorch 实现 bert 模型的一个编码层, 需要包含以下部分
1. embedding层
2. self-attention层
3. feed-forward层
"""

class bert_encoder_layer(nn.Module):
    """
    实现 bert 模型的一个 Transformer Encoder 层
    """
    def __init__(self, vocab_size, max_position_embeddings=512, embedding_dim=768, num_attention_heads=12):
        super().__init__()
        # 定义 embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.segment_embedding = nn.Embedding(2, embedding_dim)
        self.position_embedding = nn.Embedding(max_position_embeddings, embedding_dim)

        # 定义 Q, K, V
        self.q = nn.Linear(embedding_dim, embedding_dim)
        self.k = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Linear(embedding_dim, embedding_dim)

        self.d_k = embedding_dim // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.embedding_dim = embedding_dim

        # 定义 feed-forward 层
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.GELU(),
            nn.Linear(2 * embedding_dim, embedding_dim)
        )

        # 定义 Layer Normalization 层
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    # 计算嵌入层
    def calculate_embedding(self, token_ids, seg_ids, pos_ids):
        token_emb = self.embedding(token_ids)
        seg_emb = self.segment_embedding(seg_ids)
        pos_emb = self.position_embedding(pos_ids)
        return token_emb + seg_emb + pos_emb

    def calculate_multihead_attention(self, token_emb):
        # 获取输入的维度信息: batch大小, 序列长度, embedding维度
        batch_size, seq_len, embedding_dim = token_emb.size()

        # 通过三个线性层得到 Q, K, V
        Q = self.q(token_emb)
        K = self.k(token_emb)
        V = self.v(token_emb)

        # 将 Q, K, V 从 (batch, seq_len, embedding_dim) 变换为 (batch, num_heads, seq_len, d_k)
        # view 用于重塑张量形状, transpose 用于交换维度 1 和 2
        Q = Q.view(batch_size, seq_len, self.num_attention_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_attention_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_attention_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数: Q * K^T / sqrt(d_k)
        # 使用 sqrt(d_k) 进行缩放, 防止点积值过大导致 softmax 梯度消失
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 对注意力分数进行 softmax 归一化, 得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        # 用注意力权重对 V 加权求和: attention_weights * V
        attention_output = torch.matmul(attention_weights, V)

        # 将多头的结果合并: 从 (batch, num_heads, seq_len, d_k) 变回 (batch, seq_len, embedding_dim)
        # transpose 交换维度, contiguous 确保内存连续, view 重塑为原始形状
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, embedding_dim)

        return attention_output
        
        
    def feed_forward(self, token_emb):
        """
        实现 feed-forward 层: 两层全连接, 中间使用 GELU 激活
        """
        return self.ffn(token_emb)


    def forward(self, token_ids, seg_ids, pos_ids):
        # 1. 计算嵌入层
        token_emb = self.calculate_embedding(token_ids, seg_ids, pos_ids)

        # 2. Self-Attention 层 + Add & Norm (残差连接 + 层归一化)
        attention_output = self.calculate_multihead_attention(token_emb)
        token_emb = token_emb + attention_output  # 残差连接
        token_emb = self.layer_norm1(token_emb)

        # 3. Feed-Forward 层 + Add & Norm (残差连接 + 层归一化)
        ffn_output = self.feed_forward(token_emb)
        token_emb = token_emb + ffn_output  # 残差连接
        token_emb = self.layer_norm2(token_emb)

        return token_emb
        