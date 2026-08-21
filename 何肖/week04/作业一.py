import torch
import torch.nn as nn
import torch.nn.functional as F

"""
多头注意力编码器
"""
class MultiHeadTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, num_head=8, ffn_d=2048, dropout=0.5):
        """
        多头注意力编码器
        初始化
        """
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.ffn_d = ffn_d
        self.dropout = dropout
        self.d_k = d_model // num_head

        # 多头注意力权重矩阵
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.attn_out = nn.Linear(d_model, d_model)

        # 前馈网络层
        self.ffn1 = nn.Linear(d_model, ffn_d)
        self.ffn2 = nn.Linear(ffn_d, d_model)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def self_attention(self, q, k, v):
        """
        多头注意力层
        :param q: 查询向量
        :param k: 键向量
        :param v: 值向量
        :return: 注意力输出
        """
        attn_score = torch.matmul(q, k.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn_score = F.softmax(attn_score, dim=-1)
        attention = torch.matmul(attn_score, v)
        return attention


    def forward(self, x):
        """
        多头注意力编码器前向传播
        :param x: 输入序列
        :return: 输出序列
        """
        residual_1 = x
        batch_size = x.shape[0]

        # 过线性层，得到q, k, v，分头并改变形状用于并行计算 原始形状(32,512) -> (32,5,8,64) -> (32,8,5,64)
        # [batch, seq, 512] -> 分头 [batch, heads, seq, 64]
        q = self.w_q(x).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)

        # 计算attention
        attn_out = self.self_attention(q, k, v)

        # 改变形状(32,8,5,64) -> (32,5,8,64)，多头拼接-> (32, 5, 512)
        attention = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 多头拼接后，过线性层
        attn = self.attn_out(attention)

        # 自注意力残差连接 + 层归一化1
        x = self.norm1(residual_1 + self.dropout(attn))

        # 前馈层
        residual2 = x
        ffn1 = self.ffn1(x)
        ffn1 = F.relu(ffn1)
        ffn2 = self.ffn2(ffn1)

        # 前馈层残差连接 + 层归一化2
        x = self.norm2(residual2 + self.dropout(ffn2))

        return x

# ========== 测试代码 ==========
if __name__ == "__main__":
    # 模拟embedding输入序列
    batch_size = 32
    seq_len = 5
    hidden_size = 512

    x = torch.randn(batch_size, seq_len, hidden_size)

    # 初始化模型
    model = MultiHeadTransformerEncoder()

    output = model(x)

    print(f"输入的形状是: {x.shape}")
    print(f"输出的形状是: {output.shape}")
