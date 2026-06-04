'''
第四周作业：尝试用pytorch实现一个transformer层。

'''

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model #特征维度
        self.num_heads = num_heads #头参
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "每个头必须一样大，且d_model必须能被num_heads整除"
        
        #线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        #输出线性层
        self.fc_out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        #线性变换得到Q、K、V
        Q = self.W_q(x) 
        K = self.W_k(x) 
        V = self.W_v(x) 

        #分头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        #计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        #计算加权和
        attn_output = torch.matmul(attn_weights, V)
        #合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        #输出线形成层
        output = self.fc_out(attn_output)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x
    
class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(TransformerLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads) #多头注意力
        self.norm1 = nn.LayerNorm(d_model) #层归一化
        self.ffn = FeedForward(d_model, d_ff) #前馈网络
        self.norm2 = nn.LayerNorm(d_model) #层归一化
        
    def forward(self, x):
        #多头注意力
        attn_output = self.mha(x) #计算注意力输出
        x = self.norm1(x + attn_output) #残差连接和层归一化
        
        #前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output) #残差连接和层归一化
        
        return x
    
#测试代码
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4
    d_ff = 64
    
    x = torch.randn(batch_size, seq_len, d_model) #随机输入
    transformer_layer = TransformerLayer(d_model, num_heads, d_ff)
    output = transformer_layer(x)
    print(output.shape) 

'''
输出：torch.Size([2, 5, 16])
'''
