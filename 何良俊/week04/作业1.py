
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决OpenMP重复初始化问题

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

'''

使用PyTorch实现Transformer层

'''

# PyTorch实现Transformer层
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072, dropout_prob=0.1):
        super(TransformerLayer, self).__init__()
        
        # 基础参数
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        
        # Multi-Head Attention: Q/K/V线性变换
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout_prob)
        
        # 层归一化
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        # Feed Forward Network
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.ff_dropout = nn.Dropout(dropout_prob)
        self.ff_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        # GELU激活函数
        self.gelu = nn.GELU()
    
    def transpose_for_scores(self, x):
        # 将输入转换为多头格式: [batch, seq_len, hidden] → [batch, heads, seq_len, head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        # 1. 多头注意力计算
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)
        
        # 加权求和得到上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        attention_output = self.attention_output(context_layer.view(*new_context_layer_shape))
        attention_output = self.attention_dropout(attention_output)
        
        # 2. 残差连接 + 层归一化
        attention_output = self.attention_layer_norm(hidden_states + attention_output)
        
        # 3. 前馈网络
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.gelu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.ff_dropout(layer_output)
        
        # 4. 残差连接 + 层归一化
        layer_output = self.ff_layer_norm(attention_output + layer_output)
        
        return layer_output

# 测试自定义Transformer层
# ==================== 测试Transformer层 ====================
# 创建Transformer层
transformer_layer = TransformerLayer(hidden_size=768, num_attention_heads=12, intermediate_size=3072)

# 测试输入：模拟batch_size=1, seq_len=4的输入
test_input = torch.randn(1, 4, 768)  # 形状: [batch, seq_len, hidden_size]

# 前向传播
output = transformer_layer(test_input)

# 打印测试结果
print("Transformer层测试结果")
print("-" * 40)
print(f"输入形状: {test_input.shape}")
print(f"输出形状: {output.shape}")
print(f"输入均值: {test_input.mean():.4f}, 方差: {test_input.var():.4f}")
print(f"输出均值: {output.mean():.4f}, 方差: {output.var():.4f}")
print("-" * 40)
print("Transformer层测试完成！")
