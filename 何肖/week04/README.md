# Transformer 多头注意力编码器

这是一个基于 PyTorch 实现的多头注意力编码器，实现了 Transformer 架构中的核心组件。

## 功能特点

- **多头自注意力机制**：实现并行的多头注意力计算
- **前馈网络**：包含两层线性变换和 ReLU 激活
- **残差连接**：在注意力层和前馈网络后都添加残差连接
- **层归一化**：使用 Layer Normalization 稳定训练过程
- **Dropout**：防止过拟合

## 架构组成

### 1. 多头注意力层
- Query、Key、Value 线性变换层
- 缩放点积注意力计算
- 多头分割和拼接
- 输出线性层

### 2. 前馈网络 (FFN)
- 第一层线性变换 (`d_model` → `ffn_d`)
- ReLU 激活函数
- 第二层线性变换 (`ffn_d` → `d_model`)

### 3. 规范化组件
- Layer Normalization (两个)
- Dropout 层

## 参数配置

- `d_model`：模型隐藏层维度，默认 512
- `num_head`：注意力头数量，默认 8
- `ffn_d`：前馈网络中间层维度，默认 2048
- `dropout`：dropout 概率，默认 0.5

## 使用方法

```python
import torch
from transformer2 import MultiHeadTransformerEncoder

# 创建模型实例
model = MultiHeadTransformerEncoder(
    d_model=512,
    num_head=8,
    ffn_d=2048,
    dropout=0.1
)

# 准备输入张量 [batch_size, seq_len, d_model]
batch_size = 32
seq_len = 10
x = torch.randn(batch_size, seq_len, 512)

# 前向传播
output = model(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
```

## 模型结构

1. **输入处理**：通过线性层生成 Q、K、V 并分割成多头
2. **注意力计算**：计算多头注意力并合并
3. **残差连接 + 层归一化**：注意力层的残差连接
4. **前馈网络**：两层线性变换 + 激活
5. **残差连接 + 层归一化**：前馈网络的残差连接

## 计算流程

- **多头分割**：`[batch, seq, d_model]` → `[batch, heads, seq, d_k]`
- **注意力计算**：`Q @ K^T / sqrt(d_k)` → `softmax` → `(scores @ V)`
- **多头合并**：`[batch, heads, seq, d_k]` → `[batch, seq, d_model]`
- **输出变换**：通过最终线性层

## 输出

- 输入形状：`[batch_size, seq_len, d_model]`
- 输出形状：`[batch_size, seq_len, d_model]`

该编码器保留了输入的序列长度和模型维度，适用于各种序列建模任务。