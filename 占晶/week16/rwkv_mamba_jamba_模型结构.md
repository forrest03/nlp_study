## 1. RWKV

### 核心结构

- Time Mixing
- Channel Mixing
- Recurrent State
- Decay / Gate 机制

### 特点

- 不使用标准 Self-Attention
- 不需要传统 KV Cache
- 历史信息压缩进固定大小的 State
- 训练可并行，推理类似 RNN
- 长上下文显存占用更稳定

## 2. Mamba

### 核心结构

- Input Projection
- Local 1D Convolution
- Selective State Space Model（SSM）
- Gate
- Recurrent State

### 特点

- 完全不依赖 Attention
- 核心是 Selective SSM
- 根据输入动态决定“记什么、忘什么”
- 序列计算复杂度接近 O(n)
- 推理只维护固定大小 State
- 长序列效率高，但精确回忆很早之前的信息相对弱

## 3. Jamba

### 核心结构

- Mamba Layers
- 少量 Attention Layers
- MLP / MoE
- Router + Experts

### 特点

- 属于 Hybrid 混合架构
- Mamba 负责高效处理长序列
- Attention 负责精确查找历史信息
- MoE 提升模型容量，同时减少实际激活参数
- 兼顾长上下文效率和 Transformer 的检索能力
