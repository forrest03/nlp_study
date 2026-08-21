1、
Llama 3 —— 经典 Dense Transformer 路线
核心特点：
Decoder-only Transformer
Dense 架构
RMSNorm
RoPE
SwiGLU
GQA（Grouped Query Attention）
不采用 MoE
 Llama 是现代开源 LLM 生态里非常典型的 Dense Transformer 基线。
Dense：
Token
 ↓
Attention
 ↓
FFN
 ↓
下一层

MoE：

Token
 ↓
Router
 ↓
Expert 1 / Expert 2 / Expert 3 ...
 ↓
部分 Expert 被激活
 ↓
下一层



2、
Mixtral —— 经典稀疏 MoE
采用：
> Sparse Mixture-of-Experts Transformer
一个 MoE Layer 中存在多个 FFN Expert，Router 根据 Token 动态选择少数 Expert。
核心思想：
┌─ Expert 1
Token → Router ──┼─ Expert 2
                 ├─ Expert 3
                 └─ Expert 4
并不是所有 Expert 都参与计算。



3、
Mamba —— 状态空间模型 SSM
Transformer：
Token
 ↓
Self-Attention
 ↓
FFN

Mamba：
Token
 ↓
Selective State Space Model
 ↓
Hidden State
 ↓
Next Token

它的核心不是 Attention，而是 State Space Model（SSM）。
特点：
不依赖传统 Self-Attention
对长序列更加友好
通过状态空间递推处理信息
计算和内存效率




4、
Gemma 3 —— 轻量化 Transformer 路线
它的特点主要是：
Decoder-only Transformer
Dense 架构
强调较小参数规模下的性能
支持较长上下文
面向本地部署和资源受限场景
