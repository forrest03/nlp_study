# Qwen3.8-27B vs 原始 Transformer 架构对比分析报告

## 一、模型总览

| 属性 | 原始 Transformer (2017) | Qwen3.8-27B (2026) |
|------|------------------------|---------------------|
| 论文 | Attention Is All You Need | Qwen3.8 技术报告 |
| 层数 | 6 (base) | 64 |
| 隐藏维度 | 512 | 5120 |
| 注意力头数 | 8 | 24 |
| KV 头数 | 8 (MHA) | 4 (GQA) |
| 头维度 | 64 | 256 |
| 词表大小 | ~37K | 248,320 |
| 最大上下文 | 512 | 262,144 (512x) |
| 参数量 | ~65M | ~27B (415x) |
| 模态 | 纯文本 | 文本 + 图片 + 视频 |

---

## 二、核心架构差异详解

### 2.1 注意力机制：从 MHA 到混合注意力

**原始 Transformer:**
```
所有层: 标准 Multi-Head Attention (MHA)
Q = W_q · X,  K = W_k · X,  V = W_v · X
Attention = softmax(QK^T / √d_k) · V
复杂度: O(n²)  ← 序列长度 n 的平方
```

**Qwen3.8-27B:**
```
64 层混合设计:
  ├─ 16 层: Full Attention（标准 + GQA + QK Norm + Gate）
  └─ 48 层: MLA（Gated DeltaNet，线性注意力）
      每 4 层插入一个 Full Attention 层

MLA 层复杂度: O(n)  ← 线性！
```

**关键差异：**

| 特性 | 原始 Transformer | Qwen3.8 Full Attention | Qwen3.8 MLA |
|------|-----------------|----------------------|-------------|
| 注意力类型 | MHA | GQA (24Q + 4KV) | Gated DeltaNet |
| 复杂度 | O(n²) | O(n²) | **O(n)** |
| Q/K Norm | 无 | RMSNorm | 无 |
| 输出门控 | 无 | Sigmoid gate | 内置门控 |
| 长序列 | 不可行 | 可接受 | 优秀 |

**首创者与目的（这些升级分别是谁第一个用的）：**

| 技术 | 首创 | 主要目的 |
|------|------|---------|
| MHA | Transformer 原文 (2017, Vaswani) | 多头并行，捕捉不同子空间的特征 |
| GQA | Google (2023, Ainslie) | 压缩 KV 缓存（相比 MQA 单头，质量更高；相比 MHA，显存更省） |
| Q/K Norm | 社区实践 (2023, YouJiacheng 等) | 防止注意力 logit 随层深爆炸，稳定大学习率训练 |
| 输出门控 | 混合架构探索 (2024, Jamba 等) | 动态调控注意力信息流，与线性注意力配合 |

GQA 的演进链路：MHA (2017) → MQA (2020, Shazeer，全层共享 KV) → GQA (2023, 中间分组)。Qwen3.8 取 24Q/4KV 的分组，是 Llama-2 验证过的折中选择。

**Full Attention 层的 Q/K Norm + Gate:**
```python
# 原始 Transformer
Q = W_q · X
K = W_k · X
Attention = softmax(QK^T / √d) · V

# Qwen3.8 Full Attention
Q = RMSNorm(W_q · X)       # ← Q 的 RMSNorm
K = RMSNorm(W_k · X)       # ← K 的 RMSNorm
gate = sigmoid(W_g · X)    # ← 门控信号
Attention = softmax(QK^T / √d) · V
Output = gate · (W_o · Attention)  # ← 门控输出
```

### 2.2 MLA (Gated DeltaNet)：线性注意力

**MLA 是一种状态空间模型（SSM）风格的线性注意力**，其核心是 Gated Delta Rule：

```
传统 Attention:  O = softmax(QK^T/√d) · V     → O(n²) 显存
Gated DeltaNet:  S_t = g_t · S_{t-1} + k_t · v_t^T  → O(n) 显存
                  O_t = q_t · S_t + β_t · v_t
```

**为什么混合使用？**

| 层类型 | 优势 | 劣势 |
|--------|------|------|
| Full Attention | 长程依赖捕捉能力强 | O(n²) 显存，长序列贵 |
| MLA | O(n) 显存，支持 256K 上下文 | 长程依赖略弱于 Full Attention |

每 4 层交替一次，兼顾效率与效果。

**首创者与目的：**

| 技术 | 首创 | 主要目的 |
|------|------|---------|
| MLA（低秩 KV 压缩） | DeepSeek-V2 (2024) | 把 KV 缓存压缩到低秩潜空间，推理显存/速度大幅优化 |
| Gated DeltaNet（线性注意力） | Gated Delta Networks (2024, Microsoft) | 用关联记忆（delta rule）做线性注意力，O(n) 复杂度 |
| 混合注意力（Full + 线性） | Jamba (2024, AI21) 开创，Qwen3/DeepSeek 跟进 | 线性层跑长序列省显存，全注意力层保精度 |

这里有个关键点：Qwen3.8 的 **MLA 与 DeepSeek-V2 的 MLA 是两条不同路线**——
- DeepSeek-V2 的 MLA：本质仍是标准注意力，只是用低秩投影压缩 KV，复杂度还是 O(n²)，主打**显存省**
- Qwen3.8 的 MLA：是 Gated DeltaNet 线性注意力，复杂度降到 O(n)，主打**长序列快**

目的都是为了解决同一个问题：Transformer 的 KV 缓存随序列长度线性增长，上下文越长越扛不住。

### 2.3 位置编码：从 Sinusoidal 到 MRoPE

**原始 Transformer:**
```
Sinusoidal 绝对位置编码:
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
固定编码，不可学习，不支持多模态
```

**Qwen3.8-27B:**
```
MRoPE（Multi-Resolution Rotary Position Embedding）:
- 3D 旋转位置编码（时间T + 高度H + 宽度W）
- RoPE theta = 10,000,000（原始为 10,000）
- Partial Rotary: 仅 25% 的 head_dim 应用旋转
- Interleaved MRoPE: THWTHWTHW... 交错排列
```

**Partial Rotary 的设计原因：**
```
head_dim = 256
  ├─ 前 64 维 (25%): 应用 RoPE 旋转 → 捕获位置信息
  └─ 后 192 维 (75%): 不旋转 → 保留语义信息
```

**首创者与目的：**

| 技术 | 首创 | 主要目的 |
|------|------|---------|
| Sinusoidal PE | Transformer 原文 (2017) | 给无位置信息的注意力注入绝对位置，固定不学习 |
| RoPE | RoFormer (2021, 苏剑林) | 用旋转矩阵把位置信息乘进 Q/K，天然支持相对位置、外推性更好 |
| Partial Rotary | Qwen 系列 (2024, 通义) | 只旋转部分维度，保留更多语义信息（Qwen 首创并开源验证） |
| MRoPE (3D) | Qwen2-VL (2024, 通义) | 把一维旋转扩展成 时间+高度+宽度 三维，支持图片/视频 |

为什么要从 Sinusoidal 换到 RoPE？原始 Sinusoidal 是**加在输入上的绝对位置**，长度外推能力弱（超过训练长度就失效）。RoPE 把位置信息**乘进 Q/K 点积**，模型学到的是相对位置关系，天然更容易外推到更长序列——这正是 262K 超长上下文的基石之一。

### 2.4 归一化：从 LayerNorm 到 RMSNorm

**原始 Transformer:**
```python
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
# 需要计算均值和方差，两个可学习参数
```

**Qwen3.8-27B:**
```python
RMSNorm(x) = γ · x / √(mean(x²) + ε)
# 只计算均方根，一个可学习参数（无 bias）
# 计算量减少约 30%，效果相当
```

**首创者与目的：**

RMSNorm 由 Zhang & Sennrich 于 2019 年提出（论文 *Root Mean Square Layer Normalization*）。其目的很纯粹——**简化 LayerNorm**：

| 对比项 | LayerNorm (2016) | RMSNorm (2019) |
|--------|-----------------|----------------|
| 计算 | 减均值 + 除方差 | 只除均方根 |
| 参数 | γ 和 β（两个） | 只有 γ（一个） |
| 出发点 | 保证分布稳定 | 作者发现**减去均值对效果影响极小**，干脆去掉 |

省掉均值减法和 bias 后，计算量降低约 30%，训练/推理更快。这个简化 2023 年被 LLaMA 大规模采用后，成了现代 LLM 的标配归一化。

### 2.5 前馈网络：从 ReLU 到 SwiGLU

**原始 Transformer:**
```python
FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂
# 两层全连接 + ReLU 激活
```

**Qwen3.8-27B:**
```python
FFN_SwiGLU(x) = W₂ · (SiLU(W₁ · x) ⊙ W₃ · x)
# Gate + Up + Down 三路结构
# SiLU(x) = x · sigmoid(x)，比 ReLU 更平滑
```

| 对比 | 原始 | Qwen3.8 |
|------|------|---------|
| 激活函数 | ReLU | SiLU (Swish) |
| 结构 | 2 层 (up + down) | 3 层 (gate + up + down) |
| 参数量 | 2 × hidden × ff | 3 × hidden × ff |
| 效果 | 基线 | 更好（PaLM 等验证） |

**首创者与目的：**

SwiGLU 出自 Noam Shazeer 2020 年论文 *GLU Variants Improve Transformer*，把 2016 年的门控线性单元（GLU）与 Swish 激活结合。目的是**用门控增强 FFN 的非线性表达能力**：

```
ReLU:     y = max(0, x)              # 简单阈值截断
Swish:    y = x · sigmoid(x)         # 平滑的非线性
SwiGLU:   y = SiLU(W_g·x) ⊙ (W_u·x) # 一路做门控，一路做变换
```

- 门控机制让模型能**按输入动态选择**要激活的神经元，表达能力更强
- 代价：参数量多 50%（3 层 vs 2 层）
- 2023 年 LLaMA 和 PaLM 采用后成为主流，收益超过参数量增加的代价

### 2.6 归一化位置：从 Post-Norm 到 Pre-Norm

**原始 Transformer:**
```
X → Attention(X) → Add → LayerNorm → FFN → Add → LayerNorm
     ↑ Post-Norm
```

**Qwen3.8-27B:**
```
X → RMSNorm → Attention → Add → RMSNorm → FFN → Add
     ↑ Pre-Norm (更稳定)
```

Pre-Norm 训练更稳定，不需要 warmup，梯度流更通畅。

**首创者与目的：**

Pre-Norm 结构最早出现在 2018-2019 年（GPT-2 等模型）。目的是**解决深层 Transformer 训练不稳定的问题**：

```
Post-Norm（原始）:  y = Norm(x + Attn(x))
   → 残差里穿过 LayerNorm，深层时梯度反复被缩放，容易消失/爆炸
   → 需要 warmup 学习率调度辅助

Pre-Norm（现代）:   y = x + Attn(Norm(x))
   → 残差恒等直通，梯度不经过归一化，训练稳定
   → 不需要 warmup，可以用更大学习率
```

- 代价：Pre-Norm 的模型在**浅层**效果略低于 Post-Norm
- 但对 64 层这种深度，稳定性的收益远大于那点效果损失
- 2019 年 GPT-2 采用后，Pre-Norm 成为所有大模型的标准结构

### 2.7 多模态：纯文本 → 视觉-语言

**原始 Transformer:** 仅处理文本 token。

**Qwen3.8-27B:** 原生视觉-语言模型（VLM）

```
视觉编码器（27层 ViT）:
  Image → Patch Embed (16×16) → Vision Transformer → Visual Features

文本模型（64层）:
  Text → Tokenizer → Text Model ←→ Visual Features (cross-attention)

特殊 Token:
  <|vision_start|> <|image_pad|> <|vision_end|>
  <|video_pad|>
```

**首创者与目的：**

| 技术 | 首创 | 主要目的 |
|------|------|---------|
| ViT（视觉编码器） | Google (2020, Dosovitskiy) | 把图像切 patch 当 token 序列，直接进 Transformer |
| VLM 融合 | Flamingo (2022, DeepMind) / BLIP-2 (2023) | 让语言模型"看懂"图像 |
| 统一视觉 token | Qwen2-VL 等 (2024) | 把图片/视频/文本统一到同一个 token 空间 |

原始 Transformer 只能吃文本 token，图像对它来说是一堆无关的数字。ViT 证明**图像切成 16×16 patch 后就是一张 token 序列**，天然可以和文本 token 拼在一起。Qwen3.8 沿用这条路：视觉编码器把图片/视频转成视觉特征，文本模型通过 cross-attention 融合，最终让"看图说话"成为端到端能力。

### 2.8 Thinking/Response 分离

**原始 Transformer:** 直接输出文本。

**Qwen3.8-27B:** 思考过程与最终回复分离

```
<|im_start|>assistant
 thinking
(内部推理过程，用户看不到)
 response
(最终回复，用户看到)
<|im_end|>
```

通过 `enable_thinking` 参数控制。

**首创者与目的：**

| 技术 | 首创 | 主要目的 |
|------|------|---------|
| Chain-of-Thought | Google (2022, Wei) | 提示词里引导模型"一步步想"再回答 |
| 显式 Reasoning | OpenAI o1 (2024) | 推理时多花计算换取准确率（Scaling Test-time Compute） |
| Thinking 开源化 | DeepSeek-R1 (2025) | 把可复现的推理模型开源 |
| thinking 开关 | Qwen3 (2025) | 让用户可控是否输出思考过程 |

原始 Transformer 是"问什么答什么"，一次性生成答案。o1 和 DeepSeek-R1 证明：**让模型在回答前先"写草稿"推理，复杂问题的准确率显著提升**。Qwen3.8 把这一能力标准化为 thinking / response 两个标签：
- 打开：模型先输出一段不可见的推理过程，再给最终答案
- 关闭：直接输出答案（省 token，适合简单问题）

---

## 三、架构差异总结表

| 组件 | 原始 Transformer (2017) | Qwen3.8-27B (2026) |
|------|------------------------|---------------------|
| **注意力机制** | MHA (全部层) | 混合: 16 Full Attention + 48 MLA |
| **MLA 线性注意力** | 无 | Gated DeltaNet，O(n) 复杂度 |
| **KV 头数** | = Q 头数 (MHA) | 4 (GQA, 6:1 压缩) |
| **Q/K Normalization** | 无 | RMSNorm on Q & K |
| **输出门控** | 无 | Sigmoid gate on attention output |
| **位置编码** | Sinusoidal (绝对) | MRoPE (3D 旋转) |
| **Partial Rotary** | 无 | 25% 维度应用旋转 |
| **归一化** | LayerNorm (Post-Norm) | RMSNorm (Pre-Norm) |
| **激活函数** | ReLU | SiLU (SwiGLU) |
| **FFN 结构** | 2 层 | 3 层 (Gate + Up + Down) |
| **上下文长度** | 512 | 262,144 (512x) |
| **模态** | 纯文本 | 文本 + 图片 + 视频 |
| **思考分离** | 无 | Thinking / Response 双通道 |
| **工具调用** | 无 | 原生 <tool_call> 支持 |
| **词表** | ~37K (BPE) | 248K (BPE) |
| **特殊标记** | 基本 (BOS/EOS/UNK) | 30+ 特殊标记 |

---

## 四、演进路线图

```
2017 ─── Transformer ─── 基础架构
  │          MHA + FFN + LayerNorm + Sinusoidal PE
  │
  ▼
2018 ─── GPT/BERT ─── Pre-Norm + GELU
  │
  ▼
2020 ─── GPT-3 ─── 规模扩展
  │
  ▼
2021 ─── RoPE + SwiGLU ─── 位置编码 + 激活函数升级
  │
  ▼
2023 ─── GQA + RMSNorm ─── KV 压缩 + 归一化简化
  │         Llama 系列
  ▼
2024 ─── Mamba/SSM ─── 线性注意力探索
  │
  ▼
2025 ─── Qwen3.5 ─── 混合注意力 (Full + MLA)
  │         DeepSeek-V2
  ▼
2026 ─── Qwen3.8 ─── 视觉-语言 + Thinking + Agent
            Gated DeltaNet + MRoPE + 256K 上下文
```

---

## 五、关键改进总结

| 改进 | 带来的收益 |
|------|-----------|
| MLA (Gated DeltaNet) | 48/64 层使用 O(n) 注意力，支持 256K 上下文 |
| Q/K RMSNorm | 训练更稳定，梯度流更平滑 |
| Gated Attention Output | 动态控制注意力信息流 |
| MRoPE 3D | 支持多模态（图片/视频）位置编码 |
| Partial Rotary (25%) | 保留更多语义信息，位置编码更高效 |
| Pre-Norm RMSNorm | 训练收敛更快，无需 warmup |
| SwiGLU | 更强的非线性表达能力 |
| Thinking/Response | 可控的思考过程，支持 Agent 场景 |
| 原生工具调用 | 直接支持 Function Calling |
| 视觉编码器 | 端到端多模态理解 |