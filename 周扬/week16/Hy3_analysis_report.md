# Hy3 vs 原始 Transformer 架构对比分析报告

> 模型来源：[ModelScope - Tencent-Hunyuan/Hy3](https://www.modelscope.cn/models/Tencent-Hunyuan/Hy3)（腾讯混元团队，2026）

## 一、模型总览

| 属性 | 原始 Transformer (2017) | Hy3 (2026) |
|------|------------------------|------------|
| 架构 | 稠密 Dense | **稀疏 MoE（混合专家）** |
| 总参数量 | ~65M | **295B** |
| 激活参数量 | ~65M（全量激活） | **21B**（top-8 专家激活） |
| 层数 | 6 (base) | 80（+1 MTP 层，3.8B 参数） |
| 隐藏维度 | 512 | 4096 |
| 注意力头数 | 8 | 64 |
| KV 头数 | 8 (MHA) | 8 (GQA, 8:1 压缩) |
| 头维度 | 64 | 128 |
| FFN 中间维度 | 2048 | 13312（稠密 FFN）/ 1536（专家维度） |
| 专家数 | — | 192 专家，top-8 激活 + 1 共享专家 |
| 词表大小 | ~37K | 120,832 |
| 最大上下文 | 512 | 262,144 (256K) |
| 精度 | FP32 | BF16（另有 FP8 量化版） |
| 生成方式 | 逐 token 自回归 | **MTP 多 token 预测** |
| 模态 | 纯文本 | 纯文本（专注 Agent/编程/推理） |

---

## 二、核心架构差异详解

### 2.1 前馈网络：从稠密 FFN 到 MoE 稀疏（最大差异）

**原始 Transformer:**
```
每个 token 都经过同一个 FFN:
FFN(x) = W₂ · ReLU(W₁ · x) + b
参数量: 全量参与计算，dense 结构
```

**Hy3:**
```
MoE 稀疏架构:
FFN 变成 192 个专家（每个专家是一个小 FFN）+ 1 个共享专家
每个 token 只激活 top-8 个专家 + 共享专家:
  router: 计算 token 与 192 个专家的匹配度 → sigmoid → 取 top-8
  output = shared_expert(x) + Σ top-8 [ score_i · expert_i(x) ]
  router_scaling_factor = 2.826（放大路由分数）
```

**关键差异：**

| 特性 | 原始 Transformer | Hy3 MoE |
|------|-----------------|---------|
| 结构 | 1 个 FFN | 192 专家 + 1 共享专家 |
| 激活比例 | 100% | ~4%（8/192）+ 1 共享 |
| 总参数量 | 全量参与 | 295B 中只激活 21B |
| 计算成本 | 随参数线性增长 | **参数多但计算省** |

**首创者与目的：**

| 技术 | 首创 | 主要目的 |
|------|------|---------|
| MoE 概念 | Google (2017, Shazeer) | 把 FFN 拆成多个专家，路由只激活部分，**参数量大但 FLOPs 小** |
| top-k 路由 | Google (2017, Shazeer) | 每个 token 只走 k 个专家，稀疏激活 |
| 共享专家 | DeepSeek-MoE (2024) | 用一个共享专家吸收通用知识，减少专家冗余 |
| Sigmoid 路由 | DeepSeek-V3 (2024) | 每个 token 独立路由（不做 softmax 竞争），配合 norm_topk |

为什么要做 MoE？核心矛盾是**"大模型聪明但太贵"**：
- 稠密模型：参数量 = 计算量，想变聪明就要全量算
- MoE：把"知识容量"（295B）和"单次计算量"（21B）解耦——模型知道得多，但每次推理只算一小部分

Hy3 的 192 专家 top-8 激活，让 295B 模型拥有接近 21B 激活模型的计算成本，性价比是稠密模型无法比拟的（这就是它能"比肩更大尺寸旗舰"的原因）。

**Sigmoid 路由细节：**
```
传统 softmax 路由:   scores 在所有专家间归一化，专家间互相竞争
Hy3 sigmoid 路由:   每个专家独立判断"我是否合适"，取 top-8
                    避免 token 被强行塞给"最不差的"专家
```

### 2.2 生成方式：从单 token 自回归到 MTP 多 token 预测

**原始 Transformer:**
```
一次只预测下一个 token:
p(next_token | 前文) → 采样一个 token → 拼进上下文 → 再预测
逐 token 生成，串行等待
```

**Hy3:**
```
MTP (Multi-Token Prediction):
主模型后面挂一个 MTP 层（1 层，3.8B 参数）
主模型预测下一个 token 的同时，MTP 层并行预测下两个 token
```

**首创者与目的：**

| 技术 | 首创 | 主要目的 |
|------|------|---------|
| MTP 训练 | DeepSeek-V3 (2024) | 训练时一次学多个 token，**提升训练效率和数据利用率** |
| MTP 投机采样 | DeepSeek-V3 / EAGLE | 推理时用 MTP 层当"草稿模型"，**加速生成**（一次验证多个 token） |

原始 Transformer 只能逐 token 生成，一次算一个位置。MTP 让模型"多步一起想"：
- **训练阶段**：每个位置同时预测未来 1/2 个 token，模型学得更快
- **推理阶段**：配合 vLLM 的 `--speculative-config.method mtp`，用 MTP 层做投机采样，一次 draft 多个 token 再验证，生成速度大幅提升

### 2.3 注意力机制：从 MHA 到 GQA（8:1 压缩）

**原始 Transformer:**
```
MHA: 每个 query 头配一个独立的 K/V 头
8 头 → 8 份 KV，KV 缓存大
```

**Hy3:**
```
GQA: 64 个 query 头共享 8 个 KV 头
KV 缓存只有 MHA 的 1/8
head_dim = 128（比原始 64 大一倍，单头容量更大）
```

**首创者与目的：**

| 技术 | 首创 | 主要目的 |
|------|------|---------|
| MQA（1 KV 头） | Shazeer (2020) | 极致压缩 KV，但质量损失明显 |
| GQA（分组 KV） | Google (2023, Ainslie) | MQA 和质量之间的折中：KV 压缩 + 效果损失小 |
| qk_norm | 社区实践 (2023) | 稳定注意力 logit，支持大学习率 |

Hy3 采用 64Q/8KV（8:1 压缩），比 Qwen3.8 的 24Q/4KV（6:1）压缩更狠，进一步省 KV 缓存——对 256K 长上下文推理至关重要。

### 2.4 归一化：从 LayerNorm 到 RMSNorm（Pre-Norm）

**原始 Transformer:**
```python
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β   # Post-Norm
# 减均值 + 除方差 + 两个参数
```

**Hy3:**
```python
RMSNorm(x) = γ · x / √(mean(x²) + ε)          # Pre-Norm
# 只除均方根 + 一个参数，rms_norm_eps = 1e-5
```

**首创者与目的：**

| 技术 | 首创 | 主要目的 |
|------|------|---------|
| RMSNorm | Zhang & Sennrich (2019) | 省掉均值减法，计算量 -30%，效果几乎不变 |
| Pre-Norm | GPT-2 等 (2018-2019) | 残差恒等直通，深层训练稳定，免 warmup |

与 Qwen3.8 相同，这是现代 LLM 的标配。

### 2.5 前馈网络激活：从 ReLU 到 SiLU

**原始 Transformer:**
```python
FFN(x) = W₂ · ReLU(W₁ · x)   # ReLU 阈值截断
```

**Hy3:**
```python
hidden_act = "silu"
# 专家内部和稠密 FFN 都用 SiLU（Swish），配合门控结构
```

**首创者与目的：** SwiGLU/SiLU 出自 Shazeer (2020) *GLU Variants Improve Transformer*，门控非线性比 ReLU 平滑，表达能力更强；2023 年 LLaMA 大规模采用后成为主流。

### 2.6 位置编码：从 Sinusoidal 到 RoPE（大 theta 外推）

**原始 Transformer:**
```
Sinusoidal 绝对位置编码，固定不可学习，长度外推差
```

**Hy3:**
```
RoPE（旋转位置编码）+ 大 theta = 11,158,840
rope_theta ≈ 1.1e7（原始 Transformer 是 1e4）
```

**首创者与目的：**

| 技术 | 首创 | 主要目的 |
|------|------|---------|
| RoPE | RoFormer (2021, 苏剑林) | 相对位置编码，外推性好 |
| 大 rope_theta | Code Llama / YaRN (2023) | theta 越大，低频分量越多，**长距离位置区分度越好**，支撑 256K |

### 2.7 上下文长度：从 512 到 256K

```
原始 Transformer: max_position_embeddings = 512
Hy3:              max_position_embeddings = 262,144 (256K，512倍)
```

支撑 256K 的三驾马车：
1. **RoPE + 大 theta**（位置编码外推）
2. **GQA 8:1**（KV 缓存压缩）
3. **MoE**（激活参数少，长序列内存压力小）

### 2.8 Thinking/Reasoning：快慢思考分离

**原始 Transformer:** 一次性输出答案。

**Hy3:**
```
reasoning_effort 参数三种模式:
  "no_think"（默认）: 直接回复，日常对话
  "low":            轻量思考
  "high":           深度思维链（数学/编程/复杂推理）
```

**首创者与目的：**

| 技术 | 首创 | 主要目的 |
|------|------|---------|
| CoT | Google (2022, Wei) | 引导分步推理 |
| 显式 Reasoning | OpenAI o1 (2024) | 推理时多花计算换准确率 |
| Thinking 开源化 | DeepSeek-R1 (2025) | 可复现的开源推理模型 |
| 快慢思考融合 | 2025-2026（Qwen3 / Hy3 等） | 让用户按任务复杂度选择是否思考 |

与 Qwen3.8 的 `enable_thinking` 思路一致，Hy3 用 `reasoning_effort` 三档控制。

### 2.9 数值精度：从 FP32 到 BF16 / FP8

```
原始 Transformer: FP32 训练
Hy3:             BF16 训练（dtype=bfloat16）
                 另有 Hy3-FP8 量化版本，8 卡可部署
```

BF16 与 FP32 相比，指数位相同（范围一样），只是尾数精度减半，训练显存和速度显著改善。FP8 进一步把模型压到 8 卡可部署。

---

## 三、架构差异总结表

| 组件 | 原始 Transformer (2017) | Hy3 (2026) |
|------|------------------------|------------|
| **整体架构** | 稠密 Dense | 稀疏 MoE（192 专家 + 1 共享） |
| **激活参数** | 全量 | top-8 激活，295B→21B |
| **路由机制** | 无 | Sigmoid 路由 + norm_topk |
| **生成方式** | 单 token 自回归 | MTP 多 token 预测（投机采样） |
| **注意力** | MHA | GQA (64Q/8KV, 8:1) |
| **QK Norm** | 无 | 有 |
| **归一化** | LayerNorm (Post) | RMSNorm (Pre) |
| **激活函数** | ReLU | SiLU |
| **FFN** | 2 层稠密 | 192 专家（1536 维）+ 1 共享 |
| **位置编码** | Sinusoidal | RoPE（theta≈1.1e7） |
| **上下文** | 512 | 262,144 (512x) |
| **精度** | FP32 | BF16 / FP8 |
| **思考分离** | 无 | reasoning_effort: no_think/low/high |
| **工具调用** | 无 | 原生支持（vLLM `--tool-call-parser hy_v3`） |
| **词表** | ~37K | 120,832 |

---

## 四、演进路线图（MoE 技术线）

```
2017 ─── Transformer ─── 稠密自回归基础
  │         + MoE 概念（Shazeer, 仅 LSTM 实验）
  │
  ▼
2020 ─── MQA ─── KV 压缩尝试
  │
  ▼
2021 ─── RoPE + SwiGLU ─── 位置编码 + 激活升级
  │
  ▼
2023 ─── GQA + RMSNorm ─── KV 压缩折中 + 归一化简化
  │         + 大 theta RoPE（长上下文）
  │
  ▼
2024 ─── DeepSeek-V3 ─── MoE 工程化爆发
  │         Sigmoid 路由 + 共享专家 + MTP
  │         ↓
  │         DeepSeek-R1 ─── Thinking 开源化
  │
  ▼
2025 ─── MoE 成为旗舰标配 ─── 快慢思考融合
  │         （Qwen3-MoE / GLM / Hy 系列）
  │
  ▼
2026 ─── Hy3 ─── 295B MoE + MTP + 256K + Agent
            21B 激活，比肩更大尺寸旗舰
```

---

## 五、关键改进总结

| 改进 | 带来的收益 |
|------|-----------|
| MoE (192 专家 top-8) | 总参 295B 但只算 21B，性价比极高 |
| 共享专家 | 吸收通用知识，减少专家冗余 |
| Sigmoid 路由 | token 独立选专家，避免软最大竞争 |
| MTP 多 token 预测 | 训练效率 + 推理投机采样加速 |
| GQA 8:1 | KV 缓存降 8 倍，支撑 256K 上下文 |
| 大 theta RoPE | 长距离位置区分度高，外推性好 |
| RMSNorm Pre-Norm | 训练稳定、更快 |
| reasoning_effort | 快慢思考三档，按任务选 |
| BF16/FP8 | 8 卡可部署 295B 模型 |

---

## 六、Hy3 vs Qwen3.8-27B 快速对照（两条技术路线）

刚分析完 Qwen3.8-27B，正好对比一下这两大 2026 旗舰模型的技术路线差异：

| 维度 | Qwen3.8-27B | Hy3 |
|------|-------------|-----|
| 架构 | **稠密** 27B | **MoE** 295B（21B 激活） |
| 层数 | 64 | 80 (+1 MTP) |
| 隐藏维度 | 5120 | 4096 |
| 注意力 | 混合（16 Full + 48 MLA 线性） | 全 Full Attention + GQA 8:1 |
| 长序列方案 | **线性注意力 O(n)** | GQA 压缩 + MoE 省显存 |
| 上下文 | 262K | 262K |
| 生成加速 | 无 MTP | MTP 投机采样 |
| 模态 | **文本+图片+视频** | 纯文本（聚焦 Agent） |
| 思考 | enable_thinking 开关 | reasoning_effort 三档 |
| 定位 | 多模态通用助手 | 高性价比生产力/Agent 模型 |

**两条路线解决长上下文的不同哲学：**
- Qwen3.8：**改注意力本身**（MLA 线性化，把 O(n²) 降 O(n)）
- Hy3：**不改注意力**，用 GQA 压缩 KV + MoE 降低整体显存，注意力仍靠 256K 外推

两者殊途同归，都达到了 256K 上下文，但 Qwen3.8 靠架构革新，Hy3 靠稀疏化工程。
