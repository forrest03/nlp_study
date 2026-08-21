# Qwen3.5 与 HY V3 模型结构调研

> 调研对象：`modeling_qwen3_5.py`（Qwen3.5，2152 行）与 `modeling_hy_v3.py`（Tencent HunYuan V3，608 行）
> 两者均为 HuggingFace transformers 的 modular 自动生成文件，风格一致，但架构路线完全不同。

---

## 一、总体定位

| 维度 | **HY V3**（HunYuan） | **Qwen3.5** |
|---|---|---|
| 架构路线 | 纯文本 Decoder-only **MoE（稀疏专家 + 共享专家）** | **多模态** Decoder-only，**混合注意力（线性注意力 + 全注意力）** |
| 模态 | 仅文本 | 文本 + 图像 + 视频（Vision Encoder 接入 LLM） |
| 核心卖点 | DeepSeek-V2 式 MoE + 分层稀疏/稠密 MLP 混合 | Gated Delta Net 线性注意力 + M-RoPE 多模态位置编码 |
| 输出头 | `HYV3ForCausalLM` | `Qwen3_5ForCausalLM`（纯文本）、`Qwen3_5ForConditionalGeneration`（多模态）等 5 个头 |

一句话：**HY 是在"稀疏化参数"上做文章（MoE），Qwen3.5 是在"稀疏化注意力复杂度"上做文章（线性注意力）+ 多模态融合。**

---

## 二、HY V3 结构特点（modeling_hy_v3.py）

标准 Pre-Norm Transformer 骨架，逐层可选 **dense MLP 或 sparse MoE**：

```
embed_tokens
  └─ DecoderLayer × N
        input_layernorm(RMSNorm)
        └─ Attention（GQA + q/k norm + RoPE）
        residual+
        post_attention_layernorm(RMSNorm)
        └─ MLP 或 MoE（由 config.mlp_layer_types[layer_idx] 决定）
        residual+
final norm(RMSNorm) → lm_head（与 embedding 权重绑定）
```

### 2.1 注意力（HYV3Attention）
- **GQA**：`q_proj` 输出 `num_attention_heads × head_dim`，`k/v_proj` 输出 `num_key_value_heads × head_dim`，通过 `repeat_kv` 扩展。
- **QK 归一化**：对每个 head 的 q/k 做 `HYV3RMSNorm(head_dim)`（仅 head 维度），**先 norm 后 RoPE**，与 Qwen3.5 一致，均为当前主流做法。
- 可配置 `attention_bias`、`head_dim`、`attention_dropout`；`scaling = head_dim^-0.5`。
- 支持 eager / Flash / SDPA / Flex 四种注意力后端。

### 2.2 MoE（HYV3MoE + HYV3TopKRouter + HYV3Experts）— 本文件最大的特色
- **Router 用 sigmoid 而非 softmax**（DeepSeek 风格）：
  1. `router_logits = F.linear(x, weight)` → `sigmoid`
  2. 加上 **`e_score_correction_bias`**（每个专家一个可学习偏置，buffer 初始化全 0，且在 fp32 下保持）
  3. `topk` 选 top_k 个专家
  4. top-k 权重归一化后 **再乘 `router_scaling_factor`**（关键差异点，注释里明确标注）
- **专家权重以 3D 张量存储**：`gate_up_proj (E, 2*inter, hidden)` 融合了 gate/up，`down_proj (E, hidden, inter)`；按 `expert_mask` 只对命中的专家循环计算（稀疏计算）。
- **共享专家（shared_experts）**：一个稠密 MLP，`intermediate = moe_intermediate_size * num_shared_experts`，与路由输出相加 —— DeepSeek-V2 式设计，缓解路由稀疏导致的知识丢失。
- `enable_moe_fp32_combine`：可选在 fp32 下合并路由输出与共享专家输出，再转回原 dtype。
- **无辅助损失**：`aux_loss=None`，注释 "Not used in this model"（无 load-balancing loss，靠 bias 校正）。

### 2.3 其他细节
- **分层稀疏/稠密**：`mlp_layer_types[layer_idx] == "sparse"` 用 MoE，否则用 dense MLP —— 层级别混合。
- MLP 为 SwiGLU：`down(act(gate(x)) * up(x))`，可配 `mlp_bias`。
- `_keys_to_ignore_on_load_unexpected = [r"model\.layers\.80.*"]`：加载时忽略第 80 层权重（疑似原模型第 80 层为其他结构，如 MoE→dense 转换或额外模块）。
- `_keep_in_fp32_modules_strict = ["e_score_correction_bias"]`。
- RoPE：`rope_parameters["rope_type"]` 支持 default / dynamic 等，`rope_theta` 可配。

---

## 三、Qwen3.5 结构特点（modeling_qwen3_5.py）

双塔结构：**Vision Encoder（Qwen3_5VisionModel）+ 语言模型（Qwen3_5TextModel）**，由 `Qwen3_5Model` 组合，多模态位置用 **M-RoPE** 对齐。

### 3.1 文本侧：混合注意力骨干（本文件最大的特色）

```
embed_tokens
  └─ DecoderLayer × N   （config.layer_types[layer_idx] 决定）
        input_layernorm(RMSNorm)
        ├─ layer_type == "linear_attention" → Qwen3_5GatedDeltaNet（线性注意力）
        └─ layer_type == "full_attention"   → Qwen3_5Attention（全注意力）
        residual+
        post_attention_layernorm(RMSNorm)
        └─ MLP（SwiGLU，无 bias）
        residual+
final norm(RMSNorm) → lm_head
```

#### ① 线性注意力：Qwen3_5GatedDeltaNet（Gated Delta Rule）
- 采用 **Gated Delta Rule**（FLA 库的 chunk / recurrent / fused 核），是状态空间模型（SSM）思想：每个头维护常量大小状态 `k_head_dim × v_head_dim`，**每步推理 O(1) 内存**，可超长上下文。
- 结构：
  - **因果深度卷积** `Conv1d`（depthwise，`linear_conv_kernel_dim`，SiLU 激活）作用在 QKV 拼接上，捕捉局部 token 依赖。
  - SSM 式离散化参数：`dt_bias`、`A_log`（A~U(0,16) 初始化后取 log）。
  - 投影：`in_proj_qkv`（Q+K+V）、`in_proj_z`（输出门）、`in_proj_b`（beta 门）、`in_proj_a`。
  - `beta = sigmoid(b)`，衰减 `g = -exp(A_log) * softplus(a + dt_bias)`。
  - q/k 过 **L2 norm**（`use_qk_l2norm_in_kernel=True`）。
  - 输出经 **`Qwen3_5RMSNormGated`**（RMSNorm 后用 SiLU(z) 门控）。
- 缓存状态化（`_is_stateful = True`）：缓存 `conv_state` + `recurrent_state`，支持单 token 解码与 chunk 续推两条路径。
- 纯 torch 回退实现（`torch_recurrent_gated_delta_rule` / `torch_chunk_gated_delta_rule`），无内核时自动降级并告警。

#### ② 全注意力：Qwen3_5Attention
- GQA + 每 head q/k 归一化（同 HY）。
- **q_proj 输出 2×head_dim**，切分为 query 和 **gate**；注意力输出乘 `sigmoid(gate)`（Qwen3/GLM 式输出门控）。
- RoPE 支持 `partial_rotary_factor`（部分维度旋转）。

#### ③ 文本 RoPE：M-RoPE（3D 多模态 RoPE）
- `Qwen3_5TextRotaryEmbedding` 生成 **3 组频率**（T 时间 / H 高 / W 宽，`mrope_section=[11,11,10]`），`apply_interleaved_mrope` 将 chunked 布局重排为 interleaved。
- position_ids 形状为 **(4, bs, seq)**：第 0 维是纯文本位置，后 3 维是 T/H/W。

#### ④ 分层掩码
- 线性注意力层与全注意力层用**不同的 mask**：全注意力用标准 causal mask；线性注意力用 `linear_attn_mask`（左 padding 语义，padding token 状态需清零；缓存续推或全 1 mask 时置 None）。

### 3.2 视觉侧：Qwen3_5VisionModel
- **PatchEmbed**：`Conv3d`（`temporal_patch_size × patch_size × patch_size`）→ 图像和视频统一走 3D 分块。
- **动态分辨率**：按 `grid_thw`(T,H,W) 网格，`pos_embed`（可学习）用双线性插值适配任意分辨率；`cu_seqlens` 打包变长序列注意力。
- 块结构：LayerNorm（非 RMSNorm）+ 非因果全注意力 + MLP（GELU），`Qwen3_5VisionPatchMerger`（LayerNorm+MLP）把 2×2 patch 合并为 LLM 输入 token。
- 旋转位置用 `Qwen3_5VisionRotaryEmbedding`（vision 专用，dim = head_dim/2）。

### 3.3 多模态融合（Qwen3_5Model）
- `get_image_features` / `get_video_features`：视觉 token 经 merger 后，`masked_scatter` 替换输入中 `<image>`/`<video>` 占位符。
- `get_rope_index`：按 `mm_token_type_ids` 分组（text=0/image=1/video=2），为每个视觉片段计算 3D 位置并拼接，产出 `position_ids(3,bs,seq)` 与 `rope_deltas`（生成阶段续推时增量修正位置）。
- 视频特殊处理：**时间戳分隔视频**（`<t1>...<frame1>...<t2>...<frame2>`），`video_grid_thw` 按帧数 `repeat_interleave` 拆分（与 Qwen2.5-VL 的差异点，文件注释明确说明）。
- `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`：支持（但本文件未实现）MTP 多 token 预测头。

---

## 四、核心对比

| 对比项 | HY V3 | Qwen3.5 |
|---|---|---|
| 稀疏化手段 | **MoE 参数稀疏**（路由到 top-k 专家 + 共享专家） | **注意力计算稀疏**（线性注意力替换部分全注意力层） |
| 注意力 | GQA + q/k norm + RoPE，全层相同 | 混合：部分层 GQA 全注意力，部分层 Gated Delta Rule 线性注意力 |
| 位置编码 | 标准 RoPE（`rope_theta`/`rope_type`） | 文本+视觉统一 **M-RoPE（3D T/H/W）** |
| 归一化 | RMSNorm（weight 初始化为 1） | RMSNorm（weight 初始化为 0，`1+weight` 形式）+ Gated RMSNorm |
| Router | sigmoid + 专家偏置 + scaling factor，无 aux loss | 无 MoE |
| 缓存 | 标准 KV Cache | 混合：全注意力层 KV Cache + 线性层 conv/recurrent state（`_is_stateful`） |
| 模态 | 纯文本 | 文本 + 图像 + 视频（Conv3d patch + 动态分辨率） |
| 分层差异化 | `mlp_layer_types`：dense/sparse MLP 混排 | `layer_types`：linear/full attention 混排 |
| 额外模块 | `_keys_to_ignore`：layer 80 权重 | `_keys_to_ignore`：mtp 权重 |
| 输出头数量 | 1（CausalLM） | 5（VisionModel/TextModel/CausalLM/ConditionalGeneration/分类） |

## 五、关键结论

1. **架构哲学相反**：HY 用 MoE 把"参数量"做大但每次只激活一部分；Qwen3.5 用线性注意力把"上下文成本"从 O(seq²) 降到 O(seq)，并接入视觉。
2. **共同点**：两者都采用 Pre-Norm + SwiGLU MLP + GQA + 每头 q/k RMSNorm 的现代主流基座设计；都基于 transformers modular 框架生成，支持 Flash/SDPA 后端、梯度检查点、输出捕获（router_logits/hidden_states/attentions）。
3. **工程细节差异**：
   - HY 保留 `e_score_correction_bias` 于 fp32、可选 fp32 合并，说明对 MoE 数值稳定性敏感；
   - Qwen3.5 为线性注意力做了纯 torch 回退 + 状态化缓存 + 分层 mask，工程复杂度明显更高（文件行数 2152 vs 608 也印证）。
4. **适用场景推测**：HY 适合大参数量、高吞吐的稠密推理/训练（激活参数少）；Qwen3.5 适合多模态 + 超长上下文（视频/图像 + 长文本），线性注意力层提供低内存增量解码。
