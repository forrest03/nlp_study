

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadSelfAttention(nn.Module):
    """
    多头自注意力（Multi-Head Self-Attention）。

    将 d_model 维拆成 nhead 路，每路 head_dim = d_model // nhead，
    各自做缩放点积注意力后再拼接，经 out_proj 映射回 d_model。
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 nhead ({nhead}) 整除")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        # 缩放点积注意力：scores = QK^T / sqrt(d_k)，防止点积过大导致 softmax 梯度消失
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 三路线性投影；Q/K/V 形状均为 (B, L, d_model)，后续再 reshape 成多 head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # 所有 head 拼接后做一次线性，混合不同 head 的信息
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            attn_mask: 可选。(L, L) 或 (B, L, L)。bool 时 True 表示禁止注意；float 时按元素加到 scores 上。
            key_padding_mask: (B, L)，True 表示该 key 位置为 padding，整列屏蔽。

        Returns:
            (B, L, d_model)
        """
        bsz, seq_len, _ = x.shape
        h, d_h = self.nhead, self.head_dim

        # (B, L, d) -> (B, L, H, d_h) -> (B, H, L, d_h)，便于 batch 与 head 维一起并行 matmul
        q = self.q_proj(x).view(bsz, seq_len, h, d_h).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, h, d_h).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, h, d_h).transpose(1, 2)
        # q, k, v: (B, H, L, d_h)

        # scores[b,h,i,j]：batch b、head h 下，query 位置 i 对 key 位置 j 的未归一化权重
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            # (B, L) -> (B, 1, 1, L)，在 key 维（最后一维）上广播：padding 列整列变为 -inf
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        if attn_mask is not None:
            # 与 scores 对齐到 (B, H, L, L)：2D 掩码共享给 batch/head；3D/4D 则保留 batch 维
            if attn_mask.dim() == 2:
                m = attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                m = attn_mask.unsqueeze(1)
            if m.dtype == torch.bool:
                scores = scores.masked_fill(m, float("-inf"))
            else:
                scores = scores + m

        # 在最后一个维度（key 序列维）上 softmax，得到每个 query 位置对各 key 的注意力分布
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # 对 V 加权求和：(B,H,L,L) @ (B,H,L,d_h) -> (B,H,L,d_h)
        ctx = torch.matmul(attn, v)
        # 合并 head：(B, H, L, d_h) -> (B, L, H, d_h) -> (B, L, d_model)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(ctx)


class TransformerEncoderLayer(nn.Module):
    """
    单层 Encoder：手写 Multi-Head Self-Attention + Feed-Forward，均为「子层输出 + 残差 + LayerNorm」。

    注：此处为 Post-LN 形式（先子层再残差再 LayerNorm）。部分实现采用 Pre-LN（先 Norm 再子层），
    训练稳定性与原版论文略有差异，但都是常见变体。

    Args:
        d_model: 隐藏维度，如 768。
        nhead: 注意力头数，如 12。
        dim_feedforward: FFN 中间层维度，如 3072（常为 4 * d_model）。
        dropout: 子层输出上的 dropout 概率。
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiheadSelfAttention(d_model, nhead, dropout=dropout)

        # 位置无关的两层 MLP：先升维再降维，增大表达能力（FFN 通常占参数大头）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()

        # 子层 1（注意力）与子层 2（FFN）各用一个 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout1：注意力输出；dropout2：FFN 中间层；dropout3：FFN 输出（在残差相加前）
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: 可选，见 MultiheadSelfAttention。
            key_padding_mask: (batch, seq_len)，True 表示该位置为 padding。

        Returns:
            同形状 (batch, seq_len, d_model)
        """
        # 子层 1：自注意力 + 残差 + Norm
        attn_out = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # 子层 2：FFN(x) = W2 * GELU(Drop(W1 x))，再残差 + Norm
        ff = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout3(ff))
        return x


def _demo():
    """用两段不等长文字 + Embedding 生成 (B,L,d_model)，并演示 key_padding_mask。"""
    d_model, nhead = 768, 12
    texts = [
        "这段文字会转成整数编号，再经 Embedding 变成向量。",
        "短句也要过同一层编码。",
    ]

    chars = sorted({ch for t in texts for ch in t})
    char2idx = {ch: i + 1 for i, ch in enumerate(chars)}
    pad_id = 0
    vocab_size = len(chars) + 1

    max_len = max(len(t) for t in texts)
    ids: list[list[int]] = []
    key_padding_mask: list[list[bool]] = []
    for t in texts:
        ids.append([char2idx[ch] for ch in t] + [pad_id] * (max_len - len(t)))
        key_padding_mask.append([False] * len(t) + [True] * (max_len - len(t)))

    emb = nn.Embedding(vocab_size, d_model)
    x = emb(torch.tensor(ids, dtype=torch.long))
    pad = torch.tensor(key_padding_mask, dtype=torch.bool)

    layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=3072,
        dropout=0.1,
    )
    y = layer(x, key_padding_mask=pad)
    print("samples:", texts)
    print("padded length:", max_len)
    print("in :", x.shape)
    print("out:", y.shape)


if __name__ == "__main__":
    _demo()
