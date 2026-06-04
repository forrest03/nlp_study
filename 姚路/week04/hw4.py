import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    包含：
    - Multi-Head Self Attention
    - Feed Forward Network
    - Residual Connection
    - LayerNorm
    """

    def __init__(
        self,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=0.1
    ):
        super().__init__()

        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x,
        attn_mask=None,
        key_padding_mask=None
    ):
        """
        参数:
            x:
                [batch_size, seq_len, d_model]

            attn_mask:
                [seq_len, seq_len]

            key_padding_mask:
                [batch_size, seq_len]
                True 表示 padding 位置
        """

        # =========================
        # Self Attention
        # =========================
        attn_out, attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )

        # Residual + LayerNorm
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # =========================
        # Feed Forward
        # =========================
        ffn_out = self.ffn(x)

        # Residual + LayerNorm
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)

        return x


if __name__ == "__main__":

    batch_size = 2
    seq_len = 10
    d_model = 32

    model = TransformerEncoderLayer(
        d_model=d_model,
        n_heads=4,
        d_ff=128
    )

    x = torch.randn(batch_size, seq_len, d_model)

    out = model(x)

    print("input shape :", x.shape)
    print("output shape:", out.shape)
