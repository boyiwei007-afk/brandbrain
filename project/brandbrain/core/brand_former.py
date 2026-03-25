"""
BrandFormer: 品牌销量预测Transformer模型
- LSTM编码器：捕捉局部时序依赖
- Transformer编码器：捕捉长程模式
- Cross-Attention：外生变量融合
- 分位数输出：P10/P50/P90置信区间
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ──────────────────────────────────────────────
# 基础组件
# ──────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """GLU门控残差网络，参考TFT"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim * 2)  # gate + value
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        value, gate = h.chunk(2, dim=-1)
        h = value * torch.sigmoid(gate)  # GLU
        return self.norm(h + self.residual(x))


class PatchEmbedding(nn.Module):
    """将时序切成Patch并嵌入"""
    def __init__(self, patch_size: int, d_model: int, input_dim: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        returns: (B, num_patches, d_model)
        """
        B, T, C = x.shape
        p = self.patch_size
        n_patches = T // p
        x = x[:, : n_patches * p, :]   # truncate
        x = x.reshape(B, n_patches, p * C)
        x = self.norm(self.proj(x))
        return x


class ExogEmbedding(nn.Module):
    """外生变量嵌入层"""
    def __init__(self, n_exog: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_exog, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_exog) → (B, T, d_model)"""
        return self.proj(x)


# ──────────────────────────────────────────────
# LSTM时序编码器
# ──────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, input_dim)  # 映射回输入维度以便残差

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """x: (B, T, input_dim) → out: (B, T, input_dim), hidden"""
        out, hidden = self.lstm(x)
        out = self.proj(out)
        return out + x, hidden  # 残差连接


# ──────────────────────────────────────────────
# Cross-Attention融合层
# ──────────────────────────────────────────────

class ExogCrossAttention(nn.Module):
    """时序patches cross-attend to 外生变量"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,   # (B, Tp, d_model) — patch tokens
        key_value: torch.Tensor,  # (B, Te, d_model) — exog tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.cross_attn(query, key_value, key_value)
        out = self.norm(query + self.dropout(attn_out))
        return out, attn_weights   # attn_weights: (B, Tp, Te)


# ──────────────────────────────────────────────
# 主模型：BrandFormer
# ──────────────────────────────────────────────

class BrandFormer(nn.Module):
    """
    品牌销量预测Transformer

    输入：
        x_hist:   (B, lookback, 1)            — 历史销量（归一化）
        x_exog:   (B, lookback, n_exog)       — 历史外生变量
        x_future: (B, horizon, n_future_exog) — 未来已知变量（节假日、促销计划等）

    输出：
        preds:   (B, horizon, n_quantiles)    — 分位数预测
        attn:    dict                         — 注意力权重（用于可视化）
    """

    def __init__(
        self,
        n_exog: int,
        n_future_exog: int,
        patch_size: int = 7,
        lookback: int = 60,
        horizon: int = 14,
        d_model: int = 64,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        quantiles: list = None,
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.patch_size = patch_size
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        n_q = len(self.quantiles)

        # ── 嵌入层 ──
        self.patch_embed = PatchEmbedding(patch_size, d_model, input_dim=1)
        self.exog_embed = ExogEmbedding(n_exog, d_model, dropout)
        self.future_embed = ExogEmbedding(n_future_exog, d_model, dropout) if n_future_exog > 0 else None

        # ── LSTM局部编码 ──
        self.lstm_encoder = LSTMEncoder(d_model, lstm_hidden, lstm_layers, dropout)

        # ── Transformer自注意力 ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers, enable_nested_tensor=False,
        )

        # ── Cross-Attention：时序→外生变量 ──
        self.cross_attn = ExogCrossAttention(d_model, n_heads, dropout)

        # ── 位置编码 ──
        self.pos_enc = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        # ── 预测头 ──
        self.grn_out = GatedResidualNetwork(d_model, d_ff, d_model, dropout)
        self.forecast_head = nn.Linear(d_model, n_q)  # 每步输出n_quantiles个值

        # ── 解码器Query（可学习）──
        self.decoder_queries = nn.Parameter(torch.randn(1, horizon, d_model))

        # ── Decoder cross-attention（查询未来）──
        self.decoder_cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.decoder_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_hist: torch.Tensor,
        x_exog: torch.Tensor,
        x_future: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        B = x_hist.size(0)

        # 1. Patch嵌入历史销量 → (B, n_patches, d_model)
        patch_tokens = self.patch_embed(x_hist)

        # 2. 外生变量嵌入（按patch聚合，取每个patch起始处的exog值）
        p = self.patch_size
        n_patches = patch_tokens.size(1)
        # 对每个patch取中间时刻的exog特征
        exog_patch = x_exog[:, p // 2 : p // 2 + n_patches * p : p, :]  # (B, n_patches, n_exog)
        # 如果索引超出范围则截断并用最后值填充
        if exog_patch.size(1) < n_patches:
            pad = exog_patch[:, -1:, :].expand(B, n_patches - exog_patch.size(1), -1)
            exog_patch = torch.cat([exog_patch, pad], dim=1)
        exog_tokens = self.exog_embed(exog_patch)  # (B, n_patches, d_model)

        # 3. LSTM局部编码
        patch_tokens, _ = self.lstm_encoder(patch_tokens)

        # 4. Cross-Attention：patch tokens 关注 exog tokens
        patch_tokens, cross_attn_w = self.cross_attn(patch_tokens, exog_tokens)

        # 5. 加位置编码 + Transformer自注意力
        patch_tokens = self.pos_enc(patch_tokens)
        enc_out = self.transformer(patch_tokens)   # (B, n_patches, d_model)

        # 6. 解码：学习到的queries cross-attend to encoder输出
        queries = self.decoder_queries.expand(B, -1, -1)  # (B, horizon, d_model)
        if x_future is not None and self.future_embed is not None:
            future_tokens = self.future_embed(x_future)   # (B, horizon, d_model)
            queries = queries + future_tokens

        dec_out, dec_attn_w = self.decoder_cross(queries, enc_out, enc_out)
        dec_out = self.decoder_norm(queries + dec_out)    # (B, horizon, d_model)

        # 7. 输出头 → (B, horizon, n_quantiles)
        dec_out = self.grn_out(dec_out)
        preds = self.forecast_head(dec_out)

        attn_dict = {
            "cross_attn": cross_attn_w.detach(),   # (B, n_patches, n_exog_tokens)
            "decoder_attn": dec_attn_w.detach(),   # (B, horizon, n_patches)
        }
        return preds, attn_dict


# ──────────────────────────────────────────────
# 损失函数
# ──────────────────────────────────────────────

class QuantileLoss(nn.Module):
    """分位数损失（Pinball Loss）"""
    def __init__(self, quantiles: list = None):
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        preds:  (B, T, n_q)
        target: (B, T, 1) or (B, T)
        """
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            e = target - preds[..., i : i + 1]
            loss += torch.mean(torch.max(q * e, (q - 1) * e))
        return loss / len(self.quantiles)


def build_model(
    n_exog: int,
    n_future_exog: int,
    granularity: str = "daily",
    config: dict = None,
) -> BrandFormer:
    """工厂函数：根据粒度和配置创建模型"""
    from config import GRANULARITY_CONFIG, BRANDFORMER_CONFIG

    g_cfg = GRANULARITY_CONFIG[granularity]
    m_cfg = config or BRANDFORMER_CONFIG

    return BrandFormer(
        n_exog=n_exog,
        n_future_exog=n_future_exog,
        patch_size=g_cfg["patch_size"],
        lookback=g_cfg["lookback"],
        horizon=g_cfg["forecast_horizon"],
        d_model=m_cfg["d_model"],
        n_heads=m_cfg["n_heads"],
        n_transformer_layers=m_cfg["n_encoder_layers"],
        lstm_hidden=m_cfg["lstm_hidden"],
        lstm_layers=m_cfg["lstm_layers"],
        d_ff=m_cfg["d_ff"],
        dropout=m_cfg["dropout"],
        quantiles=m_cfg["quantiles"],
    )
