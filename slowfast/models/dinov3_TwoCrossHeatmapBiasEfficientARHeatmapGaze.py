import torch
import torch.nn as nn

from slowfast.models import MODEL_REGISTRY
from slowfast.models.dinov3_HeatmapBiasEfficientARHeatmapGaze import (
    DINOv3_HeatmapBiasEfficientARHeatmapGaze,
)
from slowfast.utils import logging

logger = logging.get_logger(__name__)


class TwoCrossHeatmapBiasEfficientTransformerDecoderLayer(nn.Module):
    """
    Efficient decoder layer with separate history and current-frame cross-attention.

    Self-attention is restricted to query tokens. Historical image tokens are read
    first with an additive heatmap attention bias; current image tokens are read in
    a separate cross-attention block without bias.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.history_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.current_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, tgt, history_memory, current_memory, history_attn_bias=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        if history_memory is not None and history_memory.shape[1] > 0:
            tgt2, _ = self.history_cross_attn(
                tgt,
                history_memory,
                history_memory,
                attn_mask=history_attn_bias,
            )
            tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2, _ = self.current_cross_attn(tgt, current_memory, current_memory)
        tgt = self.norm3(tgt + self.dropout3(tgt2))

        tgt2 = self.ffn(tgt)
        tgt = self.norm4(tgt + tgt2)
        return tgt


@MODEL_REGISTRY.register()
class DINOv3_TwoCrossHeatmapBiasEfficientARHeatmapGaze(
    DINOv3_HeatmapBiasEfficientARHeatmapGaze
):
    """
    Exp11: efficient AR decoder with separate history and current cross-attention.

    Historical image tokens are read with historical heatmaps as additive attention
    bias. Current-frame image tokens are read in a second cross-attention block
    without bias, so history priors and current visual evidence do not compete in a
    single cross-attention softmax.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.history_type = "image"
        self.decoder_layers = nn.ModuleList(
            [
                TwoCrossHeatmapBiasEfficientTransformerDecoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.drop,
                )
                for _ in range(self.num_decoder_layers)
            ]
        )

        logger.info("DINOv3_TwoCrossHeatmapBiasEfficientARHeatmapGaze initialized")
        logger.info(
            "  - Cross-attn 1: query -> history image tokens with heatmap bias"
        )
        logger.info("  - Cross-attn 2: query -> current frame tokens without bias")

    def _build_history_heatmap_attention_bias(self, history_heatmaps, step=None):
        if self.heatmap_bias_weight == 0.0 or len(history_heatmaps) == 0:
            self.last_attention_bias_shape = None
            self._record_attention_bias_stats(None, step=step)
            return None

        history_biases = [self._heatmap_to_patch_bias(hm) for hm in history_heatmaps]
        memory_bias = torch.cat(history_biases, dim=1)
        attn_bias = memory_bias[:, None, :].expand(-1, self.num_query_tokens, -1)
        attn_bias = attn_bias.repeat_interleave(self.nhead, dim=0)
        self.last_attention_bias_shape = tuple(attn_bias.shape)
        self._record_attention_bias_stats(attn_bias, step=step)
        return attn_bias

    def _decode_step(
        self,
        t,
        current_tokens,
        predicted_heatmaps,
        visual_features,
        pos_encoding,
        gt_heatmap=None,
        train_ar=False,
        ss_prob=0.0,
        bias_schedule_epoch=None,
    ):
        batch_size = current_tokens.shape[0]

        if self.history_length > 0:
            history_heatmaps = self._get_history_heatmaps_for_bias(
                t,
                gt_heatmap,
                predicted_heatmaps,
                batch_size,
                train_ar=train_ar,
                ss_prob=ss_prob,
                bias_schedule_epoch=bias_schedule_epoch,
            )
        else:
            history_heatmaps = []
            self._last_history_heatmap_sources = []

        if not hasattr(self, "latest_heatmap_bias_sources"):
            self.latest_heatmap_bias_sources = []
        self.latest_heatmap_bias_sources.append(
            {
                "step": int(t),
                "sources": list(getattr(self, "_last_history_heatmap_sources", [])),
            }
        )
        history_tokens = (
            self._images_to_tokens(visual_features, t, pos_encoding)
            if self.history_length > 0
            else None
        )
        history_attn_bias = self._build_history_heatmap_attention_bias(
            history_heatmaps,
            step=t,
        )

        query_type = self.token_type_embed(
            torch.tensor(1, device=current_tokens.device)
        )
        tgt = (
            self.query_tokens.expand(batch_size, -1, -1)
            + pos_encoding
            + query_type.view(1, 1, -1)
        )

        for layer in self.decoder_layers:
            tgt = layer(
                tgt,
                history_tokens,
                current_tokens,
                history_attn_bias=history_attn_bias,
            )

        return self._decode_tokens_to_heatmap(tgt, batch_size)
