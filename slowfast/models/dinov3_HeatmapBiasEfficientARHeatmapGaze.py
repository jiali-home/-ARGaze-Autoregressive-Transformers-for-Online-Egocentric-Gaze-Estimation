import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from slowfast.models import MODEL_REGISTRY
from slowfast.models.dinov3_EfficientARHeatmapGaze import DINOv3_EfficientARHeatmapGaze
from slowfast.utils import logging

logger = logging.get_logger(__name__)


class HeatmapBiasEfficientTransformerDecoderLayer(nn.Module):
    """
    Efficient decoder layer with an additive cross-attention prior.

    Self-attention stays restricted to query tokens. Cross-attention reads the
    history/current image-token memory and can receive a per-example float
    attention mask shaped for torch MultiheadAttention.
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

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, attn_bias=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=attn_bias)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + tgt2)
        return tgt


@MODEL_REGISTRY.register()
class DINOv3_HeatmapBiasEfficientARHeatmapGaze(DINOv3_EfficientARHeatmapGaze):
    """
    Exp10: efficient AR decoder with historical heatmaps as attention bias.

    Memory contains full historical image tokens plus current image tokens. The
    historical heatmaps are not concatenated as tokens; they only add a soft
    spatial prior to cross-attention over history image tokens.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.history_type = getattr(cfg.MODEL, "HISTORY_TYPE", "image")
        self.heatmap_bias_weight = float(getattr(cfg.MODEL, "HEATMAP_BIAS_WEIGHT", 1.0))
        self.heatmap_bias_eps = float(getattr(cfg.MODEL, "HEATMAP_BIAS_EPS", 1e-6))
        self.heatmap_bias_normalize = getattr(
            cfg.MODEL, "HEATMAP_BIAS_NORMALIZE", "softmax"
        ).lower()
        self.heatmap_bias_source_mode = getattr(
            cfg.MODEL, "HEATMAP_BIAS_SOURCE_MODE", "pred"
        ).lower()
        self.heatmap_bias_center_sigma = float(
            getattr(cfg.MODEL, "HEATMAP_BIAS_CENTER_SIGMA", -1.0)
        )
        self.heatmap_bias_schedule_phase_epochs = list(
            getattr(cfg.MODEL, "HEATMAP_BIAS_SCHEDULE_PHASE_EPOCHS", [10, 20])
        )
        self.heatmap_bias_schedule_ratios = [
            list(r)
            for r in getattr(
                cfg.MODEL,
                "HEATMAP_BIAS_SCHEDULE_RATIOS",
                [[0.8, 0.2, 0.0], [0.5, 0.3, 0.2], [0.3, 0.3, 0.4]],
            )
        ]
        self.heatmap_bias_jitter_shift = float(
            getattr(cfg.MODEL, "HEATMAP_BIAS_JITTER_SHIFT", 0.05)
        )
        self.heatmap_bias_jitter_scale_min = float(
            getattr(cfg.MODEL, "HEATMAP_BIAS_JITTER_SCALE_MIN", 0.9)
        )
        self.heatmap_bias_jitter_scale_max = float(
            getattr(cfg.MODEL, "HEATMAP_BIAS_JITTER_SCALE_MAX", 1.15)
        )

        self.decoder_layers = nn.ModuleList(
            [
                HeatmapBiasEfficientTransformerDecoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.drop,
                )
                for _ in range(self.num_decoder_layers)
            ]
        )
        self.last_attention_bias_shape = None
        self.latest_attention_bias_stats = []
        self.latest_heatmap_bias_sources = []

        logger.info("DINOv3_HeatmapBiasEfficientARHeatmapGaze initialized")
        logger.info(
            "  - Memory: history image tokens + current image tokens; heatmaps are attention bias only"
        )
        logger.info(
            "  - Heatmap bias: weight=%s eps=%s normalize=%s",
            self.heatmap_bias_weight,
            self.heatmap_bias_eps,
            self.heatmap_bias_normalize,
        )
        logger.info("  - Heatmap bias source mode: %s", self.heatmap_bias_source_mode)

    def _center_heatmap_for_bias(self, batch_size, device, dtype):
        sigma = self.heatmap_bias_center_sigma
        if sigma <= 0.0:
            sigma = float(self.heatmap_size) / 6.0
        coords = torch.arange(self.heatmap_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        center = (float(self.heatmap_size) - 1.0) / 2.0
        heatmap = torch.exp(
            -((xx - center) ** 2 + (yy - center) ** 2) / (2.0 * sigma * sigma)
        )
        return heatmap.view(1, 1, self.heatmap_size, self.heatmap_size).expand(
            batch_size, -1, -1, -1
        )

    def _uniform_heatmap_for_bias(self, batch_size, device, dtype):
        return torch.ones(
            batch_size,
            1,
            self.heatmap_size,
            self.heatmap_size,
            device=device,
            dtype=dtype,
        )

    def _noisy_heatmap_for_bias(self, batch_size, device, dtype):
        return torch.rand(
            batch_size,
            1,
            self.heatmap_size,
            self.heatmap_size,
            device=device,
            dtype=dtype,
        )

    def _get_heatmap_bias_schedule_ratios(self, bias_schedule_epoch=None):
        if len(self.heatmap_bias_schedule_ratios) != 3:
            raise ValueError(
                "MODEL.HEATMAP_BIAS_SCHEDULE_RATIOS must contain three "
                "[gt, jittered_gt, pred] rows."
            )
        epoch = 0 if bias_schedule_epoch is None else int(bias_schedule_epoch)
        cutoffs = self.heatmap_bias_schedule_phase_epochs
        if len(cutoffs) < 2:
            raise ValueError(
                "MODEL.HEATMAP_BIAS_SCHEDULE_PHASE_EPOCHS must contain at least two cutoffs."
            )
        if epoch < int(cutoffs[0]):
            ratios = self.heatmap_bias_schedule_ratios[0]
        elif epoch < int(cutoffs[1]):
            ratios = self.heatmap_bias_schedule_ratios[1]
        else:
            ratios = self.heatmap_bias_schedule_ratios[2]
        if len(ratios) != 3:
            raise ValueError(
                "Each MODEL.HEATMAP_BIAS_SCHEDULE_RATIOS row must be [gt, jittered_gt, pred]."
            )
        total = float(sum(ratios))
        if total <= 0.0:
            raise ValueError("MODEL.HEATMAP_BIAS_SCHEDULE_RATIOS rows must sum positive.")
        return [float(v) / total for v in ratios]

    def _jitter_heatmap_for_bias(self, heatmap):
        if not self.training:
            return heatmap
        shift = max(0.0, self.heatmap_bias_jitter_shift)
        scale_min = max(1e-6, self.heatmap_bias_jitter_scale_min)
        scale_max = max(scale_min, self.heatmap_bias_jitter_scale_max)
        if shift == 0.0 and scale_min == 1.0 and scale_max == 1.0:
            return heatmap

        batch_size = heatmap.shape[0]
        device = heatmap.device
        dtype = heatmap.dtype
        dx = (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0) * shift
        dy = (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0) * shift
        scale = torch.empty(batch_size, device=device, dtype=dtype).uniform_(
            scale_min, scale_max
        )
        inv_scale = 1.0 / scale

        theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = inv_scale
        theta[:, 1, 1] = inv_scale
        theta[:, 0, 2] = -2.0 * dx * inv_scale
        theta[:, 1, 2] = -2.0 * dy * inv_scale
        grid = F.affine_grid(theta, size=heatmap.shape, align_corners=False)
        return F.grid_sample(
            heatmap,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).clamp_min(0.0)

    def _choose_heatmap_bias_source(
        self,
        src_t,
        gt_heatmap,
        predicted_heatmaps,
        train_ar=False,
        ss_prob=0.0,
        bias_schedule_epoch=None,
    ):
        mode = self.heatmap_bias_source_mode
        has_pred = len(predicted_heatmaps) > src_t
        has_gt = gt_heatmap is not None

        if mode in {"gt", "oracle", "gt_oracle"}:
            return "gt" if has_gt else ("pred" if has_pred else "init")
        if mode in {"pred", "prediction", "predicted"}:
            return "pred" if has_pred else "init"
        if mode in {"center", "center_prior"}:
            return "center"
        if mode in {"uniform", "flat"}:
            return "uniform"
        if mode in {"noisy", "noise", "random", "uniform_noisy"}:
            return "noisy"
        if mode in {"legacy", "ss", "scheduled_sampling"}:
            if has_gt and train_ar and has_pred and torch.rand(1).item() < ss_prob:
                return "pred"
            if has_gt:
                return "gt"
            return "pred" if has_pred else "init"
        if mode in {"scheduled", "jittered_schedule", "jitter_schedule"}:
            if not (train_ar and has_gt):
                return "pred" if has_pred else ("gt" if has_gt else "init")
            gt_ratio, jitter_ratio, pred_ratio = self._get_heatmap_bias_schedule_ratios(
                bias_schedule_epoch
            )
            sample = torch.rand(1).item()
            if sample < gt_ratio:
                return "gt"
            if sample < gt_ratio + jitter_ratio:
                return "jittered_gt"
            if pred_ratio > 0.0 and has_pred:
                return "pred"
            return "gt"
        raise ValueError(f"Unsupported MODEL.HEATMAP_BIAS_SOURCE_MODE={mode}")

    def _get_history_heatmaps_for_bias(
        self,
        t,
        gt_heatmap,
        predicted_heatmaps,
        B,
        train_ar,
        ss_prob,
        bias_schedule_epoch=None,
    ):
        history = []
        sources = []
        for idx in range(self.history_length):
            src_t = t - self.history_length + idx
            if src_t < 0:
                sources.append("init")
                history.append(self.init_heatmap.expand(B, -1, -1, -1))
                continue

            source = self._choose_heatmap_bias_source(
                src_t,
                gt_heatmap,
                predicted_heatmaps,
                train_ar=train_ar,
                ss_prob=ss_prob,
                bias_schedule_epoch=bias_schedule_epoch,
            )
            sources.append(source)
            if source == "pred":
                hm = predicted_heatmaps[src_t].detach()
            elif source == "gt":
                hm = gt_heatmap[:, :, src_t, :, :]
            elif source == "jittered_gt":
                hm = self._jitter_heatmap_for_bias(gt_heatmap[:, :, src_t, :, :])
            elif source == "center":
                hm = self._center_heatmap_for_bias(
                    B,
                    self.init_heatmap.device,
                    self.init_heatmap.dtype,
                )
            elif source == "uniform":
                hm = self._uniform_heatmap_for_bias(
                    B,
                    self.init_heatmap.device,
                    self.init_heatmap.dtype,
                )
            elif source == "noisy":
                hm = self._noisy_heatmap_for_bias(
                    B,
                    self.init_heatmap.device,
                    self.init_heatmap.dtype,
                )
            else:
                hm = self.init_heatmap.expand(B, -1, -1, -1)
            history.append(hm)
        self._last_history_heatmap_sources = sources
        return history

    def _heatmap_to_patch_bias(self, heatmap):
        heatmap = F.interpolate(
            heatmap,
            size=(self.patch_h, self.patch_w),
            mode="bilinear",
            align_corners=False,
        )
        flat = heatmap.flatten(1)
        num_patches = flat.shape[1]
        valid = flat.abs().sum(dim=1, keepdim=True) > self.heatmap_bias_eps

        if self.heatmap_bias_normalize == "softmax":
            prob = F.softmax(flat, dim=1)
        elif self.heatmap_bias_normalize in {"sum", "l1"}:
            nonnegative = flat.clamp_min(0.0)
            denom = nonnegative.sum(dim=1, keepdim=True).clamp_min(
                self.heatmap_bias_eps
            )
            prob = nonnegative / denom
            valid = nonnegative.sum(dim=1, keepdim=True) > self.heatmap_bias_eps
        else:
            raise ValueError(
                f"Unsupported MODEL.HEATMAP_BIAS_NORMALIZE={self.heatmap_bias_normalize}"
            )

        uniform_log_prob = -math.log(num_patches)
        bias = torch.log(prob.clamp_min(self.heatmap_bias_eps)) - uniform_log_prob
        bias = bias * self.heatmap_bias_weight
        return torch.where(valid, bias, torch.zeros_like(bias))

    def _record_attention_bias_stats(self, attn_bias, step=None):
        if not hasattr(self, "latest_attention_bias_stats"):
            self.latest_attention_bias_stats = []
        if attn_bias is None:
            self.latest_attention_bias_stats.append(
                {
                    "step": int(step) if step is not None else None,
                    "is_none": True,
                    "shape": None,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "nan_count": 0,
                    "inf_count": 0,
                }
            )
            return

        detached = attn_bias.detach()
        finite = torch.isfinite(detached)
        finite_values = detached[finite]
        if finite_values.numel() > 0:
            mean = float(finite_values.mean().item())
            std = float(finite_values.std(unbiased=False).item())
            min_value = float(finite_values.min().item())
            max_value = float(finite_values.max().item())
        else:
            mean = std = min_value = max_value = None
        self.latest_attention_bias_stats.append(
            {
                "step": int(step) if step is not None else None,
                "is_none": False,
                "shape": list(detached.shape),
                "mean": mean,
                "std": std,
                "min": min_value,
                "max": max_value,
                "nan_count": int(torch.isnan(detached).sum().item()),
                "inf_count": int(torch.isinf(detached).sum().item()),
            }
        )

    def _build_heatmap_attention_bias(
        self, history_heatmaps, num_current_tokens, step=None
    ):
        if self.heatmap_bias_weight == 0.0 or len(history_heatmaps) == 0:
            self._record_attention_bias_stats(None, step=step)
            return None

        history_biases = [self._heatmap_to_patch_bias(hm) for hm in history_heatmaps]
        current_bias = history_biases[0].new_zeros(
            history_biases[0].shape[0], num_current_tokens
        )
        memory_bias = torch.cat(history_biases + [current_bias], dim=1)
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

        history_heatmaps = self._get_history_heatmaps_for_bias(
            t,
            gt_heatmap,
            predicted_heatmaps,
            batch_size,
            train_ar=train_ar,
            ss_prob=ss_prob,
            bias_schedule_epoch=bias_schedule_epoch,
        )
        if not hasattr(self, "latest_heatmap_bias_sources"):
            self.latest_heatmap_bias_sources = []
        self.latest_heatmap_bias_sources.append(
            {
                "step": int(t),
                "sources": list(getattr(self, "_last_history_heatmap_sources", [])),
            }
        )
        img_tokens = self._images_to_tokens(visual_features, t, pos_encoding)
        memory = torch.cat([img_tokens, current_tokens], dim=1)
        attn_bias = self._build_heatmap_attention_bias(
            history_heatmaps,
            num_current_tokens=current_tokens.shape[1],
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
            tgt = layer(tgt, memory, attn_bias=attn_bias)

        return self._decode_tokens_to_heatmap(tgt, batch_size)

    def _autoregressive_decode(
        self,
        visual_features,
        gt_heatmap,
        B,
        T,
        train_ar,
        ss_prob,
        pos_encoding,
        bias_schedule_epoch=None,
    ):
        heatmap_outputs = []
        predicted_heatmaps = []
        self.latest_attention_bias_stats = []
        self.latest_heatmap_bias_sources = []

        for t in range(T):
            heatmap_t = self._decode_step(
                t,
                visual_features[:, t, :, :],
                predicted_heatmaps,
                visual_features,
                pos_encoding,
                gt_heatmap=gt_heatmap,
                train_ar=train_ar,
                ss_prob=ss_prob,
                bias_schedule_epoch=bias_schedule_epoch,
            )
            heatmap_outputs.append(heatmap_t)
            predicted_heatmaps.append(heatmap_t.detach())

        return torch.stack(heatmap_outputs, dim=2)

    def forward(
        self,
        x,
        gt_heatmap=None,
        train_ar=True,
        ss_prob=0.0,
        bias_schedule_epoch=None,
    ):
        x = x[0] if isinstance(x, list) else x
        if x.dim() == 4:
            x = x.unsqueeze(2)
        B, C, T, H_in, W_in = x.shape
        if gt_heatmap is not None and gt_heatmap.dim() == 4:
            gt_heatmap = gt_heatmap.unsqueeze(1)

        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H_in, W_in)
        encoder_params_grad = torch.is_grad_enabled() and any(
            param.requires_grad for param in self.encoder.parameters()
        )
        context_manager = torch.enable_grad() if encoder_params_grad else torch.no_grad()

        with context_manager:
            frames = self._preprocess_frames(frames)
            outputs = self.encoder(pixel_values=frames)
            num_spatial_patches = 196
            if self.use_multiscale:
                hidden_states = outputs.hidden_states
                multiscale_features = []
                for layer_idx, proj in zip(self.multiscale_layers, self.multiscale_proj):
                    multiscale_features.append(
                        proj(hidden_states[layer_idx][:, 1 : 1 + num_spatial_patches, :])
                    )
                patch_tokens = torch.cat(multiscale_features, dim=-1)
            else:
                patch_tokens = outputs.last_hidden_state[:, 1 : 1 + num_spatial_patches, :]

        patch_tokens = self.feature_proj(patch_tokens).view(
            B, T, num_spatial_patches, self.hidden_dim
        )
        pos_encoding = self._get_2d_pos_encoding(
            self.patch_h, self.patch_w, patch_tokens.device
        )
        patch_tokens = patch_tokens + pos_encoding.unsqueeze(0)

        return self._autoregressive_decode(
            patch_tokens,
            gt_heatmap,
            B,
            T,
            train_ar,
            ss_prob,
            pos_encoding,
            bias_schedule_epoch=bias_schedule_epoch,
        )
