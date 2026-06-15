import os
import math
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from slowfast.models import MODEL_REGISTRY
from slowfast.utils import logging

logger = logging.get_logger(__name__)


class EfficientTransformerDecoderLayer(nn.Module):
    """
    Optimized Transformer decoder block.

    Self-attention: Only within the Query tokens (Spatial interaction).
    Cross-attention: Query tokens attend to a combined Memory of History + Current Frame.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Self-attention ONLY on the current query (e.g. 196 tokens)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention to Memory (History + Current visual features)
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

    def forward(self, tgt, memory):
        """
        Args:
            tgt: Query tokens (B, N_query, D)
            memory: Reference tokens (B, N_history + N_current, D)
        """
        # 1. Self-attention: Query tokens organize themselves spatially
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # 2. Cross-attention: Query tokens "read" from the history and current frame
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # 3. Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + tgt2)
        return tgt


@MODEL_REGISTRY.register()
class DINOv3_EfficientARHeatmapGaze(nn.Module):
    """
    Efficient Autoregressive heatmap-based gaze estimation.

    Architecture optimization:
    - Self-Attention is limited to Query tokens (O(N_q^2)).
    - History context is moved to Cross-Attention Memory.
    - Drastically reduces FLOPs and enables better KV-Caching in inference.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model_name = getattr(
            cfg.MODEL, "DINOV3_MODEL_NAME", "facebook/dinov3-vits16-pretrain-lvd1689m"
        )

        # =========================
        # 1. DINOv3 Encoder
        # =========================
        self.processor = None
        hf_kwargs = {}
        env_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if env_token:
            hf_kwargs["use_auth_token"] = env_token
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, **hf_kwargs)
        except OSError as err:
            logger.warning(
                "Could not load image processor for %s (%s). "
                "Continuing because ARGaze dataloaders already apply preprocessing.",
                model_name,
                err,
            )
            self.processor = None

        self.use_multiscale = getattr(cfg.MODEL, "USE_MULTISCALE_FEATURES", True)
        if self.use_multiscale:
            try:
                self.encoder = AutoModel.from_pretrained(
                    model_name, output_hidden_states=True, **hf_kwargs
                )
            except TypeError:
                self.encoder = AutoModel.from_pretrained(
                    model_name, output_hidden_states=True, token=True
                )
            self.multiscale_layers = getattr(cfg.MODEL, "MULTISCALE_LAYERS", [-3, -2, -1])
        else:
            self.encoder = AutoModel.from_pretrained(model_name, **hf_kwargs)

        freeze_encoder = getattr(cfg.MODEL, "FREEZE_ENCODER", True)
        unfreeze_last_k = getattr(cfg.MODEL, "UNFREEZE_LAST_K_LAYERS", 0)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if unfreeze_last_k > 0:
            layers = None
            if hasattr(self.encoder, "layer"):
                layers = self.encoder.layer
            elif hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                layers = self.encoder.encoder.layer

            if layers is not None:
                for block in layers[-unfreeze_last_k:]:
                    for param in block.parameters():
                        param.requires_grad = True
                logger.info(f"Unfroze last {unfreeze_last_k} encoder layers.")

        if freeze_encoder and unfreeze_last_k == 0:
            self.encoder.eval()
        else:
            self.encoder.train()

        self.hidden_dim = self.encoder.config.hidden_size
        self.drop = getattr(cfg.MODEL, "DROPOUT_RATE", 0.1)
        self.heatmap_size = getattr(cfg.MODEL, "HEATMAP_SIZE", 64)
        self.history_length = getattr(cfg.MODEL, "HISTORY_LENGTH", 3)
        self.history_type = getattr(cfg.MODEL, "HISTORY_TYPE", "heatmap")

        # =========================
        # 2. Multi-scale Feature Fusion
        # =========================
        if self.use_multiscale:
            num_scales = len(self.multiscale_layers)
            self.multiscale_proj = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(num_scales)]
            )
            self.pixel_base_dim = self.hidden_dim * num_scales
        else:
            self.pixel_base_dim = self.hidden_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(self.pixel_base_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # =========================
        # 3. Heatmap Tokenization
        # =========================
        self.patch_h = 14
        self.patch_w = 14
        self.hm_downsample = nn.Upsample(
            size=(self.patch_h, self.patch_w),
            mode="bilinear",
            align_corners=False,
        )
        self.hm_to_tokens = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim, kernel_size=1),
            nn.LayerNorm([self.hidden_dim, self.patch_h, self.patch_w]),
        )

        # Temporal and token-type embeddings
        self.temporal_pos_embed = nn.Embedding(self.history_length, self.hidden_dim)
        self.token_type_embed = nn.Embedding(3, self.hidden_dim)  # 0=heatmap, 1=query, 2=image

        # Learnable query tokens
        self.num_query_tokens = self.patch_h * self.patch_w
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_query_tokens, self.hidden_dim) * 0.02
        )

        # =========================
        # 4. Efficient Transformer Decoder
        # =========================
        self.num_decoder_layers = getattr(cfg.MODEL, "NUM_DECODER_LAYERS", 3)
        self.nhead = getattr(cfg.MODEL, "NUM_ATTENTION_HEADS", 8)
        self.dim_feedforward = getattr(cfg.MODEL, "DIM_FEEDFORWARD", self.hidden_dim * 4)

        self.decoder_layers = nn.ModuleList(
            [
                EfficientTransformerDecoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.drop,
                )
                for _ in range(self.num_decoder_layers)
            ]
        )

        # =========================
        # 5. Conv Decoder to Heatmap
        # =========================
        self.use_subpatch = getattr(cfg.MODEL, "USE_SUBPATCH", False)
        if self.use_subpatch:
            self.pixel_decoder_conv1 = nn.Sequential(
                nn.Conv2d(self.hidden_dim, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(self.drop),
                nn.Conv2d(256, 1024, kernel_size=1),
                nn.PixelShuffle(2),
                nn.ReLU(),
            )
            self.pixel_decoder_conv2 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(self.drop),
                nn.Conv2d(128, 512, kernel_size=1),
                nn.PixelShuffle(2),
                nn.ReLU(),
            )
            if self.heatmap_size >= 112:
                self.pixel_final_upsample = nn.Sequential(
                    nn.Conv2d(128, 512, kernel_size=1),
                    nn.PixelShuffle(2)
                )
            else:
                self.pixel_final_upsample = nn.Upsample(
                    size=(self.heatmap_size, self.heatmap_size),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            self.pixel_decoder_conv1 = nn.Sequential(
                nn.Conv2d(self.hidden_dim, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(self.drop),
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            )

            self.pixel_decoder_conv2 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(self.drop),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            )

            if self.heatmap_size != 56:
                if self.heatmap_size >= 112:
                    self.pixel_final_upsample = nn.Sequential(
                        nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
                    )
                else:
                    self.pixel_final_upsample = nn.Upsample(
                        size=(self.heatmap_size, self.heatmap_size),
                        mode="bilinear",
                        align_corners=False,
                    )
            else:
                self.pixel_final_upsample = nn.Identity()

        self.pixel_final_proj = nn.Conv2d(128, 1, kernel_size=1)

        self.pos_encoding_cache = {}
        self.register_buffer(
            "init_heatmap",
            torch.zeros(1, 1, self.heatmap_size, self.heatmap_size),
            persistent=False,
        )

        logger.info("DINOv3_EfficientARHeatmapGaze initialized")
        logger.info(f"  - Hidden dim: {self.hidden_dim}")
        logger.info(f"  - Efficiency Mode: Self-Attn on Query only; History in Cross-Attn Memory.")

    def _preprocess_frames(self, frames):
        if frames.max() <= 1.0:
            frames = frames * 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1)
        frames = (frames / 255.0 - mean) / std
        return frames

    def _get_2d_pos_encoding(self, height, width, device):
        cache_key = (height, width)
        if cache_key in self.pos_encoding_cache:
            return self.pos_encoding_cache[cache_key].to(device)

        pos_encoding = torch.zeros(height, width, self.hidden_dim, device=device)
        d_model = self.hidden_dim
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float()
            * -(math.log(10000.0) / d_model)
        )

        pos_h = torch.arange(height, device=device).unsqueeze(1).float()
        div_term_h = div_term[: d_model // 4].unsqueeze(0)
        pos_h_encoded = pos_h * div_term_h
        pos_encoding[:, :, 0 : d_model // 2 : 2] = (
            torch.sin(pos_h_encoded).unsqueeze(1).repeat(1, width, 1)
        )
        pos_encoding[:, :, 1 : d_model // 2 : 2] = (
            torch.cos(pos_h_encoded).unsqueeze(1).repeat(1, width, 1)
        )

        pos_w = torch.arange(width, device=device).unsqueeze(1).float()
        div_term_w = div_term[: d_model // 4].unsqueeze(0)
        pos_w_encoded = pos_w * div_term_w
        pos_encoding[:, :, d_model // 2 :: 2] = (
            torch.sin(pos_w_encoded).unsqueeze(0).repeat(height, 1, 1)
        )
        pos_encoding[:, :, d_model // 2 + 1 :: 2] = (
            torch.cos(pos_w_encoded).unsqueeze(0).repeat(height, 1, 1)
        )

        pos_encoding = pos_encoding.view(1, height * width, self.hidden_dim)
        self.pos_encoding_cache[cache_key] = pos_encoding.cpu()
        return pos_encoding

    def _heatmaps_to_tokens(self, heatmaps, pos_encoding):
        tokens = []
        hist_type = self.token_type_embed(torch.tensor(0, device=pos_encoding.device))
        for idx, hm in enumerate(heatmaps):
            hm_down = self.hm_downsample(hm)
            hm_tok = self.hm_to_tokens(hm_down)
            hm_tok = hm_tok.flatten(2).permute(0, 2, 1)
            temporal_embed = self.temporal_pos_embed(torch.tensor(idx, device=hm_tok.device))
            hm_tok = hm_tok + pos_encoding + temporal_embed.view(1, 1, -1) + hist_type.view(1, 1, -1)
            tokens.append(hm_tok)
        return torch.cat(tokens, dim=1) if len(tokens) > 0 else None

    def _images_to_tokens(self, visual_features, t, pos_encoding):
        tokens = []
        img_type = self.token_type_embed(torch.tensor(2, device=pos_encoding.device))
        for idx in range(self.history_length):
            src_t = t - self.history_length + idx
            if src_t < 0:
                B, _, N, D = visual_features.shape
                img_tok = torch.zeros(B, N, D, device=visual_features.device) + pos_encoding
            else:
                img_tok = visual_features[:, src_t, :, :]
            temporal_embed = self.temporal_pos_embed(torch.tensor(idx, device=img_tok.device))
            img_tok = img_tok + temporal_embed.view(1, 1, -1) + img_type.view(1, 1, -1)
            tokens.append(img_tok)
        return torch.cat(tokens, dim=1)

    def encode_single_frame(self, frame, context_manager=None):
        """Encode a single frame into patch tokens for streaming inference."""
        encoder_params_grad = torch.is_grad_enabled() and any(
            param.requires_grad for param in self.encoder.parameters()
        )
        if context_manager is None:
            context_manager = torch.enable_grad() if encoder_params_grad else torch.no_grad()

        with context_manager:
            frames = self._preprocess_frames(frame)
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

        return self.feature_proj(patch_tokens)

    def _decode_tokens_to_heatmap(self, tgt, batch_size):
        feat = tgt.permute(0, 2, 1).reshape(
            batch_size, self.hidden_dim, self.patch_h, self.patch_w
        )
        feat = self.pixel_decoder_conv1(feat)
        feat = self.pixel_decoder_conv2(feat)
        feat = self.pixel_final_upsample(feat)
        return self.pixel_final_proj(feat)

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
    ):
        batch_size = current_tokens.shape[0]
        memory_list = []
        if self.history_type in ["heatmap", "both"]:
            history_heatmaps = self._get_history_heatmaps(
                t, gt_heatmap, predicted_heatmaps, batch_size, train_ar=train_ar, ss_prob=ss_prob
            )
            hm_tokens = self._heatmaps_to_tokens(history_heatmaps, pos_encoding)
            if hm_tokens is not None:
                memory_list.append(hm_tokens)

        if self.history_type in ["image", "both"]:
            img_tokens = self._images_to_tokens(visual_features, t, pos_encoding)
            memory_list.append(img_tokens)

        memory_list.append(current_tokens)
        memory = torch.cat(memory_list, dim=1)

        query_type = self.token_type_embed(torch.tensor(1, device=current_tokens.device))
        tgt = self.query_tokens.expand(batch_size, -1, -1) + pos_encoding + query_type.view(1, 1, -1)

        for layer in self.decoder_layers:
            tgt = layer(tgt, memory)

        return self._decode_tokens_to_heatmap(tgt, batch_size)

    def streaming_decode_step(
        self, t, current_tokens, predicted_heatmaps, cached_frame_tokens, pos_encoding
    ):
        """Decode a single streaming step using cached frame tokens and heatmaps."""
        device = current_tokens.device
        batch_size, num_patches, hidden_dim = current_tokens.shape
        current_tokens = current_tokens + pos_encoding

        if cached_frame_tokens:
            visual_features = torch.zeros(
                batch_size,
                t + 1,
                num_patches,
                hidden_dim,
                device=device,
                dtype=current_tokens.dtype,
            )
            for idx, tokens in enumerate(cached_frame_tokens):
                visual_features[:, idx, :, :] = tokens + pos_encoding
            visual_features[:, t, :, :] = current_tokens
        else:
            visual_features = current_tokens.unsqueeze(1)

        return self._decode_step(t, current_tokens, predicted_heatmaps, visual_features, pos_encoding)

    def forward(self, x, gt_heatmap=None, train_ar=True, ss_prob=0.0):
        x = x[0] if isinstance(x, list) else x
        if x.dim() == 4: x = x.unsqueeze(2)
        B, C, T, H_in, W_in = x.shape
        if gt_heatmap is not None and gt_heatmap.dim() == 4: gt_heatmap = gt_heatmap.unsqueeze(1)

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
                    multiscale_features.append(proj(hidden_states[layer_idx][:, 1:1+num_spatial_patches, :]))
                patch_tokens = torch.cat(multiscale_features, dim=-1)
            else:
                patch_tokens = outputs.last_hidden_state[:, 1:1+num_spatial_patches, :]

        patch_tokens = self.feature_proj(patch_tokens).view(B, T, num_spatial_patches, self.hidden_dim)
        pos_encoding = self._get_2d_pos_encoding(self.patch_h, self.patch_w, patch_tokens.device)
        patch_tokens = patch_tokens + pos_encoding.unsqueeze(0)

        return self._autoregressive_decode(patch_tokens, gt_heatmap, B, T, train_ar, ss_prob, pos_encoding)

    def _get_history_heatmaps(self, t, gt_heatmap, predicted_heatmaps, B, train_ar, ss_prob):
        history = []
        for idx in range(self.history_length):
            src_t = t - self.history_length + idx
            if src_t < 0:
                history.append(self.init_heatmap.expand(B, -1, -1, -1))
                continue
            use_pred = False
            if gt_heatmap is not None:
                if train_ar and len(predicted_heatmaps) > src_t and torch.rand(1).item() < ss_prob:
                    use_pred = True
            else:
                use_pred = len(predicted_heatmaps) > src_t
            hm = predicted_heatmaps[src_t].detach() if use_pred else (gt_heatmap[:, :, src_t, :, :] if gt_heatmap is not None else self.init_heatmap.expand(B, -1, -1, -1))
            history.append(hm)
        return history

    def _autoregressive_decode(self, visual_features, gt_heatmap, B, T, train_ar, ss_prob, pos_encoding):
        heatmap_outputs = []
        predicted_heatmaps = []

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
            )
            heatmap_outputs.append(heatmap_t)
            predicted_heatmaps.append(heatmap_t.detach())

        return torch.stack(heatmap_outputs, dim=2)
