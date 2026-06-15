#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_tracked_mask(labels, dataset):
    if dataset == "egteagaze":
        fixation_idx = 1
    elif dataset in ["ego4dgaze", "ego4d_av_gaze", "Ego4dgaze"]:
        fixation_idx = 0
    elif dataset in ["Holoassistgaze", "holoassistgaze"]:
        fixation_idx = 1
    elif dataset in ["Egoexo4dgaze", "egoexo4dgaze"]:
        return labels[:, :, 2] >= 0.5
    else:
        raise NotImplementedError(f"Metrics of {dataset} is not implemented.")
    return labels[:, :, 2] == fixation_idx


def soft_argmax_2d(logits, tau=0.5):
    if logits.dim() == 5:
        logits = logits[:, 0]
    elif logits.dim() != 4:
        raise ValueError(f"Expected logits shape (B,1,T,H,W) or (B,T,H,W), got {tuple(logits.shape)}")

    B, T, H, W = logits.shape
    logits = logits.float()
    logits_flat = logits.reshape(B * T, H * W)
    probs = F.softmax(logits_flat / float(tau), dim=-1)

    ys = torch.linspace(0.0, 1.0, H, device=logits.device, dtype=logits.dtype)
    xs = torch.linspace(0.0, 1.0, W, device=logits.device, dtype=logits.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    exp_x = (probs * grid_x).sum(dim=-1)
    exp_y = (probs * grid_y).sum(dim=-1)
    coords = torch.stack([exp_x, exp_y], dim=-1).reshape(B, T, 2)
    return coords


def coord_loss_from_logits(logits, labels, dataset, tau=0.5, loss_type="smooth_l1"):
    coords_pred = soft_argmax_2d(logits, tau=tau)
    coords_gt = labels[:, :, :2].to(dtype=coords_pred.dtype)
    mask = _get_tracked_mask(labels, dataset).to(dtype=coords_pred.dtype)

    if mask.sum().item() == 0:
        return torch.zeros((), device=coords_pred.device, dtype=coords_pred.dtype)

    if loss_type == "smooth_l1":
        loss_map = F.smooth_l1_loss(coords_pred, coords_gt, reduction="none")
    elif loss_type == "l1":
        loss_map = F.l1_loss(coords_pred, coords_gt, reduction="none")
    elif loss_type == "l2":
        loss_map = F.mse_loss(coords_pred, coords_gt, reduction="none")
    else:
        raise ValueError(f"Unsupported coord loss type: {loss_type}")

    loss_map = loss_map.sum(dim=-1)
    loss = (loss_map * mask).sum() / (mask.sum() + 1e-6)
    return loss


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = - (5 * y * F.logsigmoid(x) + (1 - y) * torch.log(1 - torch.sigmoid(x)))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise NotImplementedError


class KLDiv(nn.Module):
    """
      KL divergence for 3D attention maps
    """
    def __init__(self):
        super(KLDiv, self).__init__()
        self.register_buffer('norm_scalar', torch.tensor(1, dtype=torch.float32))

    def forward(self, pred, target=None):
        # get output shape
        batch_size, T = pred.shape[0], pred.shape[2]
        H, W = pred.shape[3], pred.shape[4]

        # N T HW
        atten_map = pred.view(batch_size, T, -1)
        log_atten_map = torch.log(atten_map + 1e-10)

        if target is None:
            # uniform prior: this is really just neg entropy
            # we keep the loss scale the same here
            log_q = torch.log(self.norm_scalar / float(H * W))
            # \sum p logp - log(1/hw) -> N T
            kl_losses = (atten_map * log_atten_map).sum(dim=-1) - log_q
        else:
            # Ensure target spatial size matches prediction; resize if needed.
            # Expected target shape: (N, T, Ht, Wt)
            if target.dim() != 4:
                raise ValueError(f"KLDiv target must be 4D (N,T,H,W), got shape {tuple(target.shape)}")
            if target.shape[-2] != H or target.shape[-1] != W:
                # Resize with bilinear interpolation; preserve per-frame probability mass via renormalization.
                t = target
                Nt, Tt, Ht, Wt = t.shape
                t = t.reshape(Nt * Tt, 1, Ht, Wt)
                t = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
                t = t.view(Nt, Tt, H, W)
                # Renormalize to sum to 1 over spatial for each (N,T)
                t_flat = t.view(Nt, Tt, -1)
                t_sum = t_flat.sum(dim=-1, keepdim=True)
                t_sum = t_sum + 1e-10
                t = (t_flat / t_sum).view(Nt, Tt, H, W)
                target = t
            log_q = torch.log(target.view(batch_size, T, -1) + 1e-10)
            # \sum p logp - \sum p logq -> N T
            kl_losses = (atten_map * log_atten_map).sum(dim=-1) - (atten_map * log_q).sum(dim=-1)
        # N T -> N
        norm_scalar = T * torch.log(self.norm_scalar * H * W)
        kl_losses = kl_losses.sum(dim=-1) / norm_scalar
        kl_loss = kl_losses.mean()
        return kl_loss


class ARPointGazeLoss(nn.Module):
    """
    Loss function for ARPointGaze model.

    Combines:
    1. Token-level Cross Entropy loss (primary)
    2. Optional L2 loss on continuous coordinates (auxiliary)
    3. Optional Auxiliary Heatmap loss (spatial-aware label smoothing)

    Input:
        logits: (B, T, 2, vocab_size) - raw logits for x and y tokens
        coords_pred: (B, T, 2) - predicted continuous coordinates [0, 1]
        coords_gt: (B, T, 2) - ground truth continuous coordinates [0, 1]
        bins: int - number of bins for discretization
        coord_range: float - vocabulary range multiplier
        aux_heatmap_weight: float - weight for auxiliary heatmap loss
        aux_heatmap_sigma: float - Gaussian sigma for GT heatmap generation
    """

    def __init__(self, bins=256, coord_range=2.0, l2_weight=0.1,
                 aux_heatmap_weight=0.0, aux_heatmap_sigma=2.0, reduction='mean'):
        super(ARPointGazeLoss, self).__init__()
        self.bins = bins
        self.coord_range = coord_range
        self.coord_offset = (coord_range - 1) * 0.5
        self.l2_weight = l2_weight
        self.aux_heatmap_weight = aux_heatmap_weight
        self.aux_heatmap_sigma = aux_heatmap_sigma
        self.reduction = reduction

        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)

    def coord_to_token(self, coords):
        """Convert continuous coordinates to discrete tokens."""
        # Shift to expanded range
        coords = coords - self.coord_offset

        # Discretize
        tokens = (coords * self.bins).long()

        # Clamp to valid range
        max_token = int(self.bins * self.coord_range) - 1
        tokens = torch.clamp(tokens, 0, max_token)

        return tokens

    def generate_gaussian_heatmap(self, coords_gt, heatmap_size):
        """
        Generate Gaussian heatmap for ground truth coordinates.

        Args:
            coords_gt: (B, T, 2) - normalized coordinates [0, 1]
            heatmap_size: int - size of heatmap (e.g., 64 for 64x64)

        Returns:
            heatmap: (B, T, heatmap_size, heatmap_size) - Gaussian heatmap
        """
        B, T, _ = coords_gt.shape
        device = coords_gt.device

        # Create coordinate grid [0, heatmap_size-1]
        y_grid, x_grid = torch.meshgrid(
            torch.arange(heatmap_size, device=device),
            torch.arange(heatmap_size, device=device),
            indexing='ij'
        )  # (H, W)

        # Normalize grid to [0, 1]
        x_grid = x_grid.float() / (heatmap_size - 1)
        y_grid = y_grid.float() / (heatmap_size - 1)

        # Expand for batch and time dimensions
        x_grid = x_grid.view(1, 1, heatmap_size, heatmap_size)  # (1, 1, H, W)
        y_grid = y_grid.view(1, 1, heatmap_size, heatmap_size)  # (1, 1, H, W)

        # Get GT coordinates
        x_gt = coords_gt[:, :, 0].view(B, T, 1, 1)  # (B, T, 1, 1)
        y_gt = coords_gt[:, :, 1].view(B, T, 1, 1)  # (B, T, 1, 1)

        # Compute Gaussian heatmap
        # Distance: (x - x_gt)^2 + (y - y_gt)^2
        sigma_sq = self.aux_heatmap_sigma ** 2
        heatmap = torch.exp(-((x_grid - x_gt) ** 2 + (y_grid - y_gt) ** 2) / (2 * sigma_sq / heatmap_size))

        # Normalize to probability distribution (sum to 1 for each frame)
        heatmap = heatmap.view(B, T, -1)  # (B, T, H*W)
        heatmap = heatmap / (heatmap.sum(dim=-1, keepdim=True) + 1e-10)
        heatmap = heatmap.view(B, T, heatmap_size, heatmap_size)

        return heatmap

    def forward(self, logits, coords_pred, coords_gt):
        """
        Args:
            logits: (B, T, 2, vocab_size)
            coords_pred: (B, T, 2)
            coords_gt: (B, T, 2)

        Returns:
            loss_dict: Dictionary containing:
                - 'loss': total loss
                - 'ce_loss': cross entropy loss
                - 'l2_loss': L2 loss (if l2_weight > 0)
                - 'aux_heatmap_loss': Auxiliary heatmap loss (if aux_heatmap_weight > 0)
        """
        B, T, _, vocab_size = logits.shape

        # Convert GT coords to tokens
        tokens_gt = self.coord_to_token(coords_gt)  # (B, T, 2)

        # Reshape for cross entropy
        logits_flat = logits.reshape(B * T * 2, vocab_size)  # (B*T*2, vocab_size)
        tokens_flat = tokens_gt.reshape(B * T * 2)  # (B*T*2)

        # Cross entropy loss
        ce_loss = self.ce_loss(logits_flat, tokens_flat)

        # L2 loss on continuous coordinates
        if self.l2_weight > 0:
            l2_loss = self.l2_loss(coords_pred, coords_gt)
        else:
            l2_loss = torch.tensor(0.0, device=logits.device)

        # Auxiliary Heatmap Loss (Method 2: Reshape token logits)
        if self.aux_heatmap_weight > 0:
            # Use fixed heatmap size (consistent with CrossAttnAR)
            heatmap_size = 64

            # ARPointGaze uses separate vocab for x and y (each vocab_size = bins * coord_range + 2)
            # We need to downsample vocab distribution to heatmap_size
            logits_x = logits[:, :, 0, :]  # (B, T, vocab_size)
            logits_y = logits[:, :, 1, :]  # (B, T, vocab_size)

            # Compute probabilities
            probs_x = F.softmax(logits_x, dim=-1)  # (B, T, vocab_size)
            probs_y = F.softmax(logits_y, dim=-1)  # (B, T, vocab_size)

            # Downsample vocab distribution to heatmap_size using 1D interpolation
            # probs shape: (B, T, vocab_size) -> (B, T, heatmap_size)
            # Reshape for 1D interpolation: (B*T, 1, vocab_size)
            probs_x_reshaped = probs_x.view(B * T, 1, vocab_size)
            probs_y_reshaped = probs_y.view(B * T, 1, vocab_size)

            # Interpolate to heatmap_size
            probs_x_downsampled = F.interpolate(
                probs_x_reshaped,
                size=heatmap_size,
                mode='linear',
                align_corners=True
            )  # (B*T, 1, heatmap_size)

            probs_y_downsampled = F.interpolate(
                probs_y_reshaped,
                size=heatmap_size,
                mode='linear',
                align_corners=True
            )  # (B*T, 1, heatmap_size)

            # Reshape back: (B*T, 1, heatmap_size) -> (B, T, heatmap_size)
            probs_x_downsampled = probs_x_downsampled.view(B, T, heatmap_size)
            probs_y_downsampled = probs_y_downsampled.view(B, T, heatmap_size)

            # Renormalize after interpolation to ensure proper probability distribution
            probs_x_downsampled = probs_x_downsampled / (probs_x_downsampled.sum(dim=-1, keepdim=True) + 1e-10)
            probs_y_downsampled = probs_y_downsampled / (probs_y_downsampled.sum(dim=-1, keepdim=True) + 1e-10)

            # Outer product to get 2D heatmap: P(x,y) = P(x) ⊗ P(y)
            pred_heatmap = probs_x_downsampled.unsqueeze(-1) * probs_y_downsampled.unsqueeze(-2)  # (B, T, 64, 64)

            # Generate GT Gaussian heatmap
            gt_heatmap = self.generate_gaussian_heatmap(coords_gt, heatmap_size)

            # KL divergence loss
            aux_heatmap_loss = F.kl_div(
                torch.log(pred_heatmap.view(B * T, -1) + 1e-10),
                gt_heatmap.view(B * T, -1),
                reduction='batchmean'
            )
        else:
            aux_heatmap_loss = torch.tensor(0.0, device=logits.device)

        # Total loss
        total_loss = ce_loss + self.l2_weight * l2_loss + self.aux_heatmap_weight * aux_heatmap_loss

        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'l2_loss': l2_loss,
            'aux_heatmap_loss': aux_heatmap_loss
        }


class ARHeatmapGazeLoss(nn.Module):
    """
    Loss function for ARHeatmapGaze model with factorized heatmap output.

    Combines:
    1. Factorized Cross Entropy loss on P(x) and P(y) (primary)
    2. Optional 2D KL Divergence loss on reconstructed heatmap (auxiliary)

    Input:
        logits_x: (B, T, W) - raw logits for x distribution
        logits_y: (B, T, H) - raw logits for y distribution
        heatmap_gt: (B, 1, T, H, W) or (B, T, H, W) - ground truth heatmap
    """

    def __init__(self, kl_weight=0.1, reduction='mean'):
        super(ARHeatmapGazeLoss, self).__init__()
        self.kl_weight = kl_weight
        self.reduction = reduction

        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_loss = KLDiv()

    def _extract_marginal_distributions(self, heatmap):
        """
        Extract marginal distributions P(x) and P(y) from 2D heatmap.

        Args:
            heatmap: (B, T, H, W) 2D probability distribution

        Returns:
            prob_x: (B, T, W) - marginal over x-axis
            prob_y: (B, T, H) - marginal over y-axis
        """
        # Normalize heatmap to ensure it's a valid probability distribution
        B, T, H, W = heatmap.shape
        hm_flat = heatmap.reshape(B, T, -1)
        hm_norm = F.softmax(hm_flat, dim=-1).reshape(B, T, H, W)

        # Compute marginals by summing over the other dimension
        prob_x = hm_norm.sum(dim=2)  # Sum over height -> (B, T, W)
        prob_y = hm_norm.sum(dim=3)  # Sum over width -> (B, T, H)

        return prob_x, prob_y

    def _factorized_to_heatmap(self, logits_x, logits_y):
        """
        Convert factorized logits to 2D heatmap.

        Args:
            logits_x: (B, T, W)
            logits_y: (B, T, H)

        Returns:
            heatmap: (B, T, H, W)
        """
        # Convert to probabilities
        prob_x = F.softmax(logits_x, dim=-1)  # (B, T, W)
        prob_y = F.softmax(logits_y, dim=-1)  # (B, T, H)

        # Outer product
        heatmap = prob_y.unsqueeze(-1) * prob_x.unsqueeze(-2)  # (B, T, H, W)

        return heatmap

    def forward(self, logits_x, logits_y, heatmap_gt):
        """
        Args:
            logits_x: (B, T, W) - predicted logits for x distribution
            logits_y: (B, T, H) - predicted logits for y distribution
            heatmap_gt: (B, 1, T, H, W) or (B, T, H, W) - ground truth heatmap

        Returns:
            loss_dict: Dictionary containing:
                - 'loss': total loss
                - 'ce_x_loss': cross entropy loss for x
                - 'ce_y_loss': cross entropy loss for y
                - 'kl_loss': KL divergence loss (if kl_weight > 0)
        """
        # Ensure heatmap_gt is (B, T, H, W)
        if heatmap_gt.dim() == 5:
            heatmap_gt = heatmap_gt.squeeze(1)  # (B, T, H, W)

        # Get dimensions from predictions
        B, T, W = logits_x.shape
        _, _, H = logits_y.shape

        # Verify heatmap_gt dimensions match
        if heatmap_gt.shape[0] != B or heatmap_gt.shape[1] != T:
            raise ValueError(
                f"heatmap_gt shape {heatmap_gt.shape} doesn't match predictions: "
                f"Expected batch={B}, temporal={T}, but got batch={heatmap_gt.shape[0]}, temporal={heatmap_gt.shape[1]}"
            )

        # Extract marginal distributions from GT heatmap
        prob_x_gt, prob_y_gt = self._extract_marginal_distributions(heatmap_gt)

        # Handle resolution mismatch between GT and predictions
        # GT heatmap might be different size (e.g., 56x56) than model output (e.g., 64x64)
        H_gt, W_gt = heatmap_gt.shape[-2:]
        if W_gt != W or H_gt != H:
            # Resize marginal distributions to match prediction size
            # prob_x_gt: (B, T, W_gt) -> (B, T, W)
            # prob_y_gt: (B, T, H_gt) -> (B, T, H)
            prob_x_gt = F.interpolate(
                prob_x_gt.unsqueeze(1),  # (B, 1, T, W_gt)
                size=(T, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # (B, T, W)

            prob_y_gt = F.interpolate(
                prob_y_gt.unsqueeze(1),  # (B, 1, T, H_gt)
                size=(T, H),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # (B, T, H)

            # Re-normalize after interpolation
            prob_x_gt = prob_x_gt / prob_x_gt.sum(dim=-1, keepdim=True)
            prob_y_gt = prob_y_gt / prob_y_gt.sum(dim=-1, keepdim=True)

        # Reshape for cross entropy loss
        logits_x_flat = logits_x.reshape(B * T, W)  # (B*T, W)
        logits_y_flat = logits_y.reshape(B * T, H)  # (B*T, H)

        # Since we have soft labels (prob distributions), use KL divergence
        # CE(p, q) = H(p, q) = -sum(p * log(q))
        # For soft labels, we use log_softmax on logits and sum with GT probs

        # X marginal loss
        log_prob_x = F.log_softmax(logits_x_flat, dim=-1)  # (B*T, W)
        prob_x_gt_flat = prob_x_gt.reshape(B * T, W)  # (B*T, W)
        ce_x_loss = -(prob_x_gt_flat * log_prob_x).sum(dim=-1)  # (B*T)
        if self.reduction == 'mean':
            ce_x_loss = ce_x_loss.mean()
        elif self.reduction == 'sum':
            ce_x_loss = ce_x_loss.sum()

        # Y marginal loss
        log_prob_y = F.log_softmax(logits_y_flat, dim=-1)  # (B*T, H)
        prob_y_gt_flat = prob_y_gt.reshape(B * T, H)  # (B*T, H)
        ce_y_loss = -(prob_y_gt_flat * log_prob_y).sum(dim=-1)  # (B*T)
        if self.reduction == 'mean':
            ce_y_loss = ce_y_loss.mean()
        elif self.reduction == 'sum':
            ce_y_loss = ce_y_loss.sum()

        # Total factorized CE loss
        ce_loss = ce_x_loss + ce_y_loss

        # Optional: 2D KL divergence on reconstructed heatmap
        if self.kl_weight > 0:
            # Reconstruct 2D heatmap from factorized predictions
            heatmap_pred = self._factorized_to_heatmap(logits_x, logits_y)  # (B, T, H, W)

            # Resize GT heatmap if needed to match prediction size
            if H_gt != H or W_gt != W:
                heatmap_gt_resized = F.interpolate(
                    heatmap_gt.reshape(B * T, 1, H_gt, W_gt),  # (B*T, 1, H_gt, W_gt)
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).reshape(B, T, H, W)  # (B, T, H, W)
                # Re-normalize to sum to 1
                heatmap_gt_resized = heatmap_gt_resized / heatmap_gt_resized.sum(dim=(-2, -1), keepdim=True)
            else:
                heatmap_gt_resized = heatmap_gt

            # Prepare for KLDiv:
            # - pred needs 5D: (B, 1, T, H, W)
            # - target needs 4D: (B, T, H, W)
            heatmap_pred_5d = heatmap_pred.unsqueeze(1)  # (B, 1, T, H, W)
            heatmap_gt_4d = heatmap_gt_resized  # (B, T, H, W)

            # Compute KL divergence
            kl_loss = self.kl_loss(heatmap_pred_5d, heatmap_gt_4d)

            total_loss = ce_loss + self.kl_weight * kl_loss
        else:
            kl_loss = torch.tensor(0.0, device=logits_x.device)
            total_loss = ce_loss

        return {
            'loss': total_loss,
            'ce_x_loss': ce_x_loss,
            'ce_y_loss': ce_y_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss
        }


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,  # Alias for bce_logit
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "kldiv": KLDiv,
    "ar_point_gaze": ARPointGazeLoss,
    "ar_heatmap_gaze": ARHeatmapGazeLoss
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
