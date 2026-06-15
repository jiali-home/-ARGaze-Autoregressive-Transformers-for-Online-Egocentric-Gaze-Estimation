#!/usr/bin/env python3
"""
Student-Teacher EMA Training Utilities

This module provides utilities for student-teacher training with EMA (Exponential Moving Average).
The teacher model is updated as an EMA of the student model weights.

Key components:
- EMA momentum scheduling
- Knowledge distillation loss computation
- Softmax with temperature for 2D heatmaps
- Entropy and KL divergence metrics for heatmaps
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def cosine_ema_momentum(step, total_steps, m0=0.99, m1=0.9995):
    """
    Compute EMA momentum using cosine schedule.

    The momentum starts at m0 and increases to m1 following a cosine curve.
    Higher momentum means the teacher updates more slowly (more stable).

    Args:
        step: Current training step
        total_steps: Total number of training steps
        m0: Initial momentum (default: 0.99)
        m1: Final momentum (default: 0.9995)

    Returns:
        momentum: EMA momentum value in [m0, m1]

    Example:
        >>> # At beginning: momentum ≈ 0.99 (teacher updates faster)
        >>> m = cosine_ema_momentum(0, 10000, 0.99, 0.9995)
        >>> # At end: momentum ≈ 0.9995 (teacher updates slower, more stable)
        >>> m = cosine_ema_momentum(10000, 10000, 0.99, 0.9995)
    """
    if total_steps <= 0:
        return m1

    # Cosine annealing: goes from 1.0 to 0.0
    cos_value = 0.5 * (1 + math.cos(math.pi * step / total_steps))

    # Map to momentum range [m0, m1]
    momentum = m1 - (m1 - m0) * cos_value

    return momentum


@torch.no_grad()
def update_teacher_ema(teacher_model, student_model, momentum):
    """
    Update teacher model parameters as EMA of student model.

    The update rule is:
        θ_teacher = momentum * θ_teacher + (1 - momentum) * θ_student

    Args:
        teacher_model: Teacher model (will be updated in-place)
        student_model: Student model (source of updates)
        momentum: EMA momentum in [0, 1]. Higher = slower teacher updates.

    Example:
        >>> # Teacher updates slowly (99.95% old, 0.05% new)
        >>> update_teacher_ema(teacher, student, momentum=0.9995)
    """
    for param_t, param_s in zip(teacher_model.parameters(), student_model.parameters()):
        param_t.data.mul_(momentum).add_(param_s.data, alpha=1.0 - momentum)


def softmax_2d_heatmap(logits, temperature=1.0, eps=1e-12):
    """
    Apply softmax over spatial dimensions (H, W) of heatmap with temperature scaling.

    Temperature controls the smoothness of the distribution:
    - T > 1: Softer, more uniform distribution (good for knowledge distillation)
    - T = 1: Standard softmax
    - T < 1: Sharper, more peaked distribution

    Args:
        logits: Heatmap logits of shape (B, 1, T, H, W) or (B, T, 1, H, W) or (B, T, C, H, W)
        temperature: Temperature for softmax (default: 1.0)
        eps: Small constant to prevent numerical issues (default: 1e-12)

    Returns:
        probs: Probability distribution over H*W, same shape as input
               Each (B, T, C) slice sums to 1 over H*W

    Example:
        >>> logits = torch.randn(2, 8, 1, 64, 64)
        >>> # Soft distribution for KD
        >>> soft_probs = softmax_2d_heatmap(logits, temperature=2.0)
        >>> # Hard distribution for loss
        >>> hard_probs = softmax_2d_heatmap(logits, temperature=1.0)
    """
    # Handle different input formats
    original_shape = logits.shape
    if len(original_shape) == 5:
        if original_shape[1] == 1 and original_shape[2] > 1:
            # Format: (B, 1, T, H, W) -> convert to (B, T, 1, H, W)
            logits = logits.permute(0, 2, 1, 3, 4)
        # Now shape is (B, T, C, H, W)
        B, T, C, H, W = logits.shape
    else:
        raise ValueError(f"Expected 5D tensor, got shape {original_shape}")

    # Flatten spatial dimensions
    x = (logits / max(temperature, 1e-8)).view(B, T, C, -1)  # (B, T, C, H*W)

    # Softmax over spatial dimension with log-space for numerical stability
    x = x - x.max(dim=-1, keepdim=True)[0]  # Subtract max for numerical stability
    x = x - x.logsumexp(dim=-1, keepdim=True)  # Log-softmax
    p = x.exp().clamp_min(eps)  # Convert to probabilities

    # Reshape back to spatial dimensions
    p = p.view(B, T, C, H, W)

    # Convert back to original format if needed
    if original_shape[1] == 1 and original_shape[2] > 1:
        # Convert back: (B, T, 1, H, W) -> (B, 1, T, H, W)
        p = p.permute(0, 2, 1, 3, 4)

    return p


def kl_divergence_2d(p_target, p_pred, eps=1e-12, reduction='none'):
    """
    Compute KL divergence between two 2D heatmap distributions.

    KL(P || Q) = Σ P(x) * (log P(x) - log Q(x))

    Args:
        p_target: Target distribution (B, 1, T, H, W) or (B, T, 1, H, W) - usually from teacher or GT
        p_pred: Predicted distribution (B, 1, T, H, W) or (B, T, 1, H, W) - from student
        eps: Small constant to prevent log(0) (default: 1e-12)
        reduction: How to reduce the output:
                  - 'none': Return (B, T) - per-sample, per-timestep KL
                  - 'mean': Return scalar - average over all samples and timesteps
                  - 'sum': Return scalar - sum over all samples and timesteps

    Returns:
        kl_div: KL divergence
               - If reduction='none': (B, T) tensor
               - If reduction='mean' or 'sum': scalar tensor

    Example:
        >>> p_teacher = softmax_2d_heatmap(teacher_logits, temperature=2.0)
        >>> p_student = softmax_2d_heatmap(student_logits, temperature=2.0)
        >>> kl_loss = kl_divergence_2d(p_teacher, p_student, reduction='mean')
    """
    # Handle different input formats
    if p_target.shape[1] == 1 and p_target.shape[2] > 1:
        # Format: (B, 1, T, H, W) -> convert to (B, T, 1, H, W)
        p_target = p_target.permute(0, 2, 1, 3, 4)
    if p_pred.shape[1] == 1 and p_pred.shape[2] > 1:
        # Format: (B, 1, T, H, W) -> convert to (B, T, 1, H, W)
        p_pred = p_pred.permute(0, 2, 1, 3, 4)

    # Clamp to avoid log(0)
    p = p_target.clamp_min(eps)
    q = p_pred.clamp_min(eps)

    # KL(P || Q) = Σ P * (log P - log Q)
    log_p = p.log()
    log_q = q.log()
    kl = (p * (log_p - log_q)).sum(dim=[2, 3, 4])  # Sum over C, H, W -> (B, T)

    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    else:  # 'none'
        return kl


def entropy_2d(p, eps=1e-12, normalize=True):
    """
    Compute Shannon entropy of 2D heatmap distribution.

    H(P) = -Σ P(x) * log P(x)

    Lower entropy = more confident/peaked distribution
    Higher entropy = more uncertain/uniform distribution

    Args:
        p: Probability distribution (B, 1, T, H, W) or (B, T, 1, H, W)
        eps: Small constant to prevent log(0) (default: 1e-12)
        normalize: If True, normalize by log(H*W) to get entropy in [0, 1]
                  (default: True)

    Returns:
        entropy: (B, T) tensor of entropy values
                If normalized, values are in [0, 1] where:
                - 0 = completely peaked (one-hot)
                - 1 = completely uniform

    Example:
        >>> p = softmax_2d_heatmap(logits, temperature=1.0)
        >>> confidence = 1.0 - entropy_2d(p, normalize=True)  # (B, T)
        >>> # Use confidence as weight for KD loss
    """
    # Handle different input formats
    if p.shape[1] == 1 and p.shape[2] > 1:
        # Format: (B, 1, T, H, W) -> convert to (B, T, 1, H, W)
        p = p.permute(0, 2, 1, 3, 4)

    p = p.clamp_min(eps)
    H = -(p * p.log()).sum(dim=[2, 3, 4])  # (B, T)

    if normalize:
        # Normalize by maximum possible entropy (uniform distribution)
        num_pixels = p.shape[-1] * p.shape[-2]  # H * W
        max_entropy = math.log(max(num_pixels, 2))
        H = H / max_entropy

    return H


def compute_kd_loss(
    teacher_logits,
    student_logits,
    temperature=2.0,
    use_confidence_weighting=True,
    keyframes=None,
    reduction='mean'
):
    """
    Compute knowledge distillation loss with optional confidence weighting.

    The KD loss encourages the student to match the teacher's soft predictions.
    Optionally weights each timestep by teacher's confidence (inverse entropy).

    Args:
        teacher_logits: Teacher's raw predictions (B, T, 1, H, W)
        student_logits: Student's raw predictions (B, T, 1, H, W)
        temperature: Softmax temperature for KD (default: 2.0)
        use_confidence_weighting: If True, weight by teacher confidence (default: True)
        keyframes: Optional list/tensor of frame indices to apply KD to (e.g., [1, 3, 7])
                  If None, apply to all frames (default: None)
        reduction: How to reduce the loss ('mean' or 'sum')

    Returns:
        loss: KD loss (scalar tensor)
        info: Dictionary with auxiliary information:
              - 'mean_confidence': Average teacher confidence (for logging)
              - 'num_frames': Number of frames used for KD

    Example:
        >>> # Standard KD loss with confidence weighting
        >>> loss, info = compute_kd_loss(
        ...     teacher_logits, student_logits,
        ...     temperature=2.0,
        ...     use_confidence_weighting=True
        ... )
        >>>
        >>> # KD loss only on keyframes (e.g., every 4th frame)
        >>> keyframes = list(range(0, 8, 4))  # [0, 4]
        >>> loss, info = compute_kd_loss(
        ...     teacher_logits, student_logits,
        ...     temperature=2.0,
        ...     keyframes=keyframes
        ... )
    """
    # Handle input format: (B, 1, T, H, W) or (B, T, 1, H, W)
    if teacher_logits.shape[1] == 1 and teacher_logits.shape[2] > 1:
        # Format: (B, 1, T, H, W) - this is the actual format from model
        B, _, T = teacher_logits.shape[:3]
    else:
        # Format: (B, T, 1, H, W)
        B, T = teacher_logits.shape[:2]

    device = teacher_logits.device

    # Convert to probability distributions with temperature
    with torch.no_grad():
        p_teacher = softmax_2d_heatmap(teacher_logits, temperature=temperature)

        # Compute teacher confidence (1 - normalized_entropy)
        if use_confidence_weighting:
            # Lower entropy = higher confidence
            confidence = 1.0 - entropy_2d(
                softmax_2d_heatmap(teacher_logits, temperature=1.0),
                normalize=True
            )  # (B, T) in [0, 1]
        else:
            confidence = torch.ones(B, T, device=device)

    p_student = softmax_2d_heatmap(student_logits, temperature=temperature)

    # Compute KL divergence per frame
    kl_div = kl_divergence_2d(p_teacher, p_student, reduction='none')  # (B, T)

    # Apply keyframe mask if specified
    if keyframes is not None:
        mask = torch.zeros(T, dtype=torch.bool, device=device)
        mask[list(keyframes)] = True
        mask = mask.unsqueeze(0).expand(B, T)  # (B, T)
    else:
        mask = torch.ones(B, T, dtype=torch.bool, device=device)

    # Weight by confidence and mask
    weights = confidence * mask.float()

    # Normalize weights to sum to 1 (or number of samples)
    weight_sum = weights.sum().clamp_min(1.0)
    weighted_kl = (kl_div * weights).sum()

    if reduction == 'mean':
        loss = weighted_kl / weight_sum
    else:  # 'sum'
        loss = weighted_kl

    # Auxiliary info for logging
    info = {
        'mean_confidence': confidence[mask].mean().item() if mask.any() else 0.0,
        'num_frames': mask.sum().item(),
    }

    return loss, info


def initialize_teacher_from_student(student_model):
    """
    Create a teacher model as a deep copy of the student model.

    The teacher model will have the same architecture and initial weights
    as the student, but will be updated via EMA during training.

    Args:
        student_model: Student model to copy

    Returns:
        teacher_model: Deep copy of student model with frozen gradients

    Example:
        >>> student = build_model(cfg)
        >>> teacher = initialize_teacher_from_student(student)
        >>> # Teacher starts with same weights as student
        >>> # During training, teacher weights updated via EMA
    """
    teacher_model = deepcopy(student_model)

    # Freeze teacher model (no gradient computation needed)
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    # Set to eval mode
    teacher_model.eval()

    return teacher_model


def compute_gt_loss(pred_probs, gt_heatmap, loss_func='kldiv'):
    """
    Compute ground truth supervision loss.

    Args:
        pred_probs: Predicted heatmap probabilities (B, T, 1, H, W) or (B, 1, T, H, W)
                    Should already be normalized via softmax
        gt_heatmap: Ground truth heatmap (B, T, 1, H, W) or (B, 1, T, H, W) - should be normalized
        loss_func: Loss function to use ('kldiv' or 'mse')

    Returns:
        loss: GT supervision loss (scalar)

    Example:
        >>> student_preds = student_model(frames, override_future=None)
        >>> student_probs = frame_softmax(student_preds, temperature=2)
        >>> loss_gt = compute_gt_loss(student_probs, gt_heatmap, loss_func='kldiv')
    """
    # Handle different input formats
    # pred_probs: (B, 1, T, H, W) or (B, T, 1, H, W)
    # gt_heatmap: (B, T, H, W) or (B, T, 1, H, W) or (B, 1, T, H, W)

    if pred_probs.shape[1] == 1 and pred_probs.shape[2] > 1:
        # Format: (B, 1, T, H, W) - convert to (B, T, 1, H, W)
        pred_probs = pred_probs.permute(0, 2, 1, 3, 4)

    # Handle GT heatmap with different formats
    if gt_heatmap.dim() == 4:
        # Format: (B, T, H, W) - add channel dimension -> (B, T, 1, H, W)
        gt_heatmap = gt_heatmap.unsqueeze(2)
    elif gt_heatmap.dim() == 5 and gt_heatmap.shape[1] == 1 and gt_heatmap.shape[2] > 1:
        # Format: (B, 1, T, H, W) - convert to (B, T, 1, H, W)
        gt_heatmap = gt_heatmap.permute(0, 2, 1, 3, 4)

    B, T, C, H_pred, W_pred = pred_probs.shape
    _, _, _, H_gt, W_gt = gt_heatmap.shape

    # Resize GT heatmap if dimensions don't match
    if H_gt != H_pred or W_gt != W_pred:
        # Resize with bilinear interpolation and renormalize
        gt_resized = gt_heatmap.view(B * T, C, H_gt, W_gt)
        gt_resized = F.interpolate(
            gt_resized,
            size=(H_pred, W_pred),
            mode='bilinear',
            align_corners=False
        )
        gt_resized = gt_resized.view(B, T, C, H_pred, W_pred)

        # Renormalize to ensure it sums to 1 over spatial dimensions
        gt_flat = gt_resized.view(B, T, C, -1)
        gt_sum = gt_flat.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        gt_resized = (gt_flat / gt_sum).view(B, T, C, H_pred, W_pred)
        gt_heatmap = gt_resized

    if loss_func == 'kldiv':
        # Use the same KL divergence formulation as losses.KLDiv
        # Expects: pred (B, T, 1, H, W), gt (B, T, 1, H, W)
        # Formula: Σ_t [ Σ_hw p * log p - Σ_hw p * log q ] / (T * log(H*W))

        # Remove channel dimension and flatten spatial: (B, T, 1, H, W) -> (B, T, H*W)
        pred_flat = pred_probs.squeeze(2).view(B, T, -1)  # (B, T, H*W)
        gt_flat = gt_heatmap.squeeze(2).view(B, T, -1)    # (B, T, H*W)

        # Compute log probabilities with numerical stability
        log_pred = torch.log(pred_flat + 1e-10)
        log_gt = torch.log(gt_flat + 1e-10)

        # KL divergence per timestep: Σ_hw p * log p - Σ_hw p * log q
        kl_per_t = (gt_flat * log_gt).sum(dim=-1) - (gt_flat * log_pred).sum(dim=-1)  # (B, T)

        # Sum over time and normalize (matching losses.KLDiv line 97-98)
        norm_scalar = T * torch.log(torch.tensor(H_pred * W_pred, dtype=torch.float32, device=pred_probs.device))
        kl_per_sample = kl_per_t.sum(dim=-1) / norm_scalar  # (B,)

        loss = kl_per_sample.mean()
    elif loss_func == 'mse':
        # MSE loss
        loss = F.mse_loss(pred_probs, gt_heatmap)
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")

    return loss
