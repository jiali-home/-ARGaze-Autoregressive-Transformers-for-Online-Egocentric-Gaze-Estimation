#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""
import ipdb
import torch
import torch.nn.functional as F
import numpy as np
import math
import time

from scipy import ndimage
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


def filter_edge_frames(labels, edge_threshold, return_stats=False):
    """
    Filter out frames where gaze is near the edge of the frame.

    Args:
        labels: torch.Tensor of shape (N, 3) where labels[:, 0] = x (normalized [0,1]),
                labels[:, 1] = y (normalized [0,1]), labels[:, 2] = fixation type
        edge_threshold: float in range [0.0, 0.5], distance from edge to filter.
                        E.g., 0.1 means filter gazes within 10% of frame boundaries.
        return_stats: if True, return (valid_idx, num_filtered) instead of just valid_idx

    Returns:
        valid_idx: torch.Tensor of indices where gaze is NOT near edges
        num_filtered: (optional) number of frames filtered out
    """
    x = labels[:, 0]
    y = labels[:, 1]

    # Check if gaze is not near any edge
    # Valid range: [edge_threshold, 1.0 - edge_threshold]
    valid_mask = (
        (x >= edge_threshold) & (x <= 1.0 - edge_threshold) &
        (y >= edge_threshold) & (y <= 1.0 - edge_threshold)
    )

    valid_idx = torch.where(valid_mask)[0]

    if return_stats:
        num_filtered = labels.size(0) - valid_idx.size(0)
        return valid_idx, num_filtered

    return valid_idx


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(0), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


# for gaze estimation
def gaze_iou(preds, labels, threshold):
    # pytorch
    binary_preds = (preds.squeeze(1) > threshold).int()
    binary_labels = (labels > 0.001).int()
    intersection = (binary_preds * binary_labels).sum(dim=(2, 3))
    union = (binary_preds.sum(dim=(2, 3)) + binary_labels.sum(dim=(2, 3))) - intersection

    iou = intersection / (union + 1e-4)
    return float(iou.mean().cpu().numpy())  # need np.float64 in logging rather than np.float32

    # numpy
    # binary_preds = (preds.squeeze(1) > threshold).astype(np.int)
    # binary_labels = (labels > threshold).astype(np.int)
    # intersection = (binary_preds * binary_labels).sum(axis=(2, 3))
    # union = (binary_preds.sum(axis=(2, 3)) + binary_labels.sum(axis=(2, 3))) - intersection
    #
    # iou = intersection / (union + 1e-6)
    # return iou.mean()


# for gaze estimation
def pixel_f1(preds, labels, threshold):
    binary_preds = (preds.squeeze(1) > threshold).int()
    binary_labels = (labels > 0.001).int()
    tp = (binary_preds * binary_labels).sum(dim=(2, 3))
    fg_labels = binary_labels.sum(dim=(2, 3))
    fg_preds = binary_preds.sum(dim=(2, 3))

    # calculate per frame
    # recall = ((tp + 1e-6) / (fg_labels + 1e-6))
    # precision = ((tp + 1e-6) / (fg_preds + 1e-6))
    # f1 = ((2 * recall * precision) / (recall + precision + 1e-6)).mean()

    # calculate over average recall and precision
    recall = (tp / (fg_labels + 1e-6)).mean()
    precision = (tp / (fg_preds + 1e-6)).mean()
    f1 = (2 * recall * precision) / (recall + precision + 1e-6)

    return f1, recall, precision


# for gaze estimation
def adaptive_f1(preds, labels_hm, labels, dataset, edge_threshold=0.0):
    """
    Automatically select the threshold getting the best f1 score.

    Args:
        preds: predicted heatmaps
        labels_hm: ground truth heatmaps
        labels: ground truth gaze labels (x, y, tracking_flag)
        dataset: dataset name
        edge_threshold: if > 0, filter out frames where gaze is within this distance from edges
    """
    # Numpy
    # # thresholds = np.linspace(0, 1.0, 51)
    # thresholds = np.linspace(0, 0.2, 11)
    # # thresholds = np.array([0.5])
    # preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
    # all_preds = np.zeros(shape=(thresholds.shape + labels.shape))
    # all_labels = np.zeros(shape=(thresholds.shape + labels.shape))
    # binary_labels = (labels > 0.001).astype(np.int)
    # for i in range(thresholds.shape[0]):
    #     binary_preds = (preds.squeeze(1) > thresholds[i]).astype(np.int)
    #     all_preds[i, ...] = binary_preds
    #     all_labels[i, ...] = binary_labels
    # tp = (all_preds * all_labels).sum(axis=(3, 4))
    # fg_labels = all_labels.sum(axis=(3, 4))
    # fg_preds = all_preds.sum(axis=(3, 4))
    # recall = (tp / (fg_labels + 1e-6)).mean(axis=(1, 2))
    # precision = (tp / (fg_preds + 1e-6)).mean(axis=(1, 2))
    # f1 = (2 * recall * precision) / (recall + precision + 1e-6)
    # max_idx = np.argmax(f1)
    # return f1[max_idx], recall[max_idx], precision[max_idx], thresholds[max_idx]

    # PyTorch
    # thresholds = np.linspace(0.3, 0.5, 11)
    thresholds = np.linspace(0, 0.02, 11)  # the one we used for GLC
    # thresholds = np.array([0.5])
    # Align prediction grid to label heatmap grid if they differ

    B, _, T, Hp, Wp = preds.shape  # preds shape: (B, 1, T, Hpred, Wpred)
    Hl, Wl = labels_hm.size(-2), labels_hm.size(-1)
    all_preds = torch.zeros(size=(thresholds.shape + labels_hm.size()), device=labels_hm.device)
    all_labels = torch.zeros(size=(thresholds.shape + labels_hm.size()), device=labels_hm.device)
    binary_labels = (labels_hm > 0.001).int()  # change to 0.001
    preds_sw = preds.squeeze(1)  # (B, T, Hp, Wp)
    if (Hp, Wp) != (Hl, Wl):
        # Resize preds to (Hl, Wl) before thresholding
        preds_sw_flat = preds_sw.view(B * T, 1, Hp, Wp)
        preds_resized = F.interpolate(preds_sw_flat, size=(Hl, Wl), mode='bilinear', align_corners=False)
        preds_sw = preds_resized.view(B, T, Hl, Wl)
    for i in range(thresholds.shape[0]):  # There is some space for improvement. You can calculate f1 in the loop rather than save all preds. It consumes much memory.
        thr = float(thresholds[i])
        binary_preds = (preds_sw > thr).int()
        all_preds[i, ...] = binary_preds
        all_labels[i, ...] = binary_labels
    tp = (all_preds * all_labels).sum(dim=(3, 4))
    fg_labels = all_labels.sum(dim=(3, 4))
    fg_preds = all_preds.sum(dim=(3, 4))

    if dataset == 'egteagaze':
        fixation_idx = 1
    elif dataset == 'ego4dgaze' or dataset == 'ego4d_av_gaze' or dataset == 'Ego4dgaze':
        fixation_idx = 0
    elif dataset == 'Holoassistgaze' or dataset == 'holoassistgaze':
        fixation_idx = 1  # Use same convention as egteagaze
    elif dataset == 'Egoexo4dgaze' or dataset == 'egoexo4dgaze':
        fixation_idx = 0  # Use is_valid flag instead
    else:
        raise NotImplementedError(f'Metrics of {dataset} is not implemented.')
    labels_flat = labels.view(labels.size(0) * labels.size(1), labels.size(2))
    if dataset == 'Holoassistgaze' or dataset == 'holoassistgaze' or dataset == 'Egoexo4dgaze' or dataset == 'egoexo4dgaze':
        tracked_idx = torch.where(labels_flat[:, 2] >= 0.5)[0]
    else:
        tracked_idx = torch.where(labels_flat[:, 2] == fixation_idx)[0]

    if tracked_idx.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Apply edge filtering if threshold > 0
    if edge_threshold > 0.0:
        labels_tracked = labels_flat.index_select(0, tracked_idx)
        edge_valid_idx, num_filtered = filter_edge_frames(labels_tracked, edge_threshold, return_stats=True)
        if num_filtered > 0:
            logger.info(f"[adaptive_f1] Filtered {num_filtered}/{labels_tracked.size(0)} frames with edge gazes (threshold={edge_threshold:.3f})")
        # Map back to original tracked_idx
        tracked_idx = tracked_idx[edge_valid_idx]

    tp = tp.view(tp.size(0), tp.size(1)*tp.size(2)).index_select(1, tracked_idx)
    fg_labels = fg_labels.view(fg_labels.size(0), fg_labels.size(1)*fg_labels.size(2)).index_select(1, tracked_idx)
    fg_preds = fg_preds.view(fg_preds.size(0), fg_preds.size(1)*fg_preds.size(2)).index_select(1, tracked_idx)
    recall = (tp / (fg_labels + 1e-6)).mean(dim=1)
    precision = (tp / (fg_preds + 1e-6)).mean(dim=1)
    f1 = (2 * recall * precision) / (recall + precision + 1e-6)
    max_idx = torch.argmax(f1)

    # recall = (tp / (fg_labels + 1e-6)).mean(dim=(1, 2))
    # precision = (tp / (fg_preds + 1e-6)).mean(dim=(1, 2))
    # f1 = (2 * recall * precision) / (recall + precision + 1e-6)
    # max_idx = torch.argmax(f1)
    # ipdb.set_trace()
    return float(f1[max_idx].cpu().numpy()), float(recall[max_idx].cpu().numpy()), \
           float(precision[max_idx].cpu().numpy()), thresholds[max_idx]  # need np.float64 in logging rather than np.float32


# for gaze estimation
def average_angle_error(preds, labels, dataset):
    if dataset == 'egteagaze':
        fixation_idx = 1
    elif dataset == 'ego4dgaze' or dataset == 'ego4d_av_gaze':
        fixation_idx = 0
    elif dataset == 'Holoassistgaze' or dataset == 'holoassistgaze':
        fixation_idx = 1  # Use same convention as egteagaze
    elif dataset == 'Egoexo4dgaze' or dataset == 'egoexo4dgaze':
        fixation_idx = 0  # Use is_valid flag instead
    else:
        raise NotImplementedError(f'Metrics of {dataset} is not implemented.')
    labels = labels.view(labels.size(0) * labels.size(1), labels.size(2))
    if dataset == 'Holoassistgaze' or dataset == 'holoassistgaze' or dataset == 'Egoexo4dgaze' or dataset == 'egoexo4dgaze':
        tracked_idx = torch.where(labels[:, 2] >= 0.5)[0]
    else:
        tracked_idx = torch.where(labels[:, 2] == fixation_idx)[0]
    labels = labels.index_select(0, tracked_idx)
    preds = preds.squeeze(1)
    preds = preds.view(preds.size(0) * preds.size(1), preds.size(2), preds.size(3))
    preds = preds.index_select(0, tracked_idx).cpu().numpy()
    labels = labels.cpu().numpy()

    aae = list()
    for frame in range(preds.shape[0]):
        out_sq = preds[frame, :, :]
        predicted = ndimage.measurements.center_of_mass(out_sq)
        H, W = out_sq.shape[-2], out_sq.shape[-1]
        (i, j) = labels[frame, 1] * H, labels[frame, 0] * W
        half = min(H, W) / 2.0
        d = half / math.tan(math.pi / 6)
        r1 = np.array([predicted[0] - half, predicted[1] - half, d])
        r2 = np.array([i - half, j - half, d])
        angle = math.atan2(np.linalg.norm(np.cross(r1, r2)), np.dot(r1, r2))
        aae.append(math.degrees(angle))

    return float(np.mean(aae))


# for gaze estimation
def auc(preds, labels_hm, labels, dataset, edge_threshold=0.0):
    """
    Calculate AUC metric for gaze prediction.

    Args:
        preds: predicted heatmaps
        labels_hm: ground truth heatmaps
        labels: ground truth gaze labels (x, y, tracking_flag)
        dataset: dataset name
        edge_threshold: if > 0, filter out frames where gaze is within this distance from edges
    """
    if dataset == 'egteagaze':
        fixation_idx = 1
    elif dataset == 'ego4dgaze' or dataset == 'ego4d_av_gaze' or dataset == 'Ego4dgaze':
        fixation_idx = 0
    elif dataset == 'Holoassistgaze' or dataset == 'holoassistgaze':
        fixation_idx = 1  # Use same convention as egteagaze
    elif dataset == 'Egoexo4dgaze' or dataset == 'egoexo4dgaze':
        fixation_idx = 0  # Use is_valid flag instead
    else:
        raise NotImplementedError(f'Metrics of {dataset} is not implemented.')
    labels = labels.view(labels.size(0) * labels.size(1), labels.size(2))

    if dataset == 'Holoassistgaze' or dataset == 'holoassistgaze' or dataset == 'Egoexo4dgaze' or dataset == 'egoexo4dgaze':
        confidence_threshold = 0.5
        tracked_idx = torch.where(labels[:, 2] >= confidence_threshold)[0]
    else:
        tracked_idx = torch.where(labels[:, 2] == fixation_idx)[0]

    if tracked_idx.numel() == 0:
        return 0.0

    # Apply edge filtering if threshold > 0
    if edge_threshold > 0.0:
        labels_tracked = labels.index_select(0, tracked_idx)
        edge_valid_idx, num_filtered = filter_edge_frames(labels_tracked, edge_threshold, return_stats=True)
        if num_filtered > 0:
            logger.info(f"[auc] Filtered {num_filtered}/{labels_tracked.size(0)} frames with edge gazes (threshold={edge_threshold:.3f})")
        # Map back to original tracked_idx
        tracked_idx = tracked_idx[edge_valid_idx]

    labels = labels.index_select(0, tracked_idx)
    preds = preds.squeeze(1)
    preds = preds.view(preds.size(0) * preds.size(1), preds.size(2), preds.size(3))
    preds = preds.index_select(0, tracked_idx).cpu().numpy()
    labels = labels.cpu().numpy()

    auc = list()
    for frame in range(preds.shape[0]):
        out_sq = preds[frame, :, :]
        predicted = ndimage.measurements.center_of_mass(out_sq)
        H, W = labels_hm.size(-2), labels_hm.size(-1)
        i = int(round(labels[frame, 1] * (H - 1)))
        j = int(round(labels[frame, 0] * (W - 1)))
        i = max(0, min(H - 1, i))
        j = max(0, min(W - 1, j))

        z = np.zeros((H, W))
        if np.isnan(predicted[0]) or np.isnan(predicted[1]):  # the prediction may be nan for some algorithms
            z[H // 2, W // 2] = 1
        else:
            pi = int(round(predicted[0]))
            pj = int(round(predicted[1]))
            if 0 <= pi < H and 0 <= pj < W:
                z[pi, pj] = 1
            else:
                # If predicted center goes out of bounds, clamp it
                z[max(0, min(H - 1, pi)), max(0, min(W - 1, pj))] = 1
        z = ndimage.filters.gaussian_filter(z, 3.2)
        z = z - np.min(z)
        z = z / (np.max(z) + 1e-8)
        atgt = z[i][j]
        fpbool = z > atgt
        auc1 = 1 - float(fpbool.sum()) / preds.shape[2] / preds.shape[1]
        auc.append(auc1)

    return float(np.mean(auc))


# for gaze estimation
def l2_distance(preds, labels_hm, labels, dataset, edge_threshold=0.0, l2_mode="argmax"):
    """
    Calculate L2 distance between predicted and ground truth gaze points.
    The distance is normalized by image dimensions (max distance = sqrt(2)).

    Args:
        preds: predicted heatmaps, shape (B, 1, T, H, W)
        labels_hm: ground truth heatmaps, shape (B, T, H, W)
        labels: ground truth labels with gaze points, shape (B, T, 3) where
                labels[:, :, 0] = x (normalized), labels[:, :, 1] = y (normalized),
                labels[:, :, 2] = fixation type
        dataset: dataset name for filtering valid frames
        edge_threshold: if > 0, filter out frames where gaze is within this distance from edges
        l2_mode: "argmax" or "expectation" for predicted point extraction

    Returns:
        float: mean normalized L2 distance across all valid frames
    """
    if dataset == 'egteagaze':
        fixation_idx = 1
    elif dataset == 'ego4dgaze' or dataset == 'ego4d_av_gaze' or dataset == 'Ego4dgaze':
        fixation_idx = 0
    elif dataset == 'Holoassistgaze' or dataset == 'holoassistgaze':
        fixation_idx = 1
    elif dataset == 'Egoexo4dgaze' or dataset == 'egoexo4dgaze':
        fixation_idx = 0  # Use is_valid flag instead
    else:
        raise NotImplementedError(f'Metrics of {dataset} is not implemented.')

    labels = labels.view(labels.size(0) * labels.size(1), labels.size(2))

    # Filter for tracked/fixation frames
    if dataset == 'Holoassistgaze' or dataset == 'holoassistgaze' or dataset == 'Egoexo4dgaze' or dataset == 'egoexo4dgaze':
        tracked_idx = torch.where(labels[:, 2] >= 0.5)[0]
    else:
        tracked_idx = torch.where(labels[:, 2] == fixation_idx)[0]

    if tracked_idx.numel() == 0:
        return 0.0

    # Apply edge filtering if threshold > 0
    if edge_threshold > 0.0:
        labels_tracked = labels.index_select(0, tracked_idx)
        edge_valid_idx, num_filtered = filter_edge_frames(labels_tracked, edge_threshold, return_stats=True)
        if num_filtered > 0:
            logger.info(f"[l2_distance] Filtered {num_filtered}/{labels_tracked.size(0)} frames with edge gazes (threshold={edge_threshold:.3f})")
        # Map back to original tracked_idx
        tracked_idx = tracked_idx[edge_valid_idx]

    labels = labels.index_select(0, tracked_idx)
    preds = preds.squeeze(1)
    preds = preds.view(preds.size(0) * preds.size(1), preds.size(2), preds.size(3))
    preds = preds.index_select(0, tracked_idx).cpu().numpy()
    labels = labels.cpu().numpy()

    l2_mode = (l2_mode or "argmax").lower()
    if l2_mode not in ["argmax", "expectation"]:
        raise ValueError(f"Unsupported l2_mode: {l2_mode}")

    l2_distances = []
    for frame in range(preds.shape[0]):
        out_sq = preds[frame, :, :]
        H, W = out_sq.shape[-2], out_sq.shape[-1]

        # Ground truth gaze point (normalized coordinates)
        gt_x = labels[frame, 0]  # normalized x
        gt_y = labels[frame, 1]  # normalized y

        # Predicted gaze point (normalized coordinates)
        if l2_mode == "argmax":
            if not np.isfinite(out_sq).any():
                pred_x = 0.5
                pred_y = 0.5
            else:
                flat_idx = np.nanargmax(out_sq)
                pred_row, pred_col = np.unravel_index(flat_idx, out_sq.shape)
                pred_x = pred_col / (W - 1) if W > 1 else 0.5
                pred_y = pred_row / (H - 1) if H > 1 else 0.5
        else:
            weights = np.nan_to_num(out_sq, nan=0.0, posinf=0.0, neginf=0.0)
            total = float(weights.sum())
            if total <= 0.0:
                pred_x = 0.5
                pred_y = 0.5
            else:
                ys = np.arange(H, dtype=np.float64)
                xs = np.arange(W, dtype=np.float64)
                wy = weights.sum(axis=1)
                wx = weights.sum(axis=0)
                pred_y = float((wy * ys).sum() / total)
                pred_x = float((wx * xs).sum() / total)
                pred_x = pred_x / (W - 1) if W > 1 else 0.5
                pred_y = pred_y / (H - 1) if H > 1 else 0.5

        # Calculate L2 distance in normalized coordinate space
        # Maximum distance is sqrt(2) (from corner to corner)
        l2_dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
        l2_distances.append(l2_dist)

    return float(np.mean(l2_distances))


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        int: number of trainable parameters in millions
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def compute_flops(model, inputs, cfg):
    """
    Compute FLOPs (Floating Point Operations) for the model.

    Args:
        model: PyTorch model
        inputs: sample input tensor(s)
        cfg: configuration object

    Returns:
        float: GFLOPs (Giga Floating-Point Operations)
    """
    try:
        from fvcore.nn import FlopCountAnalysis

        # Put model in eval mode for FLOPs computation
        model.eval()

        with torch.no_grad():
            flop_counter = FlopCountAnalysis(model, inputs)
            flops = flop_counter.total()

        return flops / 1e9  # Convert to GFLOPs
    except ImportError:
        logger_metrics = logging.get_logger(__name__)
        logger_metrics.warning("fvcore not installed, cannot compute FLOPs")
        return -1.0
    except Exception as e:
        logger_metrics = logging.get_logger(__name__)
        logger_metrics.warning(f"Failed to compute FLOPs: {e}")
        return -1.0


def measure_throughput_and_latency(model, inputs, cfg, num_warmup=10, num_iterations=100):
    """
    Measure throughput (FPS) and latency (ms) of the model.

    Args:
        model: PyTorch model
        inputs: sample input tensor(s)
        cfg: configuration object
        num_warmup: number of warmup iterations
        num_iterations: number of measurement iterations

    Returns:
        tuple: (throughput in FPS, latency in ms)
    """
    model.eval()
    device = next(model.parameters()).device

    # Move inputs to the correct device
    if isinstance(inputs, (list, tuple)):
        inputs = [inp.to(device) for inp in inputs]
    else:
        inputs = inputs.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(inputs)

    # Synchronize before measurement
    if cfg.NUM_GPUS > 0:
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(inputs)
            if cfg.NUM_GPUS > 0:
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)

    avg_time = np.mean(times)
    latency_ms = avg_time * 1000  # Convert to milliseconds

    # Get batch size from inputs
    if isinstance(inputs, (list, tuple)):
        batch_size = inputs[0].size(0)
    else:
        batch_size = inputs.size(0)

    throughput_fps = batch_size / avg_time

    return throughput_fps, latency_ms


def measure_memory_footprint(model, inputs, cfg):
    """
    Measure memory footprint of the model.

    Args:
        model: PyTorch model
        inputs: sample input tensor(s)
        cfg: configuration object

    Returns:
        dict: dictionary containing memory statistics in MB
    """
    if cfg.NUM_GPUS == 0:
        return {"total_memory_mb": -1.0, "peak_memory_mb": -1.0}

    model.eval()
    device = next(model.parameters()).device

    # Move inputs to the correct device
    if isinstance(inputs, (list, tuple)):
        inputs = [inp.to(device) for inp in inputs]
    else:
        inputs = inputs.to(device)

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Measure memory before forward pass
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(device) / 1e6  # Convert to MB

    # Forward pass
    with torch.no_grad():
        _ = model(inputs)
        torch.cuda.synchronize()

    # Measure memory after forward pass
    mem_after = torch.cuda.memory_allocated(device) / 1e6  # Convert to MB
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6  # Convert to MB

    return {
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "peak_memory_mb": peak_mem,
        "activation_memory_mb": mem_after - mem_before
    }


# for action recognition
def mean_class_accuracy(preds, labels):
    y_pred = preds.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    cf = confusion_matrix(y_true, y_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt

    return np.mean(cls_acc), cls_acc


def conf_matrix(preds, labels):
    y_pred = preds.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    if y_pred.ndim == 2:  # if preds are probability
        y_pred = np.argmax(y_pred, axis=1)
    elif y_pred.ndim == 1:  # if preds are categories
        pass
    else:
        raise NotImplementedError
    cf = confusion_matrix(y_true, y_pred)
    return cf


# for persuasion strategy classification
def mean_f1_for_multilabel(preds, labels):
    y_pred = preds.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)

    return f1, f1.mean()


# def auc(preds, labels_hm, labels):
#     preds = preds.squeeze(1)
#
#     labels_flat = labels.view(labels.size(0) * labels.size(1), 4)
#     tracked_idx = torch.where(labels_flat[:, 2] == 1)
#     tracked_labels_hm = labels_hm.view(labels_hm.size(0) * labels_hm.size(1), -1).index_select(0, tracked_idx[0])
#     tracked_preds = preds.view(preds.size(0) * preds.size(1), -1).index_select(0, tracked_idx[0])
#     # ipdb.set_trace()
#
#     p = tracked_preds.squeeze(1).view(-1).cpu().numpy()
#     binary_labels = (tracked_labels_hm > 0.001).int()
#     l = binary_labels.view(-1).cpu().numpy()
#     score = roc_auc_score(y_true=l, y_score=p)
#     # ipdb.set_trace()
#
#     return score
