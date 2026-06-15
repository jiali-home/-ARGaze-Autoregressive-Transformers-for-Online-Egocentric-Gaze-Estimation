#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import copy
import numpy as np
import pprint
import torch
import torch.nn.functional as F
import cv2

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional experiment logging dependency.
    wandb = None
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainGazeMeter, ValGazeMeter, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule
from slowfast.utils.utils import frame_softmax
from slowfast.datasets import utils as data_utils

logger = logging.get_logger(__name__)


def _select_frame_indices(T: int, k: int):
    if k <= 0:
        return []
    if k >= T:
        return list(range(T))
    # Evenly spaced indices across [0, T-1]
    xs = np.linspace(0, T - 1, num=k)
    return sorted(list({int(round(v)) for v in xs}))


def _to_rgb_uint8(img_float_chw):
    """
    img_float_chw: torch.Tensor in CHW with values in [0,1].
    Return HxWx3 uint8 RGB numpy array.
    """
    img = img_float_chw.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255.0).round().astype(np.uint8)
    return img


def _prepare_frame_for_vis(x_bcthw, cfg):
    """
    Convert a [C, T, H, W] tensor to displayable [0,1] range for logging.
    If the tensor already looks like raw image data in [0,1], keep it as-is.
    Otherwise, assume dataset normalization was applied and denormalize.
    """
    x_min = float(x_bcthw.min().detach().cpu())
    x_max = float(x_bcthw.max().detach().cpu())
    if 0.0 <= x_min and x_max <= 1.0:
        return x_bcthw.clamp(0.0, 1.0)

    c = x_bcthw.shape[0]
    mean = torch.tensor(cfg.DATA.MEAN, device=x_bcthw.device, dtype=x_bcthw.dtype).view(c, 1, 1, 1)
    std = torch.tensor(cfg.DATA.STD, device=x_bcthw.device, dtype=x_bcthw.dtype).view(c, 1, 1, 1)
    return (x_bcthw * std + mean).clamp(0.0, 1.0)


def _overlay_heatmap_rgb(img_rgb_uint8, heatmap_2d_float, alpha=0.4):
    """
    img_rgb_uint8: HxWx3 uint8 RGB
    heatmap_2d_float: HxW float in [0,1] (not necessarily normalized, will be min-max scaled)
    Returns RGB uint8 image with heatmap overlay.
    """
    h, w, _ = img_rgb_uint8.shape
    hm = heatmap_2d_float
    # Min-max normalize for visualization
    hm = hm - float(hm.min())
    denom = float(hm.max())
    hm = hm / (denom + 1e-6)
    hm_resized = cv2.resize(hm.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    hm_vis = (hm_resized * 255.0).clip(0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)  # BGR
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_rgb_uint8, 1.0, hm_color, alpha, 0)
    return blended


def _log_wandb_samples(mode, inputs, labels_hm, preds, cfg, step):
    """Log a few sample frames to Weights & Biases as images.
    Shows: original frame, GT heatmap overlay, Pred heatmap overlay.
    Only supports non-detection gaze training.
    """
    if preds is None or inputs is None or labels_hm is None:
        return
    # inputs: list with one pathway. shape [B, C, T, H, W]
    if isinstance(inputs, (list,)):
        x = inputs[0]
    else:
        x = inputs
    if x.ndim != 5:
        return
    B, C, T, H, W = x.shape
    # pred heatmap: [B, 1, T, Hp, Wp]
    if preds.ndim == 5:
        _, _, Tp, Hp, Wp = preds.shape
    else:
        return
    # labels_hm: [T, Hl, Wl] or [B, T, Hl, Wl]
    if labels_hm.ndim == 3:
        labels_hm_bt = labels_hm.unsqueeze(0)  # [1, T, H, W]
    elif labels_hm.ndim == 4:
        labels_hm_bt = labels_hm  # [B, T, H, W]
    else:
        return

    # Choose first sample in batch.
    b = 0
    # Convert frames back to display space without assuming all datasets
    # use the same normalization path.
    x_denorm = _prepare_frame_for_vis(x[b], cfg)

    # Select frames
    num_show = int(getattr(cfg.WANDB, "NUM_SAMPLE_FRAMES", 1)) if getattr(cfg, "WANDB", None) else 1
    frame_idx = _select_frame_indices(T, num_show)

    panels = []
    for t in frame_idx:
        frame_rgb = _to_rgb_uint8(x_denorm[:, t])  # HxWx3

        # Pred heatmap
        if preds.shape[2] == T:
            pred_hm = preds[b, 0, t].detach().float().cpu().numpy()
        else:
            # Fallback: use last frame index alignment if dimension mismatch
            t_eff = min(t, preds.shape[2] - 1)
            pred_hm = preds[b, 0, t_eff].detach().float().cpu().numpy()

        # GT heatmap
        if labels_hm_bt.shape[0] == B:
            gt_hm = labels_hm_bt[b, t].detach().float().cpu().numpy()
        else:
            # labels provided as [T, H, W]
            t_eff = min(t, labels_hm_bt.shape[1] - 1)
            gt_hm = labels_hm_bt[0, t_eff].detach().float().cpu().numpy()

        gt_overlay = _overlay_heatmap_rgb(frame_rgb, gt_hm, alpha=0.45)
        pred_overlay = _overlay_heatmap_rgb(frame_rgb, pred_hm, alpha=0.45)

        # Concatenate as one panel: [orig | GT | Pred]
        spacer = np.ones((frame_rgb.shape[0], 8, 3), dtype=np.uint8) * 255
        panel = np.concatenate([frame_rgb, spacer, gt_overlay, spacer, pred_overlay], axis=1)
        panels.append(panel)

    if len(panels) == 0:
        return

    vis = np.concatenate(panels, axis=0)
    caption = f"{mode}: orig | GT | Pred"
    wandb.log({f"{mode}/samples": wandb.Image(vis, caption=caption)}, step=step)


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
    wb_run=None,
    steps_per_epoch=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    # steps_per_epoch defaults to legacy behaviour when not provided.
    steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else data_size
    global_step_base = cur_epoch * steps_per_epoch

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    for cur_iter, (inputs, labels, labels_hm, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            labels_hm = labels_hm.cuda()

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:  # default false
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # Pass GT heatmap to model for head prompting (if model supports it)
            # For models that don't use it, they will ignore the extra argument
            if cfg.MODEL.MODEL_NAME in ["DINOv3_AR_HP"] and getattr(cfg.MODEL, "USE_HEAD_PROMPTING", False):
                # For head prompting models, pass GT heatmap during training
                preds = model([inputs[0], labels_hm] if isinstance(inputs, list) else [inputs, labels_hm])

            else:
                preds = model(inputs)

            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)
            if cfg.MODEL.LOSS_FUNC == 'bce_logit':
                weight = torch.tensor([20.]).cuda()
                loss_fun = loss_fun(reduction='mean', pos_weight=weight)
            elif cfg.MODEL.LOSS_FUNC == 'kldiv':
                loss_fun = loss_fun()
            else:
                loss_fun = loss_fun(reduction='mean')

            coord_loss = torch.zeros((), device=preds.device)
            if getattr(cfg.MODEL, "COORD_LOSS_WEIGHT", 0.0) > 0.0 and cfg.MODEL.MODEL_NAME not in [
                "DINOv3_ARPointGaze",
                "DINOv3_ARPointGazeTemplate",
            ]:
                logits_for_coord = preds[0] if isinstance(preds, (tuple, list)) else preds
                coord_loss = losses.coord_loss_from_logits(
                    logits_for_coord,
                    labels,
                    dataset=cfg.TRAIN.DATASET,
                    tau=getattr(cfg.MODEL, "COORD_LOSS_TAU", 0.5),
                    loss_type=getattr(cfg.MODEL, "COORD_LOSS_TYPE", "smooth_l1"),
                )

            # KL-Divergence normalization
            preds = frame_softmax(preds, temperature=2)

            # Main loss (original simple version - no interpolation to avoid breaking normalization)
            loss = loss_fun(preds, labels_hm)
            loss = loss + getattr(cfg.MODEL, "COORD_LOSS_WEIGHT", 0.0) * coord_loss
            entropy_weight = getattr(cfg.MODEL, "ENTROPY_LOSS_WEIGHT", 0.0)
            if entropy_weight > 0.0:
                entropy = -(preds * torch.log(preds + 1e-10)).sum(dim=(-1, -2))
                entropy = entropy.mean()
                loss = loss + entropy_weight * entropy
            else:
                entropy = torch.zeros((), device=loss.device)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL)
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM)
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(labels, 2, dim=1, largest=True, sorted=True)
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=global_step_base + cur_iter,
                )
            # Weights & Biases logging (if enabled)
            if wb_run is not None:
                wb_step = global_step_base + cur_iter
                wandb.log({"Train/loss": loss, "Train/lr": lr}, step=wb_step)

        else:
            # # Gather all the predictions across all the devices to perform ensemble.
            # # Before any distributed gather, optionally log sample frames on master.
            # if wb_run is not None and du.is_master_proc() and (cur_iter % getattr(cfg.WANDB, "LOG_SAMPLES_EVERY", 200) == 0):
            #     wb_step = global_step_base + cur_iter
            #     logger.info(f"Logging wandb samples at iteration {cur_iter}, step {wb_step}")
            #     # logger.info(f"  inputs type: {type(inputs)}, preds shape: {preds.shape}, labels_hm shape: {labels_hm.shape}")
            #     _log_wandb_samples("Train", inputs, labels_hm, preds.detach().clone(), cfg, step=wb_step)

            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]  # average across all processes
                preds, labels_hm, labels = du.all_gather([preds, labels_hm, labels])  # gather (concatenate) across all processes

            loss = loss.item()

            # PyTorch
            preds_rescale = preds.detach().view(preds.size()[:-2] + (preds.size(-1) * preds.size(-2),))
            preds_rescale = (preds_rescale - preds_rescale.min(dim=-1, keepdim=True)[0]) / (preds_rescale.max(dim=-1, keepdim=True)[0] - preds_rescale.min(dim=-1, keepdim=True)[0] + 1e-6)
            preds_rescale = preds_rescale.view(preds.size())
            f1, recall, precision, threshold = metrics.adaptive_f1(preds_rescale, labels_hm, labels, dataset=cfg.TRAIN.DATASET)
            auc = metrics.auc(preds_rescale, labels_hm, labels, dataset=cfg.TRAIN.DATASET)

            # Update and log stats.
            train_meter.update_stats(f1, recall, precision, auc, threshold, loss, entropy.item(), lr,
                                     mb_size=inputs[0].size(0) * max(cfg.NUM_GPUS, 1))  # If running  on CPU (cfg.NUM_GPUS == 0), use 1 to represent 1 CPU.

            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/F1": f1,
                        "Train/Recall": recall,
                        "Train/Precision": precision,
                        "Train/AUC": auc,
                        "Train/Entropy": entropy.item()
                    },
                    global_step=global_step_base + cur_iter,
                )
            if wb_run is not None:
                wb_step = global_step_base + cur_iter
                wandb.log(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/F1": f1,
                        "Train/Recall": recall,
                        "Train/Precision": precision,
                        "Train/AUC": auc,
                        "Train/Entropy": entropy.item(),
                    },
                    step=wb_step,
                )

        train_meter.iter_toc()  # measure all reduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
    val_loader,
    model,
    val_meter,
    cur_epoch,
    cfg,
    writer=None,
    wb_run=None,
    steps_per_epoch=None,
    train_iters_per_epoch=None,
):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else len(val_loader)
    train_iters_per_epoch = (
        train_iters_per_epoch
        if train_iters_per_epoch is not None
        else max(0, steps_per_epoch - len(val_loader))
    )
    global_step_base = cur_epoch * steps_per_epoch + train_iters_per_epoch

    for cur_iter, (inputs, labels, labels_hm, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            labels_hm = labels_hm.cuda()

        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            # Head prompting models: use autoregressive prediction during validation (realistic inference)
            if cfg.MODEL.MODEL_NAME in ["DINOv3_AR_HP"] and getattr(cfg.MODEL, "USE_HEAD_PROMPTING", False):
                # Autoregressive inference: frame-by-frame prediction without GT heatmap
                # This simulates real-world deployment where we don't have future GT labels
                use_autoregressive_val = getattr(cfg.MODEL, "AUTOREGRESSIVE_VAL", True)

                if use_autoregressive_val:
                    # Autoregressive mode: sequential frame-by-frame prediction
                    preds = model.module.forward_autoregressive(inputs) if hasattr(model, 'module') else model.forward_autoregressive(inputs)
                else:
                    # Teacher forcing mode: use GT heatmap (faster but unrealistic)
                    preds = model([inputs[0], labels_hm] if isinstance(inputs, list) else [inputs, labels_hm])
            else:
                preds = model(inputs)
            preds = frame_softmax(preds, temperature=2)  # KLDiv

            # Optionally log validation sample frames before any distributed gather
            if wb_run is not None and du.is_master_proc() and getattr(cfg.WANDB, "LOG_VAL_SAMPLES", True):
                if cur_iter % getattr(cfg.WANDB, "LOG_SAMPLES_EVERY", 200) == 0:
                    wb_step = global_step_base + cur_iter
                    _log_wandb_samples("Val", inputs, labels_hm, preds, cfg, step=wb_step)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    preds, labels_hm, labels = du.all_gather([preds, labels_hm, labels])

                # PyTorch
                preds_rescale = preds.detach().view(preds.size()[:-2] + (preds.size(-1) * preds.size(-2),))
                preds_rescale = (preds_rescale - preds_rescale.min(dim=-1, keepdim=True)[0]) / (preds_rescale.max(dim=-1, keepdim=True)[0] - preds_rescale.min(dim=-1, keepdim=True)[0] + 1e-6)
                preds_rescale = preds_rescale.view(preds.size())
                f1, recall, precision, threshold = metrics.adaptive_f1(preds_rescale, labels_hm, labels, dataset=cfg.TRAIN.DATASET)
                auc = metrics.auc(preds_rescale, labels_hm, labels, dataset=cfg.TRAIN.DATASET)

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(f1, recall, precision, auc, labels, threshold)  # If running  on CPU (cfg.NUM_GPUS == 0), use 1 to represent 1 CPU.

                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {
                            "Val/F1": f1,
                            "Val/Recall": recall,
                            "Val/Precision": precision,
                            "Val/AUC": auc
                        },
                        global_step=global_step_base + cur_iter,
                    )
                if wb_run is not None:
                    wb_step = global_step_base + cur_iter
                    wandb.log(
                        {
                            "Val/F1": f1,
                            "Val/Recall": recall,
                            "Val/Precision": precision,
                            "Val/AUC": auc,
                        },
                        step=wb_step,
                    )

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)

    epoch_metrics = {}
    if cfg.DETECTION.ENABLE:
        epoch_metrics["map"] = getattr(val_meter, "full_map", None)
    else:
        if val_meter.num_samples == 0:
            epoch_metrics = {
                "auc": 0.0,
                "l2": 0.0,
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
            }
        else:
            recall_epoch = val_meter.recall_total / val_meter.num_samples
            precision_epoch = val_meter.precision_total / val_meter.num_samples
            f1_epoch = 2 * recall_epoch * precision_epoch / (recall_epoch + precision_epoch + 1e-6)
            epoch_metrics = {
                "auc": val_meter.auc_total / val_meter.num_samples,
                "l2": val_meter.l2_total / val_meter.num_samples,
                "recall": recall_epoch,
                "precision": precision_epoch,
                "f1": f1_epoch,
            }

    # Log epoch-level metrics to wandb
    if wb_run is not None and du.is_master_proc():
        wandb_step = global_step_base + len(val_loader)
        if cfg.DETECTION.ENABLE:
            if epoch_metrics.get("map") is not None:
                wandb.log(
                    {
                        "Val_Epoch/mAP": epoch_metrics["map"],
                        "epoch": cur_epoch,
                    },
                    step=wandb_step,
                )
        else:
            wandb.log(
                {
                    "Val_Epoch/F1": epoch_metrics["f1"],
                    "Val_Epoch/Recall": epoch_metrics["recall"],
                    "Val_Epoch/Precision": epoch_metrics["precision"],
                    "Val_Epoch/AUC": epoch_metrics["auc"],
                    "Val_Epoch/L2": epoch_metrics["l2"],
                    "epoch": cur_epoch,
                },
                step=wandb_step,
            )

    val_meter.reset()
    return epoch_metrics


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(cfg, "train", is_precise_bn=True)
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (loader.construct_loader(cfg, "train", is_precise_bn=True) if cfg.BN.USE_PRECISE_STATS else None)

    train_iters_per_epoch = len(train_loader)
    val_iters_per_epoch = len(val_loader)
    steps_per_epoch = train_iters_per_epoch + val_iters_per_epoch

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainGazeMeter(len(train_loader), cfg)
        val_meter = ValGazeMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Initialize Weights & Biases run if enabled
    wb_run = None
    if getattr(cfg, "WANDB", None) is not None and cfg.WANDB.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        if wandb is None:
            logger.warning("WANDB.ENABLE=True but wandb is not installed. Skipping Weights & Biases logging.")
        else:
            wb_kwargs = {
                "project": getattr(cfg.WANDB, "PROJECT", None),
                "entity": getattr(cfg.WANDB, "ENTITY", None),
                "name": getattr(cfg.WANDB, "RUN_NAME", None) or getattr(cfg.WANDB, "NAME", None),
                "mode": getattr(cfg.WANDB, "MODE", "online"),
            }
            # Remove None values
            wb_kwargs = {k: v for k, v in wb_kwargs.items() if v is not None}
            try:
                wb_run = wandb.init(**wb_kwargs)
                # Log key hyperparameters and raw cfg as YAML string
                wandb.config.update(
                    {
                        "NUM_GPUS": cfg.NUM_GPUS,
                        "BATCH_SIZE": cfg.TRAIN.BATCH_SIZE,
                        "DATASET": cfg.TRAIN.DATASET,
                        "BASE_LR": cfg.SOLVER.BASE_LR,
                        "MAX_EPOCH": cfg.SOLVER.MAX_EPOCH,
                        "LOSS_FUNC": cfg.MODEL.LOSS_FUNC,
                        "cfg_yaml": cfg.dump(),
                    },
                    allow_val_change=True,
                )
                # log all the config
                wandb.config.update(cfg.dump())
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to log config: {e}")
                wb_run = wandb.init(**wb_kwargs)
                wandb.config.update(
                    {
                        "NUM_GPUS": cfg.NUM_GPUS,
                        "BATCH_SIZE": cfg.TRAIN.BATCH_SIZE,
                        "DATASET": cfg.TRAIN.DATASET,
                        "BASE_LR": cfg.SOLVER.BASE_LR,
                        "MAX_EPOCH": cfg.SOLVER.MAX_EPOCH,
                        "LOSS_FUNC": cfg.MODEL.LOSS_FUNC,
                        "cfg_yaml": cfg.dump(),
                    },
                    allow_val_change=True,
                )
                
        wandb.log({
            "config": cfg.dump(),
            "model": list(model.state_dict().keys()),
        }, step=0)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    best_metric_value = None
    best_metric_name = (
        cfg.TRAIN.CHECKPOINT_BEST_METRIC
        if hasattr(cfg.TRAIN, "CHECKPOINT_BEST_METRIC")
        else "f1"
    )
    is_lower_better = "err" in best_metric_name or "loss" in best_metric_name or best_metric_name == "l2"

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                train_iters_per_epoch = len(train_loader)
                val_iters_per_epoch = len(val_loader)
                steps_per_epoch = train_iters_per_epoch + val_iters_per_epoch

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer)

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)  # Seems not work when GPU=1

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_meter=train_meter,
            cur_epoch=cur_epoch,
            cfg=cfg,
            writer=writer,
            wb_run=wb_run,
            steps_per_epoch=steps_per_epoch,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(cfg, cur_epoch, None if multigrid is None else multigrid.schedule)
        is_eval_epoch = misc.is_eval_epoch(cfg, cur_epoch, None if multigrid is None else multigrid.schedule)

        # Compute precise BN stats.
        if ((is_checkp_epoch or is_eval_epoch) and cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0):
            calculate_and_update_precise_bn(
                loader=precise_bn_loader,
                model=model,
                num_iters=min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                use_gpu=cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)  # seems no influence

        val_metrics = None

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                path_to_job=cfg.OUTPUT_DIR,
                model=model,
                optimizer=optimizer,
                epoch=cur_epoch,
                cfg=cfg,
                scaler=scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            val_metrics = eval_epoch(
                val_loader,
                model,
                val_meter,
                cur_epoch,
                cfg,
                writer,
                wb_run,
                steps_per_epoch=steps_per_epoch,
                train_iters_per_epoch=train_iters_per_epoch,
            )

            if val_metrics and best_metric_name in val_metrics:
                current_metric = val_metrics[best_metric_name]
                if current_metric is not None:
                    is_best = False
                    if best_metric_value is None:
                        best_metric_value = current_metric
                        is_best = True
                    else:
                        if is_lower_better:
                            is_best = current_metric < best_metric_value
                        else:
                            is_best = current_metric > best_metric_value

                        if is_best:
                            prev_value = best_metric_value
                            best_metric_value = current_metric
                            logger.info(
                                f"New best {best_metric_name}: {current_metric:.6f} "
                                f"(prev: {prev_value:.6f}) at epoch {cur_epoch + 1}"
                            )

                    if is_best:
                        cu.save_checkpoint(
                            path_to_job=cfg.OUTPUT_DIR,
                            model=model,
                            optimizer=optimizer,
                            epoch=cur_epoch,
                            cfg=cfg,
                            scaler=scaler if cfg.TRAIN.MIXED_PRECISION else None,
                            is_best=True,
                        )

    if writer is not None:
        writer.close()
    if wb_run is not None:
        try:
            wb_run.finish()
        except Exception:
            pass

    logger.info("Training finished!")
