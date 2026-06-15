#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch

import slowfast.utils.lr_policy as lr_policy


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    bn_parameters = []
    non_bn_parameters = []
    zero_parameters = []
    no_grad_parameters = []
    skip = {}
    if cfg.NUM_GPUS > 1:
        if hasattr(model.module, "no_weight_decay"):
            skip = model.module.no_weight_decay()
            skip = {"module." + v for v in skip}
    else:
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

    # When UNFREEZE_LAST_K_LAYERS > 0, keep unfrozen encoder params in a
    # separate group so they can use a scaled-down LR (ENCODER_LR_SCALE)
    # to avoid destroying pretrained features.
    encoder_lr_scale = getattr(cfg.MODEL, "ENCODER_LR_SCALE", 0.1)
    unfreeze_k = getattr(cfg.MODEL, "UNFREEZE_LAST_K_LAYERS", 0)
    use_encoder_group = unfreeze_k > 0 and encoder_lr_scale != 1.0

    # Collect ids of trainable encoder params for fast lookup
    encoder_param_ids = set()
    if use_encoder_group:
        prefix = "module.encoder." if cfg.NUM_GPUS > 1 else "encoder."
        for name, p in model.named_parameters():
            if p.requires_grad and name.startswith(prefix):
                encoder_param_ids.add(id(p))

    encoder_non_bn_parameters = []
    encoder_zero_parameters = []

    for name, m in model.named_modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for p in m.parameters(recurse=False):
            if not p.requires_grad:
                no_grad_parameters.append(p)
            elif is_bn:
                bn_parameters.append(p)
            elif name in skip:
                zero_parameters.append(p)
            elif cfg.SOLVER.ZERO_WD_1D_PARAM and \
                (len(p.shape) == 1 or name.endswith(".bias")):
                if id(p) in encoder_param_ids:
                    encoder_zero_parameters.append(p)
                else:
                    zero_parameters.append(p)
            elif id(p) in encoder_param_ids:
                encoder_non_bn_parameters.append(p)
            else:
                non_bn_parameters.append(p)

    optim_params = [
        {"params": bn_parameters, "weight_decay": cfg.BN.WEIGHT_DECAY, "lr_scale": 1.0},
        {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY, "lr_scale": 1.0},
        {"params": zero_parameters, "weight_decay": 0.0, "lr_scale": 1.0},
    ]
    if use_encoder_group:
        enc_lr = cfg.SOLVER.BASE_LR * encoder_lr_scale
        if len(encoder_non_bn_parameters) > 0:
            optim_params.append({
                "params": encoder_non_bn_parameters,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                "lr": enc_lr,
                "lr_scale": encoder_lr_scale,
            })
        if len(encoder_zero_parameters) > 0:
            optim_params.append({
                "params": encoder_zero_parameters,
                "weight_decay": 0.0,
                "lr": enc_lr,
                "lr_scale": encoder_lr_scale,
            })

    optim_params = [x for x in optim_params if len(x["params"])]

    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_parameters
    ) + len(zero_parameters) + len(
        no_grad_parameters
    ) + len(encoder_non_bn_parameters) + len(
        encoder_zero_parameters
    ), "parameter size does not match: {} + {} + {} + {} + {} + {} != {}".format(
        len(non_bn_parameters),
        len(bn_parameters),
        len(zero_parameters),
        len(no_grad_parameters),
        len(encoder_non_bn_parameters),
        len(encoder_zero_parameters),
        len(list(model.parameters())),
    )
    print(
        "bn {}, non_bn {}, zero {}, no_grad {}, encoder_non_bn {}, encoder_zero {}".format(
            len(bn_parameters),
            len(non_bn_parameters),
            len(zero_parameters),
            len(no_grad_parameters),
            len(encoder_non_bn_parameters),
            len(encoder_zero_parameters),
        )
    )
    if use_encoder_group:
        print(
            "  -> encoder LR scale: {:.3f}  (encoder LR = {:.2e}, decoder LR = {:.2e})".format(
                encoder_lr_scale,
                cfg.SOLVER.BASE_LR * encoder_lr_scale,
                cfg.SOLVER.BASE_LR,
            )
        )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value, respecting per-group lr_scale.

    Groups with lr_scale < 1.0 (e.g. unfrozen encoder layers) are scaled
    proportionally so the relative LR ratio is maintained throughout training.

    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new base learning rate for the decoder group.
    """
    for param_group in optimizer.param_groups:
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = new_lr * scale
