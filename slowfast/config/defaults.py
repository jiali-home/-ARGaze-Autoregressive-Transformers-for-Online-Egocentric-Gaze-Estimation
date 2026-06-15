#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

from . import custom_config

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "kinetics"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Number of recent checkpoints to keep (0 = keep all, >0 = keep last N + best + latest)
# Example: CHECKPOINT_KEEP_LAST_N = 3 will keep:
#   - checkpoint_epoch_best.pyth (best validation performance)
#   - checkpoint_epoch_latest.pyth (most recent epoch)
#   - checkpoint_epoch_XXXXX.pyth (last 3 numbered epochs)
# This saves storage space while ensuring you can resume training and have the best model
_C.TRAIN.CHECKPOINT_KEEP_LAST_N = 0

# Metric to track for best checkpoint (e.g., "top1_err", "loss", "auc", "f1")
# If metric name contains "err" or "loss", lower is better; otherwise higher is better
_C.TRAIN.CHECKPOINT_BEST_METRIC = "f1"

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)

# If True, use FP16 for activations
_C.TRAIN.MIXED_PRECISION = False
_C.TRAIN.AMP_DTYPE = "fp16"
_C.TRAIN.FORCE_FROM_SCRATCH = False

# ---------------------------------------------------------------------------- #
# Student-Teacher EMA Training options
# ---------------------------------------------------------------------------- #
# Enable student-teacher training with Exponential Moving Average (EMA) teacher
_C.TRAIN.USE_STUDENT_TEACHER = False

# Knowledge Distillation temperature for softmax
# Higher values (e.g., 2.0-4.0) produce softer probability distributions
_C.TRAIN.KD_TEMPERATURE = 2.0

# Weight for knowledge distillation loss
_C.TRAIN.KD_LAMBDA = 1.0

# Weight for ground truth supervision loss
_C.TRAIN.GT_LAMBDA = 1.0

# Initial EMA momentum (higher = teacher updates faster initially)
_C.TRAIN.EMA_MOMENTUM_INIT = 0.99

# Final EMA momentum (lower = teacher more stable at end of training)
_C.TRAIN.EMA_MOMENTUM_FINAL = 0.9995

# Total training steps for EMA momentum scheduling (0 = auto-compute)
_C.TRAIN.TOTAL_STEPS = 0

# Weight KD loss by teacher confidence (1 - normalized_entropy)
# Confident predictions get higher weight, uncertain ones get lower weight
_C.TRAIN.KD_CONFIDENCE_WEIGHTING = True

# Apply KD only to specific frame indices (e.g., [1, 3, 7])
# None means apply to all frames
_C.TRAIN.KD_KEYFRAMES = None

# ---------------------------------------------------------------------------- #
# Temporal Window Based Student-Teacher Training (Single Model)
# ---------------------------------------------------------------------------- #
# Enable temporal window based student-teacher training
# Uses same model with different temporal windows: student (causal) vs teacher (more future info)
_C.TRAIN.USE_TEMPORAL_WINDOW_ST = False

# Teacher's future window size (how many future frames teacher can see)
# Student always uses win_T_future=0 (causal/online mode)
# Recommended: 1-4 frames
_C.TRAIN.TEACHER_WIN_T_FUTURE = 1

# Student temporal window (used in temporal-window ST branch).
# -1 means unlimited past, 0 means causal no-future.
_C.TRAIN.STUDENT_WIN_T_PAST = -1
_C.TRAIN.STUDENT_WIN_T_FUTURE = 0

# Teacher temporal past window (paired with TEACHER_WIN_T_FUTURE).
# -1 means unlimited past.
_C.TRAIN.TEACHER_WIN_T_PAST = -1

# Optional temporal trim for ST losses to avoid boundary-token mismatch.
# Losses are computed on frames [trim_left, T-trim_right).
_C.TRAIN.ST_LOSS_TRIM_LEFT = 0
_C.TRAIN.ST_LOSS_TRIM_RIGHT = 0

# Whether to detach teacher gradients (recommended: True for stability)
_C.TRAIN.DETACH_TEACHER_GRAD = True

# Randomize future frames for teacher (ablation control).
# When True, the temporal order of frames beyond the student's causal window
# is randomly permuted before the teacher forward pass.  This tests whether
# the temporal *structure* of future frames matters for distillation.
_C.TRAIN.RANDOMIZE_TEACHER_FUTURE = False

# Loss weight for student-GT KL divergence
_C.TRAIN.ST_ALPHA = 1.0

# Loss weight for student-teacher KL divergence (distillation)
_C.TRAIN.ST_BETA = 0.5

# Loss weight for teacher-GT KL divergence
_C.TRAIN.ST_GAMMA = 1.0

# ---------------------------------------------------------------------------- #
# Augmentation options.
# ---------------------------------------------------------------------------- #
_C.AUG = CfgNode()

# Whether to enable randaug.
_C.AUG.ENABLE = False

# Number of repeated augmentations to used during training.
# If this is greater than 1, then the actual batch size is
# TRAIN.BATCH_SIZE * AUG.NUM_SAMPLE.
_C.AUG.NUM_SAMPLE = 1

# Not used if using randaug.
_C.AUG.COLOR_JITTER = 0.4

# RandAug parameters.
_C.AUG.AA_TYPE = "rand-m9-mstd0.5-inc1"

# Interpolation method.
_C.AUG.INTERPOLATION = "bicubic"

# Probability of random erasing.
_C.AUG.RE_PROB = 0.25

# Random erasing mode.
_C.AUG.RE_MODE = "pixel"

# Random erase count.
_C.AUG.RE_COUNT = 1

# Do not random erase first (clean) augmentation split.
_C.AUG.RE_SPLIT = False

# ---------------------------------------------------------------------------- #
# PERF options.
# ---------------------------------------------------------------------------- #
_C.PERF = CfgNode()

# Log period.
_C.PERF.LOG_PERIOD = 20
# Cheap metrics period.
_C.PERF.METRICS_PERIOD = 10
# Full metrics period.
_C.PERF.METRICS_PERIOD_FULL = 40
# Metrics batch fraction.
_C.PERF.METRICS_BATCH_FRACTION = 0.25
# Metrics downsample.
_C.PERF.METRICS_DOWNSAMPLE = 8
# Eval every epochs.
_C.PERF.EVAL_EVERY_EPOCHS = 2
# Val max steps.
_C.PERF.VAL_MAX_STEPS = 200
# Val subsample.
_C.PERF.VAL_SUBSAMPLE = 2
# Allow tf32.
_C.PERF.ALLOW_TF32 = True
# Torch compile.
_C.PERF.TORCH_COMPILE = False
# Compile mode.
_C.PERF.COMPILE_MODE = "default"
# Channels last 3d.
_C.PERF.CHANNELS_LAST_3D = False

# ---------------------------------------------------------------------------- #
# MipUp options.
# ---------------------------------------------------------------------------- #
_C.MIXUP = CfgNode()

# Whether to use mixup.
_C.MIXUP.ENABLE = False

# Mixup alpha.
_C.MIXUP.ALPHA = 0.8

# Cutmix alpha.
_C.MIXUP.CUTMIX_ALPHA = 1.0

# Probability of performing mixup or cutmix when either/both is enabled.
_C.MIXUP.PROB = 1.0

# Probability of switching to cutmix when both mixup and cutmix enabled.
_C.MIXUP.SWITCH_PROB = 0.5

# Label smoothing.
_C.MIXUP.LABEL_SMOOTH_VALUE = 0.1

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "kinetics"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 1

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 1

# Test on all frames (used for gaze estimation)
_C.TEST.FULL_FRAME_TEST = True

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"
# Path to saving prediction results file.
_C.TEST.SAVE_RESULTS_PATH = ""

# Whether to enable subsetting for testing
_C.TEST.ENABLE_SUBSET = False

# Number of samples to use for subset testing (if 0, use all samples)
_C.TEST.SUBSET_SIZE = 0

# Optional comma-separated filters for qualitative/debug evaluation. When set,
# dataset loaders keep only matching test clips/sequences before inference.
_C.TEST.CLIP_INDEX_FILTER = ""
_C.TEST.SEQUENCE_FILTER = ""

# Store all predictions/heatmaps during testing (can be memory heavy for large datasets).
# If False, metrics are computed in a streaming manner to reduce memory usage.
_C.TEST.STORE_HEATMAPS = True

# Streaming test options (clip models evaluated frame-by-frame with sliding window).
_C.TEST.STREAMING_ENABLE = False
# Warm-start strategy when window is not full: "replicate" or "zeros".
_C.TEST.STREAMING_WARM_START = "replicate"
# Collect per-frame streaming metrics (latency/fps) during testing.
_C.TEST.STREAMING_METRICS_ENABLE = True
# Number of initial frames per clip to skip for streaming metrics.
_C.TEST.STREAMING_METRICS_WARMUP = 0
# Optional causal sliding-window size used during streaming inference.
# If <= 0, defaults to DATA.NUM_FRAMES.
_C.TEST.STREAMING_WINDOW_FRAMES = 0
# Use template tokens from cached frame embeddings (faster, may affect accuracy).
_C.TEST.STREAMING_TEMPLATE_FROM_TOKENS = False

# If True, only evaluate metrics on the last frame of each clip during testing.
_C.TEST.ONLY_LAST_FRAME = False

# First temporal index to include in test-time evaluation and reporting.
# Useful for streaming/causal evaluation when early frames do not have enough history.
_C.TEST.EVAL_START_INDEX = 0

# Test-time temporal subsampling to simulate lower frame rates.
# 1 keeps every frame, 2 keeps every other frame, 4 keeps one out of every 4, etc.
_C.TEST.TEMPORAL_SUBSAMPLE_STRIDE = 1

# Optional temporal offset for subsampling.
# Example: stride=4, offset=1 keeps indices [1, 5, 9, ...].
_C.TEST.TEMPORAL_SUBSAMPLE_OFFSET = 0

# Optional metadata for reporting/plotting effective history span.
_C.TEST.BASE_FPS = 30.0

# Optional file names for machine-readable test summaries.
_C.TEST.SUMMARY_JSON = "test_metrics_summary.json"
_C.TEST.SUMMARY_CSV = "test_metrics_summary.csv"

# If True, report metrics grouped by frame index within the clip during testing.
_C.TEST.REPORT_METRICS_BY_CLIP_INDEX = False

# Dataset split for testing (dataset-specific). For EgoExo4D:
# test_iid, test_ood_site, test_ood_task, test_ood_participant.
_C.TEST.SPLIT = "test"

# Gaze edge filtering options (for removing outliers near frame boundaries)
# If True, filter out frames where GT gaze is near the edge of the frame
_C.TEST.FILTER_EDGE_GAZE = False

# Distance threshold from edge (0.0-0.5). Gazes within this distance from any edge are filtered.
# For example, 0.1 means gazes within 10% of frame width/height from edges are excluded
_C.TEST.EDGE_THRESHOLD = 0.1

# L2 calculation mode for gaze evaluation: "argmax" or "expectation".
_C.TEST.L2_MODE = "argmax"
# Per-frame visualization/metrics during inference.
_C.TEST.VISUALIZE = False
_C.TEST.VIS_DIR = "visulization"
_C.TEST.VIS_ALPHA = 0.4
_C.TEST.VIS_MAX_FRAMES = -1
_C.TEST.SAVE_PER_FRAME_METRICS = False
_C.TEST.PER_FRAME_METRICS_FILE = "per_frame_metrics.csv"
_C.TEST.DEBUG_HEATMAP_STATS = False
_C.TEST.DEBUG_HEATMAP_STATS_FILE = "heatmap_debug_stats.csv"
_C.TEST.DEBUG_HEATMAP_SUMMARY_FILE = "heatmap_debug_summary.json"
_C.TEST.DEBUG_HEATMAP_MAX_ROWS = 2000
_C.TEST.SAVE_BOOTSTRAP_ARTIFACTS = False
_C.TEST.BOOTSTRAP_ARTIFACTS_DIR = "bootstrap_artifacts"
_C.TEST.BOOTSTRAP_ARTIFACTS_MANIFEST = "manifest.json"
_C.TEST.BOOTSTRAP_ARTIFACT_DTYPE = "float16"
_C.TEST.BOOTSTRAP_SEQUENCE_UNIT = "clip"
_C.TEST.TASK_CATEGORY_KEY = "parent_task_name"
_C.TEST.TASK_METRICS_FILE = "task_metrics.csv"
# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]

# ---------------------------------------------------------------------------- #
# X3D  options
# See https://arxiv.org/abs/2004.04730 for details about X3D Networks.
# ---------------------------------------------------------------------------- #
_C.X3D = CfgNode()

# Width expansion factor.
_C.X3D.WIDTH_FACTOR = 1.0

# Depth expansion factor.
_C.X3D.DEPTH_FACTOR = 1.0

# Bottleneck expansion factor for the 3x3x3 conv.
_C.X3D.BOTTLENECK_FACTOR = 1.0  #

# Dimensions of the last linear layer before classificaiton.
_C.X3D.DIM_C5 = 2048

# Dimensions of the first 3x3 conv layer.
_C.X3D.DIM_C1 = 12

# Whether to scale the width of Res2, default is false.
_C.X3D.SCALE_RES2 = False

# Whether to use a BatchNorm (BN) layer before the classifier, default is false.
_C.X3D.BN_LIN5 = False

# Whether to use channelwise (=depthwise) convolution in the center (3x3x3)
# convolution operation of the residual blocks.
_C.X3D.CHANNELWISE_3x3x3 = True

# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slowfast"

# Model name
_C.MODEL.MODEL_NAME = "SlowFast"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 400

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Weight for KL divergence loss (used in AR Heatmap Gaze model)
_C.MODEL.KL_LOSS_WEIGHT = 0.1

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["2d", "c2d", "i3d", "slow", "x3d", "mvit"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Randomly drop rate for Res-blocks, linearly increase from res2 to res5
_C.MODEL.DROPCONNECT_RATE = 0.0

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Activation checkpointing enabled or not to save GPU memory.
_C.MODEL.ACT_CHECKPOINT = False


# -----------------------------------------------------------------------------
# DINOv3_QueryFocuser options
# -----------------------------------------------------------------------------
_C.DINOV3_QUERY_FOCUSER = CfgNode()
_C.DINOV3_QUERY_FOCUSER.NUM_QUERIES = 1
_C.DINOV3_QUERY_FOCUSER.QUERY_TEMPERATURE = 1.0


# -----------------------------------------------------------------------------
# MViT options
# -----------------------------------------------------------------------------
_C.MVIT = CfgNode()

# Options include `conv`, `max`.
_C.MVIT.MODE = "conv"

# If True, perform pool before projection in attention.
_C.MVIT.POOL_FIRST = False

# If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
_C.MVIT.CLS_EMBED_ON = True

_C.MVIT.AUDIO_BRANCH_ON = False

# Kernel size for patchtification.
_C.MVIT.PATCH_KERNEL = [3, 7, 7]

# Stride size for patchtification.
_C.MVIT.PATCH_STRIDE = [2, 4, 4]

# Padding size for patchtification.
_C.MVIT.PATCH_PADDING = [2, 4, 4]

# If True, use 2d patch, otherwise use 3d patch.
_C.MVIT.PATCH_2D = False

# Base embedding dimension for the transformer.
_C.MVIT.EMBED_DIM = 96

# Base num of heads for the transformer.
_C.MVIT.NUM_HEADS = 1

# Dimension reduction ratio for the MLP layers.
_C.MVIT.MLP_RATIO = 4.0

# If use, use bias term in attention fc layers.
_C.MVIT.QKV_BIAS = True

# Drop path rate for the tranfomer.
_C.MVIT.DROPPATH_RATE = 0.1

# Depth of the transformer.
_C.MVIT.DEPTH = 16

# Normalization layer for the transformer. Only layernorm is supported now.
_C.MVIT.NORM = "layernorm"

# Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
# the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
_C.MVIT.DIM_MUL = []

# Head number multiplication at layer i. If 2.0 is used, then the next block will
# increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
_C.MVIT.HEAD_MUL = []

# Stride size for the Pool KV at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_KV_STRIDE = None

# Initial stride size for KV at layer 1. The stride size will be further reduced with
# the raio of MVIT.DIM_MUL. If will overwrite MVIT.POOL_KV_STRIDE if not None.
_C.MVIT.POOL_KV_STRIDE_ADAPTIVE = None

# Stride size for the Pool Q at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_Q_STRIDE = []

# If not None, overwrite the KV_KERNEL and Q_KERNEL size with POOL_KVQ_CONV_SIZ.
# Otherwise the kernel_size is [s + 1 if s > 1 else s for s in stride_size].
_C.MVIT.POOL_KVQ_KERNEL = None

# If True, perform no decay on positional embedding and cls embedding.
_C.MVIT.ZERO_DECAY_POS_CLS = True

# If True, use norm after stem.
_C.MVIT.NORM_STEM = False

# If True, perform separate positional embedding.
_C.MVIT.SEP_POS_EMBED = False

# Dropout rate for the MViT backbone.
_C.MVIT.DROPOUT_RATE = 0.0


# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# The separator used between path and label.
# _C.DATA.PATH_LABEL_SEPARATOR = " "  # modified by Bolin
_C.DATA.PATH_LABEL_SEPARATOR = ","

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.DATA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.DATA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# If a imdb have been dumpped to a local file with the following format:
# `{"im_path": im_path, "class": cont_id}`
# then we can skip the construction of imdb and load it from the local file.
_C.DATA.PATH_TO_PRELOAD_IMDB = ""

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The relative scale range of Inception-style area based random resizing augmentation.
# If this is provided, DATA.TRAIN_JITTER_SCALES above is ignored.
_C.DATA.TRAIN_JITTER_SCALES_RELATIVE = []

# The relative aspect ratio range of Inception-style area based random resizing
# augmentation.
_C.DATA.TRAIN_JITTER_ASPECT_RELATIVE = []

# If True, perform stride length uniform temporal sampling.
_C.DATA.USE_OFFSET_SAMPLING = False

# Whether to apply motion shift for augmentation.
_C.DATA.TRAIN_JITTER_MOTION_SHIFT = False

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, calculate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

_C.DATA.MODE = 'offline'

_C.DATA.VIDEO_ROOT_DIR = ""
_C.DATA.FRAMES_DIR = ""
_C.DATA.FRAME_EXT = "jpg"
_C.DATA.MISSING_FRAME_OFFSETS = [2, 4, 6, 8]

_C.DATA.PATH_TO_DATA_DIR = ""
_C.DATA.PATH_TO_VAL_DIR = ""
_C.DATA.PATH_TO_TEST_DIR = ""
_C.DATA.SPLIT_ASSIGNMENTS_CSV = ""
_C.DATA.GAZE_DATA_DIR = ""
_C.DATA.GAZE_CACHE_ENABLE = True
_C.DATA.GAZE_CACHE_DIR = ''  # default '' -> use <PATH_PREFIX>/gaze_cache
_C.DATA.CACHE_DIR = ''
_C.DATA.HEATMAP_SIGMA = -1.0
_C.DATA.GAUSSIAN_KERNEL = 19
_C.DATA.WINDOW_STRIDE = 8
_C.DATA.EVAL_CONTEXT_FRAMES = 0
_C.DATA.FILL_NAN_CENTER = True
_C.DATA.INTERPOLATE_NAN = True
_C.DATA.VISUALIZE = False
_C.DATA.VIS_DIR = 'explore/gaze_previews'
_C.DATA.SKIP_FRAME_EXISTS_CHECK = True

# Cache controls
_C.DATA.SKIP_INDEX_CACHE_LOAD = False
_C.DATA.SKIP_TAKE_CACHE_LOAD = False
_C.DATA.SKIP_MISSING_FRAMES_DIR = True

_C.DATA.SUBSAMPLE_TRAIN_FRACTION = 1.0
_C.DATA.SUBSAMPLE_VAL_FRACTION = 1.0
_C.DATA.SUBSAMPLE_TEST_FRACTION = 1.0
_C.DATA.SUBSAMPLE_TEST_IID_FRACTION = 1.0
_C.DATA.SUBSAMPLE_TEST_OOD_TASK_FRACTION = 1.0
_C.DATA.SUBSAMPLE_TEST_OOD_SITE_FRACTION = 1.0
_C.DATA.SUBSAMPLE_TEST_OOD_PARTICIPANT_FRACTION = 1.0
_C.DATA.SUBSAMPLE_BY_PARENT_TASK = False
_C.DATA.SUBSET_STRATEGY = "random"  # Strategy: "random" or "stratified"
_C.DATA.SUBSET_MAX_SAMPLES = -1  # Max samples to keep (-1 for no limit)
_C.DATA.SUBSAMPLE_WRITE_CSV = False
_C.DATA.SUBSAMPLE_SAVE_CSV_PATH = ""

# EgoExo4D clip-window expansion controls. Defaults preserve the dataset loader
# behavior: train/val use one sampled window per clip, test splits use dense
# windows unless explicitly disabled.
_C.DATA.EXPAND_TRAIN_WINDOWS = False
_C.DATA.EXPAND_VAL_WINDOWS = False
_C.DATA.EXPAND_TEST_WINDOWS = True
_C.DATA.EXPAND_TEST_IID_WINDOWS = True
_C.DATA.EXPAND_TEST_OOD_TASK_WINDOWS = True
_C.DATA.EXPAND_TEST_OOD_SITE_WINDOWS = True
_C.DATA.EXPAND_TEST_OOD_PARTICIPANT_WINDOWS = True

_C.DATA.USE_VALID_FRAMES_CSV = False  # Use CSV with precomputed valid frame indices
_C.DATA.MPS_ENABLED = False
_C.DATA.SILENCE_VRS_LOGS = True
_C.DATA.FRAME_DIR = ""

# Hand mask directory for hand-conditional testing
_C.DATA.HAND_MASK_DIR = ''  # default '' -> needs to be set for hand-conditional testing

# ---------------------------------------------------------------------------- #
# PERF options
# ---------------------------------------------------------------------------- #
_C.PERF = CfgNode()
_C.PERF.LOG_PERIOD = 100
_C.PERF.METRICS_PERIOD = 100
_C.PERF.METRICS_PERIOD_FULL = 400
_C.PERF.METRICS_BATCH_FRACTION = 0.25
_C.PERF.VAL_MAX_STEPS = 100
_C.PERF.VAL_SUBSAMPLE = 2
_C.PERF.ALLOW_TF32 = True

# ---------------------------------------------------------------------------- #
# POSE options
# ---------------------------------------------------------------------------- #
_C.POSE = CfgNode()
_C.POSE.ENABLE = False
_C.POSE.BLOCK_IDXS = []
_C.POSE.DROPOUT = 0.0
_C.POSE.STEP1_EPOCHS = 10
_C.POSE.STEP1_LR = 0.0005
_C.POSE.STEP2_LAST_LR = 0.0002
_C.POSE.STEP2_POSE_LR = 0.0004
_C.POSE.LLRD_DECAY = 0.7
_C.POSE.UNFREEZE_BLOCK_START = 14
_C.POSE.FILM_L2 = 1e-5
_C.POSE.SAFE_CHECK = True

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = None
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "."

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# ---------------------------------------------------------------------------- #
# Benchmark options
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CfgNode()

# Number of epochs for data loading benchmark.
_C.BENCHMARK.NUM_EPOCHS = 5

# Log period in iters for data loading benchmark.
_C.BENCHMARK.LOG_PERIOD = 100

# If True, shuffle dataloader for epoch during benchmark.
_C.BENCHMARK.SHUFFLE = True


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode()

# Whether enable video detection.
_C.DETECTION.ENABLE = False

# Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
_C.DETECTION.ALIGNED = True

# Spatial scale factor.
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

# RoI tranformation resolution.
_C.DETECTION.ROI_XFORM_RESOLUTION = 7


# -----------------------------------------------------------------------------
# AVA Dataset options
# -----------------------------------------------------------------------------
_C.AVA = CfgNode()

# Directory path of frames.
_C.AVA.FRAME_DIR = "/mnt/fair-flash3-east/ava_trainval_frames.img/"

# Directory path for files of frame lists.
_C.AVA.FRAME_LIST_DIR = (
    "/mnt/vol/gfsai-flash3-east/ai-group/users/haoqifan/ava/frame_list/"
)

# Directory path for annotation files.
_C.AVA.ANNOTATION_DIR = (
    "/mnt/vol/gfsai-flash3-east/ai-group/users/haoqifan/ava/frame_list/"
)

# Filenames of training samples list files.
_C.AVA.TRAIN_LISTS = ["train.csv"]

# Filenames of test samples list files.
_C.AVA.TEST_LISTS = ["val.csv"]

# Filenames of box list files for training. Note that we assume files which
# contains predicted boxes will have a suffix "predicted_boxes" in the
# filename.
_C.AVA.TRAIN_GT_BOX_LISTS = ["ava_train_v2.2.csv"]
_C.AVA.TRAIN_PREDICT_BOX_LISTS = []

# Filenames of box list files for test.
_C.AVA.TEST_PREDICT_BOX_LISTS = ["ava_val_predicted_boxes.csv"]

# This option controls the score threshold for the predicted boxes to use.
_C.AVA.DETECTION_SCORE_THRESH = 0.9

# If use BGR as the format of input frames.
_C.AVA.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.AVA.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.AVA.TRAIN_PCA_JITTER_ONLY = True

# Whether to do horizontal flipping during test.
_C.AVA.TEST_FORCE_FLIP = False

# Whether to use full test set for validation split.
_C.AVA.FULL_TEST_ON_VAL = False

# The name of the file to the ava label map.
_C.AVA.LABEL_MAP_FILE = "ava_action_list_v2.2_for_activitynet_2019.pbtxt"

# The name of the file to the ava exclusion.
_C.AVA.EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv"

# The name of the file to the ava groundtruth.
_C.AVA.GROUNDTRUTH_FILE = "ava_val_v2.2.csv"

# Backend to process image, includes `pytorch` and `cv2`.
_C.AVA.IMG_PROC_BACKEND = "cv2"

# ---------------------------------------------------------------------------- #
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CfgNode()

# Multigrid training allows us to train for more epochs with fewer iterations.
# This hyperparameter specifies how many times more epochs to train.
# The default setting in paper trains for 1.5x more epochs than baseline.
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# Enable short cycles.
_C.MULTIGRID.SHORT_CYCLE = False
# Short cycle additional spatial dimensions relative to the default crop size.
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]

_C.MULTIGRID.LONG_CYCLE = False
# (Temporal, Spatial) dimensions relative to the default shape.
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    (0.25, 0.5 ** 0.5),
    (0.5, 0.5 ** 0.5),
    (0.5, 1),
    (1, 1),
]

# While a standard BN computes stats across all examples in a GPU,
# for multigrid training we fix the number of clips to compute BN stats on.
# See https://arxiv.org/abs/1912.00998 for details.
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid training epochs are not proportional to actual training time or
# computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
# evaluation. We use a multigrid-specific rule to determine when to evaluate:
# This hyperparameter defines how many times to evaluate a model per long
# cycle shape.
_C.MULTIGRID.EVAL_FREQ = 3

# No need to specify; Set automatically and used as global variables.
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0

# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = False
# Provide path to prediction results for visualization.
# This is a pickle file of [prediction_tensor, label_tensor]
_C.TENSORBOARD.PREDICTIONS_PATH = ""
# Path to directory for tensorboard logs.
# Default to to cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}.
_C.TENSORBOARD.LOG_DIR = ""
# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
# This file must be provided to enable plotting confusion matrix
# by a subset or parent categories.
_C.TENSORBOARD.CLASS_NAMES_PATH = ""

# Path to a json file for categories -> classes mapping
# in the format {"parent_class": ["child_class1", "child_class2",...], ...}.
_C.TENSORBOARD.CATEGORIES_PATH = ""

# Config for confusion matrices visualization.
_C.TENSORBOARD.CONFUSION_MATRIX = CfgNode()
# Visualize confusion matrix.
_C.TENSORBOARD.CONFUSION_MATRIX.ENABLE = False
# Figure size of the confusion matrices plotted.
_C.TENSORBOARD.CONFUSION_MATRIX.FIGSIZE = [8, 8]
# Path to a subset of categories to visualize.
# File contains class names separated by newline characters.
_C.TENSORBOARD.CONFUSION_MATRIX.SUBSET_PATH = ""

# Config for histogram visualization.
_C.TENSORBOARD.HISTOGRAM = CfgNode()
# Visualize histograms.
_C.TENSORBOARD.HISTOGRAM.ENABLE = False
# Path to a subset of classes to plot histograms.
# Class names must be separated by newline characters.
_C.TENSORBOARD.HISTOGRAM.SUBSET_PATH = ""
# Visualize top-k most predicted classes on histograms for each
# chosen true label.
_C.TENSORBOARD.HISTOGRAM.TOPK = 10
# Figure size of the histograms plotted.
_C.TENSORBOARD.HISTOGRAM.FIGSIZE = [8, 8]

# Config for layers' weights and activations visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS = CfgNode()

# If False, skip model visualization.
_C.TENSORBOARD.MODEL_VIS.ENABLE = False

# If False, skip visualizing model weights.
_C.TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS = False

# If False, skip visualizing model activations.
_C.TENSORBOARD.MODEL_VIS.ACTIVATIONS = False

# If False, skip visualizing input videos.
_C.TENSORBOARD.MODEL_VIS.INPUT_VIDEO = False


# List of strings containing data about layer names and their indexing to
# visualize weights and activations for. The indexing is meant for
# choosing a subset of activations outputed by a layer for visualization.
# If indexing is not specified, visualize all activations outputed by the layer.
# For each string, layer name and indexing is separated by whitespaces.
# e.g.: [layer1 1,2;1,2, layer2, layer3 150,151;3,4]; this means for each array `arr`
# along the batch dimension in `layer1`, we take arr[[1, 2], [1, 2]]
_C.TENSORBOARD.MODEL_VIS.LAYER_LIST = []
# Top-k predictions to plot on videos
_C.TENSORBOARD.MODEL_VIS.TOPK_PREDS = 1
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.COLORMAP = "Pastel2"
# Config for visualization video inputs with Grad-CAM.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM = CfgNode()
# Whether to run visualization using Grad-CAM technique.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE = True
# CNN layers to use for Grad-CAM. The number of layers must be equal to
# number of pathway(s).
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST = []
# If True, visualize Grad-CAM using true labels for each instances.
# If False, use the highest predicted class.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.USE_TRUE_LABEL = False
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.COLORMAP = "viridis"

# -----------------------------------------------------------------------------
# Weights & Biases (wandb) Logging Options
# -----------------------------------------------------------------------------
_C.WANDB = CfgNode()
_C.WANDB.ENABLE = False
_C.WANDB.PROJECT = "ARGaze"
_C.WANDB.ENTITY = ""
_C.WANDB.RUN_NAME = "ARGaze"
_C.WANDB.MODE = "disabled"
_C.WANDB.LOG_SAMPLES_EVERY = 200
# Optional separate cadences for train/val sample logging; if 0/None, fall back to LOG_SAMPLES_EVERY.
_C.WANDB.LOG_SAMPLES_EVERY_TRAIN = 0
_C.WANDB.LOG_SAMPLES_EVERY_VAL = 0
_C.WANDB.NUM_SAMPLE_FRAMES = 1
_C.WANDB.LOG_VAL_SAMPLES = True
_C.WANDB.LOG_IMAGES = False
_C.WANDB.IMG_LOG_PERIOD = 200
_C.WANDB.IMG_LOG_SAMPLES = 2

# Template-scale defaults (only used by template-aware models)
_C.MODEL.TEMPLATE_SCALES = [0.25]

# Number of historical frames to use as templates (default=1, only use t-1 frame)
# When > 1: Use multiple past frames as templates, providing temporal context
# Example: TEMPLATE_HISTORY_LENGTH=3 uses frames [t-3, t-2, t-1] as templates
_C.MODEL.TEMPLATE_HISTORY_LENGTH = 1

# Whether to use gaze position as crop center for templates (default=True)
# When True: Crop each template frame around its corresponding gaze position (tracking-style)
# When False: Use full frame as template (no gaze-centered cropping)
# Note: This is different from USE_FULL_FRAME_TEMPLATE (deprecated, kept for backward compatibility)
_C.MODEL.TEMPLATE_USE_GAZE_CENTER = True

# Use full previous frame as template (instead of gaze-centered crop)
# DEPRECATED: Use TEMPLATE_USE_GAZE_CENTER=False instead
# When True: template = full previous frame resized to input_size
# When False (default): template = gaze-centered crop at TEMPLATE_SCALES
_C.MODEL.USE_FULL_FRAME_TEMPLATE = False

# Use token type embeddings to distinguish history/query/template tokens
# When True (default): add token_type_embed to distinguish token types
# When False (ablation): all tokens treated equally (no type distinction)
_C.MODEL.USE_TOKEN_TYPE_EMBED = True

# Use temporal embeddings to indicate time order of history tokens
# When True (default): add temporal_pos_embed to show temporal order (t-N, ..., t-1)
# When False (ablation): history tokens don't know their relative time positions
_C.MODEL.USE_TEMPORAL_EMBED = True

# Config for visualization for wrong prediction visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.WRONG_PRED_VIS = CfgNode()
_C.TENSORBOARD.WRONG_PRED_VIS.ENABLE = False
# Folder tag to origanize model eval videos under.
_C.TENSORBOARD.WRONG_PRED_VIS.TAG = "Incorrectly classified videos."
# Subset of labels to visualize. Only wrong predictions with true labels
# within this subset is visualized.
_C.TENSORBOARD.WRONG_PRED_VIS.SUBSET_PATH = ""


# ---------------------------------------------------------------------------- #
# Demo options
# ---------------------------------------------------------------------------- #
_C.DEMO = CfgNode()

# Run model in DEMO mode.
_C.DEMO.ENABLE = False

# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
_C.DEMO.LABEL_FILE_PATH = ""

# Specify a camera device as input. This will be prioritized
# over input video if set.
# If -1, use input video instead.
_C.DEMO.WEBCAM = -1

# Path to input video for demo.
_C.DEMO.INPUT_VIDEO = ""
# Custom width for reading input video data.
_C.DEMO.DISPLAY_WIDTH = 0
# Custom height for reading input video data.
_C.DEMO.DISPLAY_HEIGHT = 0
# Path to Detectron2 object detection model configuration,
# only used for detection tasks.
_C.DEMO.DETECTRON2_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# Path to Detectron2 object detection model pre-trained weights.
_C.DEMO.DETECTRON2_WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
# Threshold for choosing predicted bounding boxes by Detectron2.
_C.DEMO.DETECTRON2_THRESH = 0.9
# Number of overlapping frames between 2 consecutive clips.
# Increase this number for more frequent action predictions.
# The number of overlapping frames cannot be larger than
# half of the sequence length `cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE`
_C.DEMO.BUFFER_SIZE = 0
# If specified, the visualized outputs will be written this a video file of
# this path. Otherwise, the visualized outputs will be displayed in a window.
_C.DEMO.OUTPUT_FILE = ""
# Frames per second rate for writing to output video file.
# If not set (-1), use fps rate from input file.
_C.DEMO.OUTPUT_FPS = -1
# Input format from demo video reader ("RGB" or "BGR").
_C.DEMO.INPUT_FORMAT = "BGR"
# Draw visualization frames in [keyframe_idx - CLIP_VIS_SIZE, keyframe_idx + CLIP_VIS_SIZE] inclusively.
_C.DEMO.CLIP_VIS_SIZE = 10
# Number of processes to run video visualizer.
_C.DEMO.NUM_VIS_INSTANCES = 2

# Path to pre-computed predicted boxes
_C.DEMO.PREDS_BOXES = ""
# Whether to run in with multi-threaded video reader.
_C.DEMO.THREAD_ENABLE = False
# Take one clip for every `DEMO.NUM_CLIPS_SKIP` + 1 for prediction and visualization.
# This is used for fast demo speed by reducing the prediction/visualiztion frequency.
# If -1, take the most recent read clip for visualization. This mode is only supported
# if `DEMO.THREAD_ENABLE` is set to True.
_C.DEMO.NUM_CLIPS_SKIP = 0
# Path to ground-truth boxes and labels (optional)
_C.DEMO.GT_BOXES = ""
# The starting second of the video w.r.t bounding boxes file.
_C.DEMO.STARTING_SECOND = 900
# Frames per second of the input video/folder of images.
_C.DEMO.FPS = 30
# Visualize with top-k predictions or predictions above certain threshold(s).
# Option: {"thres", "top-k"}
_C.DEMO.VIS_MODE = "thres"
# Threshold for common class names.
_C.DEMO.COMMON_CLASS_THRES = 0.7
# Theshold for uncommon class names. This will not be
# used if `_C.DEMO.COMMON_CLASS_NAMES` is empty.
_C.DEMO.UNCOMMON_CLASS_THRES = 0.3
# This is chosen based on distribution of examples in
# each classes in AVA dataset.
_C.DEMO.COMMON_CLASS_NAMES = [
    "watch (a person)",
    "talk to (e.g., self, a person, a group)",
    "listen to (a person)",
    "touch (an object)",
    "carry/hold (an object)",
    "walk",
    "sit",
    "lie/sleep",
    "bend/bow (at the waist)",
]
# Slow-motion rate for the visualization. The visualized portions of the
# video will be played `_C.DEMO.SLOWMO` times slower than usual speed.
_C.DEMO.SLOWMO = 1

# Whether to use causal model for online inference.
_C.MODEL.CAUSAL = False
_C.MODEL.DECODE_LAST = False

# ---------------------------------------------------------------------------- #
# DINOv3 gaze model options (to support Egtea configs)
# ---------------------------------------------------------------------------- #
# Pretrained DINOv3 model identifier for HuggingFace transformers.
_C.MODEL.DINOV3_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

# Whether to freeze the DINOv3 encoder parameters.
_C.MODEL.FREEZE_ENCODER = True

# If > 0, unfreeze the last K encoder layers (takes effect when FREEZE_ENCODER=True).
_C.MODEL.UNFREEZE_LAST_K_LAYERS = 0

# LR multiplier for unfrozen encoder layers relative to BASE_LR.
# E.g. 0.1 means encoder layers train at 0.1 * BASE_LR to avoid destroying
# pretrained features. Only used when UNFREEZE_LAST_K_LAYERS > 0.
_C.MODEL.ENCODER_LR_SCALE = 0.1

# Whether to apply ImageNet normalization inside DINOv3 feature adapters.
_C.MODEL.DINOV3_USE_PREPROCESS = True

# Optional resize for DINOv3 inputs. 0 disables resize and uses DATA crop size.
_C.MODEL.DINOV3_INPUT_SIZE = 0

# Decoder patch grid size for GLC encoder + AR heatmap decoder baseline.
_C.MODEL.DECODER_PATCH_SIZE = 14

# Template options for GLC encoder + AR heatmap decoder baseline.
_C.MODEL.USE_TEMPLATE_TOKENS = False
_C.MODEL.TEMPLATE_SCALES = [0.25]
_C.MODEL.TEMPLATE_USE_GAZE_CENTER = True
_C.MODEL.TEMPLATE_INPUT_SIZE = 0

# Output heatmap spatial resolution (heatmap_size x heatmap_size).
_C.MODEL.HEATMAP_SIZE = 64

# Activation over the predicted heatmap: "sigmoid" or "softmax".
_C.MODEL.HEATMAP_ACTIVATION = "sigmoid"

# Predict heatmap per frame (True) or only for last frame unless DECODE_LAST is set.
_C.MODEL.PREDICT_PER_FRAME = True

# Temporal aggregation mode for the DINOv3 gaze model: "none", "offline", or "causal".
_C.MODEL.TEMPORAL_MODE = "none"

# Whether to use multi-scale features from DINOv3 encoder.
_C.MODEL.USE_MULTISCALE_FEATURES = True

# Which layers to use for multi-scale features (negative indices from last layer).
_C.MODEL.MULTISCALE_LAYERS = [-3, -2, -1]

# ---------------------------------------------------------------------------- #
# GRU Temporal Aggregation options (for DINOv3_FrameWise_GRU model)
# ---------------------------------------------------------------------------- #
# Number of GRU layers for temporal aggregation (default: 1).
_C.MODEL.GRU_NUM_LAYERS = 1

# Whether to use bidirectional GRU (default: True for better temporal modeling).
_C.MODEL.GRU_BIDIRECTIONAL = True

# ---------------------------------------------------------------------------- #
# Learnable Query Token options (for DINOv3_QueryDecoder model)
# ---------------------------------------------------------------------------- #
# Enable query-based decoding with learnable query tokens (default: True).
_C.MODEL.USE_QUERY = True

# Number of learnable query tokens (default: 1).
_C.MODEL.NUM_QUERIES = 1

# Dimension of query tokens (default: same as hidden_dim, e.g., 384 for ViT-S/16).
_C.MODEL.QUERY_DIM = 384

# Query fusion mode: "dot" (query-only), "conv" (conv-only), or "dot+conv" (fusion).
_C.MODEL.QUERY_FUSION = "dot"

# Whether to use cross-attention to refine query tokens (default: False).
_C.MODEL.USE_QUERY_ATTENTION = False

# Temperature parameter for query similarity scaling (default: 1.0, fixed).
_C.MODEL.QUERY_TEMPERATURE = 1.0

# ---------------------------------------------------------------------------- #
# Sub-patch Expansion options (PixelShuffle for high-res query decoding)
# ---------------------------------------------------------------------------- #
# Enable sub-patch expansion using PixelShuffle (default: True).
_C.MODEL.USE_SUBPATCH = True

# Upsampling factor for PixelShuffle (r). For 14x14 patches, r=4 gives 56x56.
_C.MODEL.SUBPATCH_FACTOR = 4

# Embedding dimension after PixelShuffle projection (default: same as hidden_dim).
_C.MODEL.SUBPATCH_EMBED_DIM = 384

# ---------------------------------------------------------------------------- #
# Multi-scale Supervision & Regularization options
# ---------------------------------------------------------------------------- #
# Enable dual-scale supervision (high-res + low-res auxiliary loss).
_C.MODEL.USE_DUAL_SUPERVISION = True

# Weight for low-resolution auxiliary loss (lambda_low).
_C.MODEL.LAMBDA_LOW = 0.5

# Enable consistency regularization between conv and query branches.
_C.MODEL.USE_CONSISTENCY_REG = True

# Weight for consistency regularization loss (lambda_consistency).
_C.MODEL.LAMBDA_CONSISTENCY = 0.4

# Number of groups for GroupNorm in sub-patch expansion decorrelation.
_C.MODEL.SUBPATCH_NUM_GROUPS = 8

# Whether to use depthwise separable convolution in decorrelation layers.
_C.MODEL.SUBPATCH_USE_DEPTHWISE = False

# ---------------------------------------------------------------------------- #
# Fusion Warm-up options (for dot+conv mode)
# ---------------------------------------------------------------------------- #
# Number of epochs for fusion alpha warm-up (linear increase from 0 to target).
_C.MODEL.ALPHA_WARMUP_EPOCHS = 5

# Target fusion weight after warm-up (alpha_target).
_C.MODEL.ALPHA_TARGET = 0.2

# ---------------------------------------------------------------------------- #
# 2D Positional Encoding options
# ---------------------------------------------------------------------------- #
# Enable 2D sinusoidal positional encoding (default: True).
_C.MODEL.USE_POS_ENCODING = True

# ---------------------------------------------------------------------------- #
# Divided Space-Time Attention options (for DINOv3_T_S_Attention model)
# ---------------------------------------------------------------------------- #
# Number of Divided Space-Time (T+S) attention blocks to stack (default: 3).
_C.MODEL.NUM_TS_LAYERS = 3

# Temporal window size for windowed attention (default: None for full temporal range).
# Set to an integer (e.g., 4 or 8) to limit temporal attention to a sliding window.
# DEPRECATED: Use TS_WIN_T_PAST and TS_WIN_T_FUTURE for more control.
_C.MODEL.TS_WIN_T = None

# Number of past frames to attend to (default: None for unlimited).
# Set to an integer (e.g., 3, 5) to limit past temporal attention.
# Set to -1 for unlimited past frames (equivalent to None).
_C.MODEL.TS_WIN_T_PAST = None

# Number of future frames to attend to (default: None for unlimited).
# Set to an integer (e.g., 2, 3) to limit future temporal attention.
# Set to 0 for causal mode (no future frames).
# Set to -1 for unlimited future frames (equivalent to None).
_C.MODEL.TS_WIN_T_FUTURE = None

# Whether to use causal temporal masking for online inference (default: False).
# If True, frame t can only attend to frames 0 to t (past and current).
# This automatically sets TS_WIN_T_FUTURE to 0.
_C.MODEL.TS_CAUSAL = False

# Maximum temporal sequence length for learnable 1D temporal positional encoding (default: 32).
_C.MODEL.MAX_TEMPORAL_LEN = 32

# Query Focuser options (for DINOv3_T_S_Attention model)
# Enable query focuser: use learnable query tokens to focus on important regions (default: True).
_C.MODEL.USE_QUERY_FOCUSER = True

# ---------------------------------------------------------------------------- #
# Deformable Temporal Attention (DTA) options
# ---------------------------------------------------------------------------- #
# Enable Deformable Temporal Attention instead of standard temporal attention (default: False).
# DTA learns spatial offsets to track moving objects across frames.
_C.MODEL.USE_DEFORMABLE_TEMPORAL = False

# Number of sampling offset points per neighboring frame for DTA (default: 4).
# Higher values allow more flexible motion modeling but increase computation.
_C.MODEL.DTA_N_OFFSETS = 4

# Temporal radius for DTA - samples from t±1...±R frames (default: 2).
# Larger radius captures longer-range temporal dependencies.
_C.MODEL.DTA_R_FRAMES = 2

# Initial offset scale for DTA in normalized [-1,1] coordinates (default: 0.1).
# Controls the magnitude of predicted spatial offsets. Smaller values are more conservative.
_C.MODEL.DTA_OFFSET_SCALE = 0.1

_C.MODEL.USE_CLS_INIT = False
_C.MODEL.NUM_ATTENTION_HEADS = 12
_C.MODEL.ATTENTION_DROPOUT = 0.1

# ---------------------------------------------------------------------------- #
# Cross-Attention options (for DINOv3_T_S_CLS_CrossAttn model)
# ---------------------------------------------------------------------------- #
# Number of attention heads for cross-attention (default: 8).
_C.MODEL.FOCUSER_NHEAD = 8

# ---------------------------------------------------------------------------- #
# Joint Space-Time Attention options (for DINOv3_ST_Attention model)
# ---------------------------------------------------------------------------- #
# Number of Joint Space-Time (ST) attention blocks to stack (default: 3).
_C.MODEL.NUM_ST_LAYERS = 3

# Whether to use causal masking for joint spatiotemporal attention (default: True).
# If True, patches in frame t can only attend to patches in frames 0 to t (causal).
# If False, patches can attend to all frames (non-causal/offline mode).
_C.MODEL.ST_CAUSAL = True

# ---------------------------------------------------------------------------- #
# Autoregressive Gaze Estimation options (for DINOv3_ARgaze model)
# ---------------------------------------------------------------------------- #
# Number of learnable [GAZE] query tokens for autoregressive prediction (default: 1).
_C.MODEL.NUM_GAZE_QUERIES = 1

# Whether to use confidence head for prediction quality estimation (default: False).
_C.MODEL.USE_CONFIDENCE_HEAD = False

# ---------------------------------------------------------------------------- #
# Scheduled Sampling options (for autoregressive training)
# ---------------------------------------------------------------------------- #
# Initial scheduled sampling probability at the start of training (default: 0.0 = pure teacher forcing).
_C.TRAIN.SS_PROB_START = 0.0

# Final scheduled sampling probability after ramp-up (default: 0.3).
# ss_prob = 0.0: always use ground truth heatmap
# ss_prob = 0.5: 50% chance to use prediction, 50% chance to use GT
# ss_prob = 1.0: always use model's own prediction
_C.TRAIN.SS_PROB_END = 0.3

# Number of epochs to linearly ramp up scheduled sampling probability (default: 10).
# ss_prob increases linearly from SS_PROB_START to SS_PROB_END over this period.
_C.TRAIN.SS_PROB_RAMP_EPOCHS = 10

# Auxiliary Heatmap Loss weight for ARPointGaze (default: 0.0 = disabled).
# This loss reshapes token logits to 2D heatmap and compares with Gaussian GT heatmap.
# Provides spatial-aware label smoothing. Recommended: 0.5
_C.TRAIN.AUX_HEATMAP_LOSS_WEIGHT = 0.0

# Gaussian heatmap sigma for auxiliary loss (default: 2.0).
# Controls the spread of the Gaussian around GT gaze point.
# Larger sigma = more smoothing, smaller sigma = more concentrated.
_C.TRAIN.AUX_HEATMAP_SIGMA = 2.0

# Use expectation (weighted average) instead of argmax for coordinate prediction (default: False).
# When True: coord = Σ P(token_i) × coord_i (smoother predictions)
# When False: coord = coord[argmax(P)] (standard discrete prediction)
# Applies to both training (with SS) and testing.
_C.MODEL.USE_EXPECTATION = False

# ---------------------------------------------------------------------------- #
# Head Prompting options (for DINOv3_AR_HP model)
# ---------------------------------------------------------------------------- #
# Enable head prompting: use previous frame's gaze heatmap as prompt for current frame (default: False).
_C.MODEL.USE_HEAD_PROMPTING = False

# Where to inject the head prompt: "before" (before ST blocks), "after" (after ST blocks), or "both" (default: "after").
# - "before": Inject previous heatmap prompt before Joint Space-Time attention blocks
# - "after": Inject previous heatmap prompt after Joint Space-Time attention blocks
# - "both": Inject at both locations (strongest temporal guidance)
_C.MODEL.HEAD_PROMPT_LOCATION = "after"

# Use center gaussian for frame 0 initialization (True) or zeros (False) (default: True).
# Center gaussian provides a reasonable prior for the first frame where no previous gaze is available.
_C.MODEL.HEAD_PROMPT_CENTER_INIT = True

# Scheduled sampling probability for head prompting during training (default: 0.0).
# - 0.0: Pure teacher forcing (always use GT previous frame)
# - 0.3: 30% chance to use predicted previous frame, 70% chance to use GT
# - 1.0: Always use predicted previous frame (matches inference, but harder to train)
# Recommendation: Start with 0.0, gradually increase to 0.3-0.5 during training to reduce exposure bias.
_C.MODEL.SCHEDULED_SAMPLING_PROB = 0.0

# Use autoregressive inference during validation (default: True).
# - True: Autoregressive frame-by-frame prediction without GT heatmap (realistic, slower)
# - False: Teacher forcing with GT heatmap during validation (faster, unrealistic, inflates metrics)
# Recommendation: Use True for realistic performance evaluation. Only set to False for quick debugging.
_C.MODEL.AUTOREGRESSIVE_VAL = True

# Use autoregressive inference during testing (default: True).
# - True: Autoregressive frame-by-frame prediction without GT heatmap (realistic, slower)
# - False: Standard forward pass (may use zero prompts if no GT provided)
# Recommendation: Use True for realistic test evaluation. Only set to False for debugging.
_C.MODEL.AUTOREGRESSIVE_TEST = True

# GT as Prompt mode for AR models (EXP 7: Oracle Upper Bound) - No training needed!
# - True: Use ground truth heatmap as prompt during testing (measures theoretical upper bound)
# - False: Use predicted heatmap as prompt (standard autoregressive inference)
# Purpose: Measure impact of prediction error accumulation and head prompting's upper bound
# Usage: Set to True with any trained checkpoint to get oracle performance
_C.MODEL.USE_GT_AS_PROMPT = False
# Use GT heatmap only for template center during testing (ARHeatmapGazeTemplate).
# - True: template center uses GT; history uses predicted heatmaps (matches SS-trained behavior)
# - False: template center uses predicted heatmaps (standard autoregressive inference)
_C.MODEL.USE_GT_AS_TEMPLATE_CENTER = False

# ---------------------------------------------------------------------------- #
# ROI prompt options (for ARHeatmapGazeTemplate)
# ---------------------------------------------------------------------------- #
# Use binary ROI prompt on patch tokens instead of template crops.
_C.MODEL.USE_ROI_PROMPT = False
# Whether to include template tokens in cross-attention memory.
_C.MODEL.USE_TEMPLATE_TOKENS = True
# Reuse original frame patch embeddings for template crops (skip re-encoding crops).
_C.MODEL.USE_ORIGINAL_TEMPLATE_TOKENS = False
# ROI scale as fraction of image size (same convention as template scales).
_C.MODEL.ROI_SCALE = 0.25
# ROI mask grid size before resizing to patch grid.
_C.MODEL.ROI_GRID_SIZE = 16

# Ablation: Use ROI prompt instead of template encoding (mutually exclusive with USE_TEMPLATE_TOKENS).
# When True: Memory = current_tokens (with ROI marking at gaze_{t-1} location)
# When False: Memory = [template_tokens, current_tokens] (original behavior)
# This enables comparing lightweight ROI marking vs expensive template crop encoding.
_C.MODEL.USE_ROI_INSTEAD_OF_TEMPLATE = False

# ---------------------------------------------------------------------------- #
# Transformer-based Autoregressive Gaze Model options (for DINOv3_TransformerAR)
# ---------------------------------------------------------------------------- #
# Number of Transformer Decoder layers (default: 6).
# More layers = more capacity for modeling temporal dependencies but slower training.
# Recommended range: 3-12 layers
_C.MODEL.NUM_DECODER_LAYERS = 6

# Dimension of feed-forward network in Transformer Decoder (default: 1536 = 4x hidden_dim for ViT-S).
# Standard ratio is 4x hidden_dim. Can be adjusted for model capacity.
# Recommended range: 2x to 8x hidden_dim
_C.MODEL.DIM_FEEDFORWARD = 1536

# Maximum sequence length for temporal positional encoding (default: 1000).
# Should be >= maximum video sequence length expected during training/testing.
# Used to pre-compute sinusoidal temporal positional embeddings.
_C.MODEL.MAX_SEQ_LEN = 1000

# ---------------------------------------------------------------------------- #
# Baseline options (for testing without trained models)
# ---------------------------------------------------------------------------- #
# Type of baseline to use for gaze prediction (default: "none" - use actual model).
# Options: "none", "random", "center", "dataset_prior"
# - "none": Use the actual trained model for prediction
# - "random": Generate random gaze points uniformly across the frame
# - "center": Always predict gaze at the center of the frame (0.5, 0.5)
# - "dataset_prior": Use precomputed average gaze location from training split
_C.MODEL.BASELINE_TYPE = "none"

# Path to JSON file containing dataset-specific gaze priors (used when BASELINE_TYPE="dataset_prior").
# Format: {"dataset_name": {"mean_x": 0.5, "mean_y": 0.4, "std_x": 0.2, "std_y": 0.15}}
_C.MODEL.DATASET_PRIOR_PATH = ""

# Standard deviation (in pixels) for Gaussian blob in baseline heatmaps (default: 5.0).
# Controls the spread of the Gaussian centered at the predicted gaze point.
_C.MODEL.BASELINE_GAUSSIAN_STD = 5.0

# ---------------------------------------------------------------------------- #
# ARPointGaze Configuration (Autoregressive Point-based Gaze Estimation)
# ---------------------------------------------------------------------------- #
# Inspired by ARTrack (CVPR 2023) - discrete coordinate token prediction

# Number of bins for coordinate discretization (default: 256).
# Higher values = finer granularity but larger vocabulary.
# Recommended: 128 (coarse), 256 (default), 512 (fine)
_C.MODEL.BINS = 256

# Vocabulary range multiplier (default: 2.0).
# Extends coordinate support beyond [0, 1] to handle boundary cases.
# 1.0: support [0, 1] (standard)
# 2.0: support [-0.5, 1.5] (recommended, handles near-boundary gaze)
# Follows ARTrack's range expansion strategy.
_C.MODEL.COORD_RANGE = 2.0

# Historical gaze length (default: 3).
# Number of previous frames to condition on: N in G_{t-N:t-1}
# 0: No history (single-frame baseline)
# 3: Default (good balance between temporal context and overfitting)
# 5: Longer history (may capture more dynamics but risk overfitting)
_C.MODEL.HISTORY_LENGTH = 3
# Type of history tokens to use in AR models: "heatmap", "image", or "both".
_C.MODEL.HISTORY_TYPE = "heatmap"
# Optional modality-specific history lengths for AR heatmap ablations.
# Negative values fall back to MODEL.HISTORY_LENGTH.
_C.MODEL.IMAGE_HISTORY_LENGTH = -1
_C.MODEL.GAZE_HISTORY_LENGTH = -1

# L2 loss weight for continuous coordinate regularization (default: 0.1).
# Combines token CE loss with continuous L2 loss.
# 0.0: Pure token-level discrete optimization
# 0.1: Default (adds smoothness regularization)
# 0.5: Heavy continuous regularization
_C.MODEL.L2_LOSS_WEIGHT = 0.1

# Gaze crop parameters for Exp8/Exp9
_C.MODEL.GAZE_CROP_SIZE = 3
_C.MODEL.GAZE_CROP_RATIO = 0.2

# Historical heatmap attention-bias controls for Exp10 efficient AR decoder.
_C.MODEL.HEATMAP_BIAS_WEIGHT = 1.0
_C.MODEL.HEATMAP_BIAS_EPS = 1e-6
_C.MODEL.HEATMAP_BIAS_NORMALIZE = "softmax"
# Source used to build the historical heatmap attention prior.
# Supported: "pred", "gt"/"oracle", "legacy"/"ss", "scheduled",
# "center", "uniform", "noisy"/"random".
_C.MODEL.HEATMAP_BIAS_SOURCE_MODE = "pred"
_C.MODEL.HEATMAP_BIAS_CENTER_SIGMA = -1.0
_C.MODEL.HEATMAP_BIAS_SCHEDULE_PHASE_EPOCHS = [10, 20]
_C.MODEL.HEATMAP_BIAS_SCHEDULE_RATIOS = [
    [0.8, 0.2, 0.0],
    [0.5, 0.3, 0.2],
    [0.3, 0.3, 0.4],
]
_C.MODEL.HEATMAP_BIAS_JITTER_SHIFT = 0.05
_C.MODEL.HEATMAP_BIAS_JITTER_SCALE_MIN = 0.9
_C.MODEL.HEATMAP_BIAS_JITTER_SCALE_MAX = 1.15

# Coordinate regression loss from heatmap logits (soft-argmax).
# 0.0 disables the auxiliary loss.
_C.MODEL.COORD_LOSS_WEIGHT = 0.0
_C.MODEL.COORD_LOSS_TAU = 0.5
_C.MODEL.COORD_LOSS_TYPE = "smooth_l1"

# Bound-history fusion controls for the per-frame heatmap/image binding variant.
_C.MODEL.HEATMAP_BINDING_ENABLE = False
_C.MODEL.HEATMAP_BINDING_ALIGN_MODE = "bilinear"
_C.MODEL.HEATMAP_MODULATION_HIDDEN_DIM = 128
_C.MODEL.HEATMAP_MODULATION_TYPE = "scalar"
_C.MODEL.HEATMAP_MODULATION_CONV_CHANNELS = 32
_C.MODEL.HEATMAP_MODULATION_CONV_KERNEL_SIZE = 3
_C.MODEL.HEATMAP_MODULATION_ALPHA_INIT = 0.0
_C.MODEL.HIST_READER_GATE_INIT = 0.0

# Compressed streaming-memory controls for reviewer ablations.
# These keep the ARGaze encoder/decoder/head fixed and replace explicit history
# tokens with a one-token model-internal memory state.
_C.MODEL.EMA_MEMORY_ALPHA = 0.9
_C.MODEL.MEMORY_USE_GAZE = False

# Entropy regularization on predicted heatmaps (encourage peaky distributions).
# Set to 0.0 to disable.
_C.MODEL.ENTROPY_LOSS_WEIGHT = 0.0

# DPT-style skip connections from early encoder layers for spatial precision.
# Uses features from shallow ViT layers, upsampled and added to conv decoder stages.
# Requires USE_MULTISCALE_FEATURES=True (output_hidden_states=True on encoder).
_C.MODEL.USE_DPT_SKIP = False
_C.MODEL.DPT_SKIP_LAYERS = [2, 5]

# ---------------------------------------------------------------------------- #
# Hugging Face Hub Integration options
# ---------------------------------------------------------------------------- #
_C.HF = CfgNode()
_C.HF.ENABLE = False
_C.HF.REPO_ID = ""
_C.HF.BRANCH = "main"
_C.HF.PRIVATE = True
_C.HF.PATH_PREFIX = "checkpoints"
_C.HF.INCLUDE_LATEST = True

# Add custom config with default values.
custom_config.add_custom_config(_C)


def assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.NUM_GPUS == 0 or cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.NUM_GPUS == 0 or cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.WARMUP_START_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.COSINE_END_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
