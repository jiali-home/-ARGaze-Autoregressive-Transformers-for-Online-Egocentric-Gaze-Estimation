#!/usr/bin/env python3
"""
Efficiency comparison on long continuous frame segments.

Based on demo_video_eval.py but supports frame lists instead of video capture,
and logs latency/FPS over time for degradation analysis.
"""

import argparse
import bisect
import csv
import ctypes
import inspect
import json
import os
import random
import site
import struct
import time
from collections import deque
from itertools import chain
from pathlib import Path


def _preload_cublas_pair():
    """Preload libcublas + libcublasLt before importing torch.

    This mirrors tools/run_net.py. On this cluster the dinov3 env can carry
    CUDA 12.8 pip libraries while the driver supports an older CUDA minor; cuDNN
    may then abort with "Cannot load symbol cublasLtCreate" unless the matching
    cuBLAS pair is loaded first.
    """
    if os.environ.get("GLC_SKIP_CUBLAS_PRELOAD", "0") == "1":
        return

    def _driver_cuda_ver():
        try:
            from pynvml import nvmlInit, nvmlSystemGetCudaDriverVersion_v2

            nvmlInit()
            v = nvmlSystemGetCudaDriverVersion_v2()
            return (v // 1000, (v % 1000) // 10)
        except Exception:
            return None

    def _pip_cublas_ver():
        try:
            from importlib.metadata import version as pkg_ver

            v = pkg_ver("nvidia-cublas-cu12")
            p = v.split(".")
            return (int(p[0]), int(p[1]))
        except Exception:
            return None

    def _load_pair(cublas, cublaslt):
        ctypes.CDLL(cublas, mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(cublaslt, mode=ctypes.RTLD_GLOBAL)

    def _find_pip_cublas():
        for p in site.getsitepackages() + [site.getusersitepackages()]:
            d = os.path.join(p, "nvidia", "cublas", "lib")
            c = os.path.join(d, "libcublas.so.12")
            lt = os.path.join(d, "libcublasLt.so.12")
            if os.path.isfile(c) and os.path.isfile(lt):
                return c, lt
        return None, None

    sys_cuda_dirs = [
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
    ]

    def _find_sys_cublas():
        for d in sys_cuda_dirs:
            c = os.path.join(d, "libcublas.so.12")
            lt = os.path.join(d, "libcublasLt.so.12")
            if os.path.isfile(c) and os.path.isfile(lt):
                return c, lt
        return None, None

    pip_c, pip_lt = _find_pip_cublas()
    sys_c, sys_lt = _find_sys_cublas()
    drv = _driver_cuda_ver()
    pip_ver = _pip_cublas_ver()

    force_system = os.environ.get("GLC_FORCE_SYSTEM_CUBLAS", "0") == "1"
    use_system = force_system or (drv and pip_ver and pip_ver > drv)

    source = None
    if use_system and sys_c:
        _load_pair(sys_c, sys_lt)
        source = "system"
    elif pip_c:
        _load_pair(pip_c, pip_lt)
        source = "pip"
    elif sys_c:
        _load_pair(sys_c, sys_lt)
        source = "system"

    if source and os.environ.get("GLC_DEBUG_CUBLAS_PRELOAD", "0") == "1":
        print(
            f"[efficiency_compare] preloaded {source} cuBLAS: "
            f"{sys_c if source == 'system' else pip_c}",
            flush=True,
        )


_preload_cublas_pair()

import cv2
import numpy as np
import torch
import torch.nn.functional as F

if os.environ.get("GLC_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    print("[efficiency_compare] GLC_DISABLE_CUDNN=1 -> cuDNN disabled for stability.", flush=True)


def debug_log(message):
    if os.environ.get("GLC_EFFICIENCY_DEBUG", "0") == "1":
        print(f"[efficiency_compare] {message}", flush=True)


debug_log("imports complete")

# Add parent directory to path for slowfast imports and demo utilities.
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

debug_log("importing config helpers")
from slowfast.config.defaults import get_cfg
debug_log("config helpers imported")


def load_config(config_path):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    return cfg


def load_model(cfg, checkpoint_path, device):
    debug_log("importing model registry")
    import slowfast.models  # noqa: F401
    from slowfast.models.build import build_model
    debug_log("model registry imported")

    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}", flush=True)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    return model


def preprocess_frame(frame, target_size=224):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(frame_resized).float() / 255.0
    return tensor.permute(2, 0, 1)


def heatmap_to_coords(heatmap):
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)

    bsz, _, height, width = heatmap.shape
    hm_flat = heatmap.view(bsz, -1)
    prob = F.softmax(hm_flat, dim=-1).view(bsz, height, width)

    y_coords = torch.linspace(0, 1, steps=height, device=heatmap.device).view(1, height, 1)
    x_coords = torch.linspace(0, 1, steps=width, device=heatmap.device).view(1, 1, width)

    y_expect = (prob * y_coords).sum(dim=(1, 2))
    x_expect = (prob * x_coords).sum(dim=(1, 2))
    return x_expect.item(), y_expect.item()


def frame_softmax(logits, temperature=2.0):
    """Match the GLC KLDiv test pipeline's per-frame spatial softmax."""
    batch_size, time = logits.shape[0], logits.shape[2]
    height, width = logits.shape[3], logits.shape[4]
    logits = logits.view(batch_size, -1, time, height * width)
    atten_map = F.softmax(logits / temperature, dim=-1)
    return atten_map.view(batch_size, -1, time, height, width)


def heatmap_rescale_minmax(preds):
    """Per-frame spatial min-max rescale used by GLC test-time decoding."""
    preds_flat = preds.detach().view(preds.size()[:-2] + (preds.size(-1) * preds.size(-2),))
    preds_flat = (preds_flat - preds_flat.min(dim=-1, keepdim=True)[0]) / (
        preds_flat.max(dim=-1, keepdim=True)[0] - preds_flat.min(dim=-1, keepdim=True)[0] + 1e-6
    )
    return preds_flat.view(preds.size())


def heatmap_to_coords_argmax(heatmap):
    """Convert a single heatmap to normalized coordinates using argmax."""
    height, width = heatmap.shape[-2], heatmap.shape[-1]
    flat_idx = int(torch.nan_to_num(heatmap, nan=-1e6).argmax().item())
    row = flat_idx // width
    col = flat_idx % width
    pred_x = col / (width - 1) if width > 1 else 0.5
    pred_y = row / (height - 1) if height > 1 else 0.5
    return pred_x, pred_y


def prepare_heatmaps_for_eval(heatmaps):
    """Apply the standard gaze test-time heatmap postprocess."""
    return heatmap_rescale_minmax(frame_softmax(heatmaps, temperature=2.0))


def pad_streaming_window(window, target_len, warm_start="replicate"):
    """Left-pad a causal GLC window to the training clip length."""
    if window.size(2) >= target_len:
        return window
    pad_len = target_len - window.size(2)
    if warm_start == "replicate":
        pad_frame = window[:, :, :1].expand(-1, -1, pad_len, -1, -1)
    elif warm_start == "zeros":
        pad_frame = torch.zeros_like(window[:, :, :1]).expand(-1, -1, pad_len, -1, -1)
    else:
        raise ValueError(f"Unknown warm_start: {warm_start}")
    return torch.cat([pad_frame, window], dim=2)


def overlay_heatmap(frame, heatmap, alpha=0.4):
    height, width = frame.shape[:2]
    hm_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    hm_min, hm_max = hm_resized.min(), hm_resized.max()
    if hm_max > hm_min:
        hm_norm = ((hm_resized - hm_min) / (hm_max - hm_min) * 255).astype(np.uint8)
    else:
        hm_norm = np.zeros_like(hm_resized, dtype=np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1.0, hm_color, alpha, 0)


def draw_gaze_marker(frame, x, y, color=(0, 255, 0), radius=10, thickness=2):
    height, width = frame.shape[:2]
    px = int(x * width)
    py = int(y * height)
    cv2.circle(frame, (px, py), radius, color, thickness)
    cv2.line(frame, (px - radius - 5, py), (px + radius + 5, py), color, thickness)
    cv2.line(frame, (px, py - radius - 5), (px, py + radius + 5), color, thickness)
    return frame


TYPE_FRAME = 1
TYPE_GAZE = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Efficiency comparison on long frame segments.")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--gt", type=str, default=None, help="Path to GT gaze data file")
    parser.add_argument(
        "--event-file",
        type=str,
        default=None,
        help="Path to event file containing frames + gaze (preferred)",
    )
    parser.add_argument("--frames-dir", type=str, default=None, help="Directory containing extracted frames")
    parser.add_argument(
        "--frames-list",
        type=str,
        default=None,
        help="Text file with one frame path per line (absolute or relative)",
    )
    parser.add_argument(
        "--frames-glob",
        type=str,
        default="frame_*.jpg",
        help="Glob pattern under --frames-dir (default: frame_*.jpg)",
    )
    parser.add_argument(
        "--video-ids",
        type=str,
        default=None,
        help="Comma-separated list of video subdirectories under --frames-dir",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=None,
        help="Randomly sample N video subdirectories under --frames-dir",
    )
    parser.add_argument(
        "--video-sample-seed",
        type=int,
        default=0,
        help="Seed for random video sampling",
    )
    parser.add_argument(
        "--segment-manifest-in",
        type=str,
        default=None,
        help="JSON file with preselected video segments to reuse",
    )
    parser.add_argument(
        "--segment-manifest-out",
        type=str,
        default="",
        help="Write selected video segments to this JSON file",
    )
    parser.add_argument(
        "--gt-dataset",
        type=str,
        default="auto",
        choices=["auto", "ego4d", "egoexo4d", "none"],
        help="Which dataset GT loader to use for frames-dir inputs",
    )
    parser.add_argument(
        "--ego4d-gaze-dir",
        type=str,
        default=None,
        help="Path to ego4d gaze_frame_label directory (auto if omitted)",
    )
    parser.add_argument(
        "--egoexo-root",
        type=str,
        default=None,
        help="EgoExo4D root containing takes.json/captures/gaze_data",
    )
    parser.add_argument(
        "--egoexo-gaze-dir",
        type=str,
        default=None,
        help="Optional EgoExo4D gaze_data root (overrides <egoexo-root>/gaze_data)",
    )
    parser.add_argument("--frames-fps", type=float, default=30.0, help="FPS for frame timestamps")
    parser.add_argument(
        "--segment-min-sec",
        type=float,
        default=30.0,
        help="Minimum segment duration in seconds",
    )
    parser.add_argument(
        "--segment-max-sec",
        type=float,
        default=60.0,
        help="Maximum segment duration in seconds",
    )
    parser.add_argument(
        "--segment-start-idx",
        type=int,
        default=None,
        help="Start frame index (overrides random/center selection)",
    )
    parser.add_argument(
        "--segment-start-sec",
        type=float,
        default=None,
        help="Start time in seconds (overrides random/center selection)",
    )
    parser.add_argument(
        "--segment-random",
        action="store_true",
        help="Randomize segment start (seeded by --segment-seed if provided)",
    )
    parser.add_argument("--segment-seed", type=int, default=0, help="Seed for random segment selection")
    parser.add_argument("--third-video", type=str, default=None, help="Path to third-person video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video file")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Sampling FPS for inference")
    parser.add_argument("--heatmap-alpha", type=float, default=0.4, help="Heatmap overlay alpha (0-1)")
    parser.add_argument("--show", action="store_true", help="Display output window")
    parser.add_argument("--no-vis", dest="vis", action="store_false", default=True, help="Disable visualization")
    parser.add_argument("--max-frames", type=int, default=None, help="Max number of sampled frames to process")
    parser.add_argument(
        "--gaze-tolerance",
        type=float,
        default=0.05,
        help="Max time delta (sec) to match GT gaze to a video frame",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.05,
        help="Normalized distance threshold for TP (0-1 in image coords)",
    )
    parser.add_argument(
        "--cache-embeddings",
        action="store_true",
        help="Cache encoder embeddings for sliding-window reuse",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision inference (CUDA only)",
    )
    parser.add_argument(
        "--ar-streaming",
        action="store_true",
        help="Use single-frame streaming for AR model with heatmap history",
    )
    parser.add_argument(
        "--skip-metrics-frames",
        type=int,
        default=0,
        help="Skip metrics for the first N sampled frames (for warmup parity)",
    )
    parser.add_argument(
        "--perf-log",
        type=str,
        default=None,
        help="CSV path to save per-frame performance stats",
    )
    parser.add_argument(
        "--per-frame-log",
        type=str,
        default=None,
        help="CSV path to save per-frame predictions and GT metrics",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="JSON path to save summary metrics",
    )
    parser.add_argument(
        "--fps-window-sec",
        type=float,
        default=5.0,
        help="Window size for FPS-over-time logging (seconds)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N sampled frames (0 to disable)",
    )
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Print metrics summary even when GT gaze is unavailable",
    )
    parser.add_argument(
        "--disable-template",
        action="store_true",
        help="Disable template/ROI guidance for ARGaze models (override config)",
    )
    parser.add_argument(
        "--glc-align-test",
        action="store_true",
        help="GLC only: use replicate-padded causal windows and test-time argmax decoding.",
    )
    parser.add_argument("--recovery-cap", type=int, default=20, help="Recovery metric cap in frames")
    parser.add_argument(
        "--pre-spike-baseline-window",
        type=int,
        default=10,
        help="Valid frames before each spike used for recovery baseline",
    )
    parser.add_argument(
        "--spike-threshold-multiplier",
        type=float,
        default=2.0,
        help="Error spike threshold as a multiple of sequence mean L2",
    )
    parser.add_argument(
        "--recovery-threshold-multiplier",
        type=float,
        default=1.2,
        help="Recovery threshold as a multiple of the pre-spike baseline",
    )
    parser.add_argument(
        "--saccade-merge-window",
        type=int,
        default=5,
        help="Merge detected error spikes within this many frames",
    )
    parser.add_argument(
        "--simple-vis",
        action="store_true",
        help="Use simple visualization (heatmap overlay only)",
    )
    return parser.parse_args()


def collect_gaze_events(event_path):
    gaze_times = []
    gaze_points = []
    first_frame_time = None

    with open(event_path, "rb") as f:
        while True:
            b = f.read(1)
            if not b:
                break
            (typ,) = struct.unpack("<B", b)
            (t,) = struct.unpack("<d", f.read(8))

            if typ == TYPE_FRAME:
                if first_frame_time is None:
                    first_frame_time = t
                (n,) = struct.unpack("<I", f.read(4))
                f.read(n)
            elif typ == TYPE_GAZE:
                x, y = struct.unpack("<f f", f.read(8))
                gaze_times.append(t)
                gaze_points.append((x, y))
            else:
                raise ValueError(f"Unknown type tag in file: {typ}")

    if gaze_times:
        t0 = first_frame_time if first_frame_time is not None else gaze_times[0]
        gaze_times = [t - t0 for t in gaze_times]
    else:
        t0 = 0.0

    return gaze_times, gaze_points, t0


def iter_event_frames(event_path, t0):
    with open(event_path, "rb") as f:
        while True:
            b = f.read(1)
            if not b:
                break
            (typ,) = struct.unpack("<B", b)
            (t,) = struct.unpack("<d", f.read(8))
            if typ == TYPE_FRAME:
                (n,) = struct.unpack("<I", f.read(4))
                img_bytes = f.read(n)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                yield (t - t0), frame
            elif typ == TYPE_GAZE:
                f.read(8)
            else:
                raise ValueError(f"Unknown type tag in file: {typ}")


def iter_video_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        yield t_sec, frame


def load_frame_list(frames_dir, frames_list, frames_glob):
    if frames_list:
        list_path = Path(frames_list)
        root = list_path.parent
        with open(list_path, "r") as f:
            raw = [line.strip() for line in f if line.strip()]
        paths = []
        for p in raw:
            fp = Path(p)
            if not fp.is_absolute():
                fp = (root / fp).resolve()
            paths.append(fp)
        return paths
    if frames_dir:
        base = Path(frames_dir)
        paths = sorted(base.glob(frames_glob))
        if not paths:
            # Try recursive if user passed a subdir without glob
            paths = sorted(base.rglob(frames_glob))
        return paths
    return []


def list_video_dirs(frames_dir, frames_glob):
    base = Path(frames_dir)
    if not base.exists():
        return []
    vids = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and list(p.glob(frames_glob)):
            vids.append(p)
    if vids:
        return vids
    parents = {}
    for p in base.rglob(frames_glob):
        parents[p.parent] = True
    return sorted(parents.keys())


def video_dir_id(frames_root, video_dir):
    try:
        rel = Path(video_dir).relative_to(frames_root)
        return str(rel)
    except Exception:
        return Path(video_dir).name


def resolve_video_dirs(frames_dir, frames_glob, video_ids, num_videos, seed):
    all_dirs = list_video_dirs(frames_dir, frames_glob)
    if video_ids:
        resolved = []
        for vid in video_ids:
            cand = Path(frames_dir) / vid
            if cand.exists():
                resolved.append(cand)
                continue
            matched = [p for p in all_dirs if p.name == vid or video_dir_id(frames_dir, p) == vid]
            if matched:
                resolved.append(matched[0])
                continue
            raise ValueError(f"Video id not found under frames dir: {vid}")
        return resolved
    if num_videos is not None:
        if num_videos <= 0:
            raise ValueError("--num-videos must be > 0")
        rng = random.Random(seed)
        if len(all_dirs) <= num_videos:
            return all_dirs
        return rng.sample(all_dirs, num_videos)
    return all_dirs


def detect_gt_dataset(frames_dir, mode):
    if mode != "auto":
        return mode
    if frames_dir is None:
        return "none"
    path = str(frames_dir).lower()
    if "egoexo" in path:
        return "egoexo4d"
    if "ego4d" in path:
        return "ego4d"
    return "none"


def parse_frame_index(path_obj):
    stem = path_obj.stem
    if stem.startswith("frame_"):
        try:
            return int(stem.split("_")[-1])
        except Exception:
            return None
    try:
        return int(stem)
    except Exception:
        return None


def load_ego4d_labels(gaze_dir, video_id):
    label_path = Path(gaze_dir) / f"{video_id}_frame_label.csv"
    if not label_path.exists():
        return None
    rows = []
    with open(label_path, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if not row:
                continue
            rows.append([float(x) for x in row])
    if not rows:
        return None
    arr = np.array(rows)
    if arr.shape[1] >= 4:
        return arr[:, 1:4]
    if arr.shape[1] >= 3:
        return arr[:, 1:3]
    return None


def build_ego4d_gt_lookup(frame_paths, gaze_dir, video_id, width, height):
    labels = load_ego4d_labels(gaze_dir, video_id)
    if labels is None:
        return None
    lookup = {}
    for p in frame_paths:
        idx = parse_frame_index(p)
        if idx is None:
            continue
        if idx < 0 or idx >= labels.shape[0]:
            lookup[idx] = None
            continue
        row = labels[idx]
        if row.shape[0] >= 3:
            gaze_type = int(row[2])
            if gaze_type in [3, 4]:
                lookup[idx] = None
                continue
            x, y = float(row[0]), float(row[1])
        else:
            x, y = float(row[0]), float(row[1])
        if np.isnan(x) or np.isnan(y):
            lookup[idx] = None
            continue
        lookup[idx] = (x * width, y * height)
    return lookup


def build_egoexo_gt_lookup(frame_paths, frames_root, egoexo_root, gaze_root, target_fps, width, height):
    try:
        from slowfast.datasets.egoexo4d_gaze import (
            find_aria_number,
            infer_aria_number_from_timesync,
            load_take_frame_timestamps_ns,
            load_gaze_df_for_take_cached,
            get_video_wh,
        )
    except Exception:
        glc_root = Path(__file__).resolve().parents[2] / "GLC"
        if glc_root.exists():
            sys.path.insert(0, str(glc_root))
        # If slowfast was already imported from the demo tree, force reload from GLC.
        to_del = [k for k in sys.modules if k == "slowfast" or k.startswith("slowfast.")]
        for k in to_del:
            del sys.modules[k]
        from slowfast.datasets.egoexo4d_gaze import (  # noqa: F401
            find_aria_number,
            infer_aria_number_from_timesync,
            load_take_frame_timestamps_ns,
            load_gaze_df_for_take_cached,
            get_video_wh,
        )

    if egoexo_root is None:
        raise ValueError("--egoexo-root is required to load EgoExo4D GT gaze.")

    rel = Path(frame_paths[0]).parent.relative_to(frames_root)
    take_name = rel.parts[0]
    ego_exo_root = Path(egoexo_root)

    aria_num = find_aria_number(ego_exo_root, take_name) or infer_aria_number_from_timesync(ego_exo_root, take_name)
    if aria_num is None:
        raise FileNotFoundError(f"Cannot resolve aria number for take '{take_name}'")

    take_ts_ns = load_take_frame_timestamps_ns(ego_exo_root, take_name, aria_num)
    gdf = load_gaze_df_for_take_cached(ego_exo_root, take_name, Path(gaze_root) if gaze_root else None)
    if gdf is None or gdf.empty:
        return None

    gaze_ts = gdf["timestamp_ns"].to_numpy(np.int64)
    gaze_xy = np.stack([gdf["x"].to_numpy(np.float32), gdf["y"].to_numpy(np.float32)], axis=0)
    video_wh = get_video_wh("")

    order = np.argsort(gaze_ts)
    gts = gaze_ts[order]
    gx = gaze_xy[0][order]
    gy = gaze_xy[1][order]
    half_frame_ns = 0.5 * (1e9 / target_fps)

    lookup = {}
    for p in frame_paths:
        idx = parse_frame_index(p)
        if idx is None:
            continue
        if idx < 0 or idx >= len(take_ts_ns):
            lookup[idx] = None
            continue
        ts = take_ts_ns[idx]
        pos = np.searchsorted(gts, ts)
        cand = [c for c in [pos - 1, pos, pos + 1] if 0 <= c < len(gts)]
        if not cand:
            lookup[idx] = None
            continue
        best = min(cand, key=lambda i: abs(gts[i] - ts))
        if abs(gts[best] - ts) > half_frame_ns:
            lookup[idx] = None
            continue
        x_norm = np.clip(gx[best] / max(1, video_wh[0]), 0.0, 1.0)
        y_norm = np.clip(gy[best] / max(1, video_wh[1]), 0.0, 1.0)
        lookup[idx] = (x_norm * width, y_norm * height)
    return lookup


def load_segment_manifest(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_segment_manifest(path, payload):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def select_contiguous_segment(frame_paths, fps, min_sec, max_sec, start_idx, start_sec, use_random, seed):
    total = len(frame_paths)
    if total == 0:
        raise ValueError("No frames found for segment selection.")
    min_frames = max(1, int(round(min_sec * fps)))
    max_frames = max(1, int(round(max_sec * fps)))
    if total < min_frames:
        raise ValueError(
            f"Not enough frames for min segment ({min_sec:.1f}s @ {fps:.1f}fps = {min_frames} frames)."
        )
    seg_len = min(max_frames, total)

    if start_sec is not None:
        start_idx = int(round(start_sec * fps))
    if start_idx is not None:
        start_idx = max(0, min(start_idx, total - seg_len))
    elif use_random:
        rng = random.Random(seed)
        start_idx = rng.randint(0, total - seg_len)
    else:
        start_idx = max(0, (total - seg_len) // 2)

    end_idx = start_idx + seg_len
    return frame_paths[start_idx:end_idx], start_idx, end_idx, seg_len


def iter_sampled_frames(frame_paths, src_fps, target_fps):
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    target_dt = 1.0 / target_fps
    duration_sec = (len(frame_paths) - 1) / src_fps if len(frame_paths) > 1 else 0.0
    next_t = 0.0
    last_idx = -1
    while next_t <= duration_sec + 1e-9:
        idx = int(round(next_t * src_fps))
        if idx >= len(frame_paths):
            break
        if idx != last_idx:
            frame = cv2.imread(str(frame_paths[idx]), cv2.IMREAD_COLOR)
            if frame is not None:
                yield (idx / src_fps), frame, idx
            last_idx = idx
        next_t += target_dt


def find_nearest_gaze(t, gaze_times, gaze_points, max_dt):
    if not gaze_times:
        return None
    idx = bisect.bisect_left(gaze_times, t)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(gaze_times):
        candidates.append(idx)

    best = None
    best_dt = max_dt
    for cand in candidates:
        dt = abs(gaze_times[cand] - t)
        if dt <= best_dt:
            best_dt = dt
            best = cand
    if best is None:
        return None
    return gaze_points[best]


def normalized_distance(a_xy, b_xy):
    dx = a_xy[0] - b_xy[0]
    dy = a_xy[1] - b_xy[1]
    return (dx * dx + dy * dy) ** 0.5


def is_finite_xy(x, y):
    return np.isfinite(x) and np.isfinite(y)


def compute_temporal_jitter_from_records(records):
    distances = []
    by_video = {}
    for rec in records:
        if rec.get("pred_x") is None or rec.get("pred_y") is None:
            continue
        if not is_finite_xy(rec["pred_x"], rec["pred_y"]):
            continue
        by_video.setdefault(rec["video_name"], []).append(rec)

    for video_records in by_video.values():
        video_records = sorted(video_records, key=lambda r: r["frame_offset"])
        for prev, curr in zip(video_records[:-1], video_records[1:]):
            dx = curr["pred_x"] - prev["pred_x"]
            dy = curr["pred_y"] - prev["pred_y"]
            distances.append((dx * dx + dy * dy) ** 0.5)

    return float(np.mean(distances)) if distances else None


def compute_recovery_from_records(records, args):
    recovery_lengths = []
    skipped_events = 0
    by_video = {}
    for rec in records:
        if rec.get("valid", 0) != 1:
            continue
        if rec.get("l2") is None or not np.isfinite(rec["l2"]):
            continue
        by_video.setdefault(rec["video_name"], []).append(rec)

    for video_records in by_video.values():
        valid_records = sorted(video_records, key=lambda r: r["frame_offset"])
        if not valid_records:
            continue
        seq_mean_l2 = float(np.mean([r["l2"] for r in valid_records]))
        spike_threshold = args.spike_threshold_multiplier * seq_mean_l2
        raw_spikes = [i for i, rec in enumerate(valid_records) if rec["l2"] > spike_threshold]

        merged_spikes = []
        for curr_idx in raw_spikes:
            if not merged_spikes:
                merged_spikes.append(curr_idx)
                continue
            last_idx = merged_spikes[-1]
            if (
                valid_records[curr_idx]["frame_offset"] - valid_records[last_idx]["frame_offset"]
            ) > args.saccade_merge_window:
                merged_spikes.append(curr_idx)

        for spike_idx in merged_spikes:
            start_idx = max(0, spike_idx - args.pre_spike_baseline_window)
            baseline_records = valid_records[start_idx:spike_idx]
            if len(baseline_records) < 3:
                skipped_events += 1
                continue

            baseline_l2 = float(np.mean([r["l2"] for r in baseline_records]))
            recovery_threshold = args.recovery_threshold_multiplier * baseline_l2
            spike_offset = valid_records[spike_idx]["frame_offset"]
            recovery_len = args.recovery_cap

            for curr_rec in valid_records[spike_idx + 1 :]:
                dist = curr_rec["frame_offset"] - spike_offset
                if dist <= 0:
                    continue
                if dist > args.recovery_cap:
                    break
                if curr_rec["l2"] < recovery_threshold:
                    recovery_len = dist
                    break
            recovery_lengths.append(recovery_len)

    return {
        "recovery_mean_length": float(np.mean(recovery_lengths)) if recovery_lengths else None,
        "recovery_median_length": float(np.median(recovery_lengths)) if recovery_lengths else None,
        "recovery_p95_length": float(np.percentile(recovery_lengths, 95)) if recovery_lengths else None,
        "recovery_events": len(recovery_lengths),
        "recovery_skipped_events": skipped_events,
    }


def write_per_frame_records(path, records):
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "video_name",
        "frame_idx",
        "clip_index",
        "frame_offset",
        "f1",
        "recall",
        "precision",
        "l2",
        "pred_x",
        "pred_y",
        "gt_x",
        "gt_y",
        "valid",
        "gaze_type",
        "threshold",
        "t_sec",
        "latency_ms",
        "hit",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rec in records:
            writer.writerow({h: rec.get(h, "") for h in headers})
    print(f"\nSaved per-frame prediction log to {out_path}")


def write_summary_json(path, payload):
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved summary JSON to {out_path}")


def aggregate_stats(results):
    if not results:
        return {}
    total_tp = sum(r.get("tp", 0) for r in results)
    total_fp = sum(r.get("fp", 0) for r in results)
    total_fn = sum(r.get("fn", 0) for r in results)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def mean_present(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    total_valid = sum(r.get("valid_frames", 0) for r in results)
    total_sampled = sum(r.get("sampled_frames", 0) for r in results)
    total_inference = sum(r.get("inference_frames", 0) for r in results)
    total_recovery_events = sum(r.get("recovery_events", 0) for r in results)

    return {
        "sampled_frames": total_sampled,
        "inference_frames": total_inference,
        "valid_frames": total_valid,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_dist": mean_present("mean_dist"),
        "temporal_jitter": mean_present("temporal_jitter"),
        "recovery_mean_length": mean_present("recovery_mean_length"),
        "recovery_events": total_recovery_events,
        "infer_fps": mean_present("infer_fps"),
        "mean_latency_ms": mean_present("mean_latency_ms"),
        "p95_latency_ms": mean_present("p95_latency_ms"),
        "total_fps": mean_present("total_fps"),
        "max_alloc_mb": max((r.get("max_alloc_mb") or 0.0) for r in results),
        "max_reserved_mb": max((r.get("max_reserved_mb") or 0.0) for r in results),
    }


def draw_label(frame, text, origin, font_scale=0.7, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    pad = 6
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def build_text_bar(width, height):
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:] = (10, 10, 10)
    return bar


def encode_frame_tokens(model, frame_tensor, use_amp=False):
    frame_tensor = frame_tensor.unsqueeze(0)
    frame_tensor = model._preprocess_frames(frame_tensor)
    if use_amp and frame_tensor.is_cuda:
        with torch.cuda.amp.autocast():
            outputs = model.encoder(pixel_values=frame_tensor)
    else:
        outputs = model.encoder(pixel_values=frame_tensor)
    if model.use_multiscale:
        hidden_states = outputs.hidden_states
        inter_features = []
        num_spatial_patches = 196
        for layer_idx in model.multiscale_layers:
            layer_tokens = hidden_states[layer_idx][:, 1 : 1 + num_spatial_patches, :]
            inter_features.append(layer_tokens)
        multiscale_features = []
        for feat, proj in zip(inter_features, model.multiscale_proj):
            multiscale_features.append(proj(feat))
        patch_tokens = torch.cat(multiscale_features, dim=-1)
    else:
        num_spatial_patches = 196
        patch_tokens = outputs.last_hidden_state[:, 1 : 1 + num_spatial_patches, :]

    patch_tokens = model.feature_proj(patch_tokens)
    return patch_tokens.squeeze(0)


def build_center_gaussian(size, sigma, device):
    coords = torch.arange(size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (size - 1) / 2.0
    gauss = torch.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.max().clamp(min=1e-6)
    return gauss


def build_gaze_label_heatmap(height, width, kernel_size, sigma, x_norm, y_norm, device):
    """Build a GT heatmap using the EGTEA/Ego4D dataset-loader convention."""
    heatmap = np.zeros((height, width), dtype=np.float32)
    mu_x = round(float(x_norm) * width)
    mu_y = round(float(y_norm) * height)
    radius = (kernel_size - 1) // 2
    left = max(mu_x - radius, 0)
    right = min(mu_x + radius, width - 1)
    top = max(mu_y - radius, 0)
    bottom = min(mu_y + radius, height - 1)

    if left < right and top < bottom:
        kernel_1d = cv2.getGaussianKernel(
            ksize=int(kernel_size),
            sigma=float(sigma),
            ktype=cv2.CV_32F,
        )
        kernel_2d = kernel_1d * kernel_1d.T
        k_left = radius - mu_x + left
        k_right = radius + right - mu_x
        k_top = radius - mu_y + top
        k_bottom = radius + bottom - mu_y
        heatmap[top : bottom + 1, left : right + 1] = kernel_2d[
            k_top : k_bottom + 1,
            k_left : k_right + 1,
        ]

    heatmap_sum = float(heatmap.sum())
    if heatmap_sum == 0.0:
        heatmap += 1.0 / float(height * width)
    elif heatmap_sum != 1.0:
        heatmap /= heatmap_sum
    return torch.as_tensor(heatmap, dtype=torch.float32, device=device)


def resize_with_aspect(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = (0, 0, 0)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def model_expects_gaze_kwargs(model):
    try:
        sig = inspect.signature(model.forward)
        return "gt_heatmap" in sig.parameters
    except (TypeError, ValueError):
        return False


def wrap_model_input(model, frames):
    if model.__class__.__name__.startswith("GLC_") or model.__class__.__name__ == "SlowFast":
        return [frames]
    return frames


def is_glc_model(model):
    return model.__class__.__name__.startswith("GLC_")


def call_autoregressive_decode(model, **kwargs):
    sig = inspect.signature(model._autoregressive_decode)
    accepted = {
        name
        for name, param in sig.parameters.items()
        if name != "self"
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return model._autoregressive_decode(**{k: v for k, v in kwargs.items() if k in accepted})


def supports_cached_streaming_decode(model):
    if not hasattr(model, "streaming_decode_step"):
        return False
    try:
        sig = inspect.signature(model.streaming_decode_step)
    except (TypeError, ValueError):
        return False
    return "cached_frame_tokens" in sig.parameters


def run_inference_stream(
    frame_iter,
    width,
    height,
    using_frames,
    gaze_times,
    gaze_points,
    t0,
    cfg,
    model,
    device,
    glc_streaming,
    ar_streaming,
    use_cache,
    args,
    video_tag=None,
    output_path=None,
    perf_log_path=None,
    per_frame_log_path=None,
    third_video=None,
    gt_lookup=None,
):
    third_cap = None
    third_iter = None
    third_next = None
    third_frame = None
    if third_video:
        third_cap = cv2.VideoCapture(third_video)
        if not third_cap.isOpened():
            raise RuntimeError(f"Failed to open third-person video: {third_video}")
        third_iter = iter_video_frames(third_cap)
        try:
            third_next = next(third_iter)
        except StopIteration:
            third_next = None

    writer = None
    if output_path and args.vis:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if args.simple_vis:
            writer = cv2.VideoWriter(output_path, fourcc, args.target_fps, (width, height))
        else:
            top_bar_h = 70
            exo_w = max(1, width // 2)
            out_w = exo_w + width
            out_h = top_bar_h + height
            writer = cv2.VideoWriter(output_path, fourcc, args.target_fps, (out_w, out_h))

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    estimator_buffer = deque(maxlen=cfg.DATA.NUM_FRAMES)
    token_buffer_base = deque(maxlen=cfg.DATA.NUM_FRAMES)
    encode_time_buffer = deque(maxlen=cfg.DATA.NUM_FRAMES)
    streaming_predicted_heatmaps = []
    use_cached_streaming_decode = use_cache and supports_cached_streaming_decode(model)
    use_glc_test_decode = glc_streaming and args.glc_align_test
    glc_frames = [] if use_glc_test_decode else None
    inference_times = []
    sampled_frames = 0
    inference_frames = 0

    tp = fp = fn = 0
    distances = []
    per_frame_records = []
    label_hm_size = max(1, int(getattr(cfg.DATA, "TEST_CROP_SIZE", 224)) // 4)
    label_hm_kernel = int(getattr(cfg.DATA, "GAUSSIAN_KERNEL", 19))
    label_hm_sigma = float(getattr(cfg.DATA, "HEATMAP_SIGMA", -1.0))
    heatmap_thresholds = torch.linspace(0.0, 0.02, 11, device=device)
    heatmap_recall_sum = torch.zeros_like(heatmap_thresholds)
    heatmap_precision_sum = torch.zeros_like(heatmap_thresholds)
    heatmap_valid_frames = 0

    target_dt = 1.0 / max(args.target_fps, 1e-6)
    next_time = 0.0
    start_time = None

    if glc_streaming and not use_glc_test_decode:
        zero_frame = torch.zeros(3, cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE)
        for _ in range(cfg.DATA.NUM_FRAMES):
            estimator_buffer.append(zero_frame)

    history_heatmaps = None
    if ar_streaming:
        history_hm_size = cfg.MODEL.HEATMAP_SIZE
        history_sigma = getattr(cfg.DATA, "HEATMAP_SIGMA", 3.0)
        center_hm = build_center_gaussian(history_hm_size, history_sigma, device).view(
            1, 1, history_hm_size, history_hm_size
        )
        history_heatmaps = deque(
            [center_hm.clone() for _ in range(cfg.MODEL.HISTORY_LENGTH)],
            maxlen=cfg.MODEL.HISTORY_LENGTH,
        )

    perf_records = []
    window = deque()

    for item in frame_iter:
        if len(item) == 3:
            t_sec, frame, frame_idx = item
        else:
            t_sec, frame = item
            frame_idx = None
        if not using_frames and t_sec + 1e-9 < next_time:
            continue
        next_time += target_dt

        if start_time is None:
            start_time = time.perf_counter()

        sampled_frames += 1
        if args.max_frames and sampled_frames > args.max_frames:
            break
        if args.progress_every > 0 and sampled_frames % args.progress_every == 0:
            elapsed = time.perf_counter() - start_time if start_time is not None else 0.0
            fps_now = (sampled_frames / elapsed) if elapsed > 0 else 0.0
            tag = f"[{video_tag}] " if video_tag else ""
            print(f"{tag}Progress: sampled {sampled_frames} frames | elapsed {elapsed:.1f}s | fps {fps_now:.1f}")

        tensor = preprocess_frame(frame, cfg.DATA.TEST_CROP_SIZE)
        estimator_buffer.append(tensor)
        if use_glc_test_decode:
            glc_frames.append(tensor)

        if use_cache:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(device)
            enc_start = time.perf_counter()
            with torch.no_grad():
                token_buffer_base.append(encode_frame_tokens(model, tensor.to(device), use_amp=args.amp))
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(device)
            encode_time_buffer.append(time.perf_counter() - enc_start)

        gaze_pred = None
        pred_heatmap = None
        heatmap_np = None
        step_time = None
        if ar_streaming:
            frame_tensor = tensor.unsqueeze(0).to(device)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(device)
            inf_start = time.perf_counter()
            with torch.no_grad():
                if args.amp and device.type == "cuda":
                    amp_ctx = torch.cuda.amp.autocast()
                else:
                    amp_ctx = None
                if amp_ctx is not None:
                    with amp_ctx:
                        heatmaps = model.forward_streaming(frame_tensor, list(history_heatmaps))
                else:
                    heatmaps = model.forward_streaming(frame_tensor, list(history_heatmaps))
                last_heatmap = heatmaps[0, 0, -1]
                eval_heatmaps = prepare_heatmaps_for_eval(heatmaps)
                eval_last_heatmap = eval_heatmaps[0, 0, -1]
                pred_heatmap = eval_last_heatmap.detach()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(device)
            step_time = time.perf_counter() - inf_start
            inference_times.append(step_time)
            inference_frames += 1

            history_heatmaps.append(last_heatmap.detach().unsqueeze(0).unsqueeze(0))
            gaze_pred = heatmap_to_coords_argmax(eval_last_heatmap)
            heatmap_np = eval_last_heatmap.detach().cpu().numpy()
        elif use_glc_test_decode and len(glc_frames) >= 1:
            local_t = len(glc_frames) - 1
            start = max(0, local_t - cfg.DATA.NUM_FRAMES + 1)
            seg = glc_frames[start : local_t + 1]
            window_frames = torch.stack(seg, dim=1).unsqueeze(0).to(device)
            window_frames = pad_streaming_window(
                window_frames,
                cfg.DATA.NUM_FRAMES,
                warm_start="replicate",
            )
            model_input = wrap_model_input(model, window_frames)

            if args.device.startswith("cuda"):
                torch.cuda.synchronize(device)
            inf_start = time.perf_counter()
            with torch.no_grad():
                if args.amp and device.type == "cuda":
                    amp_ctx = torch.cuda.amp.autocast()
                else:
                    amp_ctx = None
                if amp_ctx is not None:
                    with amp_ctx:
                        heatmaps = model(model_input)
                else:
                    heatmaps = model(model_input)
                preds_rescale = prepare_heatmaps_for_eval(heatmaps)
                last_heatmap = preds_rescale[0, 0, -1]
                pred_heatmap = last_heatmap.detach()
                gaze_pred = heatmap_to_coords_argmax(last_heatmap)
                heatmap_np = last_heatmap.detach().cpu().numpy()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(device)
            step_time = time.perf_counter() - inf_start
            inference_times.append(step_time)
            inference_frames += 1
        elif len(estimator_buffer) == cfg.DATA.NUM_FRAMES:
            frames = torch.stack(list(estimator_buffer), dim=0)
            frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
            model_input = wrap_model_input(model, frames)

            if args.device.startswith("cuda"):
                torch.cuda.synchronize(device)
            inf_start = time.perf_counter()
            with torch.no_grad():
                if args.amp and device.type == "cuda":
                    amp_ctx = torch.cuda.amp.autocast()
                else:
                    amp_ctx = None
                if amp_ctx is not None:
                    with amp_ctx:
                        if use_cached_streaming_decode:
                            pos = model._get_2d_pos_encoding(model.patch_h, model.patch_w, device)
                            current_tokens = token_buffer_base[-1].unsqueeze(0)
                            cached_tokens = [
                                tokens.unsqueeze(0) for tokens in list(token_buffer_base)[:-1]
                            ]
                            local_t = len(cached_tokens)
                            recent_heatmaps = (
                                streaming_predicted_heatmaps[-local_t:] if local_t > 0 else []
                            )
                            heatmap_t = model.streaming_decode_step(
                                local_t,
                                current_tokens,
                                recent_heatmaps,
                                cached_tokens,
                                pos,
                            )
                            heatmaps = heatmap_t.unsqueeze(2)
                            streaming_predicted_heatmaps.append(heatmap_t.detach())
                        elif use_cache:
                            visual_base = torch.stack(list(token_buffer_base), dim=0).unsqueeze(0)
                            pos = model._get_2d_pos_encoding(model.patch_h, model.patch_w, device)
                            visual_features = visual_base + pos
                            raw_frames = frames.permute(0, 2, 1, 3, 4)
                            heatmaps = call_autoregressive_decode(
                                model,
                                visual_features=visual_features,
                                visual_features_base=visual_base,
                                raw_frames=raw_frames,
                                gt_heatmap=None,
                                gt_heatmap_center=None,
                                B=1,
                                T=cfg.DATA.NUM_FRAMES,
                                train_ar=True,
                                ss_prob=0.0,
                                pos_encoding=pos,
                                context_manager=torch.no_grad(),
                            )
                        else:
                            if model_expects_gaze_kwargs(model):
                                heatmaps = model(model_input, gt_heatmap=None, train_ar=True, ss_prob=0.0)
                            else:
                                heatmaps = model(model_input)
                else:
                    if use_cached_streaming_decode:
                        pos = model._get_2d_pos_encoding(model.patch_h, model.patch_w, device)
                        current_tokens = token_buffer_base[-1].unsqueeze(0)
                        cached_tokens = [
                            tokens.unsqueeze(0) for tokens in list(token_buffer_base)[:-1]
                        ]
                        local_t = len(cached_tokens)
                        recent_heatmaps = (
                            streaming_predicted_heatmaps[-local_t:] if local_t > 0 else []
                        )
                        heatmap_t = model.streaming_decode_step(
                            local_t,
                            current_tokens,
                            recent_heatmaps,
                            cached_tokens,
                            pos,
                        )
                        heatmaps = heatmap_t.unsqueeze(2)
                        streaming_predicted_heatmaps.append(heatmap_t.detach())
                    elif use_cache:
                        visual_base = torch.stack(list(token_buffer_base), dim=0).unsqueeze(0)
                        pos = model._get_2d_pos_encoding(model.patch_h, model.patch_w, device)
                        visual_features = visual_base + pos
                        raw_frames = frames.permute(0, 2, 1, 3, 4)
                        heatmaps = call_autoregressive_decode(
                            model,
                            visual_features=visual_features,
                            visual_features_base=visual_base,
                            raw_frames=raw_frames,
                            gt_heatmap=None,
                            gt_heatmap_center=None,
                            B=1,
                            T=cfg.DATA.NUM_FRAMES,
                            train_ar=True,
                            ss_prob=0.0,
                            pos_encoding=pos,
                            context_manager=torch.no_grad(),
                        )
                    else:
                        if model_expects_gaze_kwargs(model):
                            heatmaps = model(model_input, gt_heatmap=None, train_ar=True, ss_prob=0.0)
                        else:
                            heatmaps = model(model_input)
                last_heatmap = heatmaps[0, 0, -1]
                eval_heatmaps = prepare_heatmaps_for_eval(heatmaps)
                eval_last_heatmap = eval_heatmaps[0, 0, -1]
                pred_heatmap = eval_last_heatmap.detach()
                gaze_pred = heatmap_to_coords_argmax(eval_last_heatmap)
                heatmap_np = eval_last_heatmap.detach().cpu().numpy()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(device)
            step_time = time.perf_counter() - inf_start
            if use_cache and encode_time_buffer:
                step_time += encode_time_buffer[-1]
            inference_times.append(step_time)
            inference_frames += 1

        if step_time is not None:
            elapsed = time.perf_counter() - start_time if start_time is not None else 0.0
            window.append((elapsed, sampled_frames, inference_frames))
            window_sec = max(args.fps_window_sec, 1e-6)
            while window and (elapsed - window[0][0]) > window_sec:
                window.popleft()
            win_fps = 0.0
            win_inf_fps = 0.0
            if len(window) >= 2:
                dt = window[-1][0] - window[0][0]
                if dt > 0:
                    win_fps = (window[-1][1] - window[0][1]) / dt
                    win_inf_fps = (window[-1][2] - window[0][2]) / dt
            perf_records.append(
                {
                    "sample_idx": sampled_frames,
                    "infer_idx": inference_frames,
                    "t_sec": t_sec,
                    "elapsed_sec": elapsed,
                    "latency_ms": step_time * 1000.0,
                    "fps_e2e": (sampled_frames / elapsed) if elapsed > 0 else 0.0,
                    "fps_window": win_fps,
                    "infer_fps_window": win_inf_fps,
                }
            )

        if gt_lookup is not None and frame_idx is not None:
            gt_xy = gt_lookup.get(frame_idx, None)
        else:
            gt_xy = find_nearest_gaze(t_sec, gaze_times, gaze_points, args.gaze_tolerance)
        gt_norm = None
        if gt_xy is not None and width > 0 and height > 0:
            gt_norm = (gt_xy[0] / width, gt_xy[1] / height)

        dist = None
        hit = None
        valid_eval = sampled_frames > args.skip_metrics_frames
        if sampled_frames > args.skip_metrics_frames:
            if gaze_pred is not None and gt_norm is not None:
                dist = normalized_distance(gaze_pred, gt_norm)
                hit = dist <= args.distance_threshold
                distances.append(dist)
                if hit:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif gaze_pred is not None and gt_norm is None:
                fp += 1
            elif gaze_pred is None and gt_norm is not None:
                fn += 1

        if valid_eval:
            rec_frame_idx = frame_idx if frame_idx is not None else sampled_frames - 1
            valid = int(gaze_pred is not None and gt_norm is not None and dist is not None)
            pred_x = gaze_pred[0] if gaze_pred is not None else None
            pred_y = gaze_pred[1] if gaze_pred is not None else None
            gt_x = gt_norm[0] if gt_norm is not None else None
            gt_y = gt_norm[1] if gt_norm is not None else None
            per_frame_records.append(
                {
                    "video_name": video_tag or "stream",
                    "frame_idx": rec_frame_idx,
                    "clip_index": 0,
                    "frame_offset": rec_frame_idx,
                    "f1": 1.0 if hit else 0.0 if hit is not None else None,
                    "recall": 1.0 if hit else 0.0 if hit is not None else None,
                    "precision": 1.0 if hit else 0.0 if hit is not None else None,
                    "l2": dist,
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                    "gt_x": gt_x,
                    "gt_y": gt_y,
                    "valid": valid,
                    "gaze_type": 1 if valid else 0,
                    "threshold": args.distance_threshold,
                    "t_sec": t_sec,
                    "latency_ms": step_time * 1000.0 if step_time is not None else None,
                    "hit": int(hit) if hit is not None else None,
                }
            )

        if valid_eval and pred_heatmap is not None and gt_norm is not None:
            gt_hm = build_gaze_label_heatmap(
                label_hm_size,
                label_hm_size,
                label_hm_kernel,
                label_hm_sigma,
                gt_norm[0],
                gt_norm[1],
                pred_heatmap.device,
            )
            pred_hm = pred_heatmap.squeeze()
            if pred_hm.shape[-2:] != gt_hm.shape[-2:]:
                pred_hm = F.interpolate(
                    pred_hm.view(1, 1, *pred_hm.shape[-2:]),
                    size=gt_hm.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).view(*gt_hm.shape)
            binary_labels = gt_hm > 0.001
            binary_preds = pred_hm.unsqueeze(0) > heatmap_thresholds.view(-1, 1, 1)
            tp_hm = (binary_preds & binary_labels).sum(dim=(1, 2)).float()
            fg_labels = binary_labels.sum().float()
            fg_preds = binary_preds.sum(dim=(1, 2)).float()
            heatmap_recall_sum += tp_hm / (fg_labels + 1e-6)
            heatmap_precision_sum += tp_hm / (fg_preds + 1e-6)
            heatmap_valid_frames += 1

        if args.vis:
            vis_frame = frame.copy()
            if heatmap_np is not None:
                vis_frame = overlay_heatmap(vis_frame, heatmap_np, alpha=args.heatmap_alpha)
            if gaze_pred is not None:
                vis_frame = draw_gaze_marker(vis_frame, gaze_pred[0], gaze_pred[1], color=(0, 255, 0))
            if gt_norm is not None:
                vis_frame = draw_gaze_marker(vis_frame, gt_norm[0], gt_norm[1], color=(0, 0, 255))
            if args.simple_vis:
                if writer is not None:
                    writer.write(vis_frame)
                if args.show:
                    cv2.imshow("Gaze Offline Demo", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                top_bar_h = 70
                exo_w = max(1, width // 2)
                ego_w = width
                out_h = top_bar_h + height
                out_w = exo_w + ego_w
                out_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)

                text_bar = build_text_bar(out_w, top_bar_h)
                out_frame[0:top_bar_h, 0:out_w] = text_bar
                ego_canvas = resize_with_aspect(vis_frame, ego_w, height)
                out_frame[top_bar_h : top_bar_h + height, exo_w : exo_w + ego_w] = ego_canvas

                elapsed = time.perf_counter() - start_time if start_time is not None else 0.0
                fps_now = (sampled_frames / elapsed) if elapsed > 0 else 0.0
                if inference_times:
                    latency_ms = inference_times[-1] * 1000.0
                    latency_text = f"{latency_ms:.1f} ms"
                else:
                    latency_text = "Buffering"

                mem_text = "N/A"
                if args.device.startswith("cuda"):
                    mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    mem_text = f"{mem_mb / 1024.0:.2f} GB"

                draw_label(out_frame, "Third-person View", (10, 30))
                draw_label(out_frame, "Egocentric View (Main)", (exo_w + 10, 30))
                draw_label(
                    out_frame,
                    f"Performance: FPS(e2e) {fps_now:.1f}  Latency(model) {latency_text}  Mem {mem_text}",
                    (exo_w + 10, 60),
                    font_scale=0.55,
                )

                if third_iter is not None:
                    while third_next is not None and third_next[0] <= t_sec:
                        third_frame = third_next[1]
                        try:
                            third_next = next(third_iter)
                        except StopIteration:
                            third_next = None
                            break

                exo_h = max(1, int(height * 0.6))
                exo_y = top_bar_h + 10
                exo_x = 10
                if third_frame is None:
                    third_resized = np.zeros((exo_h, exo_w - 20, 3), dtype=np.uint8)
                else:
                    third_resized = resize_with_aspect(third_frame, exo_w - 20, exo_h)
                out_frame[exo_y : exo_y + exo_h, exo_x : exo_x + third_resized.shape[1]] = third_resized

                if writer is not None:
                    writer.write(out_frame)
                if args.show:
                    cv2.imshow("Gaze Offline Demo", out_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    end_time = time.perf_counter()
    if third_cap is not None:
        third_cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    total_time = (end_time - start_time) if start_time is not None else 0.0
    total_fps = (sampled_frames / total_time) if total_time > 0 else 0.0
    infer_time_sum = sum(inference_times)
    infer_fps = (inference_frames / infer_time_sum) if infer_time_sum > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    mean_dist = float(np.mean(distances)) if distances else 0.0
    p95_latency = float(np.percentile(inference_times, 95)) if inference_times else 0.0
    mean_latency_ms = float(np.mean(inference_times) * 1000.0) if inference_times else 0.0
    valid_frames = sum(1 for rec in per_frame_records if rec.get("valid") == 1)
    temporal_jitter = compute_temporal_jitter_from_records(per_frame_records)
    recovery_stats = compute_recovery_from_records(per_frame_records, args)
    heatmap_f1 = None
    heatmap_recall = None
    heatmap_precision = None
    heatmap_thr = None
    if heatmap_valid_frames > 0:
        avg_recall = heatmap_recall_sum / heatmap_valid_frames
        avg_precision = heatmap_precision_sum / heatmap_valid_frames
        heatmap_f1_all = (2 * avg_recall * avg_precision) / (
            avg_recall + avg_precision + 1e-6
        )
        best_idx = int(torch.argmax(heatmap_f1_all).item())
        heatmap_f1 = float(heatmap_f1_all[best_idx].item())
        heatmap_recall = float(avg_recall[best_idx].item())
        heatmap_precision = float(avg_precision[best_idx].item())
        heatmap_thr = float(heatmap_thresholds[best_idx].item())

    tag = f" ({video_tag})" if video_tag else ""
    if gaze_times or gaze_points:
        print(f"\nMetrics{tag}")
        print(f"  Frames sampled: {sampled_frames}")
        print(f"  Inference frames: {inference_frames}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        if heatmap_f1 is not None:
            print(f"  Heatmap Precision: {heatmap_precision:.4f}")
            print(f"  Heatmap Recall: {heatmap_recall:.4f}")
            print(f"  Heatmap F1: {heatmap_f1:.4f} (thr={heatmap_thr:.4f})")
        else:
            print("  Heatmap F1: N/A")
        print(f"  Mean dist (norm): {mean_dist:.4f}")
        if temporal_jitter is not None:
            print(f"  Temporal jitter: {temporal_jitter:.4f}")
        else:
            print("  Temporal jitter: N/A")
        if recovery_stats["recovery_mean_length"] is not None:
            print(
                f"  Recovery length: {recovery_stats['recovery_mean_length']:.2f} "
                f"({recovery_stats['recovery_events']} events)"
            )
        else:
            print("  Recovery length: N/A")
    elif args.print_metrics:
        print(f"\nMetrics{tag}")
        print(f"  Frames sampled: {sampled_frames}")
        print(f"  Inference frames: {inference_frames}")
        print("  Precision: N/A (no GT)")
        print("  Recall: N/A (no GT)")
        print("  F1: N/A (no GT)")
        print("  Mean dist (norm): N/A (no GT)")

    tag = f" ({video_tag})" if video_tag else ""
    print(f"\nEfficiency (Model-only){tag}")
    print(f"  Inference FPS: {infer_fps:.2f}")
    print(f"  Mean latency: {mean_latency_ms:.2f} ms")
    print(f"  P95 latency: {p95_latency * 1000:.2f} ms")
    print(f"\nEfficiency (End-to-end){tag}")
    print(f"  Total FPS: {total_fps:.2f}")

    max_alloc = None
    max_reserved = None
    if args.device.startswith("cuda"):
        max_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        print(f"  CUDA max allocated: {max_alloc:.1f} MB")
        print(f"  CUDA max reserved: {max_reserved:.1f} MB")

    if perf_log_path and perf_records:
        out_path = Path(perf_log_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "sample_idx",
            "infer_idx",
            "t_sec",
            "elapsed_sec",
            "latency_ms",
            "fps_e2e",
            "fps_window",
            "infer_fps_window",
        ]
        with open(out_path, "w") as f:
            f.write(",".join(headers) + "\n")
            for rec in perf_records:
                f.write(",".join(str(rec.get(h, "")) for h in headers) + "\n")
        print(f"\nSaved per-frame performance log to {out_path}")

    write_per_frame_records(per_frame_log_path, per_frame_records)

    return {
        "sampled_frames": sampled_frames,
        "inference_frames": inference_frames,
        "valid_frames": valid_frames,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_dist": mean_dist,
        "heatmap_f1": heatmap_f1,
        "heatmap_recall": heatmap_recall,
        "heatmap_precision": heatmap_precision,
        "heatmap_threshold": heatmap_thr,
        "heatmap_valid_frames": heatmap_valid_frames,
        "heatmap_thresholds": [float(x) for x in heatmap_thresholds.detach().cpu().tolist()],
        "heatmap_recall_sum": [float(x) for x in heatmap_recall_sum.detach().cpu().tolist()],
        "heatmap_precision_sum": [float(x) for x in heatmap_precision_sum.detach().cpu().tolist()],
        "temporal_jitter": temporal_jitter,
        **recovery_stats,
        "infer_fps": float(infer_fps),
        "mean_latency_ms": mean_latency_ms,
        "p95_latency_ms": float(p95_latency * 1000.0),
        "total_fps": float(total_fps),
        "max_alloc_mb": max_alloc,
        "max_reserved_mb": max_reserved,
    }


def main():
    debug_log("parsing args")
    args = parse_args()
    if not args.config or not args.checkpoint:
        raise ValueError("--config and --checkpoint are required for inference.")

    using_frames = args.frames_dir is not None or args.frames_list is not None
    if args.event_file is None and args.video is None and not using_frames:
        raise ValueError("Provide --event-file, --video, or --frames-dir/--frames-list.")

    debug_log(f"loading config: {args.config}")
    cfg = load_config(args.config)
    cfg.NUM_GPUS = 1
    if args.disable_template:
        cfg.MODEL.USE_TEMPLATE_TOKENS = False
        cfg.MODEL.TEMPLATE_USE_GAZE_CENTER = False
        cfg.MODEL.USE_FULL_FRAME_TEMPLATE = False
        cfg.MODEL.USE_ROI_PROMPT = False
        cfg.MODEL.USE_ROI_INSTEAD_OF_TEMPLATE = False
    device = torch.device(args.device)
    debug_log(f"loading model checkpoint: {args.checkpoint}")
    model = load_model(cfg, args.checkpoint, device)
    debug_log("model loaded")
    glc_streaming = is_glc_model(model)
    ar_streaming = args.ar_streaming and hasattr(model, "forward_streaming")
    use_cache = args.cache_embeddings and hasattr(model, "_autoregressive_decode")
    if use_cache:
        model.use_original_template_tokens = True
    elif args.cache_embeddings:
        print("Warning: --cache-embeddings is only supported for DINOv3 AR models.")
    if ar_streaming and use_cache:
        print("Warning: --cache-embeddings is ignored for AR streaming mode.")
        use_cache = False

    gaze_times = []
    gaze_points = []
    t0 = 0.0
    frame_iter = None
    src_fps = 0.0
    width = height = None
    gt_lookup_single = None

    if args.event_file:
        gaze_times, gaze_points, t0 = collect_gaze_events(args.event_file)
        if not gaze_times:
            print(f"Warning: no gaze events found in {args.event_file}")
        frame_iter = iter_event_frames(args.event_file, t0)
        first_frame = None
        for t_sec, frame in frame_iter:
            first_frame = (t_sec, frame)
            break
        if first_frame is None:
            raise RuntimeError(f"No frames found in {args.event_file}")
        t_first, frame_first = first_frame
        height, width = frame_first.shape[:2]
        src_fps = 0.0
        print(f"Event source: {width}x{height} (first t={t_first:.3f}s)")
        print(f"Sampling at {args.target_fps:.1f} FPS (GT t0={t0:.3f}s)")
        frame_iter = chain([(t_first, frame_first)], frame_iter)
    elif using_frames:
        if args.frames_list and (args.video_ids or args.num_videos):
            raise ValueError("--video-ids/--num-videos require --frames-dir (not --frames-list).")
    else:
        if args.gt is not None:
            gaze_times, gaze_points, t0 = collect_gaze_events(args.gt)
            if not gaze_times:
                print(f"Warning: no gaze events found in {args.gt}")

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {args.video}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video source: {width}x{height} @ {src_fps:.1f} FPS")
        print(f"Sampling at {args.target_fps:.1f} FPS (GT t0={t0:.3f}s)")

        def frame_stream():
            for t_sec, frame in iter_video_frames(cap):
                yield t_sec, frame

        frame_iter = frame_stream()

    if using_frames:
        if args.segment_manifest_in and not args.frames_dir:
            raise ValueError("--segment-manifest-in requires --frames-dir")

        if args.video_ids or args.num_videos or args.segment_manifest_in:
            debug_log(f"resolving video dirs under: {args.frames_dir}")
            video_ids = [v.strip() for v in args.video_ids.split(",")] if args.video_ids else None
            manifest = None
            if args.segment_manifest_in:
                manifest = load_segment_manifest(args.segment_manifest_in)
                manifest_videos = [v["video_id"] for v in manifest.get("video_segments", [])]
                video_dirs = resolve_video_dirs(args.frames_dir, args.frames_glob, manifest_videos, None, args.video_sample_seed)
            else:
                video_dirs = resolve_video_dirs(
                    args.frames_dir,
                    args.frames_glob,
                    video_ids,
                    args.num_videos,
                    args.video_sample_seed,
                )
            if not video_dirs:
                raise RuntimeError("No video directories found under --frames-dir.")
            debug_log(f"resolved {len(video_dirs)} video dirs")

            results = []
            manifest_out = {
                "frames_dir": args.frames_dir,
                "frames_glob": args.frames_glob,
                "frames_fps": args.frames_fps,
                "segment_min_sec": args.segment_min_sec,
                "segment_max_sec": args.segment_max_sec,
                "video_segments": [],
            }
            gt_dataset = detect_gt_dataset(args.frames_dir, args.gt_dataset)
            selections = []
            for vid_dir in video_dirs:
                debug_log(f"selecting segment for video: {vid_dir.name}")
                frame_paths = sorted(vid_dir.glob(args.frames_glob))
                if not frame_paths:
                    continue
                min_frames = max(1, int(round(args.segment_min_sec * args.frames_fps)))
                if len(frame_paths) < min_frames:
                    print(
                        f"Skipping {vid_dir.name}: only {len(frame_paths)} frames "
                        f"(need >= {min_frames} for {args.segment_min_sec:.1f}s @ {args.frames_fps:.1f}fps)"
                    )
                    continue
                vid_id = video_dir_id(args.frames_dir, vid_dir)
                if manifest is not None:
                    entry = next(
                        (v for v in manifest.get("video_segments", []) if v["video_id"] == vid_id),
                        None,
                    )
                    if entry is None:
                        continue
                    start_idx = int(entry["start_idx"])
                    seg_len = int(entry["seg_len"])
                    start_idx = max(0, min(start_idx, max(0, len(frame_paths) - 1)))
                    end_idx = min(len(frame_paths), start_idx + seg_len)
                else:
                    _paths, start_idx, end_idx, seg_len = select_contiguous_segment(
                        frame_paths,
                        args.frames_fps,
                        args.segment_min_sec,
                        args.segment_max_sec,
                        args.segment_start_idx,
                        args.segment_start_sec,
                        args.segment_random,
                        args.segment_seed,
                    )
                selections.append(
                    {
                        "video_id": vid_id,
                        "video_dir": vid_dir,
                        "start_idx": int(start_idx),
                        "end_idx": int(end_idx),
                        "seg_len": int(seg_len),
                    }
                )
                if manifest is None:
                    manifest_out["video_segments"].append(
                        {
                            "video_id": vid_id,
                            "start_idx": int(start_idx),
                            "end_idx": int(end_idx),
                            "seg_len": int(seg_len),
                        }
                    )

            if args.segment_manifest_out and manifest is None:
                save_segment_manifest(args.segment_manifest_out, manifest_out)
                print(f"\nSaved segment manifest to {args.segment_manifest_out}")
            debug_log(f"selected {len(selections)} video segments")

            for sel in selections:
                vid_id = sel["video_id"]
                vid_dir = sel["video_dir"]
                debug_log(f"running video: {vid_id}")
                frame_paths = sorted(vid_dir.glob(args.frames_glob))
                if not frame_paths:
                    continue
                start_idx = sel["start_idx"]
                end_idx = sel["end_idx"]
                seg_len = sel["seg_len"]
                start_idx = max(0, min(start_idx, max(0, len(frame_paths) - 1)))
                end_idx = min(len(frame_paths), start_idx + seg_len)
                frame_paths = frame_paths[start_idx:end_idx]
                if not frame_paths:
                    continue
                src_fps = args.frames_fps
                first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
                if first_frame is None:
                    raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
                height, width = first_frame.shape[:2]
                seg_dur = seg_len / src_fps
                print(
                    f"\nVideo {vid_id}: {width}x{height} @ {src_fps:.1f} FPS | segment [{start_idx}:{end_idx}) "
                    f"({seg_dur:.1f}s)"
                )
                print(f"Sampling at {args.target_fps:.1f} FPS (t0={t0:.3f}s)")

                def frame_stream():
                    for t_sec, frame, _ in iter_sampled_frames(frame_paths, src_fps, args.target_fps):
                        yield t_sec, frame, _

                gt_lookup = None
                if gt_dataset == "ego4d":
                    gaze_dir = args.ego4d_gaze_dir
                    if gaze_dir is None:
                        gaze_dir = str(Path(args.frames_dir).parent / "gaze_frame_label")
                    gt_lookup = build_ego4d_gt_lookup(frame_paths, gaze_dir, vid_dir.name, width, height)
                elif gt_dataset == "egoexo4d":
                    gt_lookup = build_egoexo_gt_lookup(
                        frame_paths,
                        Path(args.frames_dir),
                        args.egoexo_root,
                        args.egoexo_gaze_dir,
                        args.target_fps,
                        width,
                        height,
                    )

                out_path = None
                if args.vis:
                    if args.output:
                        if "{video_id}" in args.output:
                            out_path = args.output.format(video_id=vid_id)
                        else:
                            stem = Path(args.output).stem
                            suffix = Path(args.output).suffix or ".mp4"
                            out_path = str(Path(args.output).with_name(f"{stem}_{vid_dir.name}{suffix}"))
                    else:
                        out_path = f"output_{vid_dir.name}.mp4"

                perf_path = None
                if args.perf_log:
                    base = Path(args.perf_log)
                    if "{video_id}" in str(base):
                        perf_path = Path(str(base).format(video_id=vid_id))
                    else:
                        perf_path = base.with_name(f"{base.stem}_{vid_dir.name}{base.suffix or '.csv'}")

                per_frame_path = None
                if args.per_frame_log:
                    base = Path(args.per_frame_log)
                    if "{video_id}" in str(base):
                        per_frame_path = Path(str(base).format(video_id=vid_id))
                    else:
                        per_frame_path = base.with_name(
                            f"{base.stem}_{vid_dir.name}{base.suffix or '.csv'}"
                        )

                stats = run_inference_stream(
                    frame_stream(),
                    width,
                    height,
                    True,
                    gaze_times,
                    gaze_points,
                    t0,
                    cfg,
                    model,
                    device,
                    glc_streaming,
                    ar_streaming,
                    use_cache,
                    args,
                    video_tag=vid_id,
                    output_path=out_path,
                    perf_log_path=perf_path,
                    per_frame_log_path=per_frame_path,
                    third_video=None,
                    gt_lookup=gt_lookup,
                )
                results.append(stats)

            if not results:
                raise RuntimeError("No valid videos processed.")

            avg_infer_fps = sum(r["infer_fps"] for r in results) / len(results)
            avg_latency = sum(r["mean_latency_ms"] for r in results) / len(results)
            avg_p95 = sum(r["p95_latency_ms"] for r in results) / len(results)
            avg_total_fps = sum(r["total_fps"] for r in results) / len(results)
            max_alloc = max((r["max_alloc_mb"] or 0) for r in results)
            max_reserved = max((r["max_reserved_mb"] or 0) for r in results)
            aggregate = aggregate_stats(results)

            print("\nAverage over videos")
            print(f"  Valid frames: {aggregate.get('valid_frames', 0)}")
            print(f"  Precision: {aggregate.get('precision', 0.0):.4f}")
            print(f"  Recall: {aggregate.get('recall', 0.0):.4f}")
            print(f"  F1: {aggregate.get('f1', 0.0):.4f}")
            if aggregate.get("temporal_jitter") is not None:
                print(f"  Temporal jitter: {aggregate['temporal_jitter']:.4f}")
            if aggregate.get("recovery_mean_length") is not None:
                print(
                    f"  Recovery length: {aggregate['recovery_mean_length']:.2f} "
                    f"({aggregate.get('recovery_events', 0)} events)"
                )
            print(f"  Inference FPS: {avg_infer_fps:.2f}")
            print(f"  Mean latency: {avg_latency:.2f} ms")
            print(f"  P95 latency: {avg_p95:.2f} ms")
            print(f"  Total FPS: {avg_total_fps:.2f}")
            if args.device.startswith("cuda"):
                print(f"  CUDA max allocated (max): {max_alloc:.1f} MB")
                print(f"  CUDA max reserved (max): {max_reserved:.1f} MB")
            write_summary_json(
                args.summary_json,
                {
                    "aggregate": aggregate,
                    "per_video": results,
                    "config": {
                        "target_fps": args.target_fps,
                        "distance_threshold": args.distance_threshold,
                        "gaze_tolerance": args.gaze_tolerance,
                        "skip_metrics_frames": args.skip_metrics_frames,
                        "glc_align_test": args.glc_align_test,
                        "label_heatmap_size": int(getattr(cfg.DATA, "TEST_CROP_SIZE", 224)) // 4,
                        "label_heatmap_kernel": int(getattr(cfg.DATA, "GAUSSIAN_KERNEL", 19)),
                        "label_heatmap_sigma": float(getattr(cfg.DATA, "HEATMAP_SIGMA", -1.0)),
                    },
                },
            )
            return

        frame_paths = load_frame_list(args.frames_dir, args.frames_list, args.frames_glob)
        if not frame_paths:
            raise RuntimeError("No frames found for --frames-dir/--frames-list.")
        frame_paths, start_idx, end_idx, seg_len = select_contiguous_segment(
            frame_paths,
            args.frames_fps,
            args.segment_min_sec,
            args.segment_max_sec,
            args.segment_start_idx,
            args.segment_start_sec,
            args.segment_random,
            args.segment_seed,
        )
        src_fps = args.frames_fps
        first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if first_frame is None:
            raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
        height, width = first_frame.shape[:2]
        seg_dur = seg_len / src_fps
        print(
            f"Frames source: {width}x{height} @ {src_fps:.1f} FPS | segment [{start_idx}:{end_idx}) "
            f"({seg_dur:.1f}s)"
        )
        print(f"Sampling at {args.target_fps:.1f} FPS (t0={t0:.3f}s)")

        def frame_stream():
            for t_sec, frame, frame_idx in iter_sampled_frames(frame_paths, src_fps, args.target_fps):
                yield t_sec, frame, frame_idx

        frame_iter = frame_stream()
        gt_dataset = detect_gt_dataset(args.frames_dir, args.gt_dataset)
        gt_lookup_single = None
        if gt_dataset == "ego4d":
            gaze_dir = args.ego4d_gaze_dir
            if gaze_dir is None:
                gaze_dir = str(Path(args.frames_dir).parent / "gaze_frame_label")
            gt_lookup_single = build_ego4d_gt_lookup(frame_paths, gaze_dir, Path(frame_paths[0]).parent.name, width, height)
        elif gt_dataset == "egoexo4d":
            gt_lookup_single = build_egoexo_gt_lookup(
                frame_paths,
                Path(args.frames_dir),
                args.egoexo_root,
                args.egoexo_gaze_dir,
                args.target_fps,
                width,
                height,
            )

    output_path = None
    if args.vis:
        if args.output:
            output_path = args.output
        else:
            base = "output"
            if args.event_file:
                main_stem = Path(args.event_file).stem
            elif args.video:
                main_stem = Path(args.video).stem
            else:
                main_stem = "frames_segment"
            if args.third_video:
                third_stem = Path(args.third_video).stem
                output_path = f"{base}_{main_stem}_{third_stem}.mp4"
            else:
                output_path = f"{base}_{main_stem}.mp4"

    stats = run_inference_stream(
        frame_iter,
        width,
        height,
        using_frames,
        gaze_times,
        gaze_points,
        t0,
        cfg,
        model,
        device,
        glc_streaming,
        ar_streaming,
        use_cache,
        args,
        output_path=output_path,
        perf_log_path=args.perf_log,
        per_frame_log_path=args.per_frame_log,
        third_video=args.third_video,
        gt_lookup=gt_lookup_single if using_frames and not (args.video_ids or args.num_videos or args.segment_manifest_in) else None,
    )
    write_summary_json(
        args.summary_json,
        {
            "aggregate": stats,
            "per_video": [stats],
            "config": {
                "target_fps": args.target_fps,
                "distance_threshold": args.distance_threshold,
                "gaze_tolerance": args.gaze_tolerance,
                "skip_metrics_frames": args.skip_metrics_frames,
                "glc_align_test": args.glc_align_test,
                "label_heatmap_size": int(getattr(cfg.DATA, "TEST_CROP_SIZE", 224)) // 4,
                "label_heatmap_kernel": int(getattr(cfg.DATA, "GAUSSIAN_KERNEL", 19)),
                "label_heatmap_sigma": float(getattr(cfg.DATA, "HEATMAP_SIGMA", -1.0)),
            },
        },
    )

    if not args.event_file and not using_frames:
        cap.release()


if __name__ == "__main__":
    main()
