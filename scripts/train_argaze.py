#!/usr/bin/env python3
"""Train ARGaze with the released final configuration."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "egteagaze": ROOT / "configs/Egtea/DINOV3_Full_Exp11_TwoCross_HeatmapBias_EfficientARHeatmap.yaml",
    "ego4d": ROOT / "configs/Ego4d/DINOV3_Full_Exp11_TwoCross_HeatmapBias_EfficientARHeatmap.yaml",
    "egoexo4d": ROOT / "configs/Egoexo4d/DINOV3_Full_Exp11_TwoCross_HeatmapBias_EfficientARHeatmap.yaml",
}


def dataset_overrides(dataset, data_root, frames_dir):
    if not data_root:
        return []
    if dataset == "egteagaze":
        return [
            "DATA.PATH_PREFIX",
            data_root,
            "DATA.FRAMES_DIR",
            frames_dir or str(Path(data_root) / "cropped_frames"),
        ]
    if dataset == "ego4d":
        return [
            "DATA.PATH_PREFIX",
            data_root,
            "DATA.FRAMES_DIR",
            frames_dir or str(Path(data_root) / "frames_stride8_224"),
        ]
    return [
        "DATA.PATH_PREFIX",
        data_root,
        "DATA.FRAMES_DIR",
        frames_dir or str(Path(data_root) / "takes_frames"),
        "DATA.VIDEO_ROOT_DIR",
        data_root,
        "DATA.GAZE_DATA_DIR",
        str(Path(data_root) / "gaze_data"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--data-root", help="Dataset root. Can also be supplied via opts.")
    parser.add_argument("--frames-dir", help="Decoded frame directory override.")
    parser.add_argument("--output-dir", default="output/argaze_train")
    parser.add_argument("--num-gpus", default="1")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Extra SlowFast KEY VALUE overrides.")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "tools/run_net.py"),
        "--cfg",
        str(CONFIGS[args.dataset]),
        "TRAIN.ENABLE",
        "True",
        "TEST.ENABLE",
        "False",
        "NUM_GPUS",
        args.num_gpus,
        "OUTPUT_DIR",
        args.output_dir,
    ]
    cmd.extend(dataset_overrides(args.dataset, args.data_root, args.frames_dir))
    cmd.extend(args.opts)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
