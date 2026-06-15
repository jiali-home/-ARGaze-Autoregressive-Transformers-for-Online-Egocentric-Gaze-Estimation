#!/usr/bin/env python3
"""Evaluate a released ARGaze checkpoint on short clips."""

import argparse
import subprocess
import sys
from pathlib import Path

from train_argaze import CONFIGS, ROOT, dataset_overrides


def clip_index_filter(size):
    if size is None or size <= 0:
        return []
    return ["TEST.CLIP_INDEX_FILTER", ",".join(str(i) for i in range(size))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", help="Dataset root. Can also be supplied via opts.")
    parser.add_argument("--frames-dir", help="Decoded frame directory override.")
    parser.add_argument("--output-dir", default="output/argaze_shortclip")
    parser.add_argument("--num-gpus", default="1")
    parser.add_argument("--endpoint-only", action="store_true")
    parser.add_argument("--save-per-frame", action="store_true")
    parser.add_argument("--subset-size", type=int, default=0, help="Keep the first N test clips.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Extra SlowFast KEY VALUE overrides.")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "tools/run_net.py"),
        "--cfg",
        str(CONFIGS[args.dataset]),
        "TRAIN.ENABLE",
        "False",
        "TEST.ENABLE",
        "True",
        "TEST.CHECKPOINT_FILE_PATH",
        args.checkpoint,
        "NUM_GPUS",
        args.num_gpus,
        "OUTPUT_DIR",
        args.output_dir,
    ]
    if args.endpoint_only:
        cmd.extend(["TEST.ONLY_LAST_FRAME", "True"])
    if args.save_per_frame:
        cmd.extend(["TEST.SAVE_PER_FRAME_METRICS", "True"])
    cmd.extend(clip_index_filter(args.subset_size))
    cmd.extend(dataset_overrides(args.dataset, args.data_root, args.frames_dir))
    cmd.extend(args.opts)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
