#!/usr/bin/env python3
"""Run causal streaming rollout and compute long-sequence metrics."""

import argparse
import subprocess
import sys
from pathlib import Path

from train_argaze import CONFIGS, ROOT, dataset_overrides


DATASET_FOR_METRICS = {
    "egteagaze": "egtea",
    "ego4d": "ego4d",
    "egoexo4d": "egoexo4d",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", help="Dataset root. Can also be supplied via opts.")
    parser.add_argument("--frames-dir", help="Decoded frame directory override.")
    parser.add_argument("--output-dir", default="output/argaze_longseq")
    parser.add_argument("--num-gpus", default="1")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--subset-size", type=int, default=0, help="Keep the first N test clips.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Extra SlowFast KEY VALUE overrides.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    per_frame_name = "per_frame_metrics.csv"
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
        "TEST.STREAMING_ENABLE",
        "True",
        "TEST.BATCH_SIZE",
        "1",
        "TEST.SAVE_PER_FRAME_METRICS",
        "True",
        "TEST.PER_FRAME_METRICS_FILE",
        per_frame_name,
        "NUM_GPUS",
        args.num_gpus,
        "OUTPUT_DIR",
        str(output_dir),
    ]
    if args.subset_size > 0:
        cmd.extend(["TEST.CLIP_INDEX_FILTER", ",".join(str(i) for i in range(args.subset_size))])
    cmd.extend(dataset_overrides(args.dataset, args.data_root, args.frames_dir))
    cmd.extend(args.opts)
    subprocess.run(cmd, check=True)

    if not args.skip_metrics:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/compute_longseq_metrics.py"),
                "--csv",
                str(output_dir / per_frame_name),
                "--dataset",
                DATASET_FOR_METRICS[args.dataset],
                "--checkpoint",
                args.checkpoint,
                "--model-name",
                "ARGaze",
                "--output-dir",
                str(output_dir / "longseq_metrics"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
