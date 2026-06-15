# ARGaze

Public release for **ARGaze: Autoregressive Transformers for Online Egocentric Gaze Estimation**.

This repository contains the core training and evaluation code for the final ARGaze model used in the paper. Checkpoints are distributed separately through a Hugging Face model repository.

## Contents

- `slowfast/`: core model, dataset, config, and training utilities.
- `configs/`: final ARGaze configs for EGTEA Gaze+, Ego4D, and EgoExo4D.
- `scripts/`: stable public wrappers for training, short-clip evaluation, long-sequence rollout, and checkpoint download.
- `tools/`: lower-level evaluation and efficiency utilities.
- `data/splits/`: released split files and EgoExo4D benchmark split metadata.
- `checkpoints/manifest.json`: expected checkpoint names, hashes, and source mapping.

## Setup

```bash
conda env create -f environment.yml
conda activate argaze
pip install -e .
```

The DINOv3 encoder is loaded from Hugging Face:

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

If your Hugging Face account requires authentication for the model, set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`.

## Checkpoints

After the Hugging Face model repository is published, download checkpoints with:

```bash
python scripts/download_checkpoints.py \
  --repo <hf-org-or-user>/argaze \
  --output checkpoints
```

To inspect the expected layout without network access:

```bash
python scripts/download_checkpoints.py --dry-run
```

Expected local paths:

```text
checkpoints/egteagaze/best_rollout.pyth
checkpoints/ego4d/best_rollout.pyth
checkpoints/egoexo4d/iid/best_rollout.pyth
```

## Dataset Layout

Datasets must be obtained from the official EGTEA Gaze+, Ego4D, and EgoExo4D releases under their terms of use. This repository does not redistribute frames, videos, or private annotations.

Use the provided split files under `data/splits/` and override dataset roots at runtime:

```bash
python scripts/eval_shortclip.py \
  --dataset egteagaze \
  --checkpoint checkpoints/egteagaze/best_rollout.pyth \
  --data-root /path/to/egtea \
  --output-dir output/egtea_shortclip
```

For Ego4D:

```bash
python scripts/eval_shortclip.py \
  --dataset ego4d \
  --checkpoint checkpoints/ego4d/best_rollout.pyth \
  --data-root /path/to/ego4d/v2 \
  --output-dir output/ego4d_shortclip
```

For EgoExo4D:

```bash
python scripts/eval_shortclip.py \
  --dataset egoexo4d \
  --checkpoint checkpoints/egoexo4d/iid/best_rollout.pyth \
  --data-root /path/to/egoexo4d \
  --output-dir output/egoexo4d_shortclip
```

## Training

```bash
python scripts/train_argaze.py \
  --dataset egteagaze \
  --data-root /path/to/egtea \
  --output-dir output/egtea_train
```

Any SlowFast config key can be overridden after the script arguments:

```bash
python scripts/train_argaze.py \
  --dataset ego4d \
  --data-root /path/to/ego4d/v2 \
  TRAIN.BATCH_SIZE 4 SOLVER.MAX_EPOCH 1
```

## Long-Sequence Rollout

```bash
python scripts/eval_longseq_rollout.py \
  --dataset ego4d \
  --checkpoint checkpoints/ego4d/best_rollout.pyth \
  --data-root /path/to/ego4d/v2 \
  --output-dir output/ego4d_longseq
```

This writes `per_frame_metrics.csv` and then computes long-sequence metrics under `output/ego4d_longseq/longseq_metrics/`.

## Notes

- The public API is the scripts in `scripts/`. `tools/run_net.py` remains available for advanced SlowFast-style config overrides.
- W&B logging is disabled by default. Enable it explicitly through config overrides if needed.
- The release keeps the original `slowfast` package name for compatibility with existing checkpoints.
