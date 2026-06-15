# Reproducibility

This release targets the core ARGaze paper results: final model training, short-clip evaluation, and long-sequence causal rollout on EGTEA Gaze+, Ego4D, and EgoExo4D.

## Main Artifacts

- Final configs: `configs/*/DINOV3_Full_Exp11_TwoCross_HeatmapBias_EfficientARHeatmap.yaml`
- Public wrappers: `scripts/train_argaze.py`, `scripts/eval_shortclip.py`, `scripts/eval_longseq_rollout.py`
- Checkpoint manifest: `checkpoints/manifest.json`
- Split files: `data/splits/` and `data/egoexo4d_splits/`

## Protocol Defaults

- Backbone: `facebook/dinov3-vits16-pretrain-lvd1689m`
- Encoder fine-tuning: last 12 DINOv3 blocks
- Input size: `224 x 224`
- Heatmap size: `96 x 96`
- History length: `5`
- Decoder layers: `3`
- Attention heads: `8`
- Coordinate loss weight: `2.0`

## Validation Checklist

1. Download checkpoints with `scripts/download_checkpoints.py`.
2. Run `eval_shortclip.py` on each dataset and compare `test_metrics_summary.json` against the paper table.
3. Run `eval_longseq_rollout.py` for streaming metrics and inspect `longseq_metrics_summary.json`.
4. Confirm no dataset paths are hard-coded into configs; pass roots with `--data-root` or SlowFast overrides.
