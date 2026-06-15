# ARGaze Checkpoints

## Model

ARGaze is an online egocentric gaze estimation model with a DINOv3 ViT-S/16 visual encoder and an autoregressive heatmap decoder. The released final model is registered as:

```text
DINOv3_TwoCrossHeatmapBiasEfficientARHeatmapGaze
```

The decoder separates history-biased memory retrieval from current-frame grounding through two cross-attention blocks.

## Intended Use

The checkpoints are intended for research reproduction on EGTEA Gaze+, Ego4D gaze, and EgoExo4D egocentric gaze benchmarks. They are not intended for safety-critical eye tracking, biometric identification, or deployment without dataset- and domain-specific validation.

## Training Data

The checkpoints were trained on the dataset-specific training splits described in the paper. This repository does not redistribute source videos or frames. Users must obtain datasets from their official sources.

## Checkpoint Files

See `checkpoints/manifest.json` for expected Hugging Face paths, SHA-256 hashes, and local destination paths.

## Limitations

- Performance depends on matching preprocessing, frame extraction, and split definitions.
- EgoExo4D uses benchmark split metadata included in this release; verify that the split file matches the paper version before reporting final numbers.
- The DINOv3 encoder is loaded from Hugging Face at runtime.
