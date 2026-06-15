"""
tools/longseq/config.py

Central configuration dataclass for the long-sequence streaming evaluation
metrics system.  All threshold defaults are defined here and are included
verbatim in the output metadata / config snapshot (Req 8.3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MetricConfig:
    """All configuration for one Metric_Computer run.

    Attributes
    ----------
    dataset:
        One of ``"egtea"``, ``"ego4d"``, or ``"egoexo4d"`` (case-insensitive).
    model_name:
        Human-readable model identifier included in report metadata.
    checkpoint_path:
        Path to the model checkpoint; included in report metadata.
    input_csv_path:
        Path to the ``per_frame_metrics.csv`` file being evaluated.
    output_dir:
        Directory where output files are written.
    bootstrap_artifacts_dir:
        Path to the bootstrap artifact directory, or ``None`` to skip CI.
    n_bootstrap:
        Number of bootstrap resampling iterations (Req 10.2).
    bootstrap_seed:
        Random seed for reproducible bootstrap resampling.

    group_by:
        Sequence grouping strategy.  ``"clip"`` groups by
        ``(video_name, clip_index)`` (default, Req 9.3).

    min_fixation_length:
        Minimum number of frames for a fixation sub-sequence to be eligible
        for jitter / stability metrics (Req 1.3, Req 2.5).  Default: 3.
    drift_min_frames:
        Minimum number of valid tracked frames required for a sequence to be
        included in drift analysis (Req 4.1).  Default: 30.
    recovery_cap:
        Maximum number of frames to scan for recovery; if recovery is not
        found within this window, ``Recovery_Length`` is set to this value
        (Req 5.7).  Default: 20.
    pre_spike_baseline_window:
        Number of valid frames before a spike used to compute the baseline
        L2 error (Req 5.4).  Default: 10.
    spike_threshold_multiplier:
        A frame is a spike when its L2 error exceeds this multiplier times
        the sequence mean L2 (Req 5.2).  Default: 2.0.
    recovery_threshold_multiplier:
        Recovery is declared when L2 drops below this multiplier times the
        pre-spike baseline (Req 5.5).  Default: 1.2.
    saccade_velocity_threshold:
        Predicted transition onset threshold in normalised-coordinate L2
        distance per frame (Req 3.4).  Default: 0.05.
    saccade_merge_window:
        Window (in frames) for merging consecutive spike or transition
        detections into a single event (Req 3.5, Req 5.3).  Default: 5.

    enable_localization:
        Compute standard localisation metrics (Req 0).
    enable_jitter:
        Compute temporal jitter metrics (Req 1).
    enable_stability:
        Compute fixation stability score (Req 2).
    enable_saccade:
        Compute saccade transition accuracy (Req 3).
    enable_drift:
        Compute temporal drift metrics (Req 4).
    enable_recovery:
        Compute recovery-after-error metrics (Req 5).
    enable_efficiency:
        Compute streaming efficiency metrics (Req 6).
    enable_bootstrap:
        Attach bootstrap confidence intervals when artifacts are available.
    write_per_sequence_csv:
        Write ``longseq_metrics_per_sequence.csv`` (can be suppressed via
        ``--no-per-sequence-csv`` CLI flag).
    """

    # ------------------------------------------------------------------ #
    # Identity / provenance
    # ------------------------------------------------------------------ #
    dataset: str = ""
    model_name: str = ""
    checkpoint_path: str = ""
    input_csv_path: str = ""
    output_dir: str = "longseq_metrics_out"

    # ------------------------------------------------------------------ #
    # Bootstrap
    # ------------------------------------------------------------------ #
    bootstrap_artifacts_dir: Optional[str] = None
    n_bootstrap: int = 1000
    bootstrap_seed: int = 42

    # ------------------------------------------------------------------ #
    # Sequence grouping
    # ------------------------------------------------------------------ #
    group_by: str = "clip"  # "clip" → (video_name, clip_index)

    # ------------------------------------------------------------------ #
    # Thresholds and minimum lengths (all per requirements spec §Defaults)
    # ------------------------------------------------------------------ #
    min_fixation_length: int = 3
    drift_min_frames: int = 30
    recovery_cap: int = 20
    pre_spike_baseline_window: int = 10
    spike_threshold_multiplier: float = 2.0
    recovery_threshold_multiplier: float = 1.2
    saccade_velocity_threshold: float = 0.05
    saccade_merge_window: int = 5

    # ------------------------------------------------------------------ #
    # Feature flags
    # ------------------------------------------------------------------ #
    enable_localization: bool = True
    enable_jitter: bool = True
    enable_stability: bool = True
    enable_saccade: bool = True
    enable_drift: bool = True
    enable_recovery: bool = True
    enable_efficiency: bool = True
    enable_bootstrap: bool = True
    write_per_sequence_csv: bool = True

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def as_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of all config fields.

        Used to embed a config snapshot in the output report (Req 8.3).
        """
        return asdict(self)

    def __post_init__(self) -> None:
        # Normalise dataset name to lower-case for consistent comparisons.
        self.dataset = self.dataset.lower()

        valid_datasets = {"egtea", "ego4d", "egoexo4d", ""}
        if self.dataset not in valid_datasets:
            raise ValueError(
                f"Unknown dataset {self.dataset!r}. "
                f"Must be one of: {sorted(valid_datasets - {''})}."
            )

        if self.min_fixation_length < 1:
            raise ValueError("min_fixation_length must be >= 1.")
        if self.drift_min_frames < 1:
            raise ValueError("drift_min_frames must be >= 1.")
        if self.recovery_cap < 1:
            raise ValueError("recovery_cap must be >= 1.")
        if self.pre_spike_baseline_window < 1:
            raise ValueError("pre_spike_baseline_window must be >= 1.")
        if self.spike_threshold_multiplier <= 0:
            raise ValueError("spike_threshold_multiplier must be > 0.")
        if self.recovery_threshold_multiplier <= 0:
            raise ValueError("recovery_threshold_multiplier must be > 0.")
        if self.saccade_velocity_threshold <= 0:
            raise ValueError("saccade_velocity_threshold must be > 0.")
        if self.saccade_merge_window < 1:
            raise ValueError("saccade_merge_window must be >= 1.")
        if self.n_bootstrap < 1:
            raise ValueError("n_bootstrap must be >= 1.")
