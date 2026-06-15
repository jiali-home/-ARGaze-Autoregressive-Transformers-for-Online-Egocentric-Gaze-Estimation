"""
tools/longseq/metrics/__init__.py

Core metric data models shared across all metric modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Canonical metric key table
# ---------------------------------------------------------------------------
# All MetricResult.name values MUST use one of these keys.  No aliases or
# alternative spellings are permitted (see design doc §Data Models).
CANONICAL_KEYS: Dict[str, List[str]] = {
    "localization": [
        "localization.mean_f1",
        "localization.mean_precision",
        "localization.mean_recall",
        "localization.mean_l2",
        "localization.median_l2",
        "localization.mean_aae",
        "localization.median_aae",
    ],
    "temporal_jitter": [
        "temporal_jitter.mean",
        "temporal_jitter.median",
        "temporal_jitter.p95",
    ],
    "fixation_stability": [
        "fixation_stability.mean",
        "fixation_stability.median",
        "fixation_stability.p95",
    ],
    "saccade_transition": [
        "saccade_transition.mean_lag",
        "saccade_transition.median_lag",
        "saccade_transition.pct_within_1f",
        "saccade_transition.pct_within_3f",
    ],
    "temporal_drift": [
        "temporal_drift.mean_rate",
        "temporal_drift.median_rate",
        "temporal_drift.mean_late_minus_early",
        "temporal_drift.pct_positive",
    ],
    "recovery": [
        "recovery.mean_length",
        "recovery.median_length",
        "recovery.p95_length",
    ],
    "efficiency": [
        "efficiency.mean_inference_ms",
        "efficiency.fps",
        "efficiency.total_frames",
        "efficiency.peak_gpu_mb",
    ],
}

# Flat set for O(1) membership checks
_ALL_CANONICAL_KEYS: frozenset = frozenset(
    key for keys in CANONICAL_KEYS.values() for key in keys
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Result for a single scalar metric.

    Attributes:
        name:         Canonical snake_case key, e.g. ``"temporal_jitter.mean"``.
                      Must be a member of :data:`CANONICAL_KEYS`.
        value:        Computed scalar value, or ``None`` when the metric is N/A.
        ci_lower:     Lower bound of the 95 % bootstrap CI (``None`` if not computed).
        ci_upper:     Upper bound of the 95 % bootstrap CI (``None`` if not computed).
        sample_count: Number of samples (frames / sub-sequences / events) used.
        unit:         Human-readable unit string, e.g. ``"normalized"`` or ``"ms"``.
        na_reason:    Short reason string when ``value`` is ``None``,
                      e.g. ``"no_saccade_annotations"``.
    """

    name: str
    value: Optional[float]
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    sample_count: int = 0
    unit: str = ""
    na_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.name not in _ALL_CANONICAL_KEYS:
            raise ValueError(
                f"MetricResult.name {self.name!r} is not a canonical key. "
                f"Valid keys: {sorted(_ALL_CANONICAL_KEYS)}"
            )


@dataclass
class MetricBundle:
    """Container returned by each metric module.

    Attributes:
        family:        Metric family name, e.g. ``"temporal_jitter"``.
                       Must be a key in :data:`CANONICAL_KEYS`.
        results:       List of :class:`MetricResult` objects for this family.
        sample_counts: Counts of the primary sampling units used, e.g.
                       ``{"fixation_subsequences": 42}``.
        warnings:      Human-readable warning strings generated during computation.
    """

    family: str
    results: List[MetricResult] = field(default_factory=list)
    sample_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.family not in CANONICAL_KEYS:
            raise ValueError(
                f"MetricBundle.family {self.family!r} is not a recognised family. "
                f"Valid families: {sorted(CANONICAL_KEYS)}"
            )
