"""
tools/longseq/metrics/drift.py

Temporal Drift Metrics: error accumulation over long sequences (Req 4).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from ..config import MetricConfig
from ..loader import SequenceGroup, is_valid_frame
from . import MetricBundle, MetricResult

if TYPE_CHECKING:
    from ..dataset_schema import DatasetSchema


def _linear_regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the slope of the least-squares linear regression.

    Parameters
    ----------
    x:
        1D array of independent variable values (e.g. frame indices).
    y:
        1D array of dependent variable values (e.g. L2 error).

    Returns
    -------
    float
        The regression slope. Returns 0.0 if not enough distinct points.
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0

    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def compute_temporal_drift(
    groups: List[SequenceGroup],
    schema: DatasetSchema,
    config: MetricConfig,
) -> MetricBundle:
    """Compute Temporal Drift metrics (Req 4).

    Algorithm:
        1. Filter sequences with >= ``config.drift_min_frames`` valid frames.
        2. Collect (frame_index, L2) pairs for valid frames in each sequence.
        3. Divide into thirds (early, middle, late) and compute mean L2 for each.
        4. Fit linear regression over L2 vs. sequence frame index.
        5. Aggregate and report mean drift rate, pct positive, etc.

    Parameters
    ----------
    groups:
        List of :class:`SequenceGroup` objects from the CSV loader.
    schema:
        A :class:`DatasetSchema` instance for the current dataset.
    config:
        A :class:`MetricConfig` instance with thresholds.

    Returns
    -------
    MetricBundle
        Contains ``temporal_drift.mean_rate``, ``temporal_drift.median_rate``,
        ``temporal_drift.mean_late_minus_early``, ``temporal_drift.pct_positive``,
        and ``sample_counts["drift_eligible_sequences"]``.
    """
    drift_rates: List[float] = []
    late_minus_early: List[float] = []
    eligible_count = 0
    warnings: List[str] = []
    excluded_count = 0

    for group in groups:
        # Collect valid frames over the entire sequence
        valid_pairs: List[Tuple[int, float]] = []

        # Reconstruct full order from frame_offset
        frames = sorted(group.frames, key=lambda f: f.frame_offset)

        for i, record in enumerate(frames):
            if is_valid_frame(record, schema):
                valid_pairs.append((i, record.l2))

        if len(valid_pairs) < config.drift_min_frames:
            excluded_count += 1
            continue

        eligible_count += 1

        # Calculate linear regression slope
        x = np.array([p[0] for p in valid_pairs], dtype=float)
        y = np.array([p[1] for p in valid_pairs], dtype=float)
        rate = _linear_regression_slope(x, y)
        drift_rates.append(rate)

        # Split into thirds
        n_valid = len(valid_pairs)
        third_size = n_valid // 3

        if third_size > 0:
            early_l2 = sum(p[1] for p in valid_pairs[:third_size]) / third_size
            # Late third is from (n_valid - third_size) to n_valid
            late_slice = valid_pairs[-third_size:]
            late_l2 = sum(p[1] for p in late_slice) / third_size

            late_minus_early.append(late_l2 - early_l2)
        else:
            late_minus_early.append(0.0)

    if excluded_count > 0:
        warnings.append(
            f"Excluded {excluded_count} sequences with fewer than "
            f"{config.drift_min_frames} valid tracked frames."
        )

    if eligible_count == 0:
        return MetricBundle(
            family="temporal_drift",
            results=[
                MetricResult(
                    name="temporal_drift.mean_rate",
                    value=None,
                    na_reason="no_eligible_sequences",
                    unit="L2/frame",
                ),
                MetricResult(
                    name="temporal_drift.median_rate",
                    value=None,
                    na_reason="no_eligible_sequences",
                    unit="L2/frame",
                ),
                MetricResult(
                    name="temporal_drift.mean_late_minus_early",
                    value=None,
                    na_reason="no_eligible_sequences",
                    unit="L2",
                ),
                MetricResult(
                    name="temporal_drift.pct_positive",
                    value=None,
                    na_reason="no_eligible_sequences",
                    unit="%",
                ),
            ],
            sample_counts={"drift_eligible_sequences": 0},
            warnings=warnings,
        )

    # Compute statistics
    sorted_rates = sorted(drift_rates)
    mean_rate = sum(sorted_rates) / eligible_count
    median_rate = sorted_rates[eligible_count // 2] if eligible_count % 2 == 1 else (
        sorted_rates[eligible_count // 2 - 1] + sorted_rates[eligible_count // 2]
    ) / 2.0

    mean_late_minus_early = sum(late_minus_early) / eligible_count
    positive_count = sum(1 for r in drift_rates if r > 0)
    pct_positive = (positive_count / eligible_count) * 100.0

    return MetricBundle(
        family="temporal_drift",
        results=[
            MetricResult(
                name="temporal_drift.mean_rate",
                value=mean_rate,
                sample_count=eligible_count,
                unit="L2/frame",
            ),
            MetricResult(
                name="temporal_drift.median_rate",
                value=median_rate,
                sample_count=eligible_count,
                unit="L2/frame",
            ),
            MetricResult(
                name="temporal_drift.mean_late_minus_early",
                value=mean_late_minus_early,
                sample_count=eligible_count,
                unit="L2",
            ),
            MetricResult(
                name="temporal_drift.pct_positive",
                value=pct_positive,
                sample_count=eligible_count,
                unit="%",
            ),
        ],
        sample_counts={"drift_eligible_sequences": eligible_count},
        warnings=warnings,
    )
