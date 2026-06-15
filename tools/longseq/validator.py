"""
tools/longseq/validator.py

Sanity checks and validation for the Metric_Computer (Req 13).

This module provides validation functions that run after CSV loading and
after metric computation.  All functions return warning strings and never
raise exceptions — the goal is to inform the user of potential data quality
issues without halting execution.

Functions
---------
validate_loader_stats(stats, config):
    Check loader statistics for data quality issues.
validate_metric_bundles(bundles):
    Check metric bundles for statistical reliability issues.
"""

from __future__ import annotations

from typing import List

from .config import MetricConfig
from .loader import LoaderStats
from .metrics import MetricBundle


def validate_loader_stats(stats: LoaderStats, config: MetricConfig) -> List[str]:
    """Validate loader statistics and return warning messages.

    This function checks for data quality issues after CSV loading (Req 13.1, 13.2).

    Parameters
    ----------
    stats:
        A :class:`LoaderStats` instance from the CSV loader.
    config:
        A :class:`MetricConfig` instance (currently unused, reserved for future thresholds).

    Returns
    -------
    List[str]
        A list of warning message strings.  Empty list if no warnings.

    Notes
    -----
    This function never raises exceptions.  It only returns warning strings.

    Warnings generated:
        - If ``missing_prediction_valid_fraction > 0.05`` (Req 13.1)
        - If ``sequences_with_gaps / total_sequences > 0.20`` (Req 13.2)
    """
    warnings: List[str] = []

    # Guard against division by zero
    if stats.total_frames == 0:
        warnings.append("WARNING: No frames were loaded from the CSV file.")
        return warnings

    if stats.total_sequences == 0:
        warnings.append("WARNING: No sequences were reconstructed from the CSV file.")
        return warnings

    # Check 1: Missing prediction fraction (Req 13.1)
    # Warn strongly only when otherwise-valid annotation frames lack predictions.
    if stats.missing_prediction_valid_fraction > 0.05:
        warnings.append(
            f"WARNING: {stats.missing_prediction_valid_count} otherwise-valid frame(s) "
            f"({stats.missing_prediction_valid_fraction:.1%} of all rows) have missing "
            f"prediction coordinates, which exceeds the 5% threshold. "
            f"This may affect metric reliability."
        )
    if stats.missing_prediction_invalid_count > 0:
        warnings.append(
            f"INFO: {stats.missing_prediction_invalid_count} invalid frame(s) "
            f"({stats.missing_prediction_invalid_fraction:.1%} of all rows) are without "
            f"prediction coordinates; these frames are excluded by validity logic."
        )

    # Check 2: Sequences with gaps (Req 13.2)
    # Warn if more than 20% of sequences have non-consecutive frame_offset
    gap_fraction = stats.sequences_with_gaps / stats.total_sequences
    if gap_fraction > 0.20:
        warnings.append(
            f"WARNING: {stats.sequences_with_gaps} sequences ({gap_fraction:.1%}) "
            f"have non-consecutive frame_offset values, which exceeds the 20% threshold. "
            f"Temporal metrics will be computed on split sub-sequences."
        )

    return warnings


def validate_metric_bundles(bundles: List[MetricBundle]) -> List[str]:
    """Validate metric bundles and return warning messages.

    This function checks for statistical reliability issues after metric
    computation (Req 13.3, 13.4, 13.5).

    Parameters
    ----------
    bundles:
        A list of :class:`MetricBundle` instances from metric computation.

    Returns
    -------
    List[str]
        A list of warning message strings.  Empty list if no warnings.

    Notes
    -----
    This function never raises exceptions.  It only returns warning strings.

    Warnings generated:
        - If ``fixation_subsequences < 10`` (Req 13.3)
        - If ``transition_events < 10`` (Req 13.4)
        - If ``drift_eligible_sequences < 10`` (Req 13.5)
    """
    warnings: List[str] = []

    # Guard against empty bundles
    if not bundles:
        return warnings

    # Extract sample counts from relevant bundles
    fixation_subsequences = 0
    transition_events = 0
    drift_eligible_sequences = 0

    for bundle in bundles:
        # Check for fixation-based metrics (jitter, stability)
        if bundle.family in ("temporal_jitter", "fixation_stability"):
            fixation_subsequences = max(
                fixation_subsequences,
                bundle.sample_counts.get("fixation_subsequences", 0),
            )

        # Check for saccade transition metrics
        if bundle.family == "saccade_transition":
            transition_events = bundle.sample_counts.get("transition_events", 0)

        # Check for drift metrics
        if bundle.family == "temporal_drift":
            drift_eligible_sequences = bundle.sample_counts.get(
                "drift_eligible_sequences", 0
            )

    # Check 1: Fixation sub-sequences (Req 13.3)
    # Warn if fewer than 10 fixation sub-sequences contribute to fixation metrics
    if fixation_subsequences < 10 and fixation_subsequences > 0:
        warnings.append(
            f"WARNING: Only {fixation_subsequences} fixation sub-sequence(s) "
            f"contributed to jitter/stability metrics, which is below the minimum of 10. "
            f"Results may not be statistically reliable."
        )

    # Check 2: Transition events (Req 13.4)
    # Warn if fewer than 10 transition events contribute to transition metrics
    if transition_events < 10 and transition_events > 0:
        warnings.append(
            f"WARNING: Only {transition_events} transition event(s) "
            f"contributed to saccade transition metrics, which is below the minimum of 10. "
            f"Results may not be statistically reliable."
        )

    # Check 3: Drift-eligible sequences (Req 13.5)
    # Warn if fewer than 10 drift-eligible sequences contribute to drift analysis
    if drift_eligible_sequences < 10 and drift_eligible_sequences > 0:
        warnings.append(
            f"WARNING: Only {drift_eligible_sequences} drift-eligible sequence(s) "
            f"contributed to drift analysis, which is below the minimum of 10. "
            f"Results may not be statistically reliable."
        )

    return warnings
