"""
tools/longseq/metrics/jitter.py

Temporal jitter metrics: frame-to-frame variation in predicted gaze coordinates
during fixation periods (Req 1).

This module implements fixation run extraction: iterating over consecutive
sub-sequences and finding maximal contiguous runs where both
``schema.is_fixation(r)`` and ``is_valid_frame(r, schema)`` hold.

The fixation run extraction logic is centralised here and reused by the
stability module (Req 2) via the ``extract_fixation_runs`` helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from ..config import MetricConfig
from ..loader import FrameRecord, SequenceGroup, is_valid_frame
from . import MetricBundle, MetricResult

if TYPE_CHECKING:
    from ..dataset_schema import DatasetSchema


# ---------------------------------------------------------------------------
# Fixation run extraction (shared with stability.py)
# ---------------------------------------------------------------------------


@dataclass
class FixationRun:
    """A maximal contiguous run of fixation frames.

    Attributes:
        frames: List of FrameRecord objects forming the run, in temporal order.
        sequence_id: ID of the parent SequenceGroup.
    """

    frames: List[FrameRecord]
    sequence_id: str


def extract_fixation_runs(
    groups: List[SequenceGroup],
    schema: DatasetSchema,
    config: MetricConfig,
) -> List[FixationRun]:
    """Extract all eligible fixation runs from a list of sequence groups.

    This function iterates over ``consecutive_sub_sequences`` within each
    ``SequenceGroup`` and finds maximal contiguous runs where both:

    1. ``schema.is_fixation(r)`` is ``True``
    2. ``is_valid_frame(r, schema)`` is ``True``

    Runs shorter than ``config.min_fixation_length`` are discarded
    (Req 1.3, Req 2.5).

    Parameters
    ----------
    groups:
        List of :class:`SequenceGroup` objects from the CSV loader.
    schema:
        A :class:`DatasetSchema` instance for the current dataset.
    config:
        A :class:`MetricConfig` instance with ``min_fixation_length`` threshold.

    Returns
    -------
    List[FixationRun]
        All eligible fixation runs across all sequences.

    Raises
    ------
    NotImplementedError
        If ``schema.is_fixation()`` raises ``NotImplementedError`` (EgoExo4D).
        The caller should catch this and return an N/A bundle.
    """
    runs: List[FixationRun] = []

    for group in groups:
        # Iterate over consecutive sub-sequences (handles gaps in frame_offset)
        for sub_seq in group.consecutive_sub_sequences:
            # Find maximal contiguous fixation runs within this sub-sequence
            current_run: List[FrameRecord] = []

            for record in sub_seq:
                # Check both conditions (may raise NotImplementedError for EgoExo4D)
                is_fixation = schema.is_fixation(record)
                is_valid = is_valid_frame(record, schema)

                if is_fixation and is_valid:
                    # Extend the current run
                    current_run.append(record)
                else:
                    # End of a run - check if it meets minimum length
                    if len(current_run) >= config.min_fixation_length:
                        runs.append(
                            FixationRun(
                                frames=current_run,
                                sequence_id=group.sequence_id,
                            )
                        )
                    current_run = []

            # Don't forget the last run in the sub-sequence
            if len(current_run) >= config.min_fixation_length:
                runs.append(
                    FixationRun(
                        frames=current_run,
                        sequence_id=group.sequence_id,
                    )
                )

    return runs


# ---------------------------------------------------------------------------
# Temporal jitter computation
# ---------------------------------------------------------------------------


def _compute_frame_to_frame_distances(run: FixationRun) -> List[float]:
    """Compute frame-to-frame Euclidean distances for a fixation run.

    Parameters
    ----------
    run:
        A :class:`FixationRun` instance with at least 2 frames.

    Returns
    -------
    List[float]
        List of ``||pred_t - pred_{t-1}||_2`` values for consecutive frames.
    """
    distances: List[float] = []
    frames = run.frames

    for i in range(1, len(frames)):
        prev = frames[i - 1]
        curr = frames[i]

        dx = curr.pred_x - prev.pred_x
        dy = curr.pred_y - prev.pred_y
        dist = math.sqrt(dx * dx + dy * dy)
        distances.append(dist)

    return distances


def compute_temporal_jitter(
    groups: List[SequenceGroup],
    schema: DatasetSchema,
    config: MetricConfig,
) -> MetricBundle:
    """Compute temporal jitter metrics (Req 1).

    Algorithm:
        1. Extract fixation runs using :func:`extract_fixation_runs`
        2. For each eligible run, compute frame-to-frame distances
        3. Aggregate all distances and report mean, median, p95

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
        Contains ``temporal_jitter.mean``, ``temporal_jitter.median``,
        ``temporal_jitter.p95``, and ``sample_counts["fixation_subsequences"]``.

        Returns an N/A bundle if ``schema.is_fixation()`` raises
        ``NotImplementedError`` (EgoExo4D).
    """
    # Try to extract fixation runs - may raise NotImplementedError for EgoExo4D
    try:
        runs = extract_fixation_runs(groups, schema, config)
    except NotImplementedError as e:
        # EgoExo4D: jitter metrics are forbidden
        return MetricBundle(
            family="temporal_jitter",
            results=[
                MetricResult(
                    name="temporal_jitter.mean",
                    value=None,
                    na_reason="no_fixation_annotations",
                    unit="normalized",
                ),
                MetricResult(
                    name="temporal_jitter.median",
                    value=None,
                    na_reason="no_fixation_annotations",
                    unit="normalized",
                ),
                MetricResult(
                    name="temporal_jitter.p95",
                    value=None,
                    na_reason="no_fixation_annotations",
                    unit="normalized",
                ),
            ],
            sample_counts={"fixation_subsequences": 0},
            warnings=[str(e)],
        )

    # Collect all frame-to-frame distances across all runs
    all_distances: List[float] = []
    for run in runs:
        if len(run.frames) >= 2:
            distances = _compute_frame_to_frame_distances(run)
            all_distances.extend(distances)

    # Handle empty case
    if not all_distances:
        return MetricBundle(
            family="temporal_jitter",
            results=[
                MetricResult(
                    name="temporal_jitter.mean",
                    value=None,
                    na_reason="no_valid_fixation_runs",
                    unit="normalized",
                ),
                MetricResult(
                    name="temporal_jitter.median",
                    value=None,
                    na_reason="no_valid_fixation_runs",
                    unit="normalized",
                ),
                MetricResult(
                    name="temporal_jitter.p95",
                    value=None,
                    na_reason="no_valid_fixation_runs",
                    unit="normalized",
                ),
            ],
            sample_counts={"fixation_subsequences": len(runs)},
            warnings=["No valid fixation runs with >= 2 frames found."],
        )

    # Compute statistics
    sorted_distances = sorted(all_distances)
    n = len(sorted_distances)

    mean_val = sum(sorted_distances) / n
    median_val = sorted_distances[n // 2] if n % 2 == 1 else (
        sorted_distances[n // 2 - 1] + sorted_distances[n // 2]
    ) / 2.0

    # 95th percentile
    p95_idx = int(math.ceil(0.95 * n)) - 1
    p95_idx = max(0, min(p95_idx, n - 1))
    p95_val = sorted_distances[p95_idx]

    return MetricBundle(
        family="temporal_jitter",
        results=[
            MetricResult(
                name="temporal_jitter.mean",
                value=mean_val,
                sample_count=n,
                unit="normalized",
            ),
            MetricResult(
                name="temporal_jitter.median",
                value=median_val,
                sample_count=n,
                unit="normalized",
            ),
            MetricResult(
                name="temporal_jitter.p95",
                value=p95_val,
                sample_count=n,
                unit="normalized",
            ),
        ],
        sample_counts={"fixation_subsequences": len(runs)},
        warnings=[],
    )
