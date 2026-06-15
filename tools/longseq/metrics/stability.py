"""
tools/longseq/metrics/stability.py

Fixation Stability Score: per-fixation-run spatial spread (Req 2).
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING, List

from ..config import MetricConfig
from ..loader import SequenceGroup
from . import MetricBundle, MetricResult
from .jitter import FixationRun, extract_fixation_runs

if TYPE_CHECKING:
    from ..dataset_schema import DatasetSchema


def _compute_fss(run: FixationRun) -> float:
    """Compute Fixation Stability Score (FSS) for a single fixation run.

    FSS = sqrt(std(pred_x)^2 + std(pred_y)^2)

    Parameters
    ----------
    run:
        A :class:`FixationRun` instance with at least 2 frames.

    Returns
    -------
    float
        The computed FSS value, or 0.0 if the run has fewer than 2 frames.
    """
    frames = run.frames
    if len(frames) < 2:
        return 0.0

    pred_x = [f.pred_x for f in frames]
    pred_y = [f.pred_y for f in frames]

    std_x = statistics.stdev(pred_x)
    std_y = statistics.stdev(pred_y)

    return math.sqrt(std_x * std_x + std_y * std_y)


def compute_fixation_stability(
    groups: List[SequenceGroup],
    schema: DatasetSchema,
    config: MetricConfig,
) -> MetricBundle:
    """Compute Fixation Stability Score metrics (Req 2).

    Algorithm:
        1. Extract fixation runs using :func:`extract_fixation_runs`
        2. For each eligible run, compute FSS
        3. Aggregate FSS across all runs and report mean, median, p95

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
        Contains ``fixation_stability.mean``, ``fixation_stability.median``,
        ``fixation_stability.p95``, and ``sample_counts["fixation_subsequences"]``.

        Returns an N/A bundle if ``schema.is_fixation()`` raises
        ``NotImplementedError`` (EgoExo4D).
    """
    try:
        runs = extract_fixation_runs(groups, schema, config)
    except NotImplementedError as e:
        return MetricBundle(
            family="fixation_stability",
            results=[
                MetricResult(
                    name="fixation_stability.mean",
                    value=None,
                    na_reason="no_fixation_annotations",
                    unit="normalized",
                ),
                MetricResult(
                    name="fixation_stability.median",
                    value=None,
                    na_reason="no_fixation_annotations",
                    unit="normalized",
                ),
                MetricResult(
                    name="fixation_stability.p95",
                    value=None,
                    na_reason="no_fixation_annotations",
                    unit="normalized",
                ),
            ],
            sample_counts={"fixation_subsequences": 0},
            warnings=[str(e)],
        )

    all_fss: List[float] = []
    for run in runs:
        if len(run.frames) >= 2:
            fss = _compute_fss(run)
            all_fss.append(fss)

    if not all_fss:
        return MetricBundle(
            family="fixation_stability",
            results=[
                MetricResult(
                    name="fixation_stability.mean",
                    value=None,
                    na_reason="no_valid_fixation_runs",
                    unit="normalized",
                ),
                MetricResult(
                    name="fixation_stability.median",
                    value=None,
                    na_reason="no_valid_fixation_runs",
                    unit="normalized",
                ),
                MetricResult(
                    name="fixation_stability.p95",
                    value=None,
                    na_reason="no_valid_fixation_runs",
                    unit="normalized",
                ),
            ],
            sample_counts={"fixation_subsequences": len(runs)},
            warnings=["No valid fixation runs with >= 2 frames found."],
        )

    sorted_fss = sorted(all_fss)
    n = len(sorted_fss)

    mean_val = sum(sorted_fss) / n
    median_val = sorted_fss[n // 2] if n % 2 == 1 else (
        sorted_fss[n // 2 - 1] + sorted_fss[n // 2]
    ) / 2.0

    p95_idx = int(math.ceil(0.95 * n)) - 1
    p95_idx = max(0, min(p95_idx, n - 1))
    p95_val = sorted_fss[p95_idx]

    return MetricBundle(
        family="fixation_stability",
        results=[
            MetricResult(
                name="fixation_stability.mean",
                value=mean_val,
                sample_count=n,
                unit="normalized",
            ),
            MetricResult(
                name="fixation_stability.median",
                value=median_val,
                sample_count=n,
                unit="normalized",
            ),
            MetricResult(
                name="fixation_stability.p95",
                value=p95_val,
                sample_count=n,
                unit="normalized",
            ),
        ],
        sample_counts={"fixation_subsequences": len(runs)},
        warnings=[],
    )
