"""
tools/longseq/metrics/localization.py
Compute standard localisation metrics over all valid tracked frames (Req 0).
This module computes:
    - Mean framewise F1, Precision, and Recall
    - Mean and median L2 distance between predicted and ground-truth gaze
    - Mean and median approximate Average Angular Error (AAE) from CSV
      coordinates using the repository's 60-degree FOV virtual camera
All validity checks are delegated to ``loader.is_valid_frame()``; this module
does NOT re-implement the validity logic.
"""

from __future__ import annotations
import math
import statistics
from typing import List
from ..dataset_schema import DatasetSchema
from ..loader import FrameRecord, SequenceGroup, is_valid_frame
from . import MetricBundle, MetricResult
from ..config import MetricConfig


_FOV60_DISTANCE = 0.5 / math.tan(math.pi / 6)


def _approximate_aae_degrees(record: FrameRecord) -> float:
    """Approximate angular error from normalized CSV coordinates.

    This mirrors ``slowfast.utils.metrics.average_angle_error`` under a square
    virtual image with 60-degree FOV.  It is not camera-calibrated AAE.
    """
    pred_ray = (
        record.pred_y - 0.5,
        record.pred_x - 0.5,
        _FOV60_DISTANCE,
    )
    gt_ray = (
        record.gt_y - 0.5,
        record.gt_x - 0.5,
        _FOV60_DISTANCE,
    )
    cross = (
        pred_ray[1] * gt_ray[2] - pred_ray[2] * gt_ray[1],
        pred_ray[2] * gt_ray[0] - pred_ray[0] * gt_ray[2],
        pred_ray[0] * gt_ray[1] - pred_ray[1] * gt_ray[0],
    )
    cross_norm = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
    dot = pred_ray[0] * gt_ray[0] + pred_ray[1] * gt_ray[1] + pred_ray[2] * gt_ray[2]
    return math.degrees(math.atan2(cross_norm, dot))


def compute_localization(
    groups: List[SequenceGroup],
    schema: DatasetSchema,
    config: MetricConfig,
) -> MetricBundle:
    """Compute localisation metrics over all valid tracked frames.
    This function aggregates framewise F1, Precision, Recall, and L2 distance
    over all frames where ``is_valid_frame(record, schema)`` returns ``True``.
    AAE is approximated from CSV coordinates using the same 60-degree FOV
    virtual camera assumption used by ``slowfast.utils.metrics``.  This is
    not calibration-based AAE.
    Parameters
    ----------
    groups:
        List of :class:`SequenceGroup` objects from the CSV loader.
    schema:
        A :class:`DatasetSchema` instance for the current dataset.
    Returns
    -------
    MetricBundle
        A bundle containing:
        - ``localization.mean_f1``
        - ``localization.mean_precision``
        - ``localization.mean_recall``
        - ``localization.mean_l2``
        - ``localization.median_l2``
        - ``localization.mean_aae`` (fov60_approx)
        - ``localization.median_aae`` (fov60_approx)
        - ``sample_counts["valid_frames"]``
    """
    # Collect values from all valid frames
    f1_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    l2_values: List[float] = []
    aae_values: List[float] = []
    for group in groups:
        for record in group.frames:
            if is_valid_frame(record, schema):
                # Collect F1, precision, recall from the CSV columns
                if math.isfinite(record.f1):
                    f1_values.append(record.f1)
                if math.isfinite(record.precision):
                    precision_values.append(record.precision)
                if math.isfinite(record.recall):
                    recall_values.append(record.recall)
                # Compute L2 distance from pred_x/y vs gt_x/y
                # (coordinates are already clipped to [0, 1] by the loader)
                l2_dist = math.sqrt(
                    (record.pred_x - record.gt_x) ** 2
                    + (record.pred_y - record.gt_y) ** 2
                )
                l2_values.append(l2_dist)
                aae_values.append(_approximate_aae_degrees(record))
    valid_frame_count = len(l2_values)  # Number of frames with valid L2
    # Compute aggregates
    results: List[MetricResult] = []
    # Mean F1, Precision, Recall
    if f1_values:
        mean_f1 = statistics.mean(f1_values)
        results.append(
            MetricResult(
                name="localization.mean_f1",
                value=mean_f1,
                sample_count=len(f1_values),
                unit="",
            )
        )
    else:
        results.append(
            MetricResult(
                name="localization.mean_f1",
                value=None,
                sample_count=0,
                unit="",
                na_reason="no_valid_frames",
            )
        )
    if precision_values:
        mean_precision = statistics.mean(precision_values)
        results.append(
            MetricResult(
                name="localization.mean_precision",
                value=mean_precision,
                sample_count=len(precision_values),
                unit="",
            )
        )
    else:
        results.append(
            MetricResult(
                name="localization.mean_precision",
                value=None,
                sample_count=0,
                unit="",
                na_reason="no_valid_frames",
            )
        )
    if recall_values:
        mean_recall = statistics.mean(recall_values)
        results.append(
            MetricResult(
                name="localization.mean_recall",
                value=mean_recall,
                sample_count=len(recall_values),
                unit="",
            )
        )
    else:
        results.append(
            MetricResult(
                name="localization.mean_recall",
                value=None,
                sample_count=0,
                unit="",
                na_reason="no_valid_frames",
            )
        )
    # Mean and median L2
    if l2_values:
        mean_l2 = statistics.mean(l2_values)
        median_l2 = statistics.median(l2_values)
        results.append(
            MetricResult(
                name="localization.mean_l2",
                value=mean_l2,
                sample_count=valid_frame_count,
                unit="normalized",
            )
        )
        results.append(
            MetricResult(
                name="localization.median_l2",
                value=median_l2,
                sample_count=valid_frame_count,
                unit="normalized",
            )
        )
    else:
        results.append(
            MetricResult(
                name="localization.mean_l2",
                value=None,
                sample_count=0,
                unit="normalized",
                na_reason="no_valid_frames",
            )
        )
        results.append(
            MetricResult(
                name="localization.median_l2",
                value=None,
                sample_count=0,
                unit="normalized",
                na_reason="no_valid_frames",
            )
        )
    # AAE is a CSV-coordinate approximation, not calibration-based AAE.
    if aae_values:
        results.append(
            MetricResult(
                name="localization.mean_aae",
                value=statistics.mean(aae_values),
                sample_count=len(aae_values),
                unit="degrees_fov60_approx",
            )
        )
        results.append(
            MetricResult(
                name="localization.median_aae",
                value=statistics.median(aae_values),
                sample_count=len(aae_values),
                unit="degrees_fov60_approx",
            )
        )
    else:
        results.append(
            MetricResult(
                name="localization.mean_aae",
                value=None,
                sample_count=0,
                unit="degrees_fov60_approx",
                na_reason="no_valid_frames",
            )
        )
        results.append(
            MetricResult(
                name="localization.median_aae",
                value=None,
                sample_count=0,
                unit="degrees_fov60_approx",
                na_reason="no_valid_frames",
            )
        )
    return MetricBundle(
        family="localization",
        results=results,
        sample_counts={"valid_frames": valid_frame_count},
        warnings=[
            "INFO: localization.mean_aae and localization.median_aae are fov60_approx "
            "values from CSV coordinates, not calibration-based angular error."
        ],
    )
