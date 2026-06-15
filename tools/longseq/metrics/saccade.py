"""
tools/longseq/metrics/saccade.py

Saccade Transition Accuracy: predicted vs. ground-truth saccade onset timing (Req 3).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

from ..config import MetricConfig
from ..loader import FrameRecord, SequenceGroup, is_valid_frame
from . import MetricBundle, MetricResult

if TYPE_CHECKING:
    from ..dataset_schema import DatasetSchema


def compute_saccade_transition(
    groups: List[SequenceGroup],
    schema: DatasetSchema,
    config: MetricConfig,
) -> MetricBundle:
    """Compute Saccade Transition Accuracy metrics (Req 3).

    Algorithm:
        1. Identify ground-truth saccade onsets (first frame of a saccade run
           preceded by a valid fixation frame).
        2. Detect predicted saccade onsets based on frame-to-frame velocity.
        3. Merge predicted detections within ``saccade_merge_window``.
        4. Chronological 1-to-1 matching of GT and predicted onsets.
        5. Compute lag and report mean, median, percentage within 1f and 3f.

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
        Contains ``saccade_transition.mean_lag``, ``saccade_transition.median_lag``,
        ``saccade_transition.pct_within_1f``, ``saccade_transition.pct_within_3f``,
        and ``sample_counts["transition_events"]``.

        Returns an N/A bundle if ``schema.has_saccade_annotations()`` is False.
    """
    if not schema.has_saccade_annotations():
        return MetricBundle(
            family="saccade_transition",
            results=[
                MetricResult(
                    name="saccade_transition.mean_lag",
                    value=None,
                    na_reason="no_saccade_annotations",
                    unit="frames",
                ),
                MetricResult(
                    name="saccade_transition.median_lag",
                    value=None,
                    na_reason="no_saccade_annotations",
                    unit="frames",
                ),
                MetricResult(
                    name="saccade_transition.pct_within_1f",
                    value=None,
                    na_reason="no_saccade_annotations",
                    unit="%",
                ),
                MetricResult(
                    name="saccade_transition.pct_within_3f",
                    value=None,
                    na_reason="no_saccade_annotations",
                    unit="%",
                ),
            ],
            sample_counts={"transition_events": 0},
            warnings=["Dataset lacks saccade annotations."],
        )

    all_lags: List[int] = []
    total_gt_events = 0

    for group in groups:
        for sub_seq in group.consecutive_sub_sequences:
            if len(sub_seq) < 2:
                continue

            gt_onsets: List[FrameRecord] = []
            # 8.2 GT onset detection
            # find first frame of each maximal saccade run preceded by valid fixation
            prev_was_valid_fixation = False
            for record in sub_seq:
                is_valid = is_valid_frame(record, schema)
                if not is_valid:
                    prev_was_valid_fixation = False
                    continue

                is_saccade = schema.is_saccade(record)
                if is_saccade and prev_was_valid_fixation:
                    gt_onsets.append(record)
                    # Once we mark a GT onset, the rest of the saccade run 
                    # shouldn't be marked as onset again, so we need to set 
                    # prev_was_valid_fixation to False
                    prev_was_valid_fixation = False
                elif schema.is_fixation(record):
                    prev_was_valid_fixation = True
                else:
                    prev_was_valid_fixation = False

            if not gt_onsets:
                continue
            total_gt_events += len(gt_onsets)

            # 8.3 Predicted onset detection
            raw_pred_onsets: List[FrameRecord] = []
            for i in range(1, len(sub_seq)):
                prev = sub_seq[i - 1]
                curr = sub_seq[i]

                # Both need to be valid frames to compute velocity safely
                if not (is_valid_frame(prev, schema) and is_valid_frame(curr, schema)):
                    continue

                dx = curr.pred_x - prev.pred_x
                dy = curr.pred_y - prev.pred_y
                velocity = math.sqrt(dx * dx + dy * dy)

                if velocity > config.saccade_velocity_threshold:
                    raw_pred_onsets.append(curr)

            # Merge detections within saccade_merge_window frames (keep earliest)
            merged_pred_onsets: List[FrameRecord] = []
            for curr_pred in raw_pred_onsets:
                if not merged_pred_onsets:
                    merged_pred_onsets.append(curr_pred)
                else:
                    last_pred = merged_pred_onsets[-1]
                    if (curr_pred.frame_offset - last_pred.frame_offset) <= config.saccade_merge_window:
                        # Skip, it's merged into the earlier one
                        pass
                    else:
                        merged_pred_onsets.append(curr_pred)

            # 8.4 Chronological 1-to-1 matching
            consumed_preds = set()
            for gt in gt_onsets:
                best_pred: Optional[FrameRecord] = None

                for pred in merged_pred_onsets:
                    if pred.frame_offset in consumed_preds:
                        continue
                    
                    diff = abs(pred.frame_offset - gt.frame_offset)
                    if diff <= config.saccade_merge_window * 2:
                        # Keep earliest predicted onset within window? 
                        # Actually, requirements say: "find the earliest unmatched predicted onset within saccade_merge_window * 2 frames"
                        best_pred = pred
                        break
                
                if best_pred is not None:
                    consumed_preds.add(best_pred.frame_offset)
                    lag = best_pred.frame_offset - gt.frame_offset
                    all_lags.append(lag)

    if total_gt_events == 0:
        return MetricBundle(
            family="saccade_transition",
            results=[
                MetricResult(
                    name="saccade_transition.mean_lag",
                    value=None,
                    na_reason="no_ground_truth_saccades",
                    unit="frames",
                ),
                MetricResult(
                    name="saccade_transition.median_lag",
                    value=None,
                    na_reason="no_ground_truth_saccades",
                    unit="frames",
                ),
                MetricResult(
                    name="saccade_transition.pct_within_1f",
                    value=None,
                    na_reason="no_ground_truth_saccades",
                    unit="%",
                ),
                MetricResult(
                    name="saccade_transition.pct_within_3f",
                    value=None,
                    na_reason="no_ground_truth_saccades",
                    unit="%",
                ),
            ],
            sample_counts={"transition_events": 0},
            warnings=["No ground-truth saccade transitions were found."],
        )

    if not all_lags:
        return MetricBundle(
            family="saccade_transition",
            results=[
                MetricResult(
                    name="saccade_transition.mean_lag",
                    value=None,
                    na_reason="no_matched_saccades",
                    unit="frames",
                ),
                MetricResult(
                    name="saccade_transition.median_lag",
                    value=None,
                    na_reason="no_matched_saccades",
                    unit="frames",
                ),
                MetricResult(
                    name="saccade_transition.pct_within_1f",
                    value=None,
                    na_reason="no_matched_saccades",
                    unit="%",
                ),
                MetricResult(
                    name="saccade_transition.pct_within_3f",
                    value=None,
                    na_reason="no_matched_saccades",
                    unit="%",
                ),
            ],
            sample_counts={"transition_events": total_gt_events},
            warnings=["No ground-truth saccades were matched with predictions."],
        )

    # Compute statistics
    sorted_lags = sorted(all_lags)
    n = len(sorted_lags)

    mean_val = sum(sorted_lags) / n
    median_val = sorted_lags[n // 2] if n % 2 == 1 else (
        sorted_lags[n // 2 - 1] + sorted_lags[n // 2]
    ) / 2.0

    within_1f_count = sum(1 for lag in all_lags if abs(lag) <= 1)
    within_3f_count = sum(1 for lag in all_lags if abs(lag) <= 3)

    pct_within_1f = (within_1f_count / total_gt_events) * 100.0
    pct_within_3f = (within_3f_count / total_gt_events) * 100.0

    return MetricBundle(
        family="saccade_transition",
        results=[
            MetricResult(
                name="saccade_transition.mean_lag",
                value=mean_val,
                sample_count=n,
                unit="frames",
            ),
            MetricResult(
                name="saccade_transition.median_lag",
                value=median_val,
                sample_count=n,
                unit="frames",
            ),
            MetricResult(
                name="saccade_transition.pct_within_1f",
                value=pct_within_1f,
                sample_count=total_gt_events,
                unit="%",
            ),
            MetricResult(
                name="saccade_transition.pct_within_3f",
                value=pct_within_3f,
                sample_count=total_gt_events,
                unit="%",
            ),
        ],
        sample_counts={"transition_events": total_gt_events},
        warnings=[],
    )
