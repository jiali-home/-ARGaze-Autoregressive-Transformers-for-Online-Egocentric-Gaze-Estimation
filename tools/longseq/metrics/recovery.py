"""
tools/longseq/metrics/recovery.py

Recovery After Error Metrics (Req 5).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Tuple

from ..config import MetricConfig
from ..loader import FrameRecord, SequenceGroup, is_valid_frame
from . import MetricBundle, MetricResult

if TYPE_CHECKING:
    from ..dataset_schema import DatasetSchema


def compute_recovery_after_error(
    groups: List[SequenceGroup],
    schema: DatasetSchema,
    config: MetricConfig,
) -> MetricBundle:
    """Compute Recovery After Error metrics (Req 5).

    Algorithm:
        1. Compute per-sequence mean L2.
        2. Detect spike frames where l2 > config.spike_threshold_multiplier * seq_mean_l2.
        3. Merge consecutive spikes within saccade_merge_window (keep first).
        4. For each spike, calculate baseline L2 from preceding valid frames.
        5. Scan forward to find first frame where l2 < recovery_threshold_multiplier * baseline.
        6. Compute Recovery_Length, capping at config.recovery_cap.

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
        Contains recovery lengths and sample counts.
    """
    recovery_lengths: List[int] = []
    warnings: List[str] = []
    skipped_events = 0

    for group in groups:
        # Get all valid frames in sequence (sorted chronologically)
        valid_frames: List[FrameRecord] = []
        frames = sorted(group.frames, key=lambda f: f.frame_offset)

        for record in frames:
            if is_valid_frame(record, schema):
                valid_frames.append(record)

        if not valid_frames:
            continue

        seq_mean_l2 = sum(f.l2 for f in valid_frames) / len(valid_frames)
        spike_threshold = config.spike_threshold_multiplier * seq_mean_l2

        # 10.2 Detect spikes
        raw_spikes: List[int] = []  # indices in valid_frames array
        for i, record in enumerate(valid_frames):
            if record.l2 > spike_threshold:
                raw_spikes.append(i)

        # Merge consecutive spikes
        merged_spikes: List[int] = []
        for curr_idx in raw_spikes:
            if not merged_spikes:
                merged_spikes.append(curr_idx)
            else:
                last_idx = merged_spikes[-1]
                last_record = valid_frames[last_idx]
                curr_record = valid_frames[curr_idx]

                # Check frame_offset difference, not array index difference
                if (
                    curr_record.frame_offset - last_record.frame_offset
                ) <= config.saccade_merge_window:
                    pass  # merge into the last one
                else:
                    merged_spikes.append(curr_idx)

        # 10.3 For each spike event
        for spike_idx in merged_spikes:
            spike_record = valid_frames[spike_idx]

            # Collect up to pre_spike_baseline_window valid frames before the spike
            start_idx = max(0, spike_idx - config.pre_spike_baseline_window)
            baseline_frames = valid_frames[start_idx:spike_idx]

            if len(baseline_frames) < 3:
                skipped_events += 1
                continue

            baseline_l2 = sum(f.l2 for f in baseline_frames) / len(baseline_frames)
            recovery_threshold = config.recovery_threshold_multiplier * baseline_l2

            # 10.4 Scan forward
            recovery_len = config.recovery_cap

            # Scan forward through original frames (including invalid ones)
            # to find the recovery frame by frame_offset.
            # Wait, requirements say: "identify the recovery frame as the first subsequent frame where l2 < ..."
            # Is it valid tracked frames only? Requirements 5.5 say "first subsequent frame where ...".
            # It's safer to just scan valid frames. "valid tracked frames" is standard.

            for i in range(spike_idx + 1, len(valid_frames)):
                curr_record = valid_frames[i]

                # Length is number of frames, i.e. difference in frame_offset.
                # Degenerate duplicate offsets are not subsequent frames.
                dist = curr_record.frame_offset - spike_record.frame_offset
                if dist <= 0:
                    continue

                if dist > config.recovery_cap:
                    break

                if curr_record.l2 < recovery_threshold:
                    recovery_len = dist
                    break

            recovery_lengths.append(recovery_len)

    if skipped_events > 0:
        warnings.append(
            f"Skipped {skipped_events} spike events due to having fewer than "
            f"3 valid pre-spike baseline frames."
        )

    if not recovery_lengths:
        return MetricBundle(
            family="recovery",
            results=[
                MetricResult(
                    name="recovery.mean_length",
                    value=None,
                    na_reason="no_valid_spike_events",
                    unit="frames",
                ),
                MetricResult(
                    name="recovery.median_length",
                    value=None,
                    na_reason="no_valid_spike_events",
                    unit="frames",
                ),
                MetricResult(
                    name="recovery.p95_length",
                    value=None,
                    na_reason="no_valid_spike_events",
                    unit="frames",
                ),
            ],
            sample_counts={"spike_events": 0},
            warnings=warnings,
        )

    sorted_lens = sorted(recovery_lengths)
    n = len(sorted_lens)

    mean_val = sum(sorted_lens) / n
    median_val = (
        sorted_lens[n // 2]
        if n % 2 == 1
        else (sorted_lens[n // 2 - 1] + sorted_lens[n // 2]) / 2.0
    )

    p95_idx = int(math.ceil(0.95 * n)) - 1
    p95_idx = max(0, min(p95_idx, n - 1))
    p95_val = float(sorted_lens[p95_idx])

    return MetricBundle(
        family="recovery",
        results=[
            MetricResult(
                name="recovery.mean_length",
                value=mean_val,
                sample_count=n,
                unit="frames",
            ),
            MetricResult(
                name="recovery.median_length",
                value=median_val,
                sample_count=n,
                unit="frames",
            ),
            MetricResult(
                name="recovery.p95_length",
                value=p95_val,
                sample_count=n,
                unit="frames",
            ),
        ],
        sample_counts={"spike_events": n},
        warnings=warnings,
    )
