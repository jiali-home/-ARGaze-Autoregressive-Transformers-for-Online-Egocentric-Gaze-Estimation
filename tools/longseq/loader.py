"""
tools/longseq/loader.py

CSV parsing, sequence reconstruction, and centralised valid-frame logic.

This module is the sole authority for deciding whether a frame participates
in any metric computation.  All metric modules must call ``is_valid_frame()``
rather than re-implementing the validity check.

Responsibilities (Req 9, Design §1):
    - Parse ``per_frame_metrics.csv`` with ``csv.DictReader``
    - Validate required columns (exit with descriptive error if missing)
    - Clip ``pred_x``, ``pred_y`` to ``[0, 1]`` and log warnings for out-of-range
    - Group rows by ``(video_name, clip_index)`` into ``SequenceGroup`` objects
    - Sort each group by ``frame_offset`` ascending
    - Detect and log non-consecutive ``frame_offset`` gaps
    - Count and log NaN/missing prediction values, split by otherwise-valid
      vs invalid annotation frames
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .dataset_schema import DatasetSchema


# ---------------------------------------------------------------------------
# Required CSV columns (Req 9.1)
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "video_name",
    "frame_idx",
    "clip_index",
    "frame_offset",
    "f1",
    "recall",
    "precision",
    "l2",
    "pred_x",
    "pred_y",
    "gt_x",
    "gt_y",
    "valid",
    "gaze_type",
    "threshold",
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FrameRecord:
    """One row from ``per_frame_metrics.csv`` (Req 9.1).

    Mirrors the 15-column CSV schema.  All numeric fields are stored as
    Python floats/ints; missing values are represented as ``float("nan")``.
    """

    video_name: str
    frame_idx: int
    clip_index: int
    frame_offset: int
    f1: float
    recall: float
    precision: float
    l2: float
    pred_x: float
    pred_y: float
    gt_x: float
    gt_y: float
    valid: int  # 1 = valid tracked frame
    gaze_type: int  # dataset-specific encoding
    threshold: float


@dataclass
class SequenceGroup:
    """A reconstructed ordered frame sequence (Design §Data Models).

    Attributes:
        key: ``(video_name, clip_index)`` tuple identifying this sequence.
        sequence_id: Human-readable identifier ``"{video_name}__clip{clip_index}"``.
        frames: List of :class:`FrameRecord` sorted by ``frame_offset`` ascending.
        has_gaps: ``True`` if ``frame_offset`` values are non-consecutive.
        consecutive_sub_sequences: Maximal consecutive runs of frames.
    """

    key: Tuple[str, int]
    sequence_id: str
    frames: List[FrameRecord] = field(default_factory=list)
    has_gaps: bool = False
    consecutive_sub_sequences: List[List[FrameRecord]] = field(default_factory=list)


@dataclass
class LoaderStats:
    """Statistics from the CSV loading process.

    Attributes:
        total_frames: Total number of rows parsed from the CSV.
        missing_prediction_count: Number of frames with NaN/missing predictions.
        missing_prediction_fraction: ``missing_prediction_count / total_frames``.
        missing_prediction_valid_count: Number of otherwise-valid frames with
            NaN/missing predictions.
        missing_prediction_invalid_count: Number of invalid annotation frames
            with NaN/missing predictions.
        sequences_with_gaps: Number of sequences with non-consecutive ``frame_offset``.
        total_sequences: Total number of ``SequenceGroup`` objects created.
        clipped_coordinate_count: Number of pred_x/pred_y values clipped to ``[0, 1]``.
    """

    total_frames: int = 0
    missing_prediction_count: int = 0
    missing_prediction_fraction: float = 0.0
    missing_prediction_valid_count: int = 0
    missing_prediction_valid_fraction: float = 0.0
    missing_prediction_invalid_count: int = 0
    missing_prediction_invalid_fraction: float = 0.0
    sequences_with_gaps: int = 0
    total_sequences: int = 0
    clipped_coordinate_count: int = 0


# ---------------------------------------------------------------------------
# Centralised valid-frame logic
# ---------------------------------------------------------------------------


def is_valid_frame(record: FrameRecord, schema: DatasetSchema) -> bool:
    """Return ``True`` if the frame is valid for metric computation.

    This function is the **sole authority** for frame validity.  All metric
    modules must call this function rather than re-implementing the check.

    A frame is valid when **all** of the following hold (Design §1):

    1. ``record.valid == 1``
    2. ``schema.is_valid_tracked(record)`` is ``True``
    3. ``math.isfinite(record.pred_x)`` and ``math.isfinite(record.pred_y)``
    4. ``math.isfinite(record.gt_x)`` and ``math.isfinite(record.gt_y)``

    Parameters
    ----------
    record:
        A :class:`FrameRecord` instance.
    schema:
        A :class:`DatasetSchema` instance for the current dataset.

    Returns
    -------
    bool
        ``True`` if the frame should participate in metric computation.
    """
    # Condition 1: valid flag must be 1
    if record.valid != 1:
        return False

    # Condition 2: dataset-specific tracked check
    if not schema.is_valid_tracked(record):
        return False

    # Condition 3: finite predicted coordinates
    if not (math.isfinite(record.pred_x) and math.isfinite(record.pred_y)):
        return False

    # Condition 4: finite ground-truth coordinates
    if not (math.isfinite(record.gt_x) and math.isfinite(record.gt_y)):
        return False

    return True


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _parse_float(value: str, default: float = float("nan")) -> float:
    """Parse a string to float, returning ``default`` on failure.

    Handles empty strings, ``"nan"``, ``"NaN"``, and other non-numeric values.
    """
    if not value or value.strip() == "":
        return default
    try:
        result = float(value)
        return result
    except ValueError:
        return default


def _parse_int(value: str, default: int = 0) -> int:
    """Parse a string to int, returning ``default`` on failure."""
    if not value or value.strip() == "":
        return default
    try:
        return int(float(value))  # Handle "1.0" style floats
    except ValueError:
        return default


def _clip_coordinate(value: float) -> Tuple[float, bool]:
    """Clip a coordinate to ``[0, 1]``.

    Returns:
        A tuple ``(clipped_value, was_clipped)`` where ``was_clipped`` is
        ``True`` if the value was outside ``[0, 1]``.
    """
    if not math.isfinite(value):
        return value, False
    if value < 0.0:
        return 0.0, True
    if value > 1.0:
        return 1.0, True
    return value, False


def _has_valid_annotation(record: FrameRecord, schema: DatasetSchema) -> bool:
    """Return True when a frame is valid apart from prediction coordinates."""
    return (
        record.valid == 1
        and schema.is_valid_tracked(record)
        and math.isfinite(record.gt_x)
        and math.isfinite(record.gt_y)
    )


def _validate_columns(fieldnames: List[str]) -> None:
    """Validate that all required columns are present.

    Exits with a descriptive error message if any required column is missing
    (Req 9.6).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
    if missing:
        print(
            f"ERROR: Missing required columns in CSV: {missing}",
            file=sys.stderr,
        )
        print(
            f"Required columns are: {list(REQUIRED_COLUMNS)}",
            file=sys.stderr,
        )
        sys.exit(2)


def _detect_gaps(frames: List[FrameRecord]) -> Tuple[bool, List[List[FrameRecord]]]:
    """Detect non-consecutive ``frame_offset`` values and split into sub-sequences.

    Parameters
    ----------
    frames:
        List of :class:`FrameRecord` sorted by ``frame_offset`` ascending.

    Returns
    -------
    Tuple[bool, List[List[FrameRecord]]]
        A tuple ``(has_gaps, consecutive_sub_sequences)`` where:
        - ``has_gaps`` is ``True`` if any non-consecutive ``frame_offset`` was found.
        - ``consecutive_sub_sequences`` is a list of maximal consecutive runs.
    """
    if not frames:
        return False, []

    sub_sequences: List[List[FrameRecord]] = []
    current: List[FrameRecord] = [frames[0]]

    for i in range(1, len(frames)):
        prev_offset = frames[i - 1].frame_offset
        curr_offset = frames[i].frame_offset

        # Check if consecutive (current should be prev + 1)
        if curr_offset == prev_offset + 1:
            current.append(frames[i])
        else:
            # Gap detected - start a new sub-sequence
            sub_sequences.append(current)
            current = [frames[i]]

    # Don't forget the last sub-sequence
    if current:
        sub_sequences.append(current)

    has_gaps = len(sub_sequences) > 1
    return has_gaps, sub_sequences


def load_per_frame_csv(
    csv_path: str,
    schema: DatasetSchema,
) -> Tuple[List[SequenceGroup], LoaderStats]:
    """Load and parse a ``per_frame_metrics.csv`` file.

    This function:
    1. Parses the CSV with ``csv.DictReader``
    2. Validates required columns (exits on missing)
    3. Clips ``pred_x``/``pred_y`` to ``[0, 1]`` with warning count
    4. Handles NaN/missing values
    5. Groups rows by ``(video_name, clip_index)``
    6. Sorts each group by ``frame_offset`` ascending
    7. Detects gaps and builds ``consecutive_sub_sequences``

    Parameters
    ----------
    csv_path:
        Path to the ``per_frame_metrics.csv`` file.
    schema:
        A :class:`DatasetSchema` instance for the current dataset.

    Returns
    -------
    Tuple[List[SequenceGroup], LoaderStats]
        A tuple ``(groups, stats)`` where:
        - ``groups`` is a list of :class:`SequenceGroup` objects.
        - ``stats`` contains loading statistics.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    stats = LoaderStats()
    groups_dict: dict[Tuple[str, int], List[FrameRecord]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate required columns (Req 9.6)
        _validate_columns(reader.fieldnames or [])

        for row in reader:
            stats.total_frames += 1

            # Parse coordinates with clipping (Req 9.2)
            pred_x = _parse_float(row["pred_x"])
            pred_y = _parse_float(row["pred_y"])

            # Clip predictions to [0, 1] (Design §1)
            pred_x, clipped_x = _clip_coordinate(pred_x)
            pred_y, clipped_y = _clip_coordinate(pred_y)
            if clipped_x:
                stats.clipped_coordinate_count += 1
            if clipped_y:
                stats.clipped_coordinate_count += 1

            # Parse other fields
            gt_x = _parse_float(row["gt_x"])
            gt_y = _parse_float(row["gt_y"])

            # Build the FrameRecord
            record = FrameRecord(
                video_name=row["video_name"],
                frame_idx=_parse_int(row["frame_idx"]),
                clip_index=_parse_int(row["clip_index"]),
                frame_offset=_parse_int(row["frame_offset"]),
                f1=_parse_float(row["f1"]),
                recall=_parse_float(row["recall"]),
                precision=_parse_float(row["precision"]),
                l2=_parse_float(row["l2"]),
                pred_x=pred_x,
                pred_y=pred_y,
                gt_x=gt_x,
                gt_y=gt_y,
                valid=_parse_int(row["valid"]),
                gaze_type=_parse_int(row["gaze_type"]),
                threshold=_parse_float(row["threshold"]),
            )

            # Check for missing predictions (NaN pred_x or pred_y).  Split
            # otherwise-valid annotation frames from frames that metrics would
            # already exclude due to GT validity or dataset-specific status.
            if not math.isfinite(record.pred_x) or not math.isfinite(record.pred_y):
                stats.missing_prediction_count += 1
                if _has_valid_annotation(record, schema):
                    stats.missing_prediction_valid_count += 1
                else:
                    stats.missing_prediction_invalid_count += 1

            # Group by (video_name, clip_index)
            key = (record.video_name, record.clip_index)
            if key not in groups_dict:
                groups_dict[key] = []
            groups_dict[key].append(record)

    # Compute missing prediction fraction
    if stats.total_frames > 0:
        stats.missing_prediction_fraction = (
            stats.missing_prediction_count / stats.total_frames
        )
        stats.missing_prediction_valid_fraction = (
            stats.missing_prediction_valid_count / stats.total_frames
        )
        stats.missing_prediction_invalid_fraction = (
            stats.missing_prediction_invalid_count / stats.total_frames
        )

    # Build SequenceGroup objects with gap detection
    groups: List[SequenceGroup] = []
    for key, frames in groups_dict.items():
        video_name, clip_index = key

        # Sort by frame_offset ascending (Req 9.4)
        frames.sort(key=lambda r: r.frame_offset)

        # Detect gaps and build consecutive sub-sequences (Req 9.5)
        has_gaps, consecutive_sub_sequences = _detect_gaps(frames)

        sequence_id = f"{video_name}__clip{clip_index}"
        group = SequenceGroup(
            key=key,
            sequence_id=sequence_id,
            frames=frames,
            has_gaps=has_gaps,
            consecutive_sub_sequences=consecutive_sub_sequences,
        )
        groups.append(group)

        if has_gaps:
            stats.sequences_with_gaps += 1

    stats.total_sequences = len(groups)

    # Log warnings for clipping
    if stats.clipped_coordinate_count > 0:
        print(
            f"WARNING: Clipped {stats.clipped_coordinate_count} coordinate(s) to [0, 1]",
            file=sys.stderr,
        )

    # Log warnings for gaps
    if stats.sequences_with_gaps > 0:
        print(
            f"WARNING: {stats.sequences_with_gaps} sequence(s) have non-consecutive frame_offset",
            file=sys.stderr,
        )

    return groups, stats
