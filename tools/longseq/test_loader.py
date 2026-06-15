"""
Property-based tests for tools/longseq/loader.py.

Validates:
    - Sequence grouping completeness (Req 9, Design §Property-Based Testing #8)
    - Coordinate clipping invariant (Design §Property-Based Testing #1)
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from .dataset_schema import Ego4DSchema, get_schema
from .loader import (
    FrameRecord,
    LoaderStats,
    SequenceGroup,
    is_valid_frame,
    load_per_frame_csv,
)
from .config import MetricConfig
from .validator import validate_loader_stats


# ---------------------------------------------------------------------------
# Helpers for generating test data
# ---------------------------------------------------------------------------


def _write_test_csv(rows: List[dict], path: str) -> None:
    """Write test rows to a CSV file."""
    fieldnames = [
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
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_row(
    video_name: str = "video_0",
    frame_idx: int = 0,
    clip_index: int = 0,
    frame_offset: int = 0,
    f1: float = 0.5,
    recall: float = 0.5,
    precision: float = 0.5,
    l2: float = 0.1,
    pred_x: float = 0.5,
    pred_y: float = 0.5,
    gt_x: float = 0.5,
    gt_y: float = 0.5,
    valid: int = 1,
    gaze_type: int = 0,
    threshold: float = 0.5,
) -> dict:
    """Create a single CSV row with default values."""
    return {
        "video_name": video_name,
        "frame_idx": str(frame_idx),
        "clip_index": str(clip_index),
        "frame_offset": str(frame_offset),
        "f1": str(f1),
        "recall": str(recall),
        "precision": str(precision),
        "l2": str(l2),
        "pred_x": str(pred_x),
        "pred_y": str(pred_y),
        "gt_x": str(gt_x),
        "gt_y": str(gt_y),
        "valid": str(valid),
        "gaze_type": str(gaze_type),
        "threshold": str(threshold),
    }


# ---------------------------------------------------------------------------
# Property: Sequence grouping completeness
# ---------------------------------------------------------------------------


@st.composite
def csv_rows_strategy(draw):
    """Generate a list of CSV rows with consistent grouping.

    Ensures that:
    - video_name, clip_index, frame_offset form valid groups
    - frame_offset is monotonically increasing within each group
    """
    # Generate 1-5 sequences
    num_sequences = draw(st.integers(min_value=1, max_value=5))
    sequences = []

    for seq_idx in range(num_sequences):
        video_name = f"video_{seq_idx}"
        clip_index = draw(st.integers(min_value=0, max_value=10))
        # Each sequence has 3-20 frames
        num_frames = draw(st.integers(min_value=3, max_value=20))

        for frame_offset in range(num_frames):
            row = _make_row(
                video_name=video_name,
                frame_idx=frame_offset,
                clip_index=clip_index,
                frame_offset=frame_offset,
                pred_x=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
                pred_y=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
                gt_x=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
                gt_y=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
                valid=draw(st.integers(min_value=0, max_value=1)),
                gaze_type=draw(st.integers(min_value=0, max_value=2)),
            )
            sequences.append(row)

    return sequences


class TestSequenceGroupingCompleteness:
    """Validates: Requirements 9.3, 9.4 - Sequence grouping completeness."""

    @given(rows=csv_rows_strategy())
    @settings(max_examples=50, deadline=None)
    def test_total_frames_equals_csv_rows(self, rows: List[dict]):
        """Total frames across all SequenceGroup objects equals total rows in input CSV.

        **Validates: Requirements 9.3, 9.4**

        This property ensures no frames are dropped during grouping.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _write_test_csv(rows, tmp_path)
            schema = get_schema("ego4d")
            groups, stats = load_per_frame_csv(tmp_path, schema)

            # Count total frames across all groups
            total_frames_in_groups = sum(len(g.frames) for g in groups)

            # Must equal the number of input rows
            assert total_frames_in_groups == len(rows), (
                f"Frame count mismatch: {total_frames_in_groups} frames in groups "
                f"vs {len(rows)} rows in CSV"
            )

            # Stats should also match
            assert stats.total_frames == len(rows)

        finally:
            os.unlink(tmp_path)

    @given(rows=csv_rows_strategy())
    @settings(max_examples=50, deadline=None)
    def test_all_frames_accounted_in_groups(self, rows: List[dict]):
        """Every input row appears in exactly one SequenceGroup.

        **Validates: Requirements 9.3, 9.4**
        """
        if not rows:
            return

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _write_test_csv(rows, tmp_path)
            schema = get_schema("ego4d")
            groups, stats = load_per_frame_csv(tmp_path, schema)

            # Build a set of (video_name, clip_index, frame_offset) from input
            input_keys = set()
            for row in rows:
                key = (
                    row["video_name"],
                    int(row["clip_index"]),
                    int(row["frame_offset"]),
                )
                input_keys.add(key)

            # Build a set from loaded groups
            loaded_keys = set()
            for group in groups:
                for frame in group.frames:
                    key = (frame.video_name, frame.clip_index, frame.frame_offset)
                    loaded_keys.add(key)

            # Must be identical
            assert input_keys == loaded_keys, (
                f"Mismatch between input and loaded frames. "
                f"Missing: {input_keys - loaded_keys}, "
                f"Extra: {loaded_keys - input_keys}"
            )

        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Property: Coordinate clipping invariant
# ---------------------------------------------------------------------------


@st.composite
def coordinate_strategy(draw):
    """Generate a pred_x or pred_y value, including out-of-range values."""
    return draw(
        st.one_of(
            # Finite values in and out of [0, 1]
            st.floats(
                min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
            # Explicit NaN and infinity cases
            st.just(float("nan")),
            st.just(float("inf")),
            st.just(float("-inf")),
        )
    )


class TestCoordinateClippingInvariant:
    """Validates: Design §1 - Coordinate clipping invariant."""

    @given(pred_x=coordinate_strategy(), pred_y=coordinate_strategy())
    @settings(max_examples=100, deadline=None)
    def test_clipped_coordinates_in_bounds(self, pred_x: float, pred_y: float):
        """For any pred_x/y input, loaded values are always in [0, 1].

        **Validates: Design §1, Req 9.2**

        This property ensures that after loading, all prediction coordinates
        are valid normalized coordinates, even if the input was out-of-range.
        """
        rows = [
            _make_row(
                pred_x=pred_x,
                pred_y=pred_y,
                gt_x=0.5,
                gt_y=0.5,
            )
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _write_test_csv(rows, tmp_path)
            schema = get_schema("ego4d")
            groups, stats = load_per_frame_csv(tmp_path, schema)

            assert len(groups) == 1
            assert len(groups[0].frames) == 1

            frame = groups[0].frames[0]

            # If the input was finite, the output must be in [0, 1]
            if math.isfinite(pred_x):
                assert (
                    0.0 <= frame.pred_x <= 1.0
                ), f"pred_x {frame.pred_x} not in [0, 1] for input {pred_x}"

            if math.isfinite(pred_y):
                assert (
                    0.0 <= frame.pred_y <= 1.0
                ), f"pred_y {frame.pred_y} not in [0, 1] for input {pred_y}"

            # NaN inputs should remain NaN (not clipped)
            if math.isnan(pred_x):
                assert math.isnan(frame.pred_x)
            if math.isnan(pred_y):
                assert math.isnan(frame.pred_y)

        finally:
            os.unlink(tmp_path)

    def test_clipping_warning_counted(self):
        """Clipped coordinates are counted in LoaderStats."""
        # pred_x = 1.5 should be clipped to 1.0
        rows = [
            _make_row(pred_x=1.5, pred_y=0.5),  # clipped_x
            _make_row(pred_x=0.5, pred_y=-0.3),  # clipped_y
            _make_row(pred_x=2.0, pred_y=2.0),  # clipped_x and clipped_y
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _write_test_csv(rows, tmp_path)
            schema = get_schema("ego4d")
            groups, stats = load_per_frame_csv(tmp_path, schema)

            # 3 rows with 4 total clips (1 + 1 + 2)
            assert stats.clipped_coordinate_count == 4

        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Property: is_valid_frame correctness
# ---------------------------------------------------------------------------


class TestIsValidFrame:
    """Tests for the centralized is_valid_frame function."""

    def test_valid_frame_all_conditions_met(self):
        """A frame meeting all conditions is valid."""
        record = FrameRecord(
            video_name="test",
            frame_idx=0,
            clip_index=0,
            frame_offset=0,
            f1=0.5,
            recall=0.5,
            precision=0.5,
            l2=0.1,
            pred_x=0.5,
            pred_y=0.5,
            gt_x=0.5,
            gt_y=0.5,
            valid=1,
            gaze_type=0,  # Ego4D fixation
            threshold=0.5,
        )
        schema = Ego4DSchema()
        assert is_valid_frame(record, schema) is True

    def test_invalid_frame_valid_zero(self):
        """A frame with valid=0 is not valid."""
        record = FrameRecord(
            video_name="test",
            frame_idx=0,
            clip_index=0,
            frame_offset=0,
            f1=0.5,
            recall=0.5,
            precision=0.5,
            l2=0.1,
            pred_x=0.5,
            pred_y=0.5,
            gt_x=0.5,
            gt_y=0.5,
            valid=0,  # Invalid
            gaze_type=0,
            threshold=0.5,
        )
        schema = Ego4DSchema()
        assert is_valid_frame(record, schema) is False

    def test_invalid_frame_untracked(self):
        """A frame that is not tracked per schema is not valid."""
        record = FrameRecord(
            video_name="test",
            frame_idx=0,
            clip_index=0,
            frame_offset=0,
            f1=0.5,
            recall=0.5,
            precision=0.5,
            l2=0.1,
            pred_x=0.5,
            pred_y=0.5,
            gt_x=0.5,
            gt_y=0.5,
            valid=1,
            gaze_type=2,  # Ego4D out-of-bounds
            threshold=0.5,
        )
        schema = Ego4DSchema()
        assert is_valid_frame(record, schema) is False

    def test_invalid_frame_nan_pred(self):
        """A frame with NaN prediction is not valid."""
        record = FrameRecord(
            video_name="test",
            frame_idx=0,
            clip_index=0,
            frame_offset=0,
            f1=0.5,
            recall=0.5,
            precision=0.5,
            l2=0.1,
            pred_x=float("nan"),
            pred_y=0.5,
            gt_x=0.5,
            gt_y=0.5,
            valid=1,
            gaze_type=0,
            threshold=0.5,
        )
        schema = Ego4DSchema()
        assert is_valid_frame(record, schema) is False

    def test_invalid_frame_nan_gt(self):
        """A frame with NaN ground truth is not valid."""
        record = FrameRecord(
            video_name="test",
            frame_idx=0,
            clip_index=0,
            frame_offset=0,
            f1=0.5,
            recall=0.5,
            precision=0.5,
            l2=0.1,
            pred_x=0.5,
            pred_y=0.5,
            gt_x=float("nan"),
            gt_y=0.5,
            valid=1,
            gaze_type=0,
            threshold=0.5,
        )
        schema = Ego4DSchema()
        assert is_valid_frame(record, schema) is False


# ---------------------------------------------------------------------------
# Property: Gap detection
# ---------------------------------------------------------------------------


class TestGapDetection:
    """Tests for gap detection and consecutive_sub_sequences."""

    def test_no_gaps_consecutive_frames(self):
        """Consecutive frames produce a single sub-sequence."""
        rows = [
            _make_row(video_name="v1", clip_index=0, frame_offset=0),
            _make_row(video_name="v1", clip_index=0, frame_offset=1),
            _make_row(video_name="v1", clip_index=0, frame_offset=2),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _write_test_csv(rows, tmp_path)
            schema = get_schema("ego4d")
            groups, stats = load_per_frame_csv(tmp_path, schema)

            assert len(groups) == 1
            assert groups[0].has_gaps is False
            assert len(groups[0].consecutive_sub_sequences) == 1
            assert len(groups[0].consecutive_sub_sequences[0]) == 3
            assert stats.sequences_with_gaps == 0

        finally:
            os.unlink(tmp_path)

    def test_gap_detected(self):
        """Non-consecutive frames produce multiple sub-sequences."""
        rows = [
            _make_row(video_name="v1", clip_index=0, frame_offset=0),
            _make_row(video_name="v1", clip_index=0, frame_offset=1),
            _make_row(video_name="v1", clip_index=0, frame_offset=3),  # Gap at 2
            _make_row(video_name="v1", clip_index=0, frame_offset=4),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _write_test_csv(rows, tmp_path)
            schema = get_schema("ego4d")
            groups, stats = load_per_frame_csv(tmp_path, schema)

            assert len(groups) == 1
            assert groups[0].has_gaps is True
            assert len(groups[0].consecutive_sub_sequences) == 2
            assert len(groups[0].consecutive_sub_sequences[0]) == 2  # 0, 1
            assert len(groups[0].consecutive_sub_sequences[1]) == 2  # 3, 4
            assert stats.sequences_with_gaps == 1

        finally:
            os.unlink(tmp_path)

    def test_multiple_gaps(self):
        """Multiple gaps produce multiple sub-sequences."""
        rows = [
            _make_row(video_name="v1", clip_index=0, frame_offset=0),
            _make_row(video_name="v1", clip_index=0, frame_offset=2),  # Gap at 1
            _make_row(video_name="v1", clip_index=0, frame_offset=5),  # Gaps at 3, 4
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _write_test_csv(rows, tmp_path)
            schema = get_schema("ego4d")
            groups, stats = load_per_frame_csv(tmp_path, schema)

            assert len(groups) == 1
            assert groups[0].has_gaps is True
            assert len(groups[0].consecutive_sub_sequences) == 3
            assert stats.sequences_with_gaps == 1

        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Property: Missing prediction handling
# ---------------------------------------------------------------------------


class TestMissingPredictionHandling:
    """Tests for NaN/missing prediction handling."""

    def test_missing_prediction_counted(self):
        """Frames with NaN predictions are counted."""
        rows = [
            _make_row(pred_x=0.5, pred_y=0.5),  # Valid
            _make_row(pred_x=float("nan"), pred_y=0.5),  # Missing pred_x
            _make_row(pred_x=0.5, pred_y=float("nan")),  # Missing pred_y
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _write_test_csv(rows, tmp_path)
            schema = get_schema("ego4d")
            groups, stats = load_per_frame_csv(tmp_path, schema)

            assert stats.missing_prediction_count == 2
            assert stats.missing_prediction_fraction == 2 / 3
            assert stats.missing_prediction_valid_count == 2
            assert stats.missing_prediction_invalid_count == 0

        finally:
            os.unlink(tmp_path)

    def test_empty_string_as_missing(self):
        """Empty strings are treated as missing/NaN."""
        rows = [
            _make_row(pred_x="", pred_y=0.5),  # Empty pred_x
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _write_test_csv(rows, tmp_path)
            schema = get_schema("ego4d")
            groups, stats = load_per_frame_csv(tmp_path, schema)

            assert stats.missing_prediction_count == 1
            assert stats.missing_prediction_valid_count == 1

        finally:
            os.unlink(tmp_path)

    def test_invalid_missing_predictions_are_counted_separately(self):
        """Missing predictions on invalid frames are not counted as valid-frame loss."""
        rows = [
            _make_row(pred_x=float("nan"), pred_y=0.5, valid=0, gaze_type=0),
            _make_row(pred_x=0.5, pred_y=float("nan"), valid=1, gaze_type=2),
            _make_row(pred_x=float("nan"), pred_y=0.5, valid=1, gaze_type=0),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _write_test_csv(rows, tmp_path)
            schema = get_schema("ego4d")
            groups, stats = load_per_frame_csv(tmp_path, schema)

            assert stats.missing_prediction_count == 3
            assert stats.missing_prediction_valid_count == 1
            assert stats.missing_prediction_invalid_count == 2

        finally:
            os.unlink(tmp_path)

    def test_invalid_missing_predictions_do_not_trigger_high_risk_warning(self):
        stats = LoaderStats(
            total_frames=10,
            total_sequences=1,
            missing_prediction_count=8,
            missing_prediction_fraction=0.8,
            missing_prediction_valid_count=0,
            missing_prediction_valid_fraction=0.0,
            missing_prediction_invalid_count=8,
            missing_prediction_invalid_fraction=0.8,
        )
        config = MetricConfig(
            dataset="ego4d", input_csv_path="unused.csv", output_dir="unused"
        )

        warnings = validate_loader_stats(stats, config)

        assert not any(
            "WARNING:" in warning and "missing prediction" in warning
            for warning in warnings
        )
        assert any(
            "INFO:" in warning and "invalid frame" in warning for warning in warnings
        )

    def test_valid_missing_predictions_still_trigger_warning(self):
        stats = LoaderStats(
            total_frames=10,
            total_sequences=1,
            missing_prediction_count=1,
            missing_prediction_fraction=0.1,
            missing_prediction_valid_count=1,
            missing_prediction_valid_fraction=0.1,
            missing_prediction_invalid_count=0,
            missing_prediction_invalid_fraction=0.0,
        )
        config = MetricConfig(
            dataset="ego4d", input_csv_path="unused.csv", output_dir="unused"
        )

        warnings = validate_loader_stats(stats, config)

        assert any(
            "WARNING:" in warning and "otherwise-valid" in warning
            for warning in warnings
        )


# ---------------------------------------------------------------------------
# Property: Sequence ID format
# ---------------------------------------------------------------------------


class TestSequenceIdFormat:
    """Tests for sequence_id format."""

    def test_sequence_id_format(self):
        """Sequence IDs follow the format {video_name}__clip{clip_index}."""
        rows = [
            _make_row(video_name="my_video", clip_index=5, frame_offset=0),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _write_test_csv(rows, tmp_path)
            schema = get_schema("ego4d")
            groups, stats = load_per_frame_csv(tmp_path, schema)

            assert len(groups) == 1
            assert groups[0].sequence_id == "my_video__clip5"

        finally:
            os.unlink(tmp_path)
