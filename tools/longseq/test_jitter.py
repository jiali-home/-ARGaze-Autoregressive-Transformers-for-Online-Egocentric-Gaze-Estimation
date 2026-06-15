import math
from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from tools.longseq.config import MetricConfig
from tools.longseq.dataset_schema import Ego4DSchema
from tools.longseq.loader import FrameRecord, SequenceGroup
from tools.longseq.metrics.jitter import (
    FixationRun,
    _compute_frame_to_frame_distances,
    compute_temporal_jitter,
    extract_fixation_runs,
)


@st.composite
def frame_record_strategy(draw, is_fixation_force=False):
    # Generates a synthetic FrameRecord
    valid = draw(st.integers(min_value=0, max_value=1))
    
    # For Ego4D, gaze_type=0 is fixation
    if is_fixation_force:
        gaze_type = 0
        valid = 1
    else:
        gaze_type = draw(st.integers(min_value=0, max_value=3))
        
    return FrameRecord(
        video_name="test_vid",
        frame_idx=draw(st.integers(min_value=0)),
        clip_index=1,
        frame_offset=draw(st.integers(min_value=0, max_value=1000)),
        f1=draw(st.floats(min_value=0, max_value=1)),
        recall=draw(st.floats(min_value=0, max_value=1)),
        precision=draw(st.floats(min_value=0, max_value=1)),
        l2=draw(st.floats(min_value=0, max_value=1)),
        pred_x=draw(st.floats(min_value=0, max_value=1)),
        pred_y=draw(st.floats(min_value=0, max_value=1)),
        gt_x=draw(st.floats(min_value=0, max_value=1)),
        gt_y=draw(st.floats(min_value=0, max_value=1)),
        valid=valid,
        gaze_type=gaze_type,
        threshold=0.5,
    )


@st.composite
def sequence_group_strategy(draw):
    frames = draw(st.lists(frame_record_strategy(), min_size=1, max_size=50))
    # Sort frames to simulate consecutive sub-sequences (simplified)
    frames = sorted(frames, key=lambda f: f.frame_offset)
    
    return SequenceGroup(
        key=("test_vid", 1),
        sequence_id="test_vid__clip1",
        frames=frames,
        has_gaps=False,
        consecutive_sub_sequences=[frames],
    )


@given(st.lists(sequence_group_strategy(), min_size=1, max_size=5))
@settings(max_examples=100)
def test_jitter_non_negativity(groups):
    config = MetricConfig()
    schema = Ego4DSchema()
    
    bundle = compute_temporal_jitter(groups, schema, config)
    
    for result in bundle.results:
        if result.value is not None:
            assert result.value >= 0.0, f"Expected non-negative jitter, got {result.value} for {result.name}"


@given(st.lists(sequence_group_strategy(), min_size=1, max_size=5), st.integers(min_value=2, max_value=10))
@settings(max_examples=100)
def test_minimum_length_filter(groups, min_fixation_length):
    config = MetricConfig(min_fixation_length=min_fixation_length)
    schema = Ego4DSchema()
    
    runs = extract_fixation_runs(groups, schema, config)
    for run in runs:
        assert len(run.frames) >= min_fixation_length, f"Found run with {len(run.frames)} frames, less than config min {min_fixation_length}"


