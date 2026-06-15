import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tools.longseq.config import MetricConfig
from tools.longseq.dataset_schema import Ego4DSchema
from tools.longseq.loader import FrameRecord, SequenceGroup
from tools.longseq.metrics.drift import compute_temporal_drift, _linear_regression_slope

@given(st.lists(st.floats(min_value=0, max_value=1), min_size=30, max_size=100))
@settings(max_examples=100, deadline=None)
def test_drift_rate_sign_consistency(l2_values):
    # Sort L2 values to make them strictly increasing
    increasing_l2 = sorted(set(l2_values))
    if len(increasing_l2) < 30:
        return
        
    config = MetricConfig(drift_min_frames=30)
    schema = Ego4DSchema()
    
    # Create frames with increasing L2
    frames_inc = []
    for i, l2 in enumerate(increasing_l2):
        frames_inc.append(FrameRecord(
            video_name="test_vid",
            frame_idx=i,
            clip_index=1,
            frame_offset=i,
            f1=1.0, recall=1.0, precision=1.0,
            l2=l2, pred_x=0.5, pred_y=0.5, gt_x=0.5, gt_y=0.5,
            valid=1, gaze_type=0, threshold=0.5
        ))
        
    group_inc = SequenceGroup(
        key=("test_vid", 1),
        sequence_id="test_vid__clip1",
        frames=frames_inc,
        has_gaps=False,
        consecutive_sub_sequences=[frames_inc]
    )
    
    bundle_inc = compute_temporal_drift([group_inc], schema, config)
    
    mean_rate = bundle_inc.results[0].value
    assert mean_rate is not None
    assert mean_rate > 0, f"Expected positive drift rate for strictly increasing L2, got {mean_rate}"
    
    # Create frames with decreasing L2
    decreasing_l2 = list(reversed(increasing_l2))
    frames_dec = []
    for i, l2 in enumerate(decreasing_l2):
        frames_dec.append(FrameRecord(
            video_name="test_vid",
            frame_idx=i,
            clip_index=1,
            frame_offset=i,
            f1=1.0, recall=1.0, precision=1.0,
            l2=l2, pred_x=0.5, pred_y=0.5, gt_x=0.5, gt_y=0.5,
            valid=1, gaze_type=0, threshold=0.5
        ))
        
    group_dec = SequenceGroup(
        key=("test_vid", 1),
        sequence_id="test_vid__clip2",
        frames=frames_dec,
        has_gaps=False,
        consecutive_sub_sequences=[frames_dec]
    )
    
    bundle_dec = compute_temporal_drift([group_dec], schema, config)
    mean_rate_dec = bundle_dec.results[0].value
    assert mean_rate_dec is not None
    assert mean_rate_dec < 0, f"Expected negative drift rate for strictly decreasing L2, got {mean_rate_dec}"
