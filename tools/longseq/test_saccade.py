from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from tools.longseq.config import MetricConfig
from tools.longseq.dataset_schema import Ego4DSchema, EgoExo4DSchema
from tools.longseq.loader import FrameRecord, SequenceGroup
from tools.longseq.metrics.saccade import compute_saccade_transition
from tools.longseq.test_jitter import sequence_group_strategy

@given(st.lists(sequence_group_strategy(), min_size=1, max_size=5))
@settings(max_examples=100)
def test_saccade_transition_lag_sign(groups):
    config = MetricConfig(saccade_merge_window=5)
    schema = Ego4DSchema()  # Has saccades
    
    bundle = compute_saccade_transition(groups, schema, config)
    
    for result in bundle.results:
        if result.value is not None:
            if "pct" not in result.name:
                # Lag can be negative or positive but shouldn't exceed window * 2
                max_lag = config.saccade_merge_window * 2
                assert abs(result.value) <= max_lag, f"Expected lag <= {max_lag}, got {result.value} for {result.name}"

def test_saccade_na_for_egoexo4d():
    config = MetricConfig()
    schema = EgoExo4DSchema() # No saccades
    
    # Just mock an empty group
    groups = []
    bundle = compute_saccade_transition(groups, schema, config)
    
    assert bundle.results[0].value is None
    assert bundle.results[0].na_reason == "no_saccade_annotations"


def test_saccade_percentages_use_all_gt_events():
    config = MetricConfig(saccade_velocity_threshold=0.05, saccade_merge_window=1)
    schema = Ego4DSchema()

    frames = [
        FrameRecord("vid", 0, 0, 0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, 0.5),
        FrameRecord("vid", 1, 0, 1, 1.0, 1.0, 1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1, 1, 0.5),
        FrameRecord("vid", 2, 0, 2, 1.0, 1.0, 1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1, 0, 0.5),
        FrameRecord("vid", 3, 0, 3, 1.0, 1.0, 1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1, 1, 0.5),
    ]
    group = SequenceGroup(
        key=("vid", 0),
        sequence_id="vid__clip0",
        frames=frames,
        has_gaps=False,
        consecutive_sub_sequences=[frames],
    )

    bundle = compute_saccade_transition([group], schema, config)
    results = {r.name: r for r in bundle.results}

    assert bundle.sample_counts["transition_events"] == 2
    assert results["saccade_transition.mean_lag"].value == 0.0
    assert results["saccade_transition.pct_within_1f"].value == 50.0
    assert results["saccade_transition.pct_within_3f"].value == 50.0
    assert results["saccade_transition.pct_within_1f"].sample_count == 2
