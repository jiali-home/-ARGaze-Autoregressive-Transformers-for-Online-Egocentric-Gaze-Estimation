from hypothesis import given, settings
from hypothesis import strategies as st

from tools.longseq.config import MetricConfig
from tools.longseq.dataset_schema import Ego4DSchema
from tools.longseq.loader import FrameRecord, SequenceGroup
from tools.longseq.metrics.recovery import compute_recovery_after_error
from tools.longseq.test_jitter import sequence_group_strategy

@given(st.lists(sequence_group_strategy(), min_size=1, max_size=5))
@settings(max_examples=100)
def test_recovery_length_bounds(groups):
    config = MetricConfig(recovery_cap=20, spike_threshold_multiplier=2.0)
    schema = Ego4DSchema()
    
    bundle = compute_recovery_after_error(groups, schema, config)
    
    for result in bundle.results:
        if result.value is not None:
            # Result could be float (mean, median, p95)
            val = result.value
            assert 1 <= val <= config.recovery_cap, f"Expected 1 <= recovery_length <= {config.recovery_cap}, got {val}"

