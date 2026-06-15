import math
from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from tools.longseq.config import MetricConfig
from tools.longseq.dataset_schema import Ego4DSchema
from tools.longseq.loader import FrameRecord, SequenceGroup
from tools.longseq.metrics.stability import compute_fixation_stability
from tools.longseq.test_jitter import sequence_group_strategy

@given(st.lists(sequence_group_strategy(), min_size=1, max_size=5))
@settings(max_examples=100)
def test_fss_non_negativity(groups):
    config = MetricConfig()
    schema = Ego4DSchema()
    
    bundle = compute_fixation_stability(groups, schema, config)
    
    for result in bundle.results:
        if result.value is not None:
            assert result.value >= 0.0, f"Expected non-negative FSS, got {result.value} for {result.name}"

