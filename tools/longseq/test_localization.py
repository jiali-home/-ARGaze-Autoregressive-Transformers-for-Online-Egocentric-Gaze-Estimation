import math

from tools.longseq.config import MetricConfig
from tools.longseq.dataset_schema import Ego4DSchema
from tools.longseq.loader import FrameRecord, SequenceGroup
from tools.longseq.metrics.localization import compute_localization


def _record(
    pred_x=0.5,
    pred_y=0.5,
    gt_x=0.5,
    gt_y=0.5,
    valid=1,
    gaze_type=0,
):
    return FrameRecord(
        video_name="vid",
        frame_idx=0,
        clip_index=0,
        frame_offset=0,
        f1=1.0,
        recall=1.0,
        precision=1.0,
        l2=0.0,
        pred_x=pred_x,
        pred_y=pred_y,
        gt_x=gt_x,
        gt_y=gt_y,
        valid=valid,
        gaze_type=gaze_type,
        threshold=0.5,
    )


def _bundle_for(records):
    group = SequenceGroup(
        key=("vid", 0),
        sequence_id="vid__clip0",
        frames=records,
        has_gaps=False,
        consecutive_sub_sequences=[records],
    )
    return compute_localization(
        [group],
        Ego4DSchema(),
        MetricConfig(dataset="ego4d", input_csv_path="unused.csv", output_dir="unused"),
    )


def _results_by_name(bundle):
    return {result.name: result for result in bundle.results}


def test_approximate_aae_identical_coordinates_is_zero():
    bundle = _bundle_for([_record(pred_x=0.5, pred_y=0.5, gt_x=0.5, gt_y=0.5)])
    results = _results_by_name(bundle)

    assert results["localization.mean_aae"].value == 0.0
    assert results["localization.median_aae"].value == 0.0
    assert results["localization.mean_aae"].unit == "degrees_fov60_approx"
    assert "fov60_approx" in bundle.warnings[0]


def test_approximate_aae_symmetric_offsets_are_positive_finite_degrees():
    bundle = _bundle_for(
        [
            _record(pred_x=0.6, pred_y=0.5, gt_x=0.5, gt_y=0.5),
            _record(pred_x=0.4, pred_y=0.5, gt_x=0.5, gt_y=0.5),
        ]
    )
    results = _results_by_name(bundle)

    assert results["localization.mean_aae"].value > 0.0
    assert math.isfinite(results["localization.mean_aae"].value)
    assert (
        results["localization.mean_aae"].value
        == results["localization.median_aae"].value
    )


def test_approximate_aae_excludes_invalid_and_nan_frames():
    bundle = _bundle_for(
        [
            _record(pred_x=0.5, pred_y=0.5, gt_x=0.5, gt_y=0.5),
            _record(pred_x=0.9, pred_y=0.9, gt_x=0.5, gt_y=0.5, valid=0),
            _record(pred_x=float("nan"), pred_y=0.5, gt_x=0.5, gt_y=0.5),
        ]
    )
    results = _results_by_name(bundle)

    assert results["localization.mean_aae"].value == 0.0
    assert results["localization.mean_aae"].sample_count == 1
    assert bundle.sample_counts["valid_frames"] == 1
