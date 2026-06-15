import csv
import json

from tools.longseq.config import MetricConfig
from tools.longseq.dataset_schema import Ego4DSchema
from tools.longseq.loader import FrameRecord, SequenceGroup
from tools.longseq.metrics.localization import compute_localization
from tools.longseq.reporter import write_report


def test_per_sequence_report_uses_central_validity_logic(tmp_path):
    config = MetricConfig(
        dataset="ego4d",
        input_csv_path=str(tmp_path / "per_frame_metrics.csv"),
        output_dir=str(tmp_path / "out"),
    )

    frames = [
        FrameRecord(
            "vid", 0, 0, 0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1, 2, 0.5
        ),
        FrameRecord(
            "vid", 1, 0, 1, 1.0, 1.0, 1.0, 0.0, float("nan"), 0.5, 0.5, 0.5, 1, 0, 0.5
        ),
    ]
    group = SequenceGroup(
        key=("vid", 0),
        sequence_id="vid__clip0",
        frames=frames,
        has_gaps=False,
        consecutive_sub_sequences=[frames],
    )

    bundles = [compute_localization([group], Ego4DSchema(), config)]
    write_report(bundles, config, [], config.output_dir, groups=[group], write_per_sequence=True)

    with open(tmp_path / "out" / "longseq_metrics_per_sequence.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["valid_frames"] == "0"
    assert rows[0]["mean_l2"] == "N/A"
    assert rows[0]["mean_f1"] == "N/A"

    with open(tmp_path / "out" / "longseq_metrics_summary.json", encoding="utf-8") as f:
        report = json.load(f)

    assert report["sample_counts"]["valid_frames"] == 0
