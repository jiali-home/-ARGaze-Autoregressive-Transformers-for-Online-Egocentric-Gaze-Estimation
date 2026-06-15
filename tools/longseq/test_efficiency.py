import json
from pathlib import Path

from tools.longseq.config import MetricConfig
from tools.longseq.metrics.efficiency import compute_streaming_efficiency


def test_efficiency_no_timing_data(tmp_path):
    config = MetricConfig(
        input_csv_path=str(tmp_path / "per_frame_metrics.csv"),
        output_dir=str(tmp_path)
    )
    
    # Create empty CSV with no inference_time_ms
    with open(config.input_csv_path, "w") as f:
        f.write("video_name,frame_idx\nvid1,0\nvid1,1\n")
        
    bundle = compute_streaming_efficiency(config)
    
    res_dict = {r.name: r for r in bundle.results}
    assert res_dict["efficiency.mean_inference_ms"].value is None
    assert res_dict["efficiency.mean_inference_ms"].na_reason == "no_timing_data"
    assert res_dict["efficiency.fps"].value is None
    assert res_dict["efficiency.total_frames"].value is None
    assert res_dict["efficiency.total_frames"].na_reason == "no_timing_data"
    assert res_dict["efficiency.peak_gpu_mb"].value is None
    assert res_dict["efficiency.peak_gpu_mb"].na_reason == "no_timing_data"


def test_efficiency_with_timing_data(tmp_path):
    config = MetricConfig(
        input_csv_path=str(tmp_path / "per_frame_metrics.csv"),
        output_dir=str(tmp_path)
    )
    
    # Create timing_log.csv
    with open(tmp_path / "timing_log.csv", "w") as f:
        f.write("frame_idx,inference_time_ms\n0,10.0\n1,20.0\n")
        
    bundle = compute_streaming_efficiency(config)
    res_dict = {r.name: r for r in bundle.results}
    
    assert res_dict["efficiency.mean_inference_ms"].value == 15.0
    assert res_dict["efficiency.fps"].value == 1000.0 / 15.0


def test_efficiency_with_gpu_log(tmp_path):
    config = MetricConfig(
        input_csv_path=str(tmp_path / "per_frame_metrics.csv"),
        output_dir=str(tmp_path)
    )
    
    with open(tmp_path / "timing_log.csv", "w") as f:
        f.write("frame_idx,inference_time_ms\n0,12.0\n")
        
    with open(tmp_path / "gpu_memory_log.json", "w") as f:
        json.dump({"peak_gpu_memory_mb": 4096.5}, f)
        
    bundle = compute_streaming_efficiency(config)
    res_dict = {r.name: r for r in bundle.results}
    
    assert res_dict["efficiency.peak_gpu_mb"].value == 4096.5


def test_efficiency_total_frames_present_when_timing_available(tmp_path):
    config = MetricConfig(
        input_csv_path=str(tmp_path / "per_frame_metrics.csv"),
        output_dir=str(tmp_path),
    )

    with open(tmp_path / "timing_log.csv", "w") as f:
        f.write("frame_idx,inference_time_ms\n0,10.0\n1,20.0\n2,30.0\n")

    bundle = compute_streaming_efficiency(config)
    res_dict = {r.name: r for r in bundle.results}

    assert res_dict["efficiency.total_frames"].value == 3.0
    assert res_dict["efficiency.total_frames"].na_reason is None
