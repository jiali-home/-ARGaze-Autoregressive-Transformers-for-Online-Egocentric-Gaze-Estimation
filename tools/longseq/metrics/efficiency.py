"""
tools/longseq/metrics/efficiency.py

Streaming Efficiency Metrics (Req 6).
"""

import csv
import json
from pathlib import Path
from typing import List

from ..config import MetricConfig
from . import MetricBundle, MetricResult


def compute_streaming_efficiency(config: MetricConfig) -> MetricBundle:
    """Compute Streaming Efficiency metrics (Req 6).

    Algorithm:
        1. Look for timing_log.csv in the same directory as input_csv_path.
        2. Fall back to inference_time_ms column in the input CSV if present.
        3. If neither exists, return N/A for timing metrics.
        4. Compute mean_inference_ms, fps, total_frames.
        5. Check for gpu_memory_log.json in output_dir, extract peak_gpu_mb.

    Parameters
    ----------
    config:
        A :class:`MetricConfig` instance.

    Returns
    -------
    MetricBundle
        Contains efficiency metrics.
    """
    warnings: List[str] = []
    
    input_path = Path(config.input_csv_path) if config.input_csv_path else None
    output_path = Path(config.output_dir) if config.output_dir else None
    
    inference_times: List[float] = []
    total_frames = 0
    
    # 1. Try timing_log.csv
    timing_log_path = None
    if input_path and input_path.parent.exists():
        timing_log_path = input_path.parent / "timing_log.csv"
        
    if timing_log_path and timing_log_path.exists():
        try:
            with open(timing_log_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "inference_time_ms" in row:
                        val = row["inference_time_ms"].strip()
                        if val:
                            try:
                                inference_times.append(float(val))
                            except ValueError:
                                pass
                    total_frames += 1
        except Exception as e:
            warnings.append(f"Failed to read {timing_log_path}: {e}")
            
    # 2. Fall back to input CSV
    elif input_path and input_path.exists():
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "inference_time_ms" in reader.fieldnames:
                    for row in reader:
                        val = row["inference_time_ms"].strip()
                        if val:
                            try:
                                inference_times.append(float(val))
                            except ValueError:
                                pass
                        total_frames += 1
                else:
                    # Just count total frames
                    for _ in reader:
                        total_frames += 1
        except Exception as e:
            warnings.append(f"Failed to read {input_path}: {e}")

    # Compute timing metrics
    mean_inference_ms = None
    fps = None
    na_reason_timing = None
    
    if inference_times:
        mean_inference_ms = sum(inference_times) / len(inference_times)
        if mean_inference_ms > 0:
            fps = 1000.0 / mean_inference_ms
        else:
            fps = float("inf")
    else:
        na_reason_timing = "no_timing_data"
        warnings.append("No timing data found. Timing metrics will be N/A.")

    # 3. Check for gpu memory log
    peak_gpu_mb = None
    na_reason_gpu = None

    if not inference_times:
        total_frames = 0
        na_reason_gpu = "no_timing_data"
    else:
        if output_path and output_path.exists():
            gpu_log_path = output_path / "gpu_memory_log.json"
            if gpu_log_path.exists():
                try:
                    with open(gpu_log_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "peak_gpu_memory_mb" in data:
                            peak_gpu_mb = float(data["peak_gpu_memory_mb"])
                except Exception as e:
                    warnings.append(f"Failed to read {gpu_log_path}: {e}")
                    na_reason_gpu = "parse_error"
            else:
                na_reason_gpu = "no_memory_log"
        else:
            na_reason_gpu = "no_memory_log"

    return MetricBundle(
        family="efficiency",
        results=[
            MetricResult(
                name="efficiency.mean_inference_ms",
                value=mean_inference_ms,
                na_reason=na_reason_timing,
                unit="ms",
                sample_count=len(inference_times)
            ),
            MetricResult(
                name="efficiency.fps",
                value=fps,
                na_reason=na_reason_timing,
                unit="Hz",
                sample_count=len(inference_times)
            ),
            MetricResult(
                name="efficiency.total_frames",
                value=float(total_frames) if inference_times else None,
                na_reason=na_reason_timing,
                unit="frames",
                sample_count=total_frames if inference_times else 0
            ),
            MetricResult(
                name="efficiency.peak_gpu_mb",
                value=peak_gpu_mb,
                na_reason=na_reason_gpu,
                unit="MB",
                sample_count=1 if peak_gpu_mb is not None else 0
            ),
        ],
        sample_counts={"inference_time_samples": len(inference_times)},
        warnings=warnings,
    )
