"""
tools/longseq/reporter.py

Reporter: generates CSV and JSON summaries (Req 8).
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import MetricConfig
from .loader import SequenceGroup, is_valid_frame
from .metrics import MetricBundle, MetricResult


def _format_value(value: float | None, unit: str) -> str:
    """Format a metric value correctly."""
    if value is None or not math.isfinite(value):
        return "N/A"
    
    if unit == "ms":
        return f"{value:.2f}"
    elif unit == "L2/frame" or "drift" in unit.lower():
        return f"{value:.4f}"
    else:
        return f"{value:.4f}"


def write_report(
    bundles: List[MetricBundle],
    config: MetricConfig,
    warnings: List[str],
    output_dir: str,
    groups: Optional[List[SequenceGroup]] = None,
    write_per_sequence: bool = True,
) -> None:
    """Write metric reports to output directory.

    Parameters
    ----------
    bundles:
        Computed metric bundles.
    config:
        Configuration containing metadata.
    warnings:
        List of generated warnings.
    output_dir:
        Output path.
    groups:
        Sequence groups for per-sequence reporting.
    write_per_sequence:
        Whether to write per-sequence breakdown.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Summary CSV
    csv_path = out_path / "longseq_metrics_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric_name", "value", "ci_lower", "ci_upper", "sample_count", "unit"]
        )
        writer.writeheader()
        
        for bundle in bundles:
            for res in bundle.results:
                val_str = _format_value(res.value, res.unit)
                lower_str = _format_value(res.ci_lower, res.unit) if res.ci_lower is not None else ""
                upper_str = _format_value(res.ci_upper, res.unit) if res.ci_upper is not None else ""
                
                writer.writerow({
                    "metric_name": res.name,
                    "value": val_str,
                    "ci_lower": lower_str,
                    "ci_upper": upper_str,
                    "sample_count": res.sample_count,
                    "unit": res.unit,
                })

    # 2. JSON Report
    json_path = out_path / "longseq_metrics_summary.json"
    
    report: Dict[str, Any] = {
        "metadata": {
            "dataset": config.dataset,
            "model_name": config.model_name,
            "checkpoint_path": config.checkpoint_path,
            "input_csv_path": config.input_csv_path,
            "bootstrap_artifacts_path": config.bootstrap_artifacts_dir,
            "config_snapshot": {
                "group_by": config.group_by,
                "min_fixation_length": config.min_fixation_length,
                "drift_min_frames": config.drift_min_frames,
                "recovery_cap": config.recovery_cap,
                "pre_spike_baseline_window": config.pre_spike_baseline_window,
                "spike_threshold_multiplier": config.spike_threshold_multiplier,
                "recovery_threshold_multiplier": config.recovery_threshold_multiplier,
                "saccade_velocity_threshold": config.saccade_velocity_threshold,
                "saccade_merge_window": config.saccade_merge_window,
            }
        },
        "sample_counts": {},
        "feasibility_matrix": {},
        "warnings": warnings,
        "metrics": {},
    }

    # Populate json fields
    for bundle in bundles:
        report["sample_counts"].update(bundle.sample_counts)
        report["warnings"].extend(bundle.warnings)
        
        family_metrics = {}
        is_feasible = False
        
        for res in bundle.results:
            if res.value is not None:
                is_feasible = True
                
                val_str = _format_value(res.value, res.unit)
                if res.ci_lower is not None and res.ci_upper is not None:
                    lower_str = _format_value(res.ci_lower, res.unit)
                    upper_str = _format_value(res.ci_upper, res.unit)
                    formatted_val = f"{val_str} ({lower_str}, {upper_str})"
                    family_metrics[res.name] = formatted_val
                else:
                    try:
                        family_metrics[res.name] = float(val_str)
                    except ValueError:
                        family_metrics[res.name] = val_str
            else:
                family_metrics[res.name] = None
                if res.na_reason:
                    family_metrics[f"{res.name}_na_reason"] = res.na_reason

        report["metrics"][bundle.family] = family_metrics
        report["feasibility_matrix"][bundle.family] = is_feasible

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 3. Per-sequence CSV
    if write_per_sequence and groups:
        seq_csv_path = out_path / "longseq_metrics_per_sequence.csv"
        
        from .dataset_schema import get_schema
        schema = get_schema(config.dataset)
        
        # Imports for local recalculations
        from .metrics.jitter import extract_fixation_runs, _compute_frame_to_frame_distances
        from .metrics.stability import _compute_fss
        from .metrics.drift import compute_temporal_drift
        from .metrics.recovery import compute_recovery_after_error
        
        with open(seq_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "sequence_id", "valid_frames", "mean_l2", "mean_f1",
                "temporal_jitter_mean", "fixation_stability_mean", 
                "drift_rate", "recovery_length_mean"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for group in groups:
                # 1. Localization
                valid_records = [r for r in group.frames if is_valid_frame(r, schema)]
                valid_frames = len(valid_records)
                l2_vals = [r.l2 for r in valid_records if math.isfinite(r.l2)]
                f1_vals = [r.f1 for r in valid_records if math.isfinite(r.f1)]
                
                mean_l2 = sum(l2_vals) / len(l2_vals) if l2_vals else "N/A"
                mean_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else "N/A"
                
                # 2. Jitter & Stability
                jitter_mean = "N/A"
                fss_mean = "N/A"
                try:
                    runs = extract_fixation_runs([group], schema, config)
                    if runs:
                        j_vals = []
                        fss_vals = []
                        for run in runs:
                            if len(run.frames) >= 2:
                                j_vals.extend(_compute_frame_to_frame_distances(run))
                                fss_vals.append(_compute_fss(run))
                        
                        if j_vals:
                            jitter_mean = sum(j_vals) / len(j_vals)
                        if fss_vals:
                            fss_mean = sum(fss_vals) / len(fss_vals)
                except NotImplementedError:
                    pass
                    
                # 3. Drift
                drift_mean = "N/A"
                drift_bundle = compute_temporal_drift([group], schema, config)
                for res in drift_bundle.results:
                    if res.name == "temporal_drift.mean_rate" and res.value is not None:
                        drift_mean = res.value
                        break
                        
                # 4. Recovery
                recovery_mean = "N/A"
                recovery_bundle = compute_recovery_after_error([group], schema, config)
                for res in recovery_bundle.results:
                    if res.name == "recovery.mean_length" and res.value is not None:
                        recovery_mean = res.value
                        break
                
                def fmt(v):
                    return f"{v:.4f}" if isinstance(v, float) else v
                
                writer.writerow({
                    "sequence_id": group.sequence_id,
                    "valid_frames": valid_frames,
                    "mean_l2": fmt(mean_l2),
                    "mean_f1": fmt(mean_f1),
                    "temporal_jitter_mean": fmt(jitter_mean),
                    "fixation_stability_mean": fmt(fss_mean),
                    "drift_rate": fmt(drift_mean),
                    "recovery_length_mean": fmt(recovery_mean)
                })
