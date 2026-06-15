#!/usr/bin/env python3
"""
tools/compute_longseq_metrics.py

CLI entry point for Long-Sequence Streaming Evaluation Metrics.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Callable

from longseq.config import MetricConfig
from longseq.dataset_schema import get_schema
from longseq.loader import load_per_frame_csv, SequenceGroup
from longseq.metrics.localization import compute_localization
from longseq.metrics.jitter import compute_temporal_jitter
from longseq.metrics.stability import compute_fixation_stability
from longseq.metrics.saccade import compute_saccade_transition
from longseq.metrics.drift import compute_temporal_drift
from longseq.metrics.recovery import compute_recovery_after_error
from longseq.metrics.efficiency import compute_streaming_efficiency
from longseq.bootstrap import attach_bootstrap_ci
from longseq.reporter import write_report
from longseq.validator import validate_loader_stats, validate_metric_bundles


def main():
    parser = argparse.ArgumentParser(description="Compute long-sequence metrics.")
    parser.add_argument("--csv", required=True, help="Path to per_frame_metrics.csv")
    parser.add_argument("--dataset", required=True, choices=["egtea", "ego4d", "egoexo4d"])
    parser.add_argument("--output-dir", default="longseq_metrics_out")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--checkpoint", default="", dest="checkpoint_path")
    parser.add_argument("--bootstrap-artifacts", dest="bootstrap_artifacts_dir", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    
    # Feature flags
    parser.add_argument("--no-localization", action="store_false", dest="enable_localization")
    parser.add_argument("--no-jitter", action="store_false", dest="enable_jitter")
    parser.add_argument("--no-stability", action="store_false", dest="enable_stability")
    parser.add_argument("--no-saccade", action="store_false", dest="enable_saccade")
    parser.add_argument("--no-drift", action="store_false", dest="enable_drift")
    parser.add_argument("--no-recovery", action="store_false", dest="enable_recovery")
    parser.add_argument("--no-efficiency", action="store_false", dest="enable_efficiency")
    parser.add_argument("--no-per-sequence-csv", action="store_false", dest="write_per_sequence")
    
    # Thresholds
    parser.add_argument("--saccade-velocity-threshold", type=float, default=0.05)
    parser.add_argument("--min-fixation-length", type=int, default=3)
    parser.add_argument("--drift-min-frames", type=int, default=30)
    parser.add_argument("--recovery-cap", type=int, default=20)
    
    args = parser.parse_args()
    
    config = MetricConfig(
        dataset=args.dataset,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        input_csv_path=args.csv,
        output_dir=args.output_dir,
        bootstrap_artifacts_dir=args.bootstrap_artifacts_dir,
        n_bootstrap=args.n_bootstrap,
        enable_localization=args.enable_localization,
        enable_jitter=args.enable_jitter,
        enable_stability=args.enable_stability,
        enable_saccade=args.enable_saccade,
        enable_drift=args.enable_drift,
        enable_recovery=args.enable_recovery,
        enable_efficiency=args.enable_efficiency,
        saccade_velocity_threshold=args.saccade_velocity_threshold,
        min_fixation_length=args.min_fixation_length,
        drift_min_frames=args.drift_min_frames,
        recovery_cap=args.recovery_cap,
    )
    
    csv_path = Path(config.input_csv_path)
    if not csv_path.exists():
        print(f"Error: Input CSV {csv_path} not found.")
        sys.exit(1)
        
    try:
        schema = get_schema(config.dataset)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    print(f"Loading CSV: {csv_path}")
    try:
        groups, stats = load_per_frame_csv(str(csv_path), schema)
    except ValueError as e:
        print(f"Error validating CSV schema: {e}")
        sys.exit(2)
        
    all_warnings = validate_loader_stats(stats, config)
    
    bundles = []
    compute_funcs = {}
    
    print("Computing metrics...")
    if config.enable_localization:
        bundles.append(compute_localization(groups, schema, config))
        compute_funcs["localization"] = compute_localization
        
    if config.enable_jitter:
        bundles.append(compute_temporal_jitter(groups, schema, config))
        compute_funcs["temporal_jitter"] = compute_temporal_jitter
        
    if config.enable_stability:
        bundles.append(compute_fixation_stability(groups, schema, config))
        compute_funcs["fixation_stability"] = compute_fixation_stability
        
    if config.enable_saccade:
        bundles.append(compute_saccade_transition(groups, schema, config))
        compute_funcs["saccade_transition"] = compute_saccade_transition
        
    if config.enable_drift:
        bundles.append(compute_temporal_drift(groups, schema, config))
        compute_funcs["temporal_drift"] = compute_temporal_drift
        
    if config.enable_recovery:
        bundles.append(compute_recovery_after_error(groups, schema, config))
        compute_funcs["recovery"] = compute_recovery_after_error
        
    if config.enable_efficiency:
        bundles.append(compute_streaming_efficiency(config))
        # efficiency doesn't use standard SequenceGroups, omitted from compute_funcs
        
    print("Validating metric bundles...")
    all_warnings.extend(validate_metric_bundles(bundles))
    
    if config.bootstrap_artifacts_dir:
        print(f"Attaching bootstrap CI from {config.bootstrap_artifacts_dir}...")
        bundles = attach_bootstrap_ci(bundles, groups, schema, config, compute_funcs)
        
    print(f"Writing reports to {config.output_dir}...")
    write_report(
        bundles, config, all_warnings, config.output_dir, 
        groups=groups, write_per_sequence=args.write_per_sequence
    )
    print("Done!")

if __name__ == "__main__":
    main()
