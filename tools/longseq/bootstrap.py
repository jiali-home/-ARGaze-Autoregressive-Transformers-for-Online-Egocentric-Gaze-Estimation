"""
tools/longseq/bootstrap.py

Bootstrap Confidence Intervals (Req 10).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np

from .config import MetricConfig
from .loader import SequenceGroup
from .metrics import MetricBundle, MetricResult

if TYPE_CHECKING:
    from .dataset_schema import DatasetSchema


def _percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Compute (lower, upper) CI bounds using percentiles."""
    valid = samples[np.isfinite(samples)]
    if valid.size == 0:
        return float("nan"), float("nan")
    lower = float(np.percentile(valid, 100 * (alpha / 2.0)))
    upper = float(np.percentile(valid, 100 * (1.0 - alpha / 2.0)))
    return lower, upper


def attach_bootstrap_ci(
    bundles: List[MetricBundle],
    groups: List[SequenceGroup],
    schema: DatasetSchema,
    config: MetricConfig,
    compute_funcs: Dict[str, Callable[[List[SequenceGroup], DatasetSchema, MetricConfig], MetricBundle]],
) -> List[MetricBundle]:
    """Attach bootstrap CI to the bundles if configured (Req 10).

    Algorithm:
        1. If bootstrap_artifacts_dir is None, return bundles unchanged.
        2. Load artifact sequences using tools/prediction_artifacts.load_artifact_sequences().
        3. Match loaded sequence IDs with SequenceGroup IDs.
        4. If < 10 matching sequences, log warning and return point estimates only.
        5. For n_bootstrap iterations:
            a. Resample matching SequenceGroups with replacement.
            b. Recompute metrics on resampled set.
        6. Compute 95% CI and attach to corresponding MetricResult objects.

    Parameters
    ----------
    bundles:
        Point estimate bundles.
    groups:
        Original parsed groups.
    schema:
        Dataset schema.
    config:
        Config with bootstrap options.
    compute_funcs:
        Dictionary mapping family name to the metric compute function.

    Returns
    -------
    List[MetricBundle]
        Bundles with ci_lower and ci_upper attached where applicable.
    """
    if not config.bootstrap_artifacts_dir:
        return bundles

    # Lazy import to avoid hard dependency if unused
    try:
        from tools.prediction_artifacts import load_artifact_sequences
    except ImportError:
        for bundle in bundles:
            bundle.warnings.append("prediction_artifacts module not found. Skipping bootstrap.")
        return bundles

    try:
        # Req 12.2: Load artifact sequences
        artifact_seqs, _ = load_artifact_sequences(config.bootstrap_artifacts_dir)
        artifact_ids = {seq.sequence_id for seq in artifact_seqs}
    except Exception as e:
        for bundle in bundles:
            bundle.warnings.append(f"Failed to load bootstrap artifacts: {e}")
        return bundles

    # Filter our parsed groups to only those present in the artifacts
    matching_groups = [g for g in groups if g.sequence_id in artifact_ids]
    n_seqs = len(matching_groups)

    if n_seqs < 10:
        warning_msg = (
            f"Only {n_seqs} sequences found in artifacts (minimum 10 required). "
            "Returning point estimates only."
        )
        for bundle in bundles:
            bundle.warnings.append(warning_msg)
        return bundles

    # Initialize sample arrays for every metric result that has a value
    # Struct: {family: {result_name: np.ndarray(shape=(n_bootstrap,))}}
    bootstrap_samples: Dict[str, Dict[str, np.ndarray]] = {}

    for bundle in bundles:
        family = bundle.family
        if family not in compute_funcs:
            continue  # e.g., efficiency metrics aren't resampled

        bootstrap_samples[family] = {}
        for res in bundle.results:
            if res.value is not None:
                bootstrap_samples[family][res.name] = np.full(
                    (config.n_bootstrap,), np.nan, dtype=np.float64
                )

    rng = np.random.default_rng(config.bootstrap_seed)

    # 12.3 Resample and recompute
    for i in range(config.n_bootstrap):
        # Resample indices with replacement
        sampled_indices = rng.integers(0, n_seqs, size=n_seqs)
        sampled_groups = [matching_groups[idx] for idx in sampled_indices]

        # Recompute all families
        for family, func in compute_funcs.items():
            if family not in bootstrap_samples:
                continue

            resampled_bundle = func(sampled_groups, schema, config)
            
            for res in resampled_bundle.results:
                if res.name in bootstrap_samples[family] and res.value is not None:
                    bootstrap_samples[family][res.name][i] = res.value

    # 12.4 Attach CI to original bundles
    for bundle in bundles:
        family = bundle.family
        if family not in bootstrap_samples:
            continue

        for res in bundle.results:
            if res.value is not None and res.name in bootstrap_samples[family]:
                samples = bootstrap_samples[family][res.name]
                lower, upper = _percentile_ci(samples)
                res.ci_lower = lower
                res.ci_upper = upper

    return bundles
