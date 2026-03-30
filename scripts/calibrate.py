"""Calibration script: compute nonconformity scores and mu-hat distribution.

Loads a trained model, scaler, and reference efficiencies from artifacts,
then scores the calibration set (disjoint from train/val/reference data)
to produce nonconformity scores and the mu-hat calibration distribution.

Usage:
    python scripts/calibrate.py --config configs/train_higgs.yaml --model MLP
    python scripts/calibrate.py --config configs/train_toy.yaml --model GLM
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from conformal_predictions.calibration import (
    compute_mu_hat,
    compute_nonconformity_scores,
)
from conformal_predictions.config import PipelineConfig, load_config
from conformal_predictions.data.higgs import load_calib as higgs_load_calib
from conformal_predictions.data.toy import list_split_files, load_pseudo_experiment
from conformal_predictions.data_viz import (
    plot_mu_hat_distribution,
    plot_nonconformity_scores,
)
from conformal_predictions.models import load_model
from conformal_predictions.preprocessing import load_scaler
from conformal_predictions.reference import load_efficiencies

# CLI model name → internal model name
_MODEL_CLI_MAP = {
    "GLM": "GLM",
    "RF": "Random Forest",
    "MLP": "MLP",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute nonconformity scores and μ̂ distribution "
        "on the calibration set."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML pipeline config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(_MODEL_CLI_MAP.keys()),
        help="Model to calibrate (GLM, RF, or MLP).",
    )
    return parser.parse_args()


def _load_toy_calib(
    config: PipelineConfig,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[dict]]:
    """Load calibration pseudo-experiments for toy data.

    Calibration files are disjoint from train/val files by construction:
    ``list_split_files`` removes test files first, then selects calibration
    files from the remainder, and finally splits the rest into train/val.
    """
    _train_files, _val_files, calib_files, _test_files = list_split_files(
        config.data_dir,
        config.mu,
        config.test_prefixes,
        config.n_test_experiments,
        config.valid_size,
        config.calib_size,
        config.seed,
    )

    if not calib_files:
        raise ValueError("No calibration files found. Check calib_size in your config.")

    calib_data: List[Tuple[np.ndarray, np.ndarray]] = []
    calib_meta: List[dict] = []
    for fp in calib_files:
        X, y, meta = load_pseudo_experiment(fp)
        calib_data.append((X, y))
        calib_meta.append(meta)

    return calib_data, calib_meta


def _load_higgs_calib(
    config: PipelineConfig,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[dict]]:
    """Load calibration blocks for HiggsML data.

    Calibration row groups sit after train + validation + reference row
    groups in the parquet file, so the data is guaranteed to be disjoint
    from training, validation, and reference sets.
    """
    return higgs_load_calib(config)


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    model_name = _MODEL_CLI_MAP[args.model]

    artifacts_dir = Path("results") / config.output_dir / "artifacts"
    plots_dir = config.plots_dir / model_name
    stats_dir = config.stats_dir / model_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(config.seed)

    # ---- 1. Load artifacts ----
    print(f"\n[Loading model '{model_name}' from {artifacts_dir}...]")
    model = load_model(artifacts_dir, model_name)
    models = {model_name: model}

    print(f"[Loading scaler from {artifacts_dir}...]")
    scaler = load_scaler(artifacts_dir)

    print(f"[Loading reference efficiencies from {artifacts_dir}...]")
    all_efficiencies = load_efficiencies(artifacts_dir)
    ref_efficiencies = all_efficiencies.get(model_name, (1.0, 1.0))
    print(
        f"  {model_name} — eps_signal: {ref_efficiencies[0]:.4f}, "
        f"eps_background: {ref_efficiencies[1]:.4f}"
    )

    # ---- 2. Load calibration data (disjoint from train/val/ref) ----
    print(f"\n[Loading {config.data_source} calibration data...]")
    if config.data_source == "toy":
        calib_data, calib_meta = _load_toy_calib(config)
    elif config.data_source == "higgs":
        calib_data, calib_meta = _load_higgs_calib(config)
    else:
        raise ValueError(f"Unknown data_source: {config.data_source!r}")

    print(f"  {len(calib_data)} calibration blocks")

    # ---- 3. Nonconformity scores ----
    print(
        f"\n[Computing nonconformity scores "
        f"(target={config.nonconf_target}, "
        f"mu_hat_mode={config.mu_hat_mode}, "
        f"interval_mode={config.interval_mode})...]"
    )
    nonconf_scores = compute_nonconformity_scores(
        models,
        scaler,
        calib_data,
        calib_meta,
        config.threshold,
        target=config.nonconf_target,
        interval_mode=config.interval_mode,
        mu_hat_mode=config.mu_hat_mode,
        ref_efficiencies=ref_efficiencies,
    )

    for name, values in nonconf_scores.items():
        mean_score = np.mean(values) if values else float("nan")
        std_score = np.std(values) if values else float("nan")
        print(f"  {name} — mean: {mean_score:.4f} ± {std_score:.4f}")

    # ---- 4. μ̂ distribution ----
    print(
        "\n[Computing μ̂ distribution on calibration set "
        f"(mu_hat_mode={config.mu_hat_mode})...]"
    )
    mu_hat, stats = compute_mu_hat(
        models,
        scaler,
        calib_data,
        calib_meta,
        config.threshold,
        config.mu_hat_mode,
        ref_efficiencies=ref_efficiencies,
        alpha=config.resolved_alpha,
    )

    print(f"\n[Saving calibration plots to {plots_dir}...]")
    plot_nonconformity_scores(
        nonconf_scores,
        config.nonconf_target,
        output_dir=plots_dir,
    )
    if stats:
        plot_mu_hat_distribution(
            mu_hat,
            stats,
            output_dir=plots_dir,
            pred_formula=config.pred_formula,
            confidence_level=config.confidence_level,
        )

    # ---- 5. Save artifacts ----
    print(f"\n[Saving calibration artifacts to {stats_dir}...]")

    np.savez(
        stats_dir / f"{config.nonconf_target}_nonconf_scores.npz",
        **{name: np.array(scores) for name, scores in nonconf_scores.items()},
    )

    np.savez(
        stats_dir / "mu_hat_calib_distribution.npz",
        **{name: np.array(values) for name, values in mu_hat.items()},
    )

    if stats:
        df_stats = pd.DataFrame(
            [
                {
                    "Model": name,
                    "mu_hat_mode": config.mu_hat_mode,
                    "interval_mode": config.interval_mode,
                    "nonconf_target": config.nonconf_target,
                    "alpha": config.resolved_alpha,
                    "confidence_level": config.confidence_level,
                    **s,
                }
                for name, s in stats.items()
            ]
        )
        df_stats.to_csv(stats_dir / "mu_hat_calibration_stats.csv", index=False)
        print("\nStatistics Summary:")
        print(df_stats.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
