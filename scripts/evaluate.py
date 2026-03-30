"""Evaluate a trained model on the test split and compute confidence intervals.

Usage:
    python scripts/evaluate.py --config configs/train_higgs.yaml --model MLP
    python scripts/evaluate.py --config configs/train_toy.yaml --model GLM
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from conformal_predictions.config import PipelineConfig, load_config
from conformal_predictions.data.higgs import load_test as higgs_load_test
from conformal_predictions.data.toy import list_split_files, load_pseudo_experiment
from conformal_predictions.data_viz import plot_confidence_intervals
from conformal_predictions.evaluation import (
    compute_confidence_interval,
    compute_empirical_coverage,
    inference_on_test_set,
)
from conformal_predictions.models import load_model
from conformal_predictions.preprocessing import load_scaler
from conformal_predictions.reference import load_efficiencies

_MODEL_CLI_MAP = {
    "GLM": "GLM",
    "RF": "Random Forest",
    "MLP": "MLP",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run test inference and confidence-interval evaluation."
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
        help="Model to evaluate (GLM, RF, or MLP).",
    )
    return parser.parse_args()


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} artifact not found: {path}")


def _assert_disjoint_toy_splits(
    train_files: Sequence[Path],
    val_files: Sequence[Path],
    calib_files: Sequence[Path],
    test_files: Sequence[Path],
) -> None:
    groups = {
        "train": set(train_files),
        "val": set(val_files),
        "calib": set(calib_files),
        "test": set(test_files),
    }

    if not groups["test"]:
        raise ValueError("No test files found for toy evaluation.")

    comparisons = (
        ("train", "val"),
        ("train", "calib"),
        ("train", "test"),
        ("val", "calib"),
        ("val", "test"),
        ("calib", "test"),
    )
    for left_name, right_name in comparisons:
        overlap = groups[left_name] & groups[right_name]
        if overlap:
            overlap_str = ", ".join(str(path) for path in sorted(overlap))
            raise ValueError(
                f"Toy data split overlap detected between {left_name} and "
                f"{right_name}: {overlap_str}"
            )


def _load_toy_test(
    config: PipelineConfig,
) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    train_files, val_files, calib_files, test_files = list_split_files(
        config.data_dir,
        config.mu,
        config.test_prefixes,
        config.n_test_experiments,
        config.valid_size,
        config.calib_size,
        config.seed,
    )
    _assert_disjoint_toy_splits(train_files, val_files, calib_files, test_files)

    test_data: List[Tuple[np.ndarray, np.ndarray, dict]] = []
    for file_path in test_files:
        X_test, y_test, meta = load_pseudo_experiment(file_path)
        test_data.append((X_test, y_test, meta))
    return test_data


def _load_test_data(
    config: PipelineConfig,
) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    if config.data_source == "toy":
        return _load_toy_test(config)
    if config.data_source == "higgs":
        return [
            (X_block, y_block, meta)
            for X_block, y_block, meta in higgs_load_test(config)
        ]
    raise ValueError(f"Unknown data_source: {config.data_source!r}")


def _compute_mu_hat_confidence_bounds(
    mu_hat_values: Sequence[float],
    gamma_true_values: Sequence[float],
    nonconf_scores_file: Path,
    model_name: str,
    interval_mode: str,
    nonconf_target: str,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mu_hat_array = np.asarray(mu_hat_values, dtype=np.float64)

    if nonconf_target == "mu_hat":
        lower, upper = compute_confidence_interval(
            mu_hat_array,
            nonconf_scores_file,
            model_name,
            interval_mode=interval_mode,
            alpha=alpha,
        )
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    if nonconf_target == "n_pred":
        gamma_true_array = np.asarray(gamma_true_values, dtype=np.float64)
        if np.any(gamma_true_array == 0):
            raise ValueError(
                "Cannot convert n_pred confidence intervals to mu_hat when "
                "gamma_true contains zeros."
            )
        n_pred_values = mu_hat_array * gamma_true_array
        n_lower, n_upper = compute_confidence_interval(
            n_pred_values,
            nonconf_scores_file,
            model_name,
            interval_mode=interval_mode,
            alpha=alpha,
        )
        return (
            np.asarray(n_lower, dtype=np.float64) / gamma_true_array,
            np.asarray(n_upper, dtype=np.float64) / gamma_true_array,
        )

    raise ValueError(f"Unsupported nonconf_target: {nonconf_target!r}")


def _build_experiment_rows(
    model_name: str,
    mu_hat_values: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    mu_true_values: Sequence[float],
    gamma_true_values: Sequence[float],
    metrics: Sequence[dict],
    mu_hat_mode: str,
    interval_mode: str,
    alpha: float,
    confidence_level: float,
) -> List[dict]:
    rows: List[dict] = []
    for experiment_idx, (
        mu_hat,
        lower,
        upper,
        mu_true,
        gamma_true,
    ) in enumerate(
        zip(
            mu_hat_values,
            lower_bounds,
            upper_bounds,
            mu_true_values,
            gamma_true_values,
        )
    ):
        row = {
            "model": model_name,
            "experiment_idx": experiment_idx,
            "mu_hat": float(mu_hat),
            "mu_hat_lower": float(lower),
            "mu_hat_upper": float(upper),
            "mu_true": float(mu_true),
            "gamma_true": float(gamma_true),
            "mu_hat_mode": mu_hat_mode,
            "interval_mode": interval_mode,
            "alpha": alpha,
            "confidence_level": confidence_level,
            "contains_true": bool(lower < mu_true < upper),
        }
        if experiment_idx < len(metrics):
            row.update(metrics[experiment_idx])
        rows.append(row)
    return rows


def _build_performance_summary_rows(
    model_name: str,
    metrics: Sequence[dict],
    mu_hat_mode: str,
    interval_mode: str,
    alpha: float,
    confidence_level: float,
) -> List[dict]:
    if not metrics:
        return []

    metric_names = ("accuracy", "precision", "recall", "f1")
    summary = {
        "model": model_name,
        "mu_hat_mode": mu_hat_mode,
        "interval_mode": interval_mode,
        "alpha": alpha,
        "confidence_level": confidence_level,
        "n_test_blocks": len(metrics),
    }
    for metric_name in metric_names:
        values = [float(metric[metric_name]) for metric in metrics if metric_name in metric]
        if not values:
            continue
        values_array = np.asarray(values, dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(np.mean(values_array))
        summary[f"{metric_name}_std"] = float(np.std(values_array))
        summary[f"{metric_name}_min"] = float(np.min(values_array))
        summary[f"{metric_name}_max"] = float(np.max(values_array))
    return [summary]


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    model_name = _MODEL_CLI_MAP[args.model]

    artifacts_dir = Path("results") / config.output_dir / "artifacts"
    plots_dir = config.plots_dir / model_name
    stats_dir = config.stats_dir / model_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / f"{model_name}.joblib"
    scaler_path = artifacts_dir / "scaler.joblib"
    efficiencies_path = artifacts_dir / "reference_efficiencies.json"
    nonconf_scores_file = stats_dir / f"{config.nonconf_target}_nonconf_scores.npz"

    _require_file(model_path, f"model '{model_name}'")
    _require_file(scaler_path, "scaler")
    _require_file(efficiencies_path, "reference efficiencies")
    _require_file(nonconf_scores_file, "nonconformity scores")

    np.random.seed(config.seed)

    print(f"\n[Loading artifacts from {artifacts_dir}...]")
    model = load_model(artifacts_dir, model_name)
    scaler = load_scaler(artifacts_dir)
    efficiencies = load_efficiencies(artifacts_dir)
    if model_name not in efficiencies:
        raise KeyError(
            f"Reference efficiencies for model '{model_name}' not found in "
            f"{efficiencies_path}"
        )
    ref_efficiencies = efficiencies[model_name]

    print(f"\n[Loading {config.data_source} test data...]")
    test_data = _load_test_data(config)
    print(f"  Loaded {len(test_data)} test blocks")

    print(
        "\n[Running inference on test set "
        f"(mu_hat_mode={config.mu_hat_mode}, "
        f"interval_mode={config.interval_mode})...]"
    )
    mu_hat_test, mu_true_list, gamma_true_list, performance_metrics = (
        inference_on_test_set(
            {model_name: model},
            scaler,
            test_data,
            config.threshold,
            config.mu_hat_mode,
            {model_name: ref_efficiencies},
        )
    )

    coverage_rows: List[dict] = []
    experiment_rows: List[dict] = []
    performance_summary_rows: List[dict] = []

    for current_model_name, mu_hat_values in mu_hat_test.items():
        if len(mu_hat_values) != len(mu_true_list):
            raise ValueError(
                "Mismatch between number of test predictions and true mu values: "
                f"{len(mu_hat_values)} vs {len(mu_true_list)}."
            )
        if len(mu_hat_values) != len(gamma_true_list):
            raise ValueError(
                "Mismatch between number of test predictions and gamma_true values: "
                f"{len(mu_hat_values)} vs {len(gamma_true_list)}."
            )

        lower_bounds, upper_bounds = _compute_mu_hat_confidence_bounds(
            mu_hat_values,
            gamma_true_list,
            nonconf_scores_file,
            current_model_name,
            config.interval_mode,
            config.nonconf_target,
            config.resolved_alpha,
        )
        empirical_coverage = compute_empirical_coverage(
            lower_bounds,
            upper_bounds,
            mu_true_list,
        )

        print(
            f"  {current_model_name} — empirical coverage: "
            f"{empirical_coverage * 100:.2f}%"
        )
        plot_confidence_intervals(
            list(mu_hat_values),
            lower_bounds.tolist(),
            upper_bounds.tolist(),
            list(mu_true_list),
            current_model_name,
            empirical_coverage,
            config.confidence_level,
            output_dir=plots_dir,
        )

        coverage_rows.append(
            {
                "model": current_model_name,
                "nonconf_target": config.nonconf_target,
                "mu_hat_mode": config.mu_hat_mode,
                "interval_mode": config.interval_mode,
                "alpha": config.resolved_alpha,
                "confidence_level": config.confidence_level,
                "n_test_blocks": len(mu_hat_values),
                "empirical_coverage": float(empirical_coverage),
            }
        )
        experiment_rows.extend(
            _build_experiment_rows(
                current_model_name,
                mu_hat_values,
                lower_bounds,
                upper_bounds,
                mu_true_list,
                gamma_true_list,
                performance_metrics.get(current_model_name, []),
                config.mu_hat_mode,
                config.interval_mode,
                config.resolved_alpha,
                config.confidence_level,
            )
        )
        performance_summary_rows.extend(
            _build_performance_summary_rows(
                current_model_name,
                performance_metrics.get(current_model_name, []),
                config.mu_hat_mode,
                config.interval_mode,
                config.resolved_alpha,
                config.confidence_level,
            )
        )

    coverage_path = stats_dir / "test_coverage.csv"
    experiment_metrics_path = stats_dir / "test_experiment_metrics.csv"
    performance_summary_path = stats_dir / "test_performance_summary.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    pd.DataFrame(experiment_rows).to_csv(experiment_metrics_path, index=False)
    pd.DataFrame(performance_summary_rows).to_csv(
        performance_summary_path, index=False
    )

    print(f"\nSaved coverage summary to {coverage_path}")
    print(f"Saved per-experiment metrics to {experiment_metrics_path}")
    print(f"Saved performance summary to {performance_summary_path}")
    print(f"Saved CI plots to {plots_dir}")


if __name__ == "__main__":
    main()
