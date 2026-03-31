"""Evaluate a trained model on the test split with MLE-based intervals.

Usage:
    python scripts/mle_baseline.py --config configs/train_higgs.yaml --model MLP --kind wald
    python scripts/mle_baseline.py --config configs/train_toy.yaml --model GLM --kind exact
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from conformal_predictions.calibration import (
    _get_expected_background,
    _get_expected_signal,
    _get_proportionate_beta_true,
    _get_proportionate_gamma_true,
)
from conformal_predictions.config import PipelineConfig, load_config
from conformal_predictions.data.higgs import load_test as higgs_load_test
from conformal_predictions.data.toy import list_split_files, load_pseudo_experiment
from conformal_predictions.data_viz import plot_confidence_intervals
from conformal_predictions.evaluation import compute_empirical_coverage
from conformal_predictions.mle import compute_mle_interval, compute_mle_mu_hat
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
        description="Run test inference and MLE-baseline interval evaluation."
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
    parser.add_argument(
        "--kind",
        type=str,
        required=True,
        choices=("wald", "profile", "exact"),
        help="MLE interval construction kind.",
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


def _run_mle_baseline_on_test_set(
    model: object,
    scaler: object,
    test_data: Sequence[Tuple[np.ndarray, np.ndarray, dict]],
    threshold: float,
    ref_efficiencies: Sequence[float],
    alpha: float,
    kind: str,
) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    metrics: List[dict] = []

    for experiment_idx, (X_test, y_test, meta_dict) in enumerate(test_data):
        X_test_scaled = scaler.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = y_pred_proba > threshold

        gamma_true = float(_get_proportionate_gamma_true(meta_dict))
        beta_true = float(_get_proportionate_beta_true(meta_dict))
        mu_true = float(meta_dict["mu_true"])
        n_pred = int(np.sum(y_pred))

        if gamma_true <= 0.0:
            continue

        s = float(_get_expected_signal(gamma_true, ref_efficiencies[0]))
        b = float(_get_expected_background(beta_true, ref_efficiencies[1]))

        mu_hat = compute_mle_mu_hat(n_pred, s, b)
        mu_hat_lower, mu_hat_upper = compute_mle_interval(
            n_pred,
            s,
            b,
            alpha,
            kind=kind,
        )

        metric_row = {
            "accuracy": float(model.score(X_test_scaled, y_test)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        metrics.append(metric_row)

        rows.append(
            {
                "experiment_idx": experiment_idx,
                "n_obs": n_pred,
                "n_pred": n_pred,
                "mu_hat": float(mu_hat),
                "mu_hat_lower": float(mu_hat_lower),
                "mu_hat_upper": float(mu_hat_upper),
                "mu_true": mu_true,
                "gamma_true": gamma_true,
                "beta_true": beta_true,
                "s": s,
                "b": b,
                "contains_true": bool(mu_hat_lower < mu_true < mu_hat_upper),
                **metric_row,
            }
        )

    return rows, metrics


def _build_performance_summary_row(
    model_name: str,
    metrics: Sequence[dict],
    mu_hat_mode: str,
    alpha: float,
    confidence_level: float,
    kind: str,
) -> dict:
    summary = {
        "model": model_name,
        "mu_hat_mode": mu_hat_mode,
        "mle_kind": kind,
        "alpha": alpha,
        "confidence_level": confidence_level,
        "n_test_blocks": len(metrics),
    }

    if not metrics:
        return summary

    metric_names = ("accuracy", "precision", "recall", "f1")
    for metric_name in metric_names:
        values = [float(metric[metric_name]) for metric in metrics if metric_name in metric]
        if not values:
            continue
        values_array = np.asarray(values, dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(np.mean(values_array))
        summary[f"{metric_name}_std"] = float(np.std(values_array))
        summary[f"{metric_name}_min"] = float(np.min(values_array))
        summary[f"{metric_name}_max"] = float(np.max(values_array))
    return summary


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    model_name = _MODEL_CLI_MAP[args.model]

    artifacts_dir = Path("results") / config.output_dir / "artifacts"
    plots_dir = config.plots_dir / model_name / "mle_baseline" / args.kind
    stats_dir = config.stats_dir / model_name / "mle_baseline" / args.kind
    plots_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / f"{model_name}.joblib"
    scaler_path = artifacts_dir / "scaler.joblib"
    efficiencies_path = artifacts_dir / "reference_efficiencies.json"

    _require_file(model_path, f"model '{model_name}'")
    _require_file(scaler_path, "scaler")
    _require_file(efficiencies_path, "reference efficiencies")

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
        "\n[Running MLE baseline on test set "
        f"(mu_hat_mode={config.mu_hat_mode}, kind={args.kind})...]"
    )
    experiment_rows, performance_metrics = _run_mle_baseline_on_test_set(
        model,
        scaler,
        test_data,
        config.threshold,
        ref_efficiencies,
        config.resolved_alpha,
        args.kind,
    )

    mu_hat_values = [row["mu_hat"] for row in experiment_rows]
    lower_bounds = np.asarray([row["mu_hat_lower"] for row in experiment_rows], dtype=float)
    upper_bounds = np.asarray([row["mu_hat_upper"] for row in experiment_rows], dtype=float)
    mu_true_values = [row["mu_true"] for row in experiment_rows]

    empirical_coverage = compute_empirical_coverage(
        lower_bounds,
        upper_bounds,
        mu_true_values,
    )
    print(f"  {model_name} — empirical coverage: {empirical_coverage * 100:.2f}%")

    plot_confidence_intervals(
        mu_hat_values,
        lower_bounds.tolist(),
        upper_bounds.tolist(),
        mu_true_values,
        f"{model_name} [MLE-{args.kind}]",
        empirical_coverage,
        config.confidence_level,
        output_dir=plots_dir,
    )

    coverage_path = stats_dir / "test_coverage.csv"
    experiment_metrics_path = stats_dir / "test_experiment_metrics.csv"
    performance_summary_path = stats_dir / "test_performance_summary.csv"

    pd.DataFrame(
        [
            {
                "model": model_name,
                "mu_hat_mode": config.mu_hat_mode,
                "mle_kind": args.kind,
                "alpha": config.resolved_alpha,
                "confidence_level": config.confidence_level,
                "n_test_blocks": len(experiment_rows),
                "empirical_coverage": float(empirical_coverage),
            }
        ]
    ).to_csv(coverage_path, index=False)

    pd.DataFrame(
        [
            {
                "model": model_name,
                "mu_hat_mode": config.mu_hat_mode,
                "mle_kind": args.kind,
                **row,
            }
            for row in experiment_rows
        ]
    ).to_csv(experiment_metrics_path, index=False)

    pd.DataFrame(
        [
            _build_performance_summary_row(
                model_name,
                performance_metrics,
                config.mu_hat_mode,
                config.resolved_alpha,
                config.confidence_level,
                args.kind,
            )
        ]
    ).to_csv(performance_summary_path, index=False)

    print(f"\nSaved coverage summary to {coverage_path}")
    print(f"Saved per-experiment metrics to {experiment_metrics_path}")
    print(f"Saved performance summary to {performance_summary_path}")
    print(f"Saved CI plots to {plots_dir}")


if __name__ == "__main__":
    main()
