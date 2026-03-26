from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from conformal_predictions.calibration import (
    compute_mu_hat,
    compute_nonconformity_scores,
)
from conformal_predictions.data.toy import list_split_files, load_pseudo_experiment
from conformal_predictions.data_viz import (
    contourplot_data,
    plot_confidence_intervals,
    plot_mu_hat_distribution,
    plot_nonconformity_scores,
)
from conformal_predictions.evaluation import (
    compute_confidence_interval,
    evaluate_models,
    get_events_count,
    inference_on_test_set,
)
from conformal_predictions.models import build_models, fit_models

OUTPUT_DIRNAME = "test_toy-scale-easy-1000-test-2100-calib"
PLOTS_DIR = Path("results") / OUTPUT_DIRNAME / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STATS_DIR = Path("results") / OUTPUT_DIRNAME / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data") / "toy_scale_easy"
    mu: float = 1.0
    seed: int = 18
    test_prefixes: Tuple[str, ...] = ("7e39", "6fcb")
    threshold: float = 0.5
    valid_size: float = 0.2
    calib_size: float = 0.5
    nonconf_target: str = "mu_hat"  # can be "n_pred" or "mu_hat"
    n_test_experiments: int = (
        1000  # number of test pseudo-experiments to select if no prefixes match
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train conformal prediction models on toy pseudo-experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file. Overrides default Settings and output_dirname.",
    )
    args, _ = parser.parse_known_args()
    return args


def _load_config(config_path: Path) -> Tuple[Settings, str]:
    """Load Settings and output_dirname from a YAML config file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (Settings instance, output_dirname string).
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    output_dirname: str = raw.get("output_dirname", OUTPUT_DIRNAME)

    test_prefixes_raw = raw.get("test_prefixes", list(Settings.test_prefixes))
    test_prefixes: Tuple[str, ...] = tuple(test_prefixes_raw)

    cfg = Settings(
        data_dir=Path(raw["data_dir"]) if "data_dir" in raw else Settings.data_dir,
        mu=float(raw.get("mu", Settings.mu)),
        seed=int(raw.get("seed", Settings.seed)),
        test_prefixes=test_prefixes,
        threshold=float(raw.get("threshold", Settings.threshold)),
        valid_size=float(raw.get("valid_size", Settings.valid_size)),
        calib_size=float(raw.get("calib_size", Settings.calib_size)),
        nonconf_target=str(raw.get("nonconf_target", Settings.nonconf_target)),
        n_test_experiments=int(
            raw.get("n_test_experiments", Settings.n_test_experiments)
        ),
    )
    return cfg, output_dirname


def main() -> None:
    global OUTPUT_DIRNAME, PLOTS_DIR, STATS_DIR

    args = _parse_args()
    if args.config is not None:
        cfg, new_output_dirname = _load_config(args.config)
        if new_output_dirname != OUTPUT_DIRNAME:
            OUTPUT_DIRNAME = new_output_dirname
            PLOTS_DIR = Path("results") / OUTPUT_DIRNAME / "plots"
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            STATS_DIR = Path("results") / OUTPUT_DIRNAME / "stats"
            STATS_DIR.mkdir(parents=True, exist_ok=True)
    else:
        cfg = Settings()

    np.random.seed(cfg.seed)

    train_files, val_files, calib_files, test_files = list_split_files(
        cfg.data_dir,
        cfg.mu,
        cfg.test_prefixes,
        cfg.n_test_experiments,
        cfg.valid_size,
        cfg.calib_size,
        cfg.seed,
    )
    train_blocks: List[np.ndarray] = []
    val_blocks: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []
    val_labels: List[np.ndarray] = []
    calib_data: List[Tuple[np.ndarray, np.ndarray]] = []
    calib_meta: List[dict] = []
    for file_path in train_files:
        X, y, _meta = load_pseudo_experiment(file_path)
        train_blocks.append(X)
        train_labels.append(y)

    for file_path in val_files:
        X, y, _meta = load_pseudo_experiment(file_path)
        val_blocks.append(X)
        val_labels.append(y)

    for file_path in calib_files:
        X, y, _meta = load_pseudo_experiment(file_path)
        calib_data.append((X, y))
        calib_meta.append(_meta)

    X_train = np.vstack(train_blocks)
    X_val = np.vstack(val_blocks)
    y_train = np.concatenate(train_labels)
    y_val = np.concatenate(val_labels)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Starting training on:")
    print(f"  X_train shape: {X_train_scaled.shape}, y_train shape: {y_train.shape}")
    print(f"  X_val shape: {X_val_scaled.shape}, y_val shape: {y_val.shape}")
    print(
        f"  y_train: {int(np.sum(y_train))} positives "
        f"({100*np.sum(y_train)/len(y_train):.1f}%), "
        f"{len(y_train) - int(np.sum(y_train))} negatives "
        f"({100*(len(y_train)-np.sum(y_train))/len(y_train):.1f}%)"
    )
    print(
        f"  y_val: {int(np.sum(y_val))} positives "
        f"({100*np.sum(y_val)/len(y_val):.1f}%), "
        f"{len(y_val) - int(np.sum(y_val))} negatives "
        f"({100*(len(y_val)-np.sum(y_val))/len(y_val):.1f}%)"
    )

    contourplot_data(X_val, y_val, output_dir=PLOTS_DIR)

    models = build_models(cfg)
    fit_models(models, X_train_scaled, y_train)

    # print classification performance on validation set
    performance_metrics = evaluate_models(models, X_val_scaled, y_val)
    for model_name, metrics in performance_metrics.items():
        print(f"\n\n{model_name} validation metrics:")
        print(
            f"\tAccuracy: {metrics['accuracy']:.4f}"
            f"\tPrecision: {metrics['precision']:.4f}"
            f"\tRecall: {metrics['recall']:.4f}"
            f"\tF1: {metrics['f1']:.4f}"
        )

    counts = get_events_count(models, X_val_scaled, cfg.threshold)
    for model_name, count in counts.items():
        print(
            f"{model_name} N signal events (p_pred > {cfg.threshold}): "
            f"{count} / {int(np.sum(y_val))} (true)"
        )

    # nonconformity scores
    print("\nComputing nonconformity scores...")
    print(f"{len(calib_data)} calibration samples")
    print(
        f"Average calibration sample size: {int(np.array([_[0].shape[0] for _ in calib_data]).mean())} observations"
    )
    print(f"\t...using {cfg.nonconf_target} as target for nonconformity scores")
    nonconf_scores = compute_nonconformity_scores(
        models,
        scaler,
        calib_data,
        calib_meta,
        cfg.threshold,
        target=cfg.nonconf_target,
        how="diff",
    )

    for model_name, values in nonconf_scores.items():
        mean_score = np.mean(values) if values else float("nan")
        std_score = np.std(values) if values else float("nan")
        print(
            f"{model_name} nonconformity mu_hat stats: {mean_score:.4f} ± {std_score:.4f}"
        )

    plot_nonconformity_scores(
        nonconf_scores, scores_label=cfg.nonconf_target, output_dir=PLOTS_DIR
    )

    print("\nComputing mu_hat...")
    mu_hat, stats = compute_mu_hat(
        models, scaler, calib_data, calib_meta, cfg.threshold
    )
    np.savez(
        STATS_DIR / "mu_hat_calib_distribution.npz",
        **{model_name: np.array(scores) for model_name, scores in mu_hat.items()},
    )
    np.savez(
        STATS_DIR / f"{cfg.nonconf_target}_nonconf_scores.npz",
        **{
            model_name: np.array(scores)
            for model_name, scores in nonconf_scores.items()
        },
    )

    plot_mu_hat_distribution(mu_hat, stats, output_dir=PLOTS_DIR)
    df_stats = pd.DataFrame(
        [{"Model": model_name, **stats[model_name]} for model_name in stats.keys()]
    )
    print("\nStatistics Summary:")
    print(df_stats)
    df_stats.to_csv(STATS_DIR / "mu_hat_calibration_stats.csv", index=False)

    print("\nRunning inference on test set...")
    test_data = []
    for file_path in test_files:
        X_test, y_test, meta = load_pseudo_experiment(file_path)
        test_data.append((X_test, y_test, meta))

    mu_hat_test, mu_true_list, gamma_true_list, test_metrics = inference_on_test_set(
        models, scaler, test_data, cfg.threshold
    )

    # Compute confidence intervals for test set predictions
    print("\nComputing confidence intervals for test set...")

    # Load nonconformity scores for computing intervals
    print(f"\t...using {cfg.nonconf_target} nonconformity scores for CI computation.")
    nonconf_scores_file = STATS_DIR / f"{cfg.nonconf_target}_nonconf_scores.npz"

    for model_name, mu_hat_values in mu_hat_test.items():
        if cfg.nonconf_target == "mu_hat":
            mu_hat_lower_bounds, mu_hat_upper_bounds = compute_confidence_interval(
                np.array(mu_hat_values), nonconf_scores_file, model_name, how="diff"
            )
        elif cfg.nonconf_target == "n_pred":
            n_preds = mu_hat_values * np.array(gamma_true_list)
            n_lower, n_upper = compute_confidence_interval(
                n_preds, nonconf_scores_file, model_name, how="diff"
            )
            mu_hat_lower_bounds = n_lower / np.array(gamma_true_list)
            mu_hat_upper_bounds = n_upper / np.array(gamma_true_list)

        print(f"\nModel: {model_name}\n")

        empirical_coverage = np.mean(
            [
                mu_hat_lower < mu_true < mu_hat_upper
                for mu_hat_lower, mu_hat_upper, mu_true in zip(
                    mu_hat_lower_bounds,
                    mu_hat_upper_bounds,
                    mu_true_list,
                )
            ]
        )
        print(f"Empirical coverage: {empirical_coverage*100:.2f}%")
        plot_confidence_intervals(
            mu_hat_values,
            mu_hat_lower_bounds,
            mu_hat_upper_bounds,
            mu_true_list,
            model_name,
            empirical_coverage,
            output_dir=STATS_DIR,
        )
        for exp_idx, (mu_hat, mu_hat_lower, mu_hat_upper, mu_true) in enumerate(
            zip(mu_hat_values, mu_hat_lower_bounds, mu_hat_upper_bounds, mu_true_list)
        ):
            if exp_idx % 50 != 0:
                # exp_idx += 1
                continue
            print("printing every 50th experiment:")
            # Determine color based on whether CI contains mu_true
            color = "\033[92m" if mu_hat_lower < mu_true < mu_hat_upper else "\033[91m"
            reset = "\033[0m"
            print(
                f"  {color}Exp {exp_idx}: "
                f"μ̂: {mu_hat:.3f} "
                f"CI: [{mu_hat_lower:.3f}, {mu_hat_upper:.3f}] "
                f"μ_true: {mu_true:.3f}{reset}"
            )
            print(test_metrics[model_name][exp_idx])


if __name__ == "__main__":
    main()
