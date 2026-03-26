from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.preprocessing import StandardScaler

from conformal_predictions.calibration import (
    compute_mu_hat,
    compute_nonconformity_scores,
)
from conformal_predictions.data_viz import (
    contourplot_data,
    plot_confidence_intervals,
    plot_mu_hat_distribution,
    plot_nonconformity_scores,
)
from conformal_predictions.evaluation import (
    compute_confidence_interval,
    compute_empirical_coverage,
    evaluate_models,
    get_events_count,
    inference_on_test_set,
)
from conformal_predictions.models import build_models, fit_models
from conformal_predictions.reference import get_model_efficiencies

HOW = "abs"  # method for computing nonconformity scores: "diff" or "abs"
FIT_PARALLEL = False  # whether to fit models in parallel using joblib
PRED_FORMULA = r"$\hat{\mu} = \frac{n_{pred} - \epsilon_{bkg}\beta^*_{true}}{\epsilon_{sig}\gamma^*_{true}}$"  # should match training._compute_mu_hat logic; used in plot_mu_hat_distribution titles
OUTPUT_DIRNAME = "higgs-sequential-q68-10train-10valid-10ref-10calib-10test"

PLOTS_DIR = Path("results") / OUTPUT_DIRNAME / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STATS_DIR = Path("results") / OUTPUT_DIRNAME / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data") / "HiggsML/input_data/train"
    mu: float = 1.0
    seed: int = 18
    # test_prefixes: Tuple[str, ...] = ("7e39", "6fcb")
    threshold: float = 0.5
    train_size: float = 10
    valid_size: float = 10
    ref_size: float = 10
    calib_size: float = 10
    test_size: float = 10
    nonconf_target: str = "mu_hat"  # can be "n_pred" or "mu_hat"
    block_size: int = (
        10_000  # number of test pseudo-experiments to select if no prefixes match
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train conformal prediction models on the Higgs ML dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file. Overrides default Settings and top-level constants.",
    )
    args, _ = parser.parse_known_args()
    return args


def _load_config(config_path: Path) -> Tuple[Settings, str, str, bool]:
    """Load Settings and top-level constants from a YAML config file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (Settings instance, output_dirname, how, fit_parallel).
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    output_dirname: str = raw.get("output_dirname", OUTPUT_DIRNAME)
    how: str = raw.get("how", HOW)
    fit_parallel: bool = bool(raw.get("fit_parallel", FIT_PARALLEL))

    cfg = Settings(
        data_dir=Path(raw["data_dir"]) if "data_dir" in raw else Settings.data_dir,
        mu=float(raw.get("mu", Settings.mu)),
        seed=int(raw.get("seed", Settings.seed)),
        threshold=float(raw.get("threshold", Settings.threshold)),
        train_size=int(raw.get("train_size", Settings.train_size)),
        valid_size=int(raw.get("valid_size", Settings.valid_size)),
        ref_size=int(raw.get("ref_size", Settings.ref_size)),
        calib_size=int(raw.get("calib_size", Settings.calib_size)),
        test_size=int(raw.get("test_size", Settings.test_size)),
        nonconf_target=str(raw.get("nonconf_target", Settings.nonconf_target)),
        block_size=int(raw.get("block_size", Settings.block_size)),
    )
    return cfg, output_dirname, how, fit_parallel


def load_trainval(
    cfg: Settings,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read:
      - data/data.parquet
      - labels/data.labels

    Use:
      - row groups 0,1, ...train_size -> train
      - row group train_size, ..., train_size + valid_size -> validation
    """
    base_dir = cfg.data_dir
    parquet_path = base_dir / "data" / "data.parquet"
    labels_path = base_dir / "labels" / "data.labels"
    pf = pq.ParquetFile(parquet_path)
    train_tables = [pf.read_row_group(i) for i in range(cfg.train_size)]
    val_table = [
        pf.read_row_group(i)
        for i in range(cfg.train_size, cfg.train_size + cfg.valid_size)
    ]
    ref_table = [
        pf.read_row_group(i)
        for i in range(
            cfg.train_size + cfg.valid_size,
            cfg.train_size + cfg.valid_size + cfg.ref_size,
        )
    ]
    # convert to numpy
    X_train = np.vstack([t.to_pandas().to_numpy() for t in train_tables])
    X_val = np.vstack([t.to_pandas().to_numpy() for t in val_table])
    X_ref = np.vstack([t.to_pandas().to_numpy() for t in ref_table])
    # read labels
    y_all = np.loadtxt(labels_path)
    # slicing labels
    y_train = y_all[: X_train.shape[0]]
    y_val = y_all[X_train.shape[0] : X_train.shape[0] + X_val.shape[0]]
    y_ref = y_all[
        X_train.shape[0]
        + X_val.shape[0] : X_train.shape[0]
        + X_val.shape[0]
        + X_ref.shape[0]
    ]
    return X_train, y_train, X_val, y_val, X_ref, y_ref


def load_calib(
    cfg: Settings, calib_start_label_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Legge:
      - data/data.parquet
      - labels/data.labels

    Usa:
      - row groups 0,1, ...train_size -> train
      - row group train_size, ..., train_size + valid_size -> validation
    """
    base_dir = cfg.data_dir
    parquet_path = base_dir / "data" / "data.parquet"
    labels_path = base_dir / "labels" / "data.labels"

    pf = pq.ParquetFile(parquet_path)
    calib_start_idx = cfg.train_size + cfg.valid_size + cfg.ref_size
    calib_tables = [
        pf.read_row_group(i)
        for i in range(calib_start_idx, calib_start_idx + cfg.calib_size)
    ]
    # convert to numpy
    X_calib = np.vstack([t.to_pandas().to_numpy() for t in calib_tables])
    # read labels
    y_all = np.loadtxt(labels_path)
    y_calib = y_all[calib_start_label_idx : calib_start_label_idx + X_calib.shape[0]]
    calib_data = []
    metadata = []
    for i in range(0, X_calib.shape[0], cfg.block_size):
        block_length = X_calib[i : i + cfg.block_size].shape[0]
        calib_data.append(
            (X_calib[i : i + cfg.block_size], y_calib[i : i + cfg.block_size])
        )
        gamma_true = 336.0 * (
            block_length / 1000
        )  # scale gamma_true proportionally to block size
        meta_dict = {
            "mu_true": 1.0,
            "gamma_true": gamma_true,
            "beta_true": block_length - gamma_true,
            "nu_expected": block_length,
            "n_signal": y_calib[i : i + cfg.block_size].sum(),
            "n_background": (1 - y_calib[i : i + cfg.block_size]).sum(),
            "n_total": block_length,
            # "weights": weights,
        }
        metadata.append(meta_dict)
    print(
        f"Calibration sample has: {X_calib.shape[0]} events and {y_calib.shape[0]} labels."
    )
    print(
        f"Calibration events are splitted into {len(calib_data)} blocks of size {cfg.block_size} (last block may be smaller)"
    )
    return calib_data, metadata


def load_test(
    cfg: Settings, test_start_label_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Legge:
      - data/data.parquet
      - labels/data.labels

    Usa:
      - row groups 0,1, ...train_size -> train
      - row group train_size, ..., train_size + valid_size -> validation
    """
    base_dir = cfg.data_dir
    parquet_path = base_dir / "data" / "data.parquet"
    labels_path = base_dir / "labels" / "data.labels"
    pf = pq.ParquetFile(parquet_path)
    test_start_idx = cfg.train_size + cfg.valid_size + cfg.ref_size + cfg.calib_size
    test_tables = [
        pf.read_row_group(i)
        for i in range(test_start_idx, test_start_idx + cfg.test_size)
    ]
    # convert to numpy
    X_test = np.vstack([t.to_pandas().to_numpy() for t in test_tables])
    # read labels
    y_all = np.loadtxt(labels_path)
    y_test = y_all[test_start_label_idx : test_start_label_idx + X_test.shape[0]]
    test_data = []
    for i in range(0, X_test.shape[0], cfg.block_size):
        _X_test_blocks = X_test[i : i + cfg.block_size]
        _y_test_blocks = y_test[i : i + cfg.block_size]
        block_length = X_test[i : i + cfg.block_size].shape[0]
        gamma_true = 336.0 * (
            block_length / 1000
        )  # scale gamma_true proportionally to block size
        _meta_dict = {
            "mu_true": 1.0,
            "gamma_true": gamma_true,
            "beta_true": block_length - gamma_true,
            "nu_expected": block_length,
            "n_signal": y_test[i : i + cfg.block_size].sum(),
            "n_background": (1 - y_test[i : i + cfg.block_size]).sum(),
            "n_total": block_length,
            # "weights": weights,
        }
        test_data.append([_X_test_blocks, _y_test_blocks, _meta_dict])
    print(f"Test sample has: {X_test.shape[0]} events and {y_test.shape[0]} labels.")
    print(
        f"Test events are splitted into {len(test_data)} blocks of size {cfg.block_size} (last block may be smaller)"
    )
    return test_data


# get_model_efficiencies moved to conformal_predictions.reference


def main() -> None:
    global OUTPUT_DIRNAME, PLOTS_DIR, STATS_DIR, HOW, FIT_PARALLEL

    args = _parse_args()
    if args.config is not None:
        cfg, new_output_dirname, new_how, new_fit_parallel = _load_config(args.config)
        HOW = new_how
        FIT_PARALLEL = new_fit_parallel
        if new_output_dirname != OUTPUT_DIRNAME:
            OUTPUT_DIRNAME = new_output_dirname
            PLOTS_DIR = Path("results") / OUTPUT_DIRNAME / "plots"
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            STATS_DIR = Path("results") / OUTPUT_DIRNAME / "stats"
            STATS_DIR.mkdir(parents=True, exist_ok=True)
    else:
        cfg = Settings()

    start_time = datetime.now()
    print(f"Script started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    init_time = start_time

    np.random.seed(cfg.seed)

    # STEP 1: Load data
    step_start = datetime.now()
    print("\n[Data preparation...]")
    X_train, y_train, X_val, y_val, X_ref, y_ref = load_trainval(cfg)
    n_trainval = X_train.shape[0] + X_val.shape[0]
    n_ref = X_ref.shape[0]
    # X_train, y_train, X_val, y_val, X_ref, y_ref = (
    #     X_train[:10000],
    #     y_train[:10000],
    #     X_val[:5000],
    #     y_val[:5000],
    #     X_ref[:5000],
    #     y_ref[:5000],
    # )
    calib_data, calib_meta = load_calib(cfg, calib_start_label_idx=n_trainval + n_ref)
    n_calib = np.sum([_[0].shape[0] for _ in calib_data])
    # calib_data, calib_meta = calib_data[:100], calib_meta[:1000]
    test_data = load_test(cfg, test_start_label_idx=n_trainval + n_ref + n_calib)
    n_test = np.sum([_[0].shape[0] for _ in test_data])
    # test_data = test_data[:1000]
    print(f"Total train+val events: {n_trainval}")
    print(f"Total reference events: {n_ref}")
    print(f"Total calib events: {n_calib}\t{len(calib_data)} blocks")
    print(f"Total test events: {n_test}\t{len(test_data)} blocks")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_ref_scaled = scaler.transform(X_ref)

    print("Training data:")
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

    contourplot_data(X_val_scaled, y_val, output_dir=PLOTS_DIR)
    step_duration = (datetime.now() - step_start).total_seconds()
    print(
        f"Data preparation completed in {int(step_duration // 3600):02d}:{int((step_duration % 3600) // 60):02d}:{int(step_duration % 60):02d}"
    )

    # STEP 2: Train models
    step_start = datetime.now()
    print("\n[Model training...]")
    n_jobs_model = 1 if FIT_PARALLEL else -1
    models = build_models(cfg, n_jobs=n_jobs_model)
    fit_models(
        models,
        X_train_scaled,
        y_train,
        parallel=FIT_PARALLEL,
        n_jobs=-1,
        memmap_threshold="200M",
    )
    step_duration = (datetime.now() - step_start).total_seconds()
    print(
        f"Model training completed in {int(step_duration // 3600):02d}:{int((step_duration % 3600) // 60):02d}:{int(step_duration % 60):02d}"
    )

    # STEP 3: Evaluate models on validation set
    step_start = datetime.now()
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
    step_duration = (datetime.now() - step_start).total_seconds()
    print(
        f"Model scoring completed in {int(step_duration // 3600):02d}:{int((step_duration % 3600) // 60):02d}:{int(step_duration % 60):02d}"
    )

    # STEP 4: Get reference efficiencies on reference set
    step_start = datetime.now()

    ref_efficiencies_dict = {}
    for model_name, model in models.items():
        eps_signal, eps_background = get_model_efficiencies(
            model, X_ref_scaled, y_ref, cfg
        )
        ref_efficiencies_dict[model_name] = (eps_signal, eps_background)
        print(
            f"{model_name} reference efficiencies: "
            f"{eps_signal=:.4f}, "
            f"{eps_background=:.4f}"
        )

    step_duration = (datetime.now() - step_start).total_seconds()
    print(
        f"Reference efficiencies computation completed in {int(step_duration // 3600):02d}:{int((step_duration % 3600) // 60):02d}:{int(step_duration % 60):02d}"
    )

    # STEP 5: Calibration and nonconformity scores computation
    step_start = datetime.now()
    print("\n[Calibration...]")
    print("\nComputing nonconformity scores...")
    print(f"{len(calib_data)} calibration samples")
    print(
        f"Average calibration sample size: {int(np.array([_[0].shape[0] for _ in calib_data]).mean())} observations"
    )
    print(f"\t...using {cfg.nonconf_target} as target for nonconformity scores")
    nonconf_scores = {}
    for model_name, model in models.items():
        model_scores = compute_nonconformity_scores(
            {model_name: model},
            scaler,
            calib_data,
            calib_meta,
            cfg.threshold,
            target=cfg.nonconf_target,
            how=HOW,
            ref_efficiencies=ref_efficiencies_dict.get(model_name, (1.0, 1.0)),
        )
        nonconf_scores[model_name] = model_scores[model_name]

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

    mu_hat = {}
    stats = {}
    for model_name, model in models.items():
        model_mu_hat, model_stats = compute_mu_hat(
            {model_name: model},
            scaler,
            calib_data,
            calib_meta,
            cfg.threshold,
            ref_efficiencies=ref_efficiencies_dict.get(model_name, (1.0, 1.0)),
        )
        mu_hat[model_name] = model_mu_hat[model_name]
        if model_name in model_stats:
            stats[model_name] = model_stats[model_name]
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

    plot_mu_hat_distribution(
        mu_hat,
        stats,
        output_dir=PLOTS_DIR,
        pred_formula=PRED_FORMULA,
    )
    df_stats = pd.DataFrame(
        [{"Model": model_name, **stats[model_name]} for model_name in stats.keys()]
    )
    print("\nStatistics Summary:")
    print(df_stats)
    df_stats.to_csv(STATS_DIR / "mu_hat_calibration_stats.csv", index=False)
    step_duration = (datetime.now() - step_start).total_seconds()
    print(
        f"Calibration completed in {int(step_duration // 3600):02d}:{int((step_duration % 3600) // 60):02d}:{int(step_duration % 60):02d}"
    )

    step_start = datetime.now()
    print("\nRunning inference on test set...")

    mu_hat_test, mu_true_list, gamma_true_list, test_metrics = inference_on_test_set(
        models, scaler, test_data, cfg.threshold, ref_efficiencies_dict
    )

    # Compute confidence intervals for test set predictions
    print("\nComputing confidence intervals for test set...")

    # Load nonconformity scores for computing intervals
    print(f"\t...using {cfg.nonconf_target} nonconformity scores for CI computation.")
    nonconf_scores_file = STATS_DIR / f"{cfg.nonconf_target}_nonconf_scores.npz"

    for model_name, mu_hat_values in mu_hat_test.items():
        if cfg.nonconf_target == "mu_hat":
            mu_hat_lower_bounds, mu_hat_upper_bounds = compute_confidence_interval(
                np.array(mu_hat_values),
                nonconf_scores_file,
                model_name,
                how=HOW,
            )
        elif cfg.nonconf_target == "n_pred":
            n_preds = mu_hat_values * np.array(gamma_true_list)
            n_lower, n_upper = compute_confidence_interval(
                n_preds, nonconf_scores_file, model_name, how=HOW
            )
            mu_hat_lower_bounds = n_lower / np.array(gamma_true_list)
            mu_hat_upper_bounds = n_upper / np.array(gamma_true_list)
        print(f"\nModel: {model_name}\n")
        empirical_coverage = compute_empirical_coverage(
            mu_hat_lower_bounds, mu_hat_upper_bounds, mu_true_list
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
                continue
            print("printing every 50th experiment:")
            # Determine color based on whether CI contains mu_true
            color = "\033[92m" if mu_hat_lower < mu_true < mu_hat_upper else "\033[91m"
            reset = "\033[0m"
            print(
                f"  {color}Exp {exp_idx}: "
                f"$\hat{{\mu}}$: {mu_hat:.3f} "
                f"CI: [{mu_hat_lower:.3f}, {mu_hat_upper:.3f}] "
                f"$\mu_{{true}}$: {mu_true:.3f}{reset}"
            )
            print(test_metrics[model_name][exp_idx])

    step_duration = (datetime.now() - step_start).total_seconds()
    print(
        f"Testing completed in {int(step_duration // 3600):02d}:{int((step_duration % 3600) // 60):02d}:{int(step_duration % 60):02d}"
    )
    print(f"Script finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    total_duration = (datetime.now() - init_time).total_seconds()
    print(
        f"Total script duration: {int(total_duration // 3600):02d}:{int((total_duration % 3600) // 60):02d}:{int(total_duration % 60):02d}"
    )


if __name__ == "__main__":
    main()
