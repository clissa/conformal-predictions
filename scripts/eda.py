"""EDA / data-visualization script for toy and HiggsML experiments.

Usage:
    python scripts/explore.py --config configs/train_toy.yaml
    python scripts/explore.py --config configs/train_higgs.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for batch runs

from conformal_predictions.config import PipelineConfig, load_config
from conformal_predictions.data_viz import contourplot_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis for toy or HiggsML data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML pipeline config file.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loaders (dispatch by data_source)
# ---------------------------------------------------------------------------
def _load_toy_data(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load toy pseudo-experiments and stack them into a single (X, y) pair."""
    from conformal_predictions.data.toy import list_split_files, load_pseudo_experiment

    train_files, val_files, _, _ = list_split_files(
        data_dir=cfg.data_dir,
        mu=cfg.mu,
        test_prefixes=cfg.test_prefixes,
        n_test_experiments=cfg.n_test_experiments,
        valid_size=cfg.valid_size,
        calib_size=cfg.calib_size,
        seed=cfg.seed,
    )
    files = train_files + val_files
    if not files:
        raise FileNotFoundError(
            f"No .npz files found for mu={cfg.mu} in {cfg.data_dir}"
        )

    Xs, ys = [], []
    for f in files:
        X_i, y_i, _ = load_pseudo_experiment(f)
        Xs.append(X_i)
        ys.append(y_i)
    return np.vstack(Xs), np.concatenate(ys)


def _load_higgs_data(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load a subset of HiggsML data (train + val row groups)."""
    import pyarrow.parquet as pq

    base_dir = cfg.data_dir
    parquet_path = base_dir / "data" / "data.parquet"
    labels_path = base_dir / "labels" / "data.labels"

    pf = pq.ParquetFile(str(parquet_path))
    n_groups = cfg.train_size or 1
    if cfg.valid_size is not None:
        n_groups += int(cfg.valid_size)
    n_groups = min(n_groups, pf.metadata.num_row_groups)

    tables = [pf.read_row_group(i) for i in range(n_groups)]
    X = np.vstack([t.to_pandas().to_numpy() for t in tables])
    y_all = np.loadtxt(str(labels_path))
    y = y_all[: X.shape[0]]
    return X, y


def load_data(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    if cfg.data_source == "toy":
        return _load_toy_data(cfg)
    elif cfg.data_source == "higgs":
        return _load_higgs_data(cfg)
    else:
        raise ValueError(f"Unknown data_source: {cfg.data_source!r}")


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------
def print_class_balance(y: np.ndarray, output_dir: Path) -> None:
    """Print and save class-balance statistics."""
    n_total = len(y)
    n_signal = int((y == 1).sum())
    n_background = int((y == 0).sum())
    sig_frac = n_signal / n_total if n_total else 0.0
    bkg_frac = n_background / n_total if n_total else 0.0

    lines = [
        "Class balance",
        "─" * 40,
        f"  Total events:      {n_total:>10,}",
        f"  Signal  (y=1):     {n_signal:>10,}  ({sig_frac:6.2%})",
        f"  Background (y=0):  {n_background:>10,}  ({bkg_frac:6.2%})",
        f"  Ratio sig/bkg:     {n_signal / max(n_background, 1):>10.4f}",
    ]
    report = "\n".join(lines)
    print(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "class_balance.txt").write_text(report + "\n")


def plot_feature_distributions(X: np.ndarray, y: np.ndarray, output_dir: Path) -> None:
    """Plot per-feature histograms split by signal/background."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_features = X.shape[1]

    # Limit to at most 30 features to keep plots manageable
    n_plot = min(n_features, 30)

    ncols = min(3, n_plot)
    nrows = int(np.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    signal_mask = y == 1
    bkg_mask = y == 0

    for idx in range(n_plot):
        ax = axes.flat[idx]
        ax.hist(
            X[bkg_mask, idx],
            bins=50,
            alpha=0.5,
            density=True,
            color="blue",
            label="Background",
        )
        ax.hist(
            X[signal_mask, idx],
            bins=50,
            alpha=0.5,
            density=True,
            color="red",
            label="Signal",
        )
        ax.set_title(f"Feature {idx}")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    # Hide unused axes
    for idx in range(n_plot, len(axes.flat)):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature distributions saved to {output_dir / 'feature_distributions.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    output_dir = cfg.plots_dir

    print(f"Data source: {cfg.data_source}")
    print(f"Data dir:    {cfg.data_dir}")
    print(f"Output:      {output_dir}\n")

    X, y = load_data(cfg)
    print(f"Loaded {X.shape[0]:,} events, {X.shape[1]} features\n")

    # 1. Class balance
    print_class_balance(y, cfg.stats_dir)
    print()

    # 2. Contour plot (signal vs background density)
    contourplot_data(X, y, output_dir=output_dir)
    print(f"Contour plot saved to {output_dir / 'data_contour.png'}")

    # 3. Feature distributions
    plot_feature_distributions(X, y, output_dir=output_dir)


if __name__ == "__main__":
    main()
