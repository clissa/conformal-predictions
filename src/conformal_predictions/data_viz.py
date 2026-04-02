from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

mpl.rcParams["font.size"] = 14


def plot_mu_hat_distribution(
    mu_hat: Dict[str, List[float]],
    stats: Dict[str, Dict[str, float]],
    output_dir: Path = Path("plots"),
    pred_formula: str = r"$\hat{\mu} = n_{pred} / \gamma_{true}$",
    confidence_level: float = 0.6826894921370859,
) -> None:
    """Plot mu_hat distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(mu_hat.keys())
    q_low = (1 - confidence_level) / 2
    q_high = 1 - q_low
    q_central = confidence_level

    for model_name in model_names:
        values = mu_hat[model_name]
        if len(values) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_box_aspect(1)
        ax.hist(values, bins=50, alpha=0.99, color="royalblue", density=True)

        if len(values) > 1:
            density = gaussian_kde(values)
            xs = np.linspace(min(values), max(values), 200)
            ax.plot(xs, density(xs), "k-", linewidth=2, label="KDE")

        s = stats[model_name]
        ax.axvline(
            s["quantile_lower"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"q{q_low*100:.2f}: {s['quantile_lower']:.3f}",
        )
        ax.axvline(
            s["mu_median"],
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"median: {s['mu_median']:.3f}",
        )
        ax.axvline(
            s["mu_mean"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"mean: {s['mu_mean']:.3f}",
        )
        ax.axvline(
            s["central_quantile"],
            color="tab:cyan",
            linestyle="--",
            linewidth=2,
            label=f"q{q_central*100:.2f}: {s['central_quantile']:.3f}",
        )
        ax.axvline(
            s["quantile_upper"],
            color="purple",
            linestyle="--",
            linewidth=2,
            label=f"q{q_high*100:.2f}: {s['quantile_upper']:.3f}",
        )
        ax.axvline(
            s["map"],
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"MAP: {s['map']:.3f}",
        )

        ax.set_title(
            f"$\\hat{{\\mu}}$ Distribution: {model_name} "
            f"({confidence_level * 100:.2f}% central)"
        )
        ax.set_xlabel(pred_formula)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"mu_hat_distribution_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_nonconformity_scores(
    nonconf_scores: Dict[str, List[int]],
    scores_label: str,
    output_dir: Path = Path("plots"),
) -> None:
    """Plot nonconformity score distributions for each model and comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(nonconf_scores.keys())

    for model_name in model_names:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_box_aspect(1)
        scores = nonconf_scores[model_name]

        ax.hist(scores, bins=50, alpha=0.99, color="royalblue", density=True)

        if len(scores) > 1:
            density = gaussian_kde(scores)
            xs = np.linspace(min(scores), max(scores), 200)
            ax.plot(xs, density(xs), "k-", linewidth=2, label="KDE")

        ax.set_title(f"Nonconformity Scores Distribution: {model_name}")
        ax.set_xlabel(f"Nonconformity Score ({scores_label})")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{scores_label}_scores_distribution_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    if len(model_names) > 1:
        fig, axs = plt.subplots(len(model_names), 1, figsize=(10, 5 * len(model_names)))
        if len(model_names) == 1:
            axs = [axs]

        colors = ["#4472C4", "#59A14F", "#E15759"]
        for i, model_name in enumerate(model_names):
            scores = nonconf_scores[model_name]
            bin_width = 1
            axs[i].hist(
                scores,
                bins=np.arange(min(scores), max(scores) + bin_width, bin_width),
                alpha=0.99,
                color=colors[i % len(colors)],
                density=True,
            )

            if len(scores) > 1:
                density = gaussian_kde(scores)
                xs = np.linspace(min(scores), max(scores), 200)
                axs[i].plot(xs, density(xs), "k-", linewidth=2, label="KDE")

            axs[i].set_title(f"{model_name}")
            axs[i].set_xlabel(f"Nonconformity Score ({scores_label})")
            axs[i].set_ylabel("Density")
            axs[i].legend()
            axs[i].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{scores_label}_scores_distribution_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_CI(
    y_coord: float,
    mu_hat: float,
    mu_hat_lower: float,
    mu_hat_upper: float,
    mu_true: float,
    ax: plt.Axes,
) -> plt.Axes:
    """Plot a single confidence interval on the given Axes."""
    fontsize = 11

    TRUE_COLOR = "#4E79A7"  # desaturated blue
    CORRECT_COLOR = "#59A14F"  # green (Tableau safe)
    WRONG_COLOR = "#E15759"  # muted red (Tableau safe)

    pred_color = CORRECT_COLOR if mu_hat_lower < mu_true < mu_hat_upper else WRONG_COLOR
    pred_marker = "o" if mu_hat_lower < mu_true < mu_hat_upper else "X"

    # draw CI
    ax.hlines(y_coord, mu_hat_lower, mu_hat_upper, colors=pred_color, linewidth=2)

    # mu_hat
    ax.plot(
        mu_hat,
        y_coord,
        pred_marker,
        color=pred_color,
        markersize=10,
    )
    ax.text(
        mu_hat,
        y_coord + 0.5,
        f"$\hat{{\mu}}={mu_hat:.2f}$",
        ha="center",
        va="center",
        fontsize=fontsize,
        color=pred_color,
    )

    ax.plot(mu_hat_lower, y_coord, "|", color=pred_color, linewidth=5, markersize=12)
    ax.plot(mu_hat_upper, y_coord, "|", color=pred_color, linewidth=5, markersize=12)
    ax.text(
        mu_hat_lower,
        y_coord + 0.5,
        f"CI: [{mu_hat_lower:.2f}, {mu_hat_upper:.2f}]",
        ha="left",
        va="center",
        fontsize=fontsize,
        color=pred_color,
    )

    # draw mu_true marker
    ax.plot(mu_true, y_coord, "*", color=TRUE_COLOR, markersize=12)
    ax.text(
        mu_true,
        y_coord - 0.5,
        f"$\mu={mu_true:.2f}$",
        ha="center",
        va="top",
        fontsize=fontsize,
        color=TRUE_COLOR,
    )
    return ax


LINE_SPACING = 3
PAD = 1  # top/bottom padding in "spacing units"


def plot_confidence_intervals(
    mu_hat_values: List[float],
    mu_hat_lower_bounds: List[float],
    mu_hat_upper_bounds: List[float],
    mu_true_list: List[float],
    model_name: str,
    empirical_coverage: float,
    confidence_level: float,
    output_dir: Path = Path("plots"),
) -> None:
    """Plot confidence intervals for mu_hat estimates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _experiment_idxs = range(len(mu_hat_values))

    n_plots = 5
    n_CI_per_plot = min(10, len(mu_hat_values))
    CI_y_coords = LINE_SPACING * np.arange(n_CI_per_plot)
    for plot_id in range(1, n_plots + 1):
        if len(_experiment_idxs) == 0:
            break

        # draw randomly from the remaining experiments to plot
        if len(_experiment_idxs) > n_CI_per_plot:
            _experiments_to_plot = np.random.choice(
                _experiment_idxs, size=n_CI_per_plot, replace=False
            )
        # take all remaining experiments if less than n_plots
        else:
            _experiments_to_plot = _experiment_idxs
        _experiments_to_plot = np.sort(_experiments_to_plot)

        # discard plotted experiments
        _experiment_idxs = [
            id_to_keep
            for id_to_keep in _experiment_idxs
            if id_to_keep not in _experiments_to_plot
        ]

        # prepare data for plotting
        mu_hat_values_to_plot = np.array(mu_hat_values)[_experiments_to_plot].tolist()
        mu_true_list_to_plot = np.array(mu_true_list)[_experiments_to_plot].tolist()
        mu_hat_lower_bounds_to_plot = np.array(mu_hat_lower_bounds)[
            _experiments_to_plot
        ].tolist()
        mu_hat_upper_bounds_to_plot = np.array(mu_hat_upper_bounds)[
            _experiments_to_plot
        ].tolist()
        mu_hat_lower_bounds_to_plot = np.array(mu_hat_lower_bounds)[
            _experiments_to_plot
        ].tolist()

        # create plot with proper spacing and loop through CIs
        fig, ax = plt.subplots(figsize=(12, n_CI_per_plot * LINE_SPACING / 2 + PAD))
        for CI_y, mu_hat, mu_hat_lower, mu_hat_upper, mu_true in zip(
            CI_y_coords,
            mu_hat_values_to_plot,
            mu_hat_lower_bounds_to_plot,
            mu_hat_upper_bounds_to_plot,
            mu_true_list_to_plot,
        ):
            ax = plot_CI(CI_y, mu_hat, mu_hat_lower, mu_hat_upper, mu_true, ax)

        coverage_legend = (
            f"Empirical coverage: {empirical_coverage*100:.3f}%\n"
            f"Confidence level (1-$\\alpha$): {confidence_level*100:.2f}%"
        )
        ax.text(
            0.95,
            0.99,
            coverage_legend,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_yticks(CI_y_coords)
        ax.set_yticklabels([f"{exp_id}" for exp_id in _experiments_to_plot])

        ax.set_xlabel("$\mu$ value")
        ax.set_ylabel("Experiment ID")
        ax.set_title(f"Confidence Intervals: {model_name}")
        ax.grid(alpha=0.3, axis="x")
        ax.set_ylim(-PAD * LINE_SPACING, (n_CI_per_plot - 1 + PAD) * LINE_SPACING)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"test_CI_plots-{plot_id}_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def contourplot_data(
    X: np.ndarray, y: np.ndarray, output_dir: Path = Path("plots")
) -> None:
    """
    Plot 2D density contours with overlaid scatter points for
    background (y=0) and signal (y=1).

    Background is drawn first, then signal.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # downsample if too large for better visualization and performance
    max_n_points = 500_000
    if len(X) > max_n_points:
        idx = np.random.choice(len(X), size=max_n_points, replace=False)
        X = X[idx]
        y = y[idx]

    if X.shape[1] < 2:
        return
    elif X.shape[1] > 2:
        # use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2, random_state=0)
        X = pca.fit_transform(X)
        var_ratio = pca.explained_variance_ratio_
        feature_names = [
            f"PCA 1 ({var_ratio[0]*100:.1f}%)",
            f"PCA 2 ({var_ratio[1]*100:.1f}%)",
        ]

    signal = X[y == 1][:, :2]
    background = X[y == 0][:, :2]

    fig, ax = plt.subplots(figsize=(9, 7))
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    dx = 0.05 * (x_max - x_min)
    dy = 0.05 * (y_max - y_min)
    x_grid, y_grid = np.mgrid[
        (x_min - dx) : (x_max + dx) : 200j,
        (y_min - dy) : (y_max + dy) : 200j,
    ]
    grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
    background_kde = gaussian_kde(background.T)
    signal_kde = gaussian_kde(signal.T)
    background_density = background_kde(grid).reshape(x_grid.shape)
    signal_density = signal_kde(grid).reshape(x_grid.shape)
    background_levels = np.percentile(background_density, [50, 75, 90, 95, 99])
    signal_levels = np.percentile(signal_density, [50, 75, 90, 95, 99])
    ax.contourf(
        x_grid,
        y_grid,
        background_density,
        levels=background_levels,
        cmap="Blues",
        alpha=0.6,
    )
    ax.contour(
        x_grid,
        y_grid,
        background_density,
        levels=background_levels,
        colors="blue",
        linewidths=1.5,
    )
    ax.scatter(
        background[:, 0],
        background[:, 1],
        s=8,
        c="blue",
        alpha=0.25,
        label=f"Background ({len(background)} events)",
        rasterized=True,
    )
    ax.contourf(
        x_grid,
        y_grid,
        signal_density,
        levels=signal_levels,
        cmap="Reds",
        alpha=0.6,
    )
    ax.contour(
        x_grid,
        y_grid,
        signal_density,
        levels=signal_levels,
        colors="red",
        linewidths=1.5,
    )
    ax.scatter(
        signal[:, 0],
        signal[:, 1],
        s=10,
        c="red",
        alpha=0.25,
        label=f"Signal ({len(signal)} events)",
        rasterized=True,
    )

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(f"Signal vs Background Density ({min(len(X), max_n_points)} events)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "data_contour.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
