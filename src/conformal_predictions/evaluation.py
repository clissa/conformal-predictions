from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from conformal_predictions.calibration import (
    _get_expected_background,
    _get_expected_signal,
    _get_proportionate_beta_true,
    _get_proportionate_gamma_true,
)


def evaluate_models(
    models: Dict[str, object], X: np.ndarray, y: np.ndarray
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        y_pred = model.predict(X)
        results[name] = {
            "accuracy": float(model.score(X, y)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
        }
    return results


def get_events_count(
    models: Dict[str, object], X: np.ndarray, threshold: float
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X)[:, 1]
        counts[name] = int(np.sum(y_pred_proba > threshold))
    return counts


def inference_on_test_set(
    models: Dict[str, object],
    scaler: StandardScaler,
    test_data: Sequence[Tuple[np.ndarray, np.ndarray, dict]],
    threshold: float,
    mu_hat_mode: str,
    ref_efficiencies_dict: Dict[str, Sequence[float]] = None,
    debug: bool = False,
) -> Tuple[
    Dict[str, List[float]],
    List[float],
    List[int],
    Dict[str, List[Dict[str, float]]],
]:
    """
    Compute mu_hat estimates on test set pseudo-experiments.

    Args:
        models: Dictionary of trained models
        scaler: Fitted StandardScaler for feature normalization
        test_data: Sequence of (X, y, meta_dict) tuples for test experiments
        threshold: Decision threshold for classification

    Returns:
        mu_hat_test: Dict mapping model names to lists of
            mu_hat values (one per experiment)
        mu_true_list: List of mu_true values (one per experiment)
        gamma_true_list: List of gamma_true values (one per experiment)
        performance_metrics: Dictionary mapping model names to lists of metric dicts
    """
    mu_hat_test: Dict[str, List[float]] = {name: [] for name in models}
    performance_metrics: Dict[str, List[Dict[str, float]]] = {
        name: [] for name in models
    }
    mu_true_list: List[float] = []
    gamma_true_list: List[int] = []

    for X_test, y_test, meta_dict in tqdm(test_data, desc="Inference on test set"):
        X_test_scaled = scaler.transform(X_test)

        gamma_true = _get_proportionate_gamma_true(meta_dict)
        beta_true = _get_proportionate_beta_true(meta_dict)
        mu_true = meta_dict["mu_true"]
        mu_true_list.append(float(mu_true))
        gamma_true_list.append(int(gamma_true))

        if gamma_true == 0:
            continue

        if ref_efficiencies_dict is None:
            ref_efficiencies_dict = {model_name: (1.0, 1.0) for model_name in models}
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = y_pred_proba > threshold

            # classification metrics
            _metrics = {
                "accuracy": float(model.score(X_test_scaled, y_test)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            }
            performance_metrics[name].append(_metrics)

            # counting metrics
            n_pred = int(np.sum(y_pred))
            if mu_hat_mode == "raw":
                mu_hat = n_pred / gamma_true if gamma_true > 0 else 0.0
            elif mu_hat_mode == "mle":
                expected_signal = _get_expected_signal(
                    gamma_true, ref_efficiencies_dict[name][0]
                )
                expected_background = _get_expected_background(
                    beta_true, ref_efficiencies_dict[name][1]
                )
                mu_hat = (
                    (n_pred - expected_background) / expected_signal
                    if expected_signal > 0
                    else 0.0
                )
            else:
                raise ValueError(f"Unknown mu_hat_mode: {mu_hat_mode}")
            mu_hat_test[name].append(mu_hat)

            if debug:
                print("\nDebug prints:", name)
                n_obs = int(np.sum(y_test))
                print(
                    f"\tExperiment: mu_true={mu_true:.4f}, "
                    f"gamma_true={gamma_true}, n_obs={n_obs}, "
                    f"n_pred={n_pred}, mu_hat={mu_hat:.4f}"
                )

    return mu_hat_test, mu_true_list, gamma_true_list, performance_metrics


def compute_confidence_interval(
    y_pred,
    nonconf_scores_file: Path,
    model_name: str,
    interval_mode: str,
    alpha: float,
) -> Tuple[float, float]:
    """
    Compute confidence interval from calibration nonconformity scores.

    Args:
        y_pred: Predicted value for which to compute the confidence interval
        nonconf_scores_file: Path to .npz file containing nonconformity scores
        model_name: Name of the model to extract scores for
        interval_mode: Interval construction mode ("symmetric" or "free_form")
        alpha: Miscoverage level for the central interval
    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    data = np.load(nonconf_scores_file)
    if model_name not in data:
        raise KeyError(f"Model '{model_name}' not found in {nonconf_scores_file}")

    scores = data[model_name]

    if interval_mode == "free_form":
        q_low = float(np.percentile(scores, 100.0 * alpha / 2.0))
        q_high = float(np.percentile(scores, 100.0 * (1.0 - alpha / 2.0)))
    elif interval_mode == "symmetric":
        q = float(np.percentile(scores, 100.0 * (1.0 - alpha)))
        q_low = -q
        q_high = q
    else:
        raise ValueError(f"Unknown interval_mode: {interval_mode}")

    lower_bound = y_pred + q_low
    upper_bound = y_pred + q_high

    return lower_bound, upper_bound


def compute_empirical_coverage(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    true_values: Sequence[float],
) -> float:
    """Fraction of experiments where the true value falls within the CI.

    Args:
        lower_bounds: Lower CI bounds (one per experiment).
        upper_bounds: Upper CI bounds (one per experiment).
        true_values: True parameter values (one per experiment).

    Returns:
        Empirical coverage in [0, 1].
    """
    return float(
        np.mean(
            [
                lo < mu < hi
                for lo, hi, mu in zip(lower_bounds, upper_bounds, true_values)
            ]
        )
    )
