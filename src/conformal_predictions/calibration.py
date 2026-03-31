from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def _random_perturbation_for_numerical_stability() -> float:
    return np.random.normal(0, 1e-6)


def _nonconformity_scores(pred, target, interval_mode: str) -> float:
    """Compute a nonconformity score for the requested interval mode."""
    if interval_mode == "free_form":
        score = target - pred
    elif interval_mode == "symmetric":
        score = abs(target - pred)
    else:
        raise ValueError(f"Unknown interval_mode: {interval_mode}")
    return score + _random_perturbation_for_numerical_stability()


def _get_proportionate_gamma_true(meta: dict) -> float:
    return meta["gamma_true"] / meta["nu_expected"] * meta["n_total"]


def _get_expected_signal(gamma_true: float, eps_signal: float) -> float:
    return gamma_true * eps_signal


def _get_proportionate_beta_true(meta: dict) -> float:
    return meta["beta_true"] / meta["nu_expected"] * meta["n_total"]


def _get_expected_background(beta_true: int, eps_background: float) -> float:
    return beta_true * eps_background


def _compute_mu_hat(
    n_pred: int,
    meta: dict,
    mu_hat_mode: str,
    ref_efficiencies: Optional[Sequence[float]] = None,
) -> float:
    gamma_true = _get_proportionate_gamma_true(meta)
    if mu_hat_mode == "raw":
        return n_pred / gamma_true if gamma_true > 0 else 0.0

    if mu_hat_mode != "mle":
        raise ValueError(f"Unknown mu_hat_mode: {mu_hat_mode}")

    if ref_efficiencies is None:
        raise ValueError("ref_efficiencies are required for mu_hat_mode='mle'")

    beta_true = _get_proportionate_beta_true(meta)
    expected_signal = _get_expected_signal(gamma_true, ref_efficiencies[0])
    expected_background = _get_expected_background(beta_true, ref_efficiencies[1])
    mu_hat = (
        (n_pred - expected_background) / expected_signal if expected_signal > 0 else 0.0
    )
    return mu_hat


def compute_nonconformity_scores(
    models: Dict[str, object],
    scaler: StandardScaler,
    calib_data: Sequence[Tuple[np.ndarray, np.ndarray]],
    calib_meta: Sequence[dict],
    threshold: float,
    *,
    target: str = "mu_hat",  # can be "n_pred" or "mu_hat",
    interval_mode: str,
    mu_hat_mode: str,
    ref_efficiencies: Optional[Sequence[float]] = None,
) -> Dict[str, List[int]]:
    scores: Dict[str, List[int]] = {name: [] for name in models}
    for (X_calib, y_calib), _meta in tqdm(
        zip(calib_data, calib_meta),
        total=len(calib_data),
        desc="Computing nonconformity scores",
    ):

        X_calib = scaler.transform(X_calib)
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_calib)[:, 1]
            n_pred = int(np.sum(y_pred_proba > threshold))
            if target == "mu_hat":
                mu_true = _meta["mu_true"]
                mu_hat = _compute_mu_hat(
                    n_pred,
                    _meta,
                    mu_hat_mode,
                    ref_efficiencies,
                )
                scores[name].append(
                    _nonconformity_scores(mu_hat, mu_true, interval_mode)
                )
            elif target == "n_pred":
                n_obs = int(np.sum(y_calib))
                scores[name].append(_nonconformity_scores(n_pred, n_obs, interval_mode))
    return scores


def compute_mu_hat(
    models: Dict[str, object],
    scaler: StandardScaler,
    calib_data: Sequence[Tuple[np.ndarray, np.ndarray]],
    calib_meta: Sequence[dict],
    threshold: float,
    mu_hat_mode: str,
    ref_efficiencies: Sequence[float],
    alpha: float,
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    mu_hat: Dict[str, List[float]] = {name: [] for name in models}
    for (X_calib, y_calib), meta in zip(calib_data, calib_meta):
        X_calib = scaler.transform(X_calib)
        # gamma_true = _get_proportionate_gamma_true(meta)
        if meta["gamma_true"] == 0:
            continue
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_calib)[:, 1]
            n_pred = int(np.sum(y_pred_proba > threshold))
            mu_pred = _compute_mu_hat(
                n_pred,
                meta,
                mu_hat_mode,
                ref_efficiencies,
            )
            mu_hat[name].append(mu_pred)

    stats: Dict[str, Dict[str, float]] = {}
    for name, values in mu_hat.items():
        if len(values) > 0:
            density = gaussian_kde(values)
            xs = np.linspace(min(values), max(values), 1000)
            density_vals = density(xs)
            map_estimate = float(xs[np.argmax(density_vals)])
            lower_percentile = 100.0 * alpha / 2.0
            upper_percentile = 100.0 * (1.0 - alpha / 2.0)
            central_percentile = 100.0 * (1.0 - alpha)

            stats[name] = {
                "quantile_lower": float(np.percentile(values, lower_percentile)),
                "map": map_estimate,
                "mu_median": float(np.median(values)),
                "mu_mean": float(np.mean(values)),
                "central_quantile": float(np.percentile(values, central_percentile)),
                "quantile_upper": float(np.percentile(values, upper_percentile)),
            }

    return mu_hat, stats
