from __future__ import annotations

import math
from typing import Tuple

from scipy.optimize import brentq
from scipy.stats import chi2, norm


def compute_mle_mu_hat(n_obs: int, s: float, b: float) -> float:
    """Return the unconstrained Poisson MLE for mu."""
    if s <= 0.0:
        raise ValueError(f"Expected positive signal yield scale s, got {s!r}.")
    return (float(n_obs) - float(b)) / float(s)


def _poisson_log_likelihood(lam: float, n_obs: int) -> float:
    if lam < 0.0:
        raise ValueError(f"Expected non-negative Poisson mean, got {lam!r}.")
    if lam == 0.0:
        if n_obs == 0:
            return 0.0
        return float("-inf")
    return float(n_obs) * math.log(lam) - lam


def _compute_lambda_interval_wald(n_obs: int, alpha: float) -> Tuple[float, float]:
    z_value = float(norm.ppf(1.0 - alpha / 2.0))
    lambda_hat = float(n_obs)
    sigma = math.sqrt(max(lambda_hat, 1e-12))
    return max(0.0, lambda_hat - z_value * sigma), lambda_hat + z_value * sigma


def _compute_lambda_interval_exact(n_obs: int, alpha: float) -> Tuple[float, float]:
    if n_obs < 0:
        raise ValueError(f"Expected non-negative observed count, got {n_obs!r}.")
    if n_obs == 0:
        lower = 0.0
    else:
        lower = 0.5 * float(chi2.ppf(alpha / 2.0, 2 * n_obs))
    upper = 0.5 * float(chi2.ppf(1.0 - alpha / 2.0, 2 * (n_obs + 1)))
    return lower, upper


def _profile_likelihood_statistic(lam: float, n_obs: int) -> float:
    lambda_hat = float(n_obs)
    return 2.0 * (
        _poisson_log_likelihood(lambda_hat, n_obs)
        - _poisson_log_likelihood(lam, n_obs)
    )


def _compute_lambda_interval_profile(n_obs: int, alpha: float) -> Tuple[float, float]:
    threshold = float(chi2.ppf(1.0 - alpha, df=1))

    if n_obs == 0:
        upper = 0.5 * threshold
        return 0.0, upper

    lambda_hat = float(n_obs)

    def objective(lam: float) -> float:
        return _profile_likelihood_statistic(lam, n_obs) - threshold

    lower = float(brentq(objective, 1e-12, lambda_hat))

    upper_left = lambda_hat
    upper_right = max(lambda_hat + 1.0, 2.0 * lambda_hat)
    while objective(upper_right) < 0.0:
        upper_right *= 2.0

    upper = float(brentq(objective, upper_left, upper_right))
    return lower, upper


def compute_mle_interval(
    n_obs: int,
    s: float,
    b: float,
    alpha: float,
    *,
    kind: str,
) -> Tuple[float, float]:
    """Compute an interval for mu from a Poisson counting model.

    The observed count ``n_obs`` is modeled as ``Poisson(mu * s + b)``.
    Intervals are constructed on the Poisson mean and then translated to ``mu``.
    """
    if s <= 0.0:
        raise ValueError(f"Expected positive signal yield scale s, got {s!r}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"Expected alpha in (0, 1), got {alpha!r}.")
    if n_obs < 0:
        raise ValueError(f"Expected non-negative observed count, got {n_obs!r}.")

    if kind == "wald":
        lambda_lower, lambda_upper = _compute_lambda_interval_wald(n_obs, alpha)
    elif kind == "profile":
        lambda_lower, lambda_upper = _compute_lambda_interval_profile(n_obs, alpha)
    elif kind == "exact":
        lambda_lower, lambda_upper = _compute_lambda_interval_exact(n_obs, alpha)
    else:
        raise ValueError(f"Unknown MLE interval kind: {kind!r}")

    mu_lower = (lambda_lower - float(b)) / float(s)
    mu_upper = (lambda_upper - float(b)) / float(s)
    return float(mu_lower), float(mu_upper)
