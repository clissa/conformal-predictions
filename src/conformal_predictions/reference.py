"""Reference efficiency computation and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from conformal_predictions.config import PipelineConfig


def get_model_efficiencies(
    model: Any,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    config: Any,
) -> Tuple[float, float]:
    """Compute signal and background efficiencies on a reference set.

    Parameters
    ----------
    model : fitted sklearn estimator
        A classifier exposing ``predict_proba``.
    X_ref, y_ref : np.ndarray
        Reference features and binary labels (1 = signal, 0 = background).
    config : PipelineConfig or any object with a ``threshold`` attribute
        Pipeline configuration (uses ``config.threshold``).

    Returns
    -------
    (eps_signal, eps_background) : Tuple[float, float]
    """
    y_pred = (model.predict_proba(X_ref)[:, 1] > config.threshold).astype(int)

    gamma_raw = np.sum(y_pred * (y_ref == 1))  # true positives
    beta_raw = np.sum(y_pred * (y_ref == 0))  # false positives

    n_signal = np.sum(y_ref == 1)
    n_background = np.sum(y_ref == 0)

    eps_signal = gamma_raw / n_signal if n_signal > 0 else 0.0
    eps_background = beta_raw / n_background if n_background > 0 else 0.0

    return float(eps_signal), float(eps_background)


def get_all_model_efficiencies(
    models: Dict[str, Any],
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    config: PipelineConfig,
) -> Dict[str, Tuple[float, float]]:
    """Compute reference efficiencies for every model in *models*.

    For toy data (``config.data_source == "toy"``), returns ``(1.0, 1.0)``
    for each model without touching the reference data.

    Parameters
    ----------
    models : Dict[str, fitted model]
    X_ref, y_ref : np.ndarray
    config : PipelineConfig

    Returns
    -------
    Dict[str, (eps_signal, eps_background)]
    """
    if config.data_source == "toy":
        return {name: (1.0, 1.0) for name in models}

    return {
        name: get_model_efficiencies(model, X_ref, y_ref, config)
        for name, model in models.items()
    }


# -- persistence ---------------------------------------------------------


def save_efficiencies(
    efficiencies: Dict[str, Tuple[float, float]],
    output_dir: Path,
) -> Path:
    """Save efficiencies to JSON, merging with any existing entries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "reference_efficiencies.json"

    serializable = {
        name: {"eps_signal": eps_s, "eps_background": eps_b}
        for name, (eps_s, eps_b) in efficiencies.items()
    }

    if path.exists():
        with open(path) as fh:
            existing = json.load(fh)
        existing.update(serializable)
        serializable = existing

    with open(path, "w") as fh:
        json.dump(serializable, fh, indent=2)
    return path


def load_efficiencies(output_dir: Path) -> Dict[str, Tuple[float, float]]:
    """Load efficiencies from a JSON file saved by
    :func:`save_efficiencies`."""
    path = Path(output_dir) / "reference_efficiencies.json"
    with open(path) as fh:
        raw = json.load(fh)
    return {
        name: (vals["eps_signal"], vals["eps_background"]) for name, vals in raw.items()
    }
