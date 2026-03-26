"""Preprocessing utilities: StandardScaler fit, save, load."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training data and return it."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_scaler(scaler: StandardScaler, output_dir: Path) -> Path:
    """Save a fitted scaler to *output_dir*/scaler.joblib."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scaler.joblib"
    joblib.dump(scaler, path)
    return path


def load_scaler(output_dir: Path) -> StandardScaler:
    """Load a scaler from *output_dir*/scaler.joblib."""
    path = Path(output_dir) / "scaler.joblib"
    return joblib.load(path)
