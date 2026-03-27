"""Model factory, fitting, and persistence utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tqdm.auto import tqdm

from conformal_predictions.config import PipelineConfig

# Default hyperparameters used when none are supplied via config.
_DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "GLM": {
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    "Random Forest": {
        "n_estimators": 50,
        "criterion": "gini",
    },
    "MLP": {
        "hidden_layer_sizes": (32, 16),
        "activation": "relu",
        "max_iter": 1000,
    },
}

_MODEL_CLASSES: Dict[str, type] = {
    "GLM": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "MLP": MLPClassifier,
}


def build_models(
    config: PipelineConfig,
    *,
    model_names: Sequence[str] | None = None,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Instantiate (unfitted) models according to *config*.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.  ``config.seed`` is used for
        reproducibility and ``config.model_hyperparams`` (if set)
        overrides the default hyperparameters per model.
    model_names : sequence of str, optional
        If given, only build models whose names appear in this sequence.
        Names must match keys in ``_MODEL_CLASSES`` (e.g. ``"GLM"``,
        ``"Random Forest"``, ``"MLP"``).  When *None* (default), all
        available models are built.
    n_jobs : int, optional
        Number of parallel jobs for models that support it (e.g.
        ``RandomForestClassifier``).  When fitting in parallel via
        :func:`fit_models`, callers typically pass ``n_jobs=1`` here so
        that parallelism happens at the model level instead.

    Returns
    -------
    Dict[str, model]
        Mapping of model name → unfitted sklearn estimator.
    """
    hyperparams = getattr(config, "model_hyperparams", None) or _DEFAULT_HYPERPARAMS
    selected = (
        _MODEL_CLASSES.items()
        if model_names is None
        else [(n, c) for n, c in _MODEL_CLASSES.items() if n in model_names]
    )
    models: Dict[str, Any] = {}
    for name, cls in selected:
        params = dict(hyperparams.get(name, {}))
        params["random_state"] = config.seed
        if name == "Random Forest":
            params.setdefault("n_jobs", n_jobs)
        models[name] = cls(**params)
    return models


def _fit_one(name: str, model: Any, X: np.ndarray, y: np.ndarray) -> Tuple[str, Any]:
    """Fit a single model and return ``(name, fitted_model)``."""
    model.fit(X, y)
    return name, model


def fit_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    parallel: bool = False,
    n_jobs: int = -1,
    prefer_threads: bool = False,
    memmap_threshold: str = "200M",
    blas_threads: int = 1,
) -> None:
    """Fit all *models* in-place, either sequentially or in parallel.

    Parameters
    ----------
    models : Dict[str, model]
        Mapping returned by :func:`build_models`.  Models are fitted
        **in-place** (the dict values are mutated).
    X_train, y_train : np.ndarray
        Training features and labels.
    parallel : bool
        If ``True``, use :mod:`joblib` to fit models concurrently.
    n_jobs, prefer_threads, memmap_threshold, blas_threads
        Forwarded to :class:`joblib.Parallel` when *parallel* is True.
    """
    if parallel:
        _fit_parallel(
            models,
            X_train,
            y_train,
            n_jobs=n_jobs,
            prefer_threads=prefer_threads,
            memmap_threshold=memmap_threshold,
            blas_threads=blas_threads,
        )
    else:
        for model in tqdm(models.values(), desc="Training models"):
            model.fit(X_train, y_train)


def _fit_parallel(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_jobs: int = -1,
    prefer_threads: bool = False,
    memmap_threshold: str = "200M",
    blas_threads: int = 1,
) -> None:
    """Fit models in parallel using joblib (internal helper)."""
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits

    # Avoid nested parallelism: force per-model n_jobs=1
    for model in models.values():
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1

    backend = "threading" if prefer_threads else "loky"
    items = list(models.items())

    with threadpool_limits(limits=blas_threads):
        results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            max_nbytes=memmap_threshold,
        )(
            delayed(_fit_one)(name, model, X_train, y_train)
            for name, model in tqdm(items, desc=f"Training models ({backend})")
        )

    models.update(dict(results))


# -- persistence ---------------------------------------------------------


def save_models(models: Dict[str, Any], output_dir: Path) -> None:
    """Save each fitted model to *output_dir* as ``<name>.joblib``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = output_dir / f"{name}.joblib"
        joblib.dump(model, path)


def load_models(output_dir: Path) -> Dict[str, Any]:
    """Load all ``.joblib`` models from *output_dir*."""
    output_dir = Path(output_dir)
    models: Dict[str, Any] = {}
    for path in sorted(output_dir.glob("*.joblib")):
        models[path.stem] = joblib.load(path)
    return models


def load_model(output_dir: Path, name: str) -> Any:
    """Load a single model by *name* from *output_dir*."""
    path = Path(output_dir) / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model file found at {path}")
    return joblib.load(path)
