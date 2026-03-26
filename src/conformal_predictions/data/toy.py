from __future__ import annotations

import json
import uuid
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from numpy.typing import NDArray
from typing_extensions import TypeAlias

# -------------------------
# Type aliases
# -------------------------
Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
Int64Array: TypeAlias = NDArray[np.int64]
MetaDict: TypeAlias = Dict[str, Any]


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class ToyConfig:
    """
    Toy data generator config.

    Core physics-inspired model:
        nu = mu * gamma + beta
        N_signal ~ Poisson(mu * gamma)
        N_background ~ Poisson(beta)

    Once n_signal and n_background are drawn, features are sampled from
    class-conditional multivariate normals. This generates one pseudo-experiment.

    Attributes signal_weight and background_weight define per-class expected yields
    (event-equivalent totals) in the pseudo-experiment, implemented via per-event weights:
        sum(weights_signal)     = signal_weight
        sum(weights_background) = background_weight

    Notes on reproducibility:
    - seed is deterministic by default.
    - set seed=None for statistical (non-deterministic) runs.
    """

    # Physics / counting parameters
    mu: float  # signal strength
    gamma: float  # expected signal events
    beta: float  # expected background events

    # Feature space
    n_features: int  # number of features (dimensions)

    # Signal feature distribution parameters
    signal_mean: Float64Array  # shape: (n_features,)
    signal_std: Float64Array  # shape: (n_features,)
    signal_rho: Float64Array  # shape: (n_features-1,)

    # Background feature distribution parameters
    background_mean: Float64Array  # shape: (n_features,)
    background_std: Float64Array  # shape: (n_features,)
    background_rho: Float64Array  # shape: (n_features-1,)

    # Expected yields used for per-event weights (yield-conserving MC)
    signal_weight: float
    background_weight: float

    # Random seed (None → statistical reproducibility only)
    seed: Optional[int] = 4


# -------------------------
# Public API
# -------------------------
def generate_pseudo_experiment(
    cfg: ToyConfig,
    pseudo_experiment_id: Optional[str] = None,
) -> Tuple[Float32Array, Int64Array, MetaDict]:
    """
    Generate a single pseudo-experiment according to ToyConfig.

    pseudo_experiment_id:
        Hex string identifier (16 chars). If None, a random one is generated.

    Returns:
        X: shape (n_total, n_features), dtype float32
        y: shape (n_total,), dtype int64 (1=signal, 0=background)
        meta: dict containing:
            - pseudo_experiment_id
            - mu_true, gamma_true, beta_true, nu_expected
            - n_signal, n_background, n_total
            - weights: per-sample weights (float32)
            - feature_params: means/std/rho/cov used (float64)
            - seed used (base seed, can be None)
    """
    if pseudo_experiment_id is None:
        pseudo_experiment_id = uuid.uuid4().hex[:16]

    _validate_pseudo_experiment_inputs(cfg, pseudo_experiment_id)

    if cfg.seed is None:
        rng = np.random.default_rng()  # fully random
    else:
        experiment_seed = (int(cfg.seed) + int(pseudo_experiment_id, 16)) % (2**32)
        rng = np.random.default_rng(experiment_seed)

    n_signal = int(rng.poisson(lam=float(cfg.mu * cfg.gamma)))
    n_background = int(rng.poisson(lam=float(cfg.beta)))
    n_total = n_signal + n_background

    if n_total == 0:
        X_empty: Float32Array = np.empty((0, cfg.n_features), dtype=np.float32)
        y_empty: Int64Array = np.empty((0,), dtype=np.int64)
        w_empty: Float32Array = np.empty((0,), dtype=np.float32)
        meta_empty: MetaDict = {
            "pseudo_experiment_id": pseudo_experiment_id,
            "mu_true": float(cfg.mu),
            "gamma_true": float(cfg.gamma),
            "beta_true": float(cfg.beta),
            "nu_expected": float(cfg.mu * cfg.gamma + cfg.beta),
            "n_signal": 0,
            "n_background": 0,
            "n_total": 0,
            "weights": w_empty,
            "feature_params": _feature_params_dict(cfg),
            "seed": cfg.seed,
        }
        return X_empty, y_empty, meta_empty

    signal_cov = _covariance_from_std_and_adjacent_rho(
        standard_deviations=cfg.signal_std,
        adjacent_correlations=cfg.signal_rho,
    )
    background_cov = _covariance_from_std_and_adjacent_rho(
        standard_deviations=cfg.background_std,
        adjacent_correlations=cfg.background_rho,
    )

    feature_blocks: list[Float32Array] = []
    label_blocks: list[Int64Array] = []
    weight_blocks: list[Float32Array] = []

    # Signal block
    if n_signal > 0:
        signal_features: Float32Array = rng.multivariate_normal(
            mean=cfg.signal_mean,
            cov=signal_cov,
            size=n_signal,
        ).astype(np.float32, copy=False)
        signal_labels: Int64Array = np.ones((n_signal,), dtype=np.int64)

        # yield-conserving weights: sum = signal_weight
        signal_event_weight = float(cfg.signal_weight / n_signal)
        signal_weights: Float32Array = np.full(
            (n_signal,), fill_value=signal_event_weight, dtype=np.float32
        )

        feature_blocks.append(signal_features)
        label_blocks.append(signal_labels)
        weight_blocks.append(signal_weights)

    # Background block
    if n_background > 0:
        background_features: Float32Array = rng.multivariate_normal(
            mean=cfg.background_mean,
            cov=background_cov,
            size=n_background,
        ).astype(np.float32, copy=False)
        background_labels: Int64Array = np.zeros((n_background,), dtype=np.int64)

        # yield-conserving weights: sum = background_weight
        background_event_weight = float(cfg.background_weight / n_background)
        background_weights: Float32Array = np.full(
            (n_background,), fill_value=background_event_weight, dtype=np.float32
        )

        feature_blocks.append(background_features)
        label_blocks.append(background_labels)
        weight_blocks.append(background_weights)

    X: Float32Array = np.vstack(feature_blocks).astype(np.float32, copy=False)
    y: Int64Array = np.concatenate(label_blocks).astype(np.int64, copy=False)
    weights: Float32Array = np.concatenate(weight_blocks).astype(np.float32, copy=False)

    shuffle_indices: Int64Array = rng.permutation(n_total).astype(np.int64, copy=False)
    X = X[shuffle_indices]
    y = y[shuffle_indices]
    weights = weights[shuffle_indices]

    meta: MetaDict = {
        "pseudo_experiment_id": pseudo_experiment_id,
        "mu_true": float(cfg.mu),
        "gamma_true": float(cfg.gamma),
        "beta_true": float(cfg.beta),
        "nu_expected": float(cfg.mu * cfg.gamma + cfg.beta),
        "n_signal": int(n_signal),
        "n_background": int(n_background),
        "n_total": int(n_total),
        "weights": weights,
        "feature_params": _feature_params_dict(
            cfg, signal_cov=signal_cov, background_cov=background_cov
        ),
        "seed": cfg.seed,
    }
    return X, y, meta


def load_toy_config_from_yaml(yaml_path: str | Path) -> ToyConfig:
    """
    Read YAML config and build a ToyConfig instance.

    - Checks required fields automatically from ToyConfig dataclass (no hardcoded list)
    - Errors on unknown fields (catches typos)
    - Converts array-like fields to float64 numpy arrays (simulation-friendly)
    """
    yaml_path = Path(yaml_path)
    raw = yaml.safe_load(yaml_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping (dict-like).")

    field_names = {f.name for f in fields(ToyConfig)}
    required_names = {
        f.name
        for f in fields(ToyConfig)
        if f.default is MISSING and f.default_factory is MISSING
    }

    missing = sorted(required_names - raw.keys())
    if missing:
        raise ValueError(f"Missing required ToyConfig fields in YAML: {missing}")

    unknown = sorted(set(raw.keys()) - field_names)
    if unknown:
        raise ValueError(f"Unknown fields in YAML (typos?): {unknown}")

    for name in (
        "signal_mean",
        "signal_std",
        "signal_rho",
        "background_mean",
        "background_std",
        "background_rho",
    ):
        if name in raw:
            raw[name] = np.asarray(raw[name], dtype=np.float64)

    return ToyConfig(**raw)  # type: ignore[arg-type]


def generate_pseudo_experiment_from_yaml(
    yaml_path: str | Path,
    pseudo_experiment_id: Optional[str] = None,
) -> Tuple[Float32Array, Int64Array, MetaDict]:
    cfg = load_toy_config_from_yaml(yaml_path)
    return generate_pseudo_experiment(
        cfg=cfg, pseudo_experiment_id=pseudo_experiment_id
    )


def save_pseudo_experiment(
    output_dir: str | Path,
    X: Float32Array,
    y: Int64Array,
    meta: MetaDict,
    filename: str | None = None,
) -> Path:
    """
    Save a pseudo-experiment to disk as a single .npz file:
      - X (float32)
      - y (int64)
      - weights (float32) taken from meta["weights"]
      - meta_json (string) containing the remaining metadata

    Returns:
      Path to the saved file.
    """
    output_dir = Path(output_dir) / f"mu={meta['mu_true']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    pseudo_experiment_id = meta.get("pseudo_experiment_id", "unknown")
    if filename is None:
        filename = f"experiment_{pseudo_experiment_id}.npz"

    weights = meta.get("weights", None)
    if weights is None:
        raise ValueError('meta must contain "weights" to save a pseudo-experiment.')

    meta_to_serialize = _meta_to_jsonable(meta)
    meta_json = json.dumps(meta_to_serialize, separators=(",", ":"), sort_keys=True)

    output_path = output_dir / filename
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        weights=np.asarray(weights),
        meta_json=np.array(meta_json),
    )
    return output_path


def load_pseudo_experiment(
    npz_path: str | Path,
) -> Tuple[Float32Array, Int64Array, MetaDict]:
    """
    Load a pseudo-experiment saved with save_pseudo_experiment.

    Returns:
      X, y, meta (with meta["weights"] restored as ndarray)
    """
    npz_path = Path(npz_path)

    with np.load(npz_path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.int64, copy=False)
        weights = data["weights"].astype(np.float32, copy=False)
        meta_json = str(data["meta_json"].item())

    meta: MetaDict = json.loads(meta_json)
    meta["weights"] = weights
    return X, y, meta


# -------------------------
# Internals
# -------------------------
def _validate_pseudo_experiment_inputs(
    config: ToyConfig, pseudo_experiment_id: str
) -> None:
    # pseudo_experiment_id checks: type, length, hex
    if not isinstance(pseudo_experiment_id, str):
        raise ValueError("pseudo_experiment_id must be a HEX string of 16 chars")
    if len(pseudo_experiment_id) != 16:
        raise ValueError("pseudo_experiment_id must be a HEX string of 16 chars")
    try:
        int(pseudo_experiment_id, 16)
    except ValueError as exc:
        raise ValueError(
            "pseudo_experiment_id must be a HEX string of 16 chars"
        ) from exc

    # physics / counting parameters
    if config.mu < 0:
        raise ValueError("mu must be >= 0")
    if config.gamma < 0:
        raise ValueError("gamma must be >= 0")
    if config.beta < 0:
        raise ValueError("beta must be >= 0")
    if config.n_features <= 0:
        raise ValueError("n_features must be > 0")

    expected_vector_shape = (config.n_features,)
    expected_rho_shape = (config.n_features - 1,)

    # signal shapes
    if np.asarray(config.signal_mean).shape != expected_vector_shape:
        raise ValueError(f"signal_mean must have shape {expected_vector_shape}")
    if np.asarray(config.signal_std).shape != expected_vector_shape:
        raise ValueError(f"signal_std must have shape {expected_vector_shape}")
    if np.asarray(config.signal_rho).shape != expected_rho_shape:
        raise ValueError(f"signal_rho must have shape {expected_rho_shape}")

    # background shapes
    if np.asarray(config.background_mean).shape != expected_vector_shape:
        raise ValueError(f"background_mean must have shape {expected_vector_shape}")
    if np.asarray(config.background_std).shape != expected_vector_shape:
        raise ValueError(f"background_std must have shape {expected_vector_shape}")
    if np.asarray(config.background_rho).shape != expected_rho_shape:
        raise ValueError(f"background_rho must have shape {expected_rho_shape}")

    # std checks
    if np.any(np.asarray(config.signal_std) < 0):
        raise ValueError("signal_std must be >= 0 for all features")
    if np.any(np.asarray(config.background_std) < 0):
        raise ValueError("background_std must be >= 0 for all features")

    # rho checks
    signal_rho = np.asarray(config.signal_rho)
    background_rho = np.asarray(config.background_rho)
    if np.any((signal_rho < -1) | (signal_rho > 1)):
        raise ValueError("signal_rho values must be in [-1, 1]")
    if np.any((background_rho < -1) | (background_rho > 1)):
        raise ValueError("background_rho values must be in [-1, 1]")


def _covariance_from_std_and_adjacent_rho(
    standard_deviations: Float64Array,
    adjacent_correlations: Float64Array,
) -> Float64Array:
    """
    Build covariance matrix from:
      - standard_deviations: shape (d,)
      - adjacent_correlations: shape (d-1,)

    Interpretation:
      adjacent_correlations[i] is Corr(X_i, X_{i+1})
      all other off-diagonal correlations are set to 0.

    Note: This produces a symmetric tridiagonal matrix. It is not guaranteed PSD for d>2
    for arbitrary adjacent_correlations; this is acceptable for the current simulation phase.
    """
    d = int(standard_deviations.shape[0])

    cov: Float64Array = np.zeros((d, d), dtype=np.float64)
    np.fill_diagonal(cov, standard_deviations * standard_deviations)

    if adjacent_correlations.shape != (d - 1,):
        raise ValueError(
            f"adjacent_correlations must have shape ({d - 1},), got {adjacent_correlations.shape}"
        )

    for i in range(d - 1):
        rho = float(adjacent_correlations[i])
        cov_ij = rho * float(standard_deviations[i]) * float(standard_deviations[i + 1])
        cov[i, i + 1] = cov_ij
        cov[i + 1, i] = cov_ij

    return cov


def _feature_params_dict(
    cfg: ToyConfig,
    signal_cov: Optional[Float64Array] = None,
    background_cov: Optional[Float64Array] = None,
) -> MetaDict:
    # Build covariances if not provided (used in empty-case meta)
    if signal_cov is None:
        signal_cov = _covariance_from_std_and_adjacent_rho(
            cfg.signal_std, cfg.signal_rho
        )
    if background_cov is None:
        background_cov = _covariance_from_std_and_adjacent_rho(
            cfg.background_std, cfg.background_rho
        )

    return {
        "n_features": int(cfg.n_features),
        "signal": {
            "mean": np.asarray(cfg.signal_mean, dtype=np.float64),
            "std": np.asarray(cfg.signal_std, dtype=np.float64),
            "rho": np.asarray(cfg.signal_rho, dtype=np.float64),
            "covariance": np.asarray(signal_cov, dtype=np.float64),
        },
        "background": {
            "mean": np.asarray(cfg.background_mean, dtype=np.float64),
            "std": np.asarray(cfg.background_std, dtype=np.float64),
            "rho": np.asarray(cfg.background_rho, dtype=np.float64),
            "covariance": np.asarray(background_cov, dtype=np.float64),
        },
    }


def _meta_to_jsonable(meta: MetaDict) -> MetaDict:
    """
    Convert meta into a JSON-serializable dict.
    Numpy arrays -> lists; numpy scalars -> python scalars.
    """

    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    # Do not inline weights in JSON (saved separately as array)
    meta_copy = dict(meta)
    meta_copy.pop("weights", None)
    return convert(meta_copy)


# ---------------------------------------------------------------------------
# Split helpers  (moved from training.py in Step 6)
# ---------------------------------------------------------------------------


def _experiment_prefix(path: Path) -> str:
    stem = path.stem
    if stem.startswith("experiment_"):
        stem = stem[len("experiment_") :]
    return stem[:4]


def list_split_files(
    data_dir: Path,
    mu: float,
    test_prefixes: Optional[Sequence[str]],
    n_test_experiments: Optional[int],
    valid_size: float,
    calib_size: float,
    seed: int,
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    if valid_size < 0 or calib_size < 0 or valid_size + calib_size >= 1:
        raise ValueError("valid_size and calib_size must be >= 0 and sum to < 1.")
    mu_dir = data_dir / f"mu={mu}"
    files = sorted(mu_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {mu_dir}")
    rng = np.random.default_rng(seed)
    test_files: List[Path] = []
    if test_prefixes:
        test_files = [
            path for path in files if _experiment_prefix(path) in test_prefixes
        ]
    if not test_files:
        print(
            "No test files found with the specified prefixes. Falling back to random files."
        )
        if not n_test_experiments:
            raise ValueError("n_test_experiments must be set when no prefixes match.")
        n_test = min(n_test_experiments, len(files))
        test_files = list(rng.choice(files, size=n_test, replace=False))
    remaining_files = [path for path in files if path not in test_files]
    if not remaining_files:
        raise ValueError("No files remain after test split.")
    n_calib = int(np.floor(calib_size * len(remaining_files)))
    calib_files: List[Path] = []
    if n_calib > 0:
        calib_files = list(rng.choice(remaining_files, size=n_calib, replace=False))
    train_val_files = [path for path in remaining_files if path not in calib_files]
    if not train_val_files:
        raise ValueError("No files remain after calibration split.")
    n_val = int(np.floor(valid_size * len(train_val_files)))
    val_files: List[Path] = []
    if n_val > 0:
        val_files = list(rng.choice(train_val_files, size=n_val, replace=False))
    train_files = [path for path in train_val_files if path not in val_files]
    if not train_files:
        raise ValueError("No training files remain after train/val split.")
    return train_files, val_files, calib_files, test_files
