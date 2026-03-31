"""Unified pipeline configuration for toy and HiggsML experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration covering both toy and HiggsML pipelines."""

    # --- common --------------------------------------------------------
    data_source: str  # "toy" or "higgs"
    data_dir: Path
    mu: float = 1.0
    seed: int = 18
    threshold: float = 0.5
    mu_hat_mode: str = "mle"  # "raw" or "mle"
    interval_mode: str = "symmetric"  # "symmetric" or "free_form"
    nonconf_target: str = "mu_hat"  # "mu_hat" or "n_pred"
    output_dir: str = "results"
    fit_parallel: bool = False
    valid_size: float = 0.2
    calib_size: float = 0.5
    alpha: Optional[float] = None
    n_sigma: Optional[float] = None

    # --- higgs-specific ------------------------------------------------
    train_size: Optional[float] = None
    ref_size: Optional[float] = None
    test_size: Optional[float] = None
    block_size: Optional[int] = None

    # --- model hyperparameters ------------------------------------------
    model_hyperparams: Optional[Dict[str, Dict[str, Any]]] = None

    # --- toy-specific --------------------------------------------------
    n_test_experiments: Optional[int] = None
    test_prefixes: Optional[Tuple[str, ...]] = None

    # --- derived -------------------------------------------------------
    @property
    def plots_dir(self) -> Path:
        return Path("results") / self.output_dir / "plots"

    @property
    def stats_dir(self) -> Path:
        return Path("results") / self.output_dir / "stats"

    @property
    def pred_formula(self) -> str:
        if self.mu_hat_mode == "raw":
            return r"$\hat{\mu} = \frac{n_{pred}}{\gamma^*_{true}}$"
        if self.mu_hat_mode == "mle":
            return (
                r"$\hat{\mu} = \frac{n_{pred} - \epsilon_{bkg}\beta^*_{true}}"
                r"{\epsilon_{sig}\gamma^*_{true}}$"
            )
        raise ValueError(f"Unknown mu_hat_mode: {self.mu_hat_mode!r}")

    @property
    def resolved_alpha(self) -> float:
        if self.alpha is not None:
            return self.alpha
        if self.n_sigma is not None:
            return 1.0 - math.erf(self.n_sigma / math.sqrt(2.0))
        return 1.0 - math.erf(1.0 / math.sqrt(2.0))

    @property
    def confidence_level(self) -> float:
        return 1.0 - self.resolved_alpha


def _resolve_output_dir_template(raw: Dict[str, Any]) -> None:
    """Format ``output_dir`` with values from the same YAML mapping."""
    output_dir = raw.get("output_dir")
    if not isinstance(output_dir, str) or "{" not in output_dir:
        return

    try:
        raw["output_dir"] = output_dir.format(**raw)
    except KeyError as exc:
        missing_key = exc.args[0]
        raise ValueError(
            f"Unknown output_dir placeholder '{missing_key}' in config template: "
            f"{output_dir!r}"
        ) from exc


def _validate_choice(raw: Dict[str, Any], key: str, allowed: tuple[str, ...]) -> None:
    value = raw.get(key)
    if value not in allowed:
        allowed_str = ", ".join(repr(option) for option in allowed)
        raise ValueError(f"Invalid {key}={value!r}. Expected one of: {allowed_str}.")


def _resolve_confidence_config(raw: Dict[str, Any]) -> None:
    alpha = raw.get("alpha")
    n_sigma = raw.get("n_sigma")

    if alpha is not None and n_sigma is not None:
        raise ValueError("Configuration accepts either alpha or n_sigma, not both.")

    if alpha is None and n_sigma is None:
        raw["n_sigma"] = 1.0
        return

    if alpha is not None:
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError(f"Invalid alpha={alpha!r}. Expected 0 < alpha < 1.")
        return

    if not float(n_sigma) > 0.0:
        raise ValueError(f"Invalid n_sigma={n_sigma!r}. Expected n_sigma > 0.")


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config file and return a *PipelineConfig* instance."""
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    _validate_choice(raw, "mu_hat_mode", ("raw", "mle"))
    _validate_choice(raw, "interval_mode", ("symmetric", "free_form"))
    _resolve_confidence_config(raw)

    _resolve_output_dir_template(raw)

    # Convert data_dir string to Path
    if "data_dir" in raw:
        raw["data_dir"] = Path(raw["data_dir"])

    # Convert test_prefixes list to tuple
    if "test_prefixes" in raw and raw["test_prefixes"] is not None:
        raw["test_prefixes"] = tuple(raw["test_prefixes"])

    # Convert list values in model_hyperparams to tuples where needed
    if "model_hyperparams" in raw and raw["model_hyperparams"] is not None:
        for model_name, params in raw["model_hyperparams"].items():
            for key, value in params.items():
                if isinstance(value, list):
                    params[key] = tuple(value)

    return PipelineConfig(**raw)
