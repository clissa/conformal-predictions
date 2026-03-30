"""Unified pipeline configuration for toy and HiggsML experiments."""

from __future__ import annotations

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
    how: str = "abs"  # "diff" or "abs"
    nonconf_target: str = "mu_hat"  # "mu_hat" or "n_pred"
    output_dir: str = "results"
    fit_parallel: bool = False
    valid_size: float = 0.2
    calib_size: float = 0.5

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
        return (
            r"$\hat{\mu} = \frac{n_{pred} - \epsilon_{bkg}\beta^*_{true}}"
            r"{\epsilon_{sig}\gamma^*_{true}}$"
        )


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


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config file and return a *PipelineConfig* instance."""
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)

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
