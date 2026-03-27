"""Compute reference efficiencies on a held-out reference set.

For toy data the efficiencies are trivially (1.0, 1.0).
For HiggsML the script loads a trained model + scaler, scores the reference
data, and persists signal/background efficiencies to JSON.

Usage:
    python scripts/reference.py --config configs/train_higgs.yaml --model MLP
    python scripts/reference.py --config configs/train_toy.yaml --model GLM
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from conformal_predictions.config import load_config
from conformal_predictions.data.higgs import load_ref as higgs_load_ref
from conformal_predictions.models import load_model
from conformal_predictions.preprocessing import load_scaler
from conformal_predictions.reference import (
    get_model_efficiencies,
    load_efficiencies,
    save_efficiencies,
)

# CLI model name → internal model name
_MODEL_CLI_MAP = {
    "GLM": "GLM",
    "RF": "Random Forest",
    "MLP": "MLP",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute reference efficiencies for a trained model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML pipeline config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(_MODEL_CLI_MAP.keys()),
        help="Model to compute reference efficiencies for (GLM, RF, or MLP).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    model_name = _MODEL_CLI_MAP[args.model]

    artifacts_dir = Path("results") / config.output_dir / "artifacts"

    # ---- Toy: trivial efficiencies ------------------------------------
    if config.data_source == "toy":
        print("[Toy data] Reference efficiencies are trivially (1.0, 1.0).")
        efficiencies = {model_name: (1.0, 1.0)}
        save_efficiencies(efficiencies, artifacts_dir)
        print(f"Saved to {artifacts_dir / 'reference_efficiencies.json'}")
        return

    # ---- HiggsML: compute from held-out reference set -----------------
    print(f"\n[Loading reference data from {config.data_dir}...]")
    X_ref, y_ref = higgs_load_ref(config)
    print(f"  X_ref: {X_ref.shape}, y_ref: {y_ref.shape}")
    print(
        f"  y_ref: {int(np.sum(y_ref))} signal, "
        f"{int(np.sum(y_ref == 0))} background"
    )

    print(f"\n[Loading model '{model_name}' from {artifacts_dir}...]")
    model = load_model(artifacts_dir, model_name)

    print(f"[Loading scaler from {artifacts_dir}...]")
    scaler = load_scaler(artifacts_dir)
    X_ref_scaled = scaler.transform(X_ref)

    print("\n[Computing reference efficiencies...]")
    eps_signal, eps_background = get_model_efficiencies(
        model, X_ref_scaled, y_ref, config
    )
    print(
        f"  {model_name} — eps_signal: {eps_signal:.4f}, "
        f"eps_background: {eps_background:.4f}"
    )

    # Merge with existing efficiencies if present
    try:
        existing = load_efficiencies(artifacts_dir)
    except FileNotFoundError:
        existing = {}
    existing[model_name] = (eps_signal, eps_background)

    save_efficiencies(existing, artifacts_dir)
    print(f"\nSaved to {artifacts_dir / 'reference_efficiencies.json'}")


if __name__ == "__main__":
    main()
