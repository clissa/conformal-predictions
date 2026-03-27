"""Unified training script: load data → preprocess → fit one model → save artifacts.

Usage:
    python scripts/train.py --config configs/train_toy.yaml --model GLM
    python scripts/train.py --config configs/train_higgs.yaml --model MLP
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from conformal_predictions.config import PipelineConfig, load_config
from conformal_predictions.data.higgs import load_trainval as higgs_load_trainval
from conformal_predictions.data.toy import list_split_files, load_pseudo_experiment
from conformal_predictions.evaluation import evaluate_models, get_events_count
from conformal_predictions.models import build_models, fit_models, save_models
from conformal_predictions.preprocessing import fit_scaler, save_scaler
from conformal_predictions.reference import (
    get_all_model_efficiencies,
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
        description="Train a single conformal prediction model and save artifacts."
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
        help="Model to train (GLM, RF, or MLP).",
    )
    parser.add_argument(
        "--compute-reference",
        action="store_true",
        default=True,
        dest="compute_reference",
        help="Compute reference efficiencies after training (default).",
    )
    parser.add_argument(
        "--no-compute-reference",
        action="store_false",
        dest="compute_reference",
        help="Skip reference efficiency computation "
        "(use scripts/reference.py instead).",
    )
    return parser.parse_args()


def _load_toy_data(config: PipelineConfig):
    """Load and stack train/val data from toy pseudo-experiments."""
    train_files, val_files, _calib_files, _test_files = list_split_files(
        config.data_dir,
        config.mu,
        config.test_prefixes,
        config.n_test_experiments,
        config.valid_size,
        config.calib_size,
        config.seed,
    )

    train_blocks: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []
    for fp in train_files:
        X, y, _ = load_pseudo_experiment(fp)
        train_blocks.append(X)
        train_labels.append(y)

    val_blocks: List[np.ndarray] = []
    val_labels: List[np.ndarray] = []
    for fp in val_files:
        X, y, _ = load_pseudo_experiment(fp)
        val_blocks.append(X)
        val_labels.append(y)

    X_train = np.vstack(train_blocks)
    y_train = np.concatenate(train_labels)
    X_val = np.vstack(val_blocks)
    y_val = np.concatenate(val_labels)

    # Toy has no reference set (efficiencies are trivially 1.0)
    X_ref = np.empty((0, X_train.shape[1]))
    y_ref = np.empty(0)

    return X_train, y_train, X_val, y_val, X_ref, y_ref


def _load_higgs_data(config: PipelineConfig):
    """Load train/val/ref data from HiggsML parquet files."""
    return higgs_load_trainval(config)


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    model_name = _MODEL_CLI_MAP[args.model]

    artifacts_dir = Path("results") / config.output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(config.seed)

    # ---- 1. Load data ----
    print(f"\n[Loading {config.data_source} data...]")
    if config.data_source == "toy":
        X_train, y_train, X_val, y_val, X_ref, y_ref = _load_toy_data(config)
    elif config.data_source == "higgs":
        X_train, y_train, X_val, y_val, X_ref, y_ref = _load_higgs_data(config)
    else:
        raise ValueError(f"Unknown data_source: {config.data_source!r}")

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(
        f"  y_train: {int(np.sum(y_train))} positives "
        f"({100*np.sum(y_train)/len(y_train):.1f}%), "
        f"{len(y_train) - int(np.sum(y_train))} negatives "
        f"({100*(len(y_train)-np.sum(y_train))/len(y_train):.1f}%)"
    )

    # ---- 2. Preprocess ----
    print("\n[Preprocessing...]")
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ---- 3. Build & fit model ----
    print(f"\n[Training {model_name}...]")
    n_jobs_model = 1 if config.fit_parallel else -1
    models = build_models(config, model_names=[model_name], n_jobs=n_jobs_model)
    fit_models(
        models,
        X_train_scaled,
        y_train,
        parallel=config.fit_parallel,
    )

    # ---- 4. Validation metrics ----
    print("\n[Validation metrics...]")
    metrics = evaluate_models(models, X_val_scaled, y_val)
    for name, m in metrics.items():
        print(
            f"  {name} — Acc: {m['accuracy']:.4f}  "
            f"Prec: {m['precision']:.4f}  "
            f"Rec: {m['recall']:.4f}  "
            f"F1: {m['f1']:.4f}"
        )

    counts = get_events_count(models, X_val_scaled, config.threshold)
    for name, count in counts.items():
        print(
            f"  {name} predicted signal events (p > {config.threshold}): "
            f"{count} / {int(np.sum(y_val))} true"
        )

    # ---- 5. Reference efficiencies ----
    if args.compute_reference:
        print("\n[Computing reference efficiencies...]")
        if config.data_source == "higgs" and X_ref.shape[0] > 0:
            X_ref_scaled = scaler.transform(X_ref)
        else:
            X_ref_scaled = X_ref

        efficiencies = get_all_model_efficiencies(models, X_ref_scaled, y_ref, config)
        for name, (eps_s, eps_b) in efficiencies.items():
            print(f"  {name} — eps_signal: {eps_s:.4f}, eps_background: {eps_b:.4f}")
    else:
        print("\n[Skipping reference efficiencies (use scripts/reference.py)]")
        efficiencies = None

    # ---- 6. Save artifacts ----
    print(f"\n[Saving artifacts to {artifacts_dir}]")
    save_models(models, artifacts_dir)
    save_scaler(scaler, artifacts_dir)
    if efficiencies is not None:
        save_efficiencies(efficiencies, artifacts_dir)

    print("Done.")


if __name__ == "__main__":
    main()
