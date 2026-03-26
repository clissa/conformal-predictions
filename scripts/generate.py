from __future__ import annotations

import argparse
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Tuple

from conformal_predictions.data.toy import (
    generate_pseudo_experiment,
    load_toy_config_from_yaml,
    save_pseudo_experiment,
)


def _make_deterministic_experiment_id(cfg_seed: int, global_index: int) -> str:
    return uuid.uuid5(uuid.NAMESPACE_OID, f"{cfg_seed}-{global_index}").hex[:16]


def _iter_chunks(total: int, n_chunks: int) -> Iterable[Tuple[int, int]]:
    base, extra = divmod(total, n_chunks)
    start = 0
    for i in range(n_chunks):
        count = base + (1 if i < extra else 0)
        if count == 0:
            continue
        yield start, count
        start += count


def _generate_batch(
    *,
    yaml_path: str,
    outdir: str,
    start_index: int,
    count: int,
    deterministic_ids: bool,
) -> int:
    cfg = load_toy_config_from_yaml(yaml_path)
    if deterministic_ids and cfg.seed is None:
        raise ValueError("--deterministic-ids requires YAML seed to be set (not null).")

    for local_i in range(count):
        global_i = start_index + local_i
        if deterministic_ids:
            pseudo_experiment_id = _make_deterministic_experiment_id(cfg.seed, global_i)
        else:
            pseudo_experiment_id = None
        X, y, meta = generate_pseudo_experiment(
            cfg, pseudo_experiment_id=pseudo_experiment_id
        )
        save_pseudo_experiment(outdir, X, y, meta)
    return count


def _generate_single(
    *,
    yaml_path: str,
    outdir: str | None,
    experiment_id: str | None,
) -> None:
    cfg = load_toy_config_from_yaml(yaml_path)
    X, y, meta = generate_pseudo_experiment(cfg, pseudo_experiment_id=experiment_id)

    print("Generated pseudo-experiment")
    print(f" \tid: {meta['pseudo_experiment_id']}")
    print(f"\tmu_true: {meta['mu_true']}")
    print(f"\tgamma_true: {meta['gamma_true']}")
    print(f"\tbeta_true: {meta['beta_true']}")
    print("\nYields and sizes:")
    print(f"\tn_signal: {meta['n_signal']}")
    print(f"\tn_background: {meta['n_background']}")
    print(f"\tn_total: {meta['n_total']}")

    if outdir is not None:
        saved_path = save_pseudo_experiment(outdir, X, y, meta)
        print(f"Saved to: {saved_path}")

    print("---------------")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate toy pseudo-experiments from a YAML config.\n\n"
            "With --n-experiments 1 (default), generates a single experiment and "
            "prints its details. With --n-experiments > 1, generates a batch "
            "(optionally in parallel)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file (e.g. configs/toy_default.yaml).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory root for generated experiments. "
        "Required when --n-experiments > 1.",
    )
    parser.add_argument(
        "--n-experiments",
        type=int,
        default=1,
        help="Total number of pseudo-experiments to generate (default: 1).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of worker processes for batch generation (default: 1).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting global pseudo-experiment index (default: 0).",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Optional 16-char hex pseudo_experiment_id. "
        "Only valid with --n-experiments 1.",
    )
    parser.add_argument(
        "--deterministic-ids",
        action="store_true",
        help="Use deterministic IDs derived from (seed, global_index). "
        "Requires seed not None in config. Ignored with --n-experiments 1.",
    )
    args = parser.parse_args()

    if args.n_experiments <= 0:
        raise ValueError("--n-experiments must be > 0")
    if args.n_workers <= 0:
        raise ValueError("--n-workers must be > 0")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")

    # Single-experiment mode
    if args.n_experiments == 1 and args.id is not None:
        _generate_single(
            yaml_path=str(args.config),
            outdir=str(args.outdir) if args.outdir else None,
            experiment_id=args.id,
        )
        return

    # Batch mode (including n_experiments == 1 without --id)
    if args.outdir is None:
        raise ValueError("--outdir is required when generating experiments.")

    if args.id is not None and args.n_experiments > 1:
        raise ValueError("--id can only be used with --n-experiments 1.")

    # Single experiment without explicit --id: use batch path for consistency
    if args.n_experiments == 1:
        _generate_single(
            yaml_path=str(args.config),
            outdir=str(args.outdir),
            experiment_id=None,
        )
        return

    n_workers = min(args.n_workers, args.n_experiments)
    chunk_specs: List[Tuple[int, int]] = []
    for offset, count in _iter_chunks(args.n_experiments, n_workers):
        chunk_specs.append((args.start_index + offset, count))

    print(
        f"Generating {args.n_experiments} pseudo-experiments "
        f"with n_workers={n_workers}"
    )
    print(f"Output root: {args.outdir}")
    if args.deterministic_ids:
        print("Deterministic IDs: ON")
    else:
        print("Deterministic IDs: OFF (random IDs)")

    generated = 0
    if n_workers == 1:
        generated += _generate_batch(
            yaml_path=str(args.config),
            outdir=str(args.outdir),
            start_index=args.start_index,
            count=args.n_experiments,
            deterministic_ids=args.deterministic_ids,
        )
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(
                    _generate_batch,
                    yaml_path=str(args.config),
                    outdir=str(args.outdir),
                    start_index=start_index,
                    count=count,
                    deterministic_ids=args.deterministic_ids,
                )
                for start_index, count in chunk_specs
            ]
            for fut in futures:
                generated += fut.result()

    print(f"Done. Generated {generated} pseudo-experiments.")


if __name__ == "__main__":
    main()
