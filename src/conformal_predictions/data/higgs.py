"""HiggsML data loading utilities.

Provides functions to load training/validation/reference, calibration, and
test data from the HiggsML parquet + labels format used by the conformal
prediction pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq

from conformal_predictions.config import PipelineConfig


def load_trainval(
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training, validation, and reference splits from HiggsML data.

    Parameters
    ----------
    config : PipelineConfig
        Must have ``train_size``, ``valid_size``, ``ref_size`` (as integer
        row-group counts) and ``data_dir`` pointing to the HiggsML input
        directory.

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_ref, y_ref)
    """
    base_dir = config.data_dir
    parquet_path = base_dir / "data" / "data.parquet"
    labels_path = base_dir / "labels" / "data.labels"

    train_size = int(config.train_size)
    valid_size = int(config.valid_size)
    ref_size = int(config.ref_size) if config.ref_size is not None else 0

    pf = pq.ParquetFile(parquet_path)

    train_tables = [pf.read_row_group(i) for i in range(train_size)]
    val_tables = [
        pf.read_row_group(i) for i in range(train_size, train_size + valid_size)
    ]
    ref_tables = [
        pf.read_row_group(i)
        for i in range(train_size + valid_size, train_size + valid_size + ref_size)
    ]

    X_train = np.vstack([t.to_pandas().to_numpy() for t in train_tables])
    X_val = np.vstack([t.to_pandas().to_numpy() for t in val_tables])
    X_ref = (
        np.vstack([t.to_pandas().to_numpy() for t in ref_tables])
        if ref_tables
        else np.empty((0, X_train.shape[1]))
    )

    y_all = np.loadtxt(labels_path)
    y_train = y_all[: X_train.shape[0]]
    y_val = y_all[X_train.shape[0] : X_train.shape[0] + X_val.shape[0]]
    y_ref = y_all[
        X_train.shape[0]
        + X_val.shape[0] : X_train.shape[0]
        + X_val.shape[0]
        + X_ref.shape[0]
    ]

    return X_train, y_train, X_val, y_val, X_ref, y_ref


def load_ref(
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load only the reference split from HiggsML data.

    Reference row groups sit immediately after train + validation row groups,
    so the loaded data is guaranteed to be disjoint from training and
    validation data.

    Parameters
    ----------
    config : PipelineConfig
        Must have ``train_size``, ``valid_size``, ``ref_size`` and
        ``data_dir`` pointing to the HiggsML input directory.

    Returns
    -------
    (X_ref, y_ref)
    """
    base_dir = config.data_dir
    parquet_path = base_dir / "data" / "data.parquet"
    labels_path = base_dir / "labels" / "data.labels"

    train_size = int(config.train_size)
    valid_size = int(config.valid_size)
    ref_size = int(config.ref_size) if config.ref_size is not None else 0

    if ref_size == 0:
        raise ValueError(
            "ref_size is 0; set ref_size > 0 in your config to use a reference set."
        )

    pf = pq.ParquetFile(parquet_path)
    ref_start = train_size + valid_size

    # Compute label offset from row-group metadata (no data reads needed)
    label_offset = sum(pf.metadata.row_group(i).num_rows for i in range(ref_start))

    ref_tables = [pf.read_row_group(i) for i in range(ref_start, ref_start + ref_size)]
    X_ref = np.vstack([t.to_pandas().to_numpy() for t in ref_tables])

    y_all = np.loadtxt(labels_path)
    y_ref = y_all[label_offset : label_offset + X_ref.shape[0]]

    return X_ref, y_ref


def load_calib(
    config: PipelineConfig,
    calib_start_label_idx: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Dict[str, Any]]]:
    """Load calibration blocks from HiggsML data.

    Parameters
    ----------
    config : PipelineConfig
    calib_start_label_idx : int
        Cumulative label index where calibration rows begin in the flat
        labels file.

    Returns
    -------
    (calib_data, metadata)
        *calib_data* is a list of ``(X, y)`` tuples (one per block).
        *metadata* is a list of dicts with physics quantities per block.
    """
    base_dir = config.data_dir
    parquet_path = base_dir / "data" / "data.parquet"
    labels_path = base_dir / "labels" / "data.labels"

    train_size = int(config.train_size)
    valid_size = int(config.valid_size)
    ref_size = int(config.ref_size) if config.ref_size is not None else 0
    calib_size = int(config.calib_size)
    block_size = int(config.block_size)

    pf = pq.ParquetFile(parquet_path)
    calib_start_idx = train_size + valid_size + ref_size
    calib_tables = [
        pf.read_row_group(i)
        for i in range(calib_start_idx, calib_start_idx + calib_size)
    ]

    X_calib = np.vstack([t.to_pandas().to_numpy() for t in calib_tables])
    y_all = np.loadtxt(labels_path)
    y_calib = y_all[calib_start_label_idx : calib_start_label_idx + X_calib.shape[0]]

    calib_data: List[Tuple[np.ndarray, np.ndarray]] = []
    metadata: List[Dict[str, Any]] = []
    for i in range(0, X_calib.shape[0], block_size):
        X_block = X_calib[i : i + block_size]
        y_block = y_calib[i : i + block_size]
        block_length = X_block.shape[0]
        gamma_true = 336.0 * (block_length / 1000)
        meta_dict = {
            "mu_true": 1.0,
            "gamma_true": gamma_true,
            "beta_true": block_length - gamma_true,
            "nu_expected": block_length,
            "n_signal": y_block.sum(),
            "n_background": (1 - y_block).sum(),
            "n_total": block_length,
        }
        calib_data.append((X_block, y_block))
        metadata.append(meta_dict)

    print(
        f"Calibration sample has: {X_calib.shape[0]} events "
        f"and {y_calib.shape[0]} labels."
    )
    print(
        f"Calibration events are split into {len(calib_data)} blocks "
        f"of size {block_size} (last block may be smaller)"
    )
    return calib_data, metadata


def load_test(
    config: PipelineConfig,
    test_start_label_idx: int,
) -> List[List]:
    """Load test blocks from HiggsML data.

    Parameters
    ----------
    config : PipelineConfig
    test_start_label_idx : int
        Cumulative label index where test rows begin in the flat labels file.

    Returns
    -------
    list of [X, y, meta_dict]
        One entry per block.
    """
    base_dir = config.data_dir
    parquet_path = base_dir / "data" / "data.parquet"
    labels_path = base_dir / "labels" / "data.labels"

    train_size = int(config.train_size)
    valid_size = int(config.valid_size)
    ref_size = int(config.ref_size) if config.ref_size is not None else 0
    calib_size = int(config.calib_size)
    test_size = int(config.test_size)
    block_size = int(config.block_size)

    pf = pq.ParquetFile(parquet_path)
    test_start_idx = train_size + valid_size + ref_size + calib_size
    test_tables = [
        pf.read_row_group(i) for i in range(test_start_idx, test_start_idx + test_size)
    ]

    X_test = np.vstack([t.to_pandas().to_numpy() for t in test_tables])
    y_all = np.loadtxt(labels_path)
    y_test = y_all[test_start_label_idx : test_start_label_idx + X_test.shape[0]]

    test_data: List[List] = []
    for i in range(0, X_test.shape[0], block_size):
        X_block = X_test[i : i + block_size]
        y_block = y_test[i : i + block_size]
        block_length = X_block.shape[0]
        gamma_true = 336.0 * (block_length / 1000)
        meta_dict = {
            "mu_true": 1.0,
            "gamma_true": gamma_true,
            "beta_true": block_length - gamma_true,
            "nu_expected": block_length,
            "n_signal": y_block.sum(),
            "n_background": (1 - y_block).sum(),
            "n_total": block_length,
        }
        test_data.append([X_block, y_block, meta_dict])

    print(f"Test sample has: {X_test.shape[0]} events and {y_test.shape[0]} labels.")
    print(
        f"Test events are split into {len(test_data)} blocks "
        f"of size {block_size} (last block may be smaller)"
    )
    return test_data
