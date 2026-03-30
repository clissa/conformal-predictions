from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

import pytest

from conformal_predictions.config import PipelineConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def _load_script_module(module_name: str, script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _DummyModel:
    def fit(self, _X: np.ndarray, _y: np.ndarray) -> "_DummyModel":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_signal = np.full(X.shape[0], 0.6, dtype=np.float64)
        return np.column_stack([1.0 - p_signal, p_signal])

    def score(self, _X: np.ndarray, _y: np.ndarray) -> float:
        return 1.0


class _IdentityScaler:
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


def _fake_load_pseudo_experiment(_path: Path):
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 0], dtype=np.int64)
    meta = {
        "mu_true": 1.0,
        "gamma_true": 10.0,
        "beta_true": 20.0,
        "nu_expected": 30.0,
        "n_total": 3,
    }
    return X, y, meta


def _write_pipeline_config(
    path: Path,
    *,
    data_source: str = "toy",
    output_dir: str = "test-output",
    nonconf_target: str = "n_pred",
) -> Path:
    path.write_text(
        f"""\
data_source: {data_source}
data_dir: {path.parent / "data"}
mu: 1.0
seed: 18
threshold: 0.5
how: abs
nonconf_target: {nonconf_target}
output_dir: {output_dir}
fit_parallel: false
valid_size: 0.2
calib_size: 0.5
n_test_experiments: 1
test_prefixes:
  - "7e39"
train_size: 1
ref_size: 1
test_size: 1
block_size: 2
"""
    )
    return path


def test_load_config_formats_output_dir_template(tmp_path):
    config_path = tmp_path / "templated_config.yaml"
    config_path.write_text(
        """\
data_source: higgs
data_dir: data/HiggsML/input_data/train
mu: 1.0
seed: 18
threshold: 0.5
how: abs
nonconf_target: mu_hat
output_dir: higgs-{how}-{train_size}train-{valid_size}valid-{ref_size}ref-{calib_size}calib-{test_size}test
fit_parallel: false
valid_size: 5
calib_size: 10
train_size: 10
ref_size: 2
test_size: 10
block_size: 10000
"""
    )

    config = load_config(config_path)

    assert config.output_dir == "higgs-abs-10train-5valid-2ref-10calib-10test"


def test_load_config_rejects_unknown_output_dir_template_key(tmp_path):
    config_path = tmp_path / "invalid_templated_config.yaml"
    config_path.write_text(
        """\
data_source: higgs
data_dir: data/HiggsML/input_data/train
output_dir: higgs-{missing_key}
"""
    )

    with pytest.raises(ValueError, match="Unknown output_dir placeholder 'missing_key'"):
        load_config(config_path)


def test_generate_single_smoke(tmp_path, monkeypatch):
    module = _load_script_module(
        "script_generate_single_smoke",
        SCRIPTS_DIR / "generate.py",
    )

    output_dir = tmp_path / "gen_single"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--config",
            str(ROOT / "configs" / "toy_default_easy.yaml"),
            "--outdir",
            str(output_dir),
            "--n-experiments",
            "1",
        ],
    )

    module.main()

    generated_files = list((output_dir / "mu=1.0").glob("experiment_*.npz"))
    assert len(generated_files) == 1


def test_generate_batch_smoke(tmp_path, monkeypatch):
    module = _load_script_module(
        "script_generate_batch_smoke",
        SCRIPTS_DIR / "generate.py",
    )

    output_dir = tmp_path / "gen_batch"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--config",
            str(ROOT / "configs" / "toy_default_easy.yaml"),
            "--outdir",
            str(output_dir),
            "--n-experiments",
            "3",
            "--n-workers",
            "1",
        ],
    )

    module.main()

    generated_files = list((output_dir / "mu=1.0").glob("experiment_*.npz"))
    assert len(generated_files) == 3


def test_explore_command_smoke(tmp_path, monkeypatch):
    module = _load_script_module(
        "script_explore_smoke",
        SCRIPTS_DIR / "explore.py",
    )

    config_path = _write_pipeline_config(
        tmp_path / "explore_config.yaml",
        output_dir="explore-smoke",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["explore.py", "--config", str(config_path)])

    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)
    monkeypatch.setattr(module, "load_data", lambda _cfg: (X, y))

    def fake_contourplot_data(_X, _y, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data_contour.png").write_text("plot")

    monkeypatch.setattr(module, "contourplot_data", fake_contourplot_data)
    monkeypatch.setattr(
        module,
        "plot_feature_distributions",
        lambda _X, _y, output_dir: (output_dir / "feature_distributions.png").write_text(
            "plot"
        ),
    )

    module.main()

    assert (tmp_path / "results" / "explore-smoke" / "plots" / "data_contour.png").exists()
    assert (
        tmp_path
        / "results"
        / "explore-smoke"
        / "plots"
        / "feature_distributions.png"
    ).exists()
    assert (
        tmp_path / "results" / "explore-smoke" / "stats" / "class_balance.txt"
    ).exists()


def test_train_command_smoke(tmp_path, monkeypatch):
    module = _load_script_module(
        "script_train_smoke",
        SCRIPTS_DIR / "train.py",
    )

    config_path = _write_pipeline_config(
        tmp_path / "train_config.yaml",
        output_dir="train-smoke",
        nonconf_target="mu_hat",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--model", "GLM"],
    )

    train_file = tmp_path / "train.npz"
    val_file = tmp_path / "val.npz"
    calib_file = tmp_path / "calib.npz"
    test_file = tmp_path / "test.npz"

    monkeypatch.setattr(
        module,
        "list_split_files",
        lambda *_args, **_kwargs: (
            [train_file],
            [val_file],
            [calib_file],
            [test_file],
        ),
    )
    monkeypatch.setattr(module, "load_pseudo_experiment", _fake_load_pseudo_experiment)
    monkeypatch.setattr(module, "fit_scaler", lambda _X: _IdentityScaler())
    monkeypatch.setattr(
        module,
        "build_models",
        lambda _cfg, model_names, n_jobs: {"GLM": _DummyModel()},
    )
    monkeypatch.setattr(module, "fit_models", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "evaluate_models",
        lambda *_args, **_kwargs: {
            "GLM": {
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
        },
    )
    monkeypatch.setattr(module, "get_events_count", lambda *_a, **_k: {"GLM": 1})
    monkeypatch.setattr(
        module,
        "get_all_model_efficiencies",
        lambda *_args, **_kwargs: {"GLM": (1.0, 1.0)},
    )

    def fake_save_models(models, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in models:
            (output_dir / f"{name}.joblib").write_text("model")

    monkeypatch.setattr(module, "save_models", fake_save_models)
    monkeypatch.setattr(
        module,
        "save_scaler",
        lambda _scaler, output_dir: (output_dir / "scaler.joblib").write_text("scaler"),
    )
    monkeypatch.setattr(
        module,
        "save_efficiencies",
        lambda _eff, output_dir: (
            output_dir / "reference_efficiencies.json"
        ).write_text("{}"),
    )

    module.main()

    artifacts_dir = tmp_path / "results" / "train-smoke" / "artifacts"
    assert (artifacts_dir / "GLM.joblib").exists()
    assert (artifacts_dir / "scaler.joblib").exists()
    assert (artifacts_dir / "reference_efficiencies.json").exists()


def test_reference_command_smoke(tmp_path, monkeypatch):
    module = _load_script_module(
        "script_reference_smoke",
        SCRIPTS_DIR / "reference.py",
    )

    config_path = _write_pipeline_config(
        tmp_path / "reference_config.yaml",
        data_source="higgs",
        output_dir="reference-smoke",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["reference.py", "--config", str(config_path), "--model", "GLM"],
    )

    X_ref = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
    y_ref = np.array([1, 0], dtype=np.int64)
    monkeypatch.setattr(module, "higgs_load_ref", lambda _cfg: (X_ref, y_ref))
    monkeypatch.setattr(module, "load_model", lambda *_a, **_k: _DummyModel())
    monkeypatch.setattr(module, "load_scaler", lambda *_a, **_k: _IdentityScaler())
    monkeypatch.setattr(
        module,
        "get_model_efficiencies",
        lambda *_a, **_k: (0.8, 0.2),
    )
    monkeypatch.setattr(
        module,
        "load_efficiencies",
        lambda *_a, **_k: {"MLP": (0.7, 0.3)},
    )

    def fake_save_efficiencies(efficiencies, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        assert efficiencies["GLM"] == (0.8, 0.2)
        assert efficiencies["MLP"] == (0.7, 0.3)
        (output_dir / "reference_efficiencies.json").write_text("saved")

    monkeypatch.setattr(module, "save_efficiencies", fake_save_efficiencies)

    module.main()

    artifacts_dir = tmp_path / "results" / "reference-smoke" / "artifacts"
    assert (artifacts_dir / "reference_efficiencies.json").exists()


def test_calibrate_command_smoke(tmp_path, monkeypatch):
    module = _load_script_module(
        "script_calibrate_smoke",
        SCRIPTS_DIR / "calibrate.py",
    )

    config_path = _write_pipeline_config(
        tmp_path / "calibrate_config.yaml",
        output_dir="calibrate-smoke",
        nonconf_target="mu_hat",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["calibrate.py", "--config", str(config_path), "--model", "GLM"],
    )

    calib_data = [
        (
            np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
        )
    ]
    calib_meta = [
        {
            "mu_true": 1.0,
            "gamma_true": 10.0,
            "beta_true": 20.0,
            "nu_expected": 30.0,
            "n_total": 2,
        }
    ]
    monkeypatch.setattr(module, "_load_toy_calib", lambda _cfg: (calib_data, calib_meta))
    monkeypatch.setattr(module, "load_model", lambda *_a, **_k: _DummyModel())
    monkeypatch.setattr(module, "load_scaler", lambda *_a, **_k: _IdentityScaler())
    monkeypatch.setattr(
        module,
        "load_efficiencies",
        lambda *_a, **_k: {"GLM": (1.0, 1.0)},
    )
    monkeypatch.setattr(
        module,
        "compute_nonconformity_scores",
        lambda *_a, **_k: {"GLM": [0.1, 0.2, 0.3]},
    )
    monkeypatch.setattr(
        module,
        "compute_mu_hat",
        lambda *_a, **_k: (
            {"GLM": [1.0, 1.2]},
            {
                "GLM": {
                    "q16": 0.9,
                    "map": 1.0,
                    "mu_median": 1.1,
                    "mu_mean": 1.1,
                    "q68": 1.2,
                    "q84": 1.3,
                }
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "plot_nonconformity_scores",
        lambda _scores, _label, output_dir: (
            output_dir.mkdir(parents=True, exist_ok=True),
            (output_dir / "mu_hat_scores_distribution_GLM.png").write_text("plot"),
            (output_dir / "mu_hat_scores_distribution_comparison.png").write_text(
                "plot"
            ),
        ),
    )
    monkeypatch.setattr(
        module,
        "plot_mu_hat_distribution",
        lambda _mu_hat, _stats, output_dir, pred_formula: (
            output_dir.mkdir(parents=True, exist_ok=True),
            (output_dir / "mu_hat_distribution_GLM.png").write_text("plot"),
        ),
    )

    module.main()

    stats_dir = tmp_path / "results" / "calibrate-smoke" / "stats" / "GLM"
    plots_dir = tmp_path / "results" / "calibrate-smoke" / "plots" / "GLM"
    assert (stats_dir / "mu_hat_nonconf_scores.npz").exists()
    assert (stats_dir / "mu_hat_calib_distribution.npz").exists()
    assert (stats_dir / "mu_hat_calibration_stats.csv").exists()
    assert (plots_dir / "mu_hat_scores_distribution_GLM.png").exists()
    assert (plots_dir / "mu_hat_scores_distribution_comparison.png").exists()
    assert (plots_dir / "mu_hat_distribution_GLM.png").exists()


def test_evaluate_command_smoke(tmp_path, monkeypatch):
    module = _load_script_module(
        "script_evaluate_smoke",
        SCRIPTS_DIR / "evaluate.py",
    )

    config_path = _write_pipeline_config(
        tmp_path / "evaluate_config.yaml",
        output_dir="eval-smoke",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--config", str(config_path), "--model", "GLM"],
    )

    artifacts_dir = tmp_path / "results" / "eval-smoke" / "artifacts"
    stats_dir = tmp_path / "results" / "eval-smoke" / "stats" / "GLM"
    plots_dir = tmp_path / "results" / "eval-smoke" / "plots" / "GLM"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    (artifacts_dir / "GLM.joblib").write_text("placeholder")
    (artifacts_dir / "scaler.joblib").write_text("placeholder")
    (artifacts_dir / "reference_efficiencies.json").write_text("{}")
    np.savez(stats_dir / "n_pred_nonconf_scores.npz", GLM=np.array([0.1, 0.2]))

    train_file = tmp_path / "train.npz"
    val_file = tmp_path / "val.npz"
    calib_file = tmp_path / "calib.npz"
    test_file = tmp_path / "test.npz"

    monkeypatch.setattr(
        module,
        "list_split_files",
        lambda *_args, **_kwargs: (
            [train_file],
            [val_file],
            [calib_file],
            [test_file],
        ),
    )

    def fake_load_pseudo_experiment(path: Path):
        assert path == test_file
        X = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
        y = np.array([1, 0], dtype=np.int64)
        meta = {
            "mu_true": 1.0,
            "gamma_true": 10.0,
            "beta_true": 20.0,
            "nu_expected": 30.0,
            "n_total": 2,
        }
        return X, y, meta

    monkeypatch.setattr(module, "load_pseudo_experiment", fake_load_pseudo_experiment)
    monkeypatch.setattr(module, "load_model", lambda *_a, **_k: _DummyModel())
    monkeypatch.setattr(module, "load_scaler", lambda *_a, **_k: _IdentityScaler())
    monkeypatch.setattr(
        module,
        "load_efficiencies",
        lambda *_a, **_k: {"GLM": (1.0, 1.0)},
    )

    def fake_inference_on_test_set(
        models,
        scaler,
        test_data,
        threshold,
        ref_efficiencies_dict,
        debug=False,
    ):
        assert list(models.keys()) == ["GLM"]
        assert isinstance(scaler, _IdentityScaler)
        assert threshold == 0.5
        assert ref_efficiencies_dict == {"GLM": (1.0, 1.0)}
        assert debug is False
        assert len(test_data) == 1
        X_test, y_test, meta = test_data[0]
        assert X_test.shape == (2, 2)
        assert y_test.tolist() == [1, 0]
        assert meta["mu_true"] == 1.0
        return (
            {"GLM": [1.0]},
            [1.0],
            [10.0],
            {
                "GLM": [
                    {
                        "accuracy": 1.0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    }
                ]
            },
        )

    monkeypatch.setattr(module, "inference_on_test_set", fake_inference_on_test_set)
    monkeypatch.setattr(
        module,
        "compute_confidence_interval",
        lambda y_pred, *_args, **_kwargs: (
            np.asarray(y_pred, dtype=np.float64) - 2.0,
            np.asarray(y_pred, dtype=np.float64) + 2.0,
        ),
    )

    def fake_plot_confidence_intervals(
        mu_hat_values,
        mu_hat_lower_bounds,
        mu_hat_upper_bounds,
        mu_true_list,
        model_name,
        empirical_coverage,
        output_dir,
    ):
        assert mu_hat_values == [1.0]
        assert mu_hat_lower_bounds == [0.8]
        assert mu_hat_upper_bounds == [1.2]
        assert mu_true_list == [1.0]
        assert model_name == "GLM"
        assert empirical_coverage == 1.0
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "test_CI_plots-1_GLM.png").write_text("plot")

    monkeypatch.setattr(
        module,
        "plot_confidence_intervals",
        fake_plot_confidence_intervals,
    )

    module.main()

    coverage_path = stats_dir / "test_coverage.csv"
    experiment_metrics_path = stats_dir / "test_experiment_metrics.csv"
    performance_summary_path = stats_dir / "test_performance_summary.csv"
    assert coverage_path.exists()
    assert experiment_metrics_path.exists()
    assert performance_summary_path.exists()
    assert (plots_dir / "test_CI_plots-1_GLM.png").exists()

    with coverage_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["model"] == "GLM"
    assert rows[0]["nonconf_target"] == "n_pred"
    assert float(rows[0]["empirical_coverage"]) == 1.0

    with experiment_metrics_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["model"] == "GLM"
    assert float(rows[0]["mu_hat"]) == 1.0
    assert float(rows[0]["mu_hat_lower"]) == 0.8
    assert float(rows[0]["mu_hat_upper"]) == 1.2
    assert rows[0]["contains_true"] == "True"

    with performance_summary_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["model"] == "GLM"
    assert rows[0]["n_test_blocks"] == "1"
    assert float(rows[0]["accuracy_mean"]) == 1.0
    assert float(rows[0]["precision_mean"]) == 1.0
    assert float(rows[0]["recall_mean"]) == 1.0
    assert float(rows[0]["f1_mean"]) == 1.0


def test_higgs_load_test_supports_automatic_and_explicit_offsets(tmp_path, monkeypatch):
    import conformal_predictions.data.higgs as higgs_data

    row_groups = [
        np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
        np.array([[2.0, 2.1], [2.2, 2.3]], dtype=np.float32),
        np.array([[3.0, 3.1], [3.2, 3.3]], dtype=np.float32),
        np.array([[4.0, 4.1], [4.2, 4.3]], dtype=np.float32),
        np.array([[5.0, 5.1], [5.2, 5.3]], dtype=np.float32),
    ]
    labels = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int64)

    class _FakeRowGroupMeta:
        def __init__(self, num_rows: int) -> None:
            self.num_rows = num_rows

    class _FakeMetadata:
        def __init__(self, groups: list[np.ndarray]) -> None:
            self._groups = groups

        def row_group(self, index: int) -> _FakeRowGroupMeta:
            return _FakeRowGroupMeta(self._groups[index].shape[0])

    class _FakeTable:
        def __init__(self, values: np.ndarray) -> None:
            self._values = values

        def to_pandas(self):
            return self

        def to_numpy(self) -> np.ndarray:
            return self._values

    class _FakeParquetFile:
        def __init__(self, groups: list[np.ndarray]) -> None:
            self._groups = groups
            self.metadata = _FakeMetadata(groups)

        def read_row_group(self, index: int) -> _FakeTable:
            return _FakeTable(self._groups[index])

    fake_pq = type(
        "_FakePQ",
        (),
        {"ParquetFile": lambda *_args, **_kwargs: _FakeParquetFile(row_groups)},
    )()
    monkeypatch.setattr(higgs_data, "pq", fake_pq)
    monkeypatch.setattr(higgs_data.np, "loadtxt", lambda *_args, **_kwargs: labels)

    config = PipelineConfig(
        data_source="higgs",
        data_dir=tmp_path,
        mu=1.0,
        seed=18,
        threshold=0.5,
        how="abs",
        nonconf_target="mu_hat",
        output_dir="unused",
        fit_parallel=False,
        valid_size=1,
        calib_size=1,
        train_size=1,
        ref_size=1,
        test_size=1,
        block_size=2,
    )

    test_blocks_auto = higgs_data.load_test(config)
    test_blocks_explicit = higgs_data.load_test(config, test_start_label_idx=8)

    assert len(test_blocks_auto) == 1
    assert len(test_blocks_explicit) == 1

    X_auto, y_auto, meta_auto = test_blocks_auto[0]
    X_explicit, y_explicit, meta_explicit = test_blocks_explicit[0]

    assert np.array_equal(X_auto, row_groups[4])
    assert np.array_equal(X_auto, X_explicit)
    assert np.array_equal(y_auto, labels[8:10])
    assert np.array_equal(y_auto, y_explicit)
    assert meta_auto == meta_explicit
