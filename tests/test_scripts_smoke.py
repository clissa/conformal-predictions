from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from conformal_predictions.config import PipelineConfig

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


def _fake_confidence_interval(y_pred, *_args, **_kwargs):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return y_pred - 0.1, y_pred + 0.1


def test_generate_experiments_command_smoke(tmp_path, monkeypatch):
    module = _load_script_module(
        "script_generate_experiments_smoke",
        SCRIPTS_DIR / "generate_experiments.py",
    )

    output_dir = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_experiments.py",
            "--config",
            str(ROOT / "configs" / "toy_default_easy.yaml"),
            "--outdir",
            str(output_dir),
            "--n-experiments",
            "1",
            "--n-workers",
            "1",
        ],
    )

    module.main()

    generated_files = list((output_dir / "mu=1.0").glob("experiment_*.npz"))
    assert len(generated_files) == 1


def test_generate_single_smoke(tmp_path, monkeypatch):
    """generate.py with --n-experiments 1 (single-experiment mode)."""
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
    """generate.py with --n-experiments > 1 (batch mode)."""
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


def test_train_command_smoke(tmp_path, monkeypatch):
    module = _load_script_module(
        "script_train_smoke",
        SCRIPTS_DIR / "train.py",
    )

    plots_dir = tmp_path / "plots_train"
    stats_dir = tmp_path / "stats_train"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "PLOTS_DIR", plots_dir)
    monkeypatch.setattr(module, "STATS_DIR", stats_dir)

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

    monkeypatch.setattr(module, "contourplot_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "plot_nonconformity_scores", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "plot_mu_hat_distribution", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "plot_confidence_intervals", lambda *_a, **_k: None)

    monkeypatch.setattr(module, "build_models", lambda _cfg: {"Dummy": _DummyModel()})
    monkeypatch.setattr(module, "fit_models", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(
        module,
        "evaluate_models",
        lambda *_args, **_kwargs: {
            "Dummy": {
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
        },
    )
    monkeypatch.setattr(
        module,
        "get_events_count",
        lambda *_args, **_kwargs: {"Dummy": 1},
    )
    monkeypatch.setattr(
        module,
        "compute_nonconformity_scores",
        lambda *_args, **_kwargs: {"Dummy": [0.1, 0.2]},
    )
    monkeypatch.setattr(
        module,
        "compute_mu_hat",
        lambda *_args, **_kwargs: (
            {"Dummy": [1.0, 1.1]},
            {
                "Dummy": {
                    "q16": 0.9,
                    "map": 1.0,
                    "mu_median": 1.0,
                    "mu_mean": 1.05,
                    "q68": 1.1,
                    "q84": 1.2,
                }
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "inference_on_test_set",
        lambda *_args, **_kwargs: (
            {"Dummy": [1.0]},
            [1.0],
            [10.0],
            {"Dummy": [{"accuracy": 1.0}]},
        ),
    )
    monkeypatch.setattr(
        module, "compute_confidence_interval", _fake_confidence_interval
    )

    module.main()

    assert (stats_dir / "mu_hat_calib_distribution.npz").exists()
    assert (stats_dir / "mu_hat_nonconf_scores.npz").exists()
    assert (stats_dir / "mu_hat_calibration_stats.csv").exists()


def test_train_load_config(tmp_path):
    """Verify that _load_config correctly parses a YAML file for train.py."""
    module = _load_script_module(
        "script_train_load_config",
        SCRIPTS_DIR / "train.py",
    )

    yaml_content = """\
output_dirname: "custom-output"
data_dir: "data/custom_dir"
mu: 2.0
seed: 99
test_prefixes:
  - "aaaa"
  - "bbbb"
valid_size: 0.1
calib_size: 0.3
nonconf_target: "n_pred"
n_test_experiments: 500
threshold: 0.7
"""
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(yaml_content)

    cfg, output_dirname = module._load_config(config_path)

    assert output_dirname == "custom-output"
    assert str(cfg.data_dir) == "data/custom_dir"
    assert cfg.mu == 2.0
    assert cfg.seed == 99
    assert cfg.test_prefixes == ("aaaa", "bbbb")
    assert cfg.valid_size == 0.1
    assert cfg.calib_size == 0.3
    assert cfg.nonconf_target == "n_pred"
    assert cfg.n_test_experiments == 500
    assert cfg.threshold == 0.7


def test_train_command_smoke_with_config(tmp_path, monkeypatch):
    """Verify that main() in train.py correctly loads settings from a YAML config."""
    module = _load_script_module(
        "script_train_smoke_with_config",
        SCRIPTS_DIR / "train.py",
    )

    yaml_content = """\
output_dirname: "custom-output-for-test"
data_dir: "data/toy_scale_easy"
mu: 1.0
seed: 18
test_prefixes:
  - "7e39"
  - "6fcb"
valid_size: 0.2
calib_size: 0.5
nonconf_target: "mu_hat"
n_test_experiments: 1000
threshold: 0.5
"""
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(yaml_content)

    # Change CWD to tmp_path so output dirs are created under tmp_path/results/...
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path)],
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
    monkeypatch.setattr(module, "contourplot_data", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "plot_nonconformity_scores", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "plot_mu_hat_distribution", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "plot_confidence_intervals", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "build_models", lambda _cfg: {"Dummy": _DummyModel()})
    monkeypatch.setattr(module, "fit_models", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "evaluate_models",
        lambda *_a, **_k: {
            "Dummy": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
        },
    )
    monkeypatch.setattr(module, "get_events_count", lambda *_a, **_k: {"Dummy": 1})
    monkeypatch.setattr(
        module,
        "compute_nonconformity_scores",
        lambda *_a, **_k: {"Dummy": [0.1, 0.2]},
    )
    monkeypatch.setattr(
        module,
        "compute_mu_hat",
        lambda *_a, **_k: (
            {"Dummy": [1.0, 1.1]},
            {
                "Dummy": {
                    "q16": 0.9,
                    "map": 1.0,
                    "mu_median": 1.0,
                    "mu_mean": 1.05,
                    "q68": 1.1,
                    "q84": 1.2,
                }
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "inference_on_test_set",
        lambda *_a, **_k: (
            {"Dummy": [1.0]},
            [1.0],
            [10.0],
            {"Dummy": [{"accuracy": 1.0}]},
        ),
    )
    monkeypatch.setattr(
        module, "compute_confidence_interval", _fake_confidence_interval
    )

    module.main()

    stats_dir = tmp_path / "results" / "custom-output-for-test" / "stats"
    assert (stats_dir / "mu_hat_calib_distribution.npz").exists()
    assert (stats_dir / "mu_hat_nonconf_scores.npz").exists()
    assert (stats_dir / "mu_hat_calibration_stats.csv").exists()


def test_train_higgs_command_smoke(tmp_path, monkeypatch):
    module = _load_script_module(
        "script_train_higgs_smoke",
        SCRIPTS_DIR / "train_higgs.py",
    )

    plots_dir = tmp_path / "plots_higgs"
    stats_dir = tmp_path / "stats_higgs"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "PLOTS_DIR", plots_dir)
    monkeypatch.setattr(module, "STATS_DIR", stats_dir)

    def fake_load_trainval(_cfg):
        X_train = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8]], dtype=np.float32)
        y_train = np.array([0, 1, 0], dtype=np.int64)
        X_val = np.array([[0.3, 0.7], [0.7, 0.3]], dtype=np.float32)
        y_val = np.array([0, 1], dtype=np.int64)
        X_ref = np.array([[0.4, 0.6], [0.6, 0.4]], dtype=np.float32)
        y_ref = np.array([0, 1], dtype=np.int64)
        return X_train, y_train, X_val, y_val, X_ref, y_ref

    def fake_load_calib(_cfg, calib_start_label_idx):
        _ = calib_start_label_idx
        X = np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
        y = np.array([0, 1], dtype=np.int64)
        meta = {
            "mu_true": 1.0,
            "gamma_true": 10.0,
            "beta_true": 20.0,
            "nu_expected": 30.0,
            "n_total": 2,
        }
        return [(X, y)], [meta]

    def fake_load_test(_cfg, test_start_label_idx):
        _ = test_start_label_idx
        X = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
        y = np.array([0, 1], dtype=np.int64)
        meta = {
            "mu_true": 1.0,
            "gamma_true": 10.0,
            "beta_true": 20.0,
            "nu_expected": 30.0,
            "n_total": 2,
        }
        return [[X, y, meta]]

    monkeypatch.setattr(module, "load_trainval", fake_load_trainval)
    monkeypatch.setattr(module, "load_calib", fake_load_calib)
    monkeypatch.setattr(module, "load_test", fake_load_test)

    monkeypatch.setattr(module, "contourplot_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "plot_nonconformity_scores", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "plot_mu_hat_distribution", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "plot_confidence_intervals", lambda *_a, **_k: None)

    monkeypatch.setattr(
        module,
        "build_models",
        lambda _cfg, n_jobs: {"Dummy": _DummyModel()},
    )
    monkeypatch.setattr(module, "fit_models", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(
        module,
        "evaluate_models",
        lambda *_args, **_kwargs: {
            "Dummy": {
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
        },
    )
    monkeypatch.setattr(
        module,
        "get_events_count",
        lambda *_args, **_kwargs: {"Dummy": 1},
    )
    monkeypatch.setattr(
        module,
        "get_model_efficiencies",
        lambda *_args, **_kwargs: (0.5, 0.1),
    )

    def fake_nonconformity_scores(models, *_args, **_kwargs):
        model_name = next(iter(models))
        return {model_name: [0.1, 0.2, 0.3]}

    def fake_compute_mu_hat(models, *_args, **_kwargs):
        model_name = next(iter(models))
        return {model_name: [1.0, 1.2]}, {
            model_name: {
                "q16": 0.9,
                "map": 1.0,
                "mu_median": 1.1,
                "mu_mean": 1.1,
                "q68": 1.2,
                "q84": 1.3,
            }
        }

    monkeypatch.setattr(
        module, "compute_nonconformity_scores", fake_nonconformity_scores
    )
    monkeypatch.setattr(module, "compute_mu_hat", fake_compute_mu_hat)
    monkeypatch.setattr(
        module,
        "inference_on_test_set",
        lambda *_args, **_kwargs: (
            {"Dummy": [1.05]},
            [1.0],
            [10.0],
            {"Dummy": [{"accuracy": 1.0}]},
        ),
    )
    monkeypatch.setattr(
        module, "compute_confidence_interval", _fake_confidence_interval
    )

    module.main()

    assert (stats_dir / "mu_hat_calib_distribution.npz").exists()
    assert (stats_dir / "mu_hat_nonconf_scores.npz").exists()
    assert (stats_dir / "mu_hat_calibration_stats.csv").exists()


def test_evaluate_command_smoke(tmp_path, monkeypatch):
    module = _load_script_module(
        "script_evaluate_smoke",
        SCRIPTS_DIR / "evaluate.py",
    )

    output_dir = "eval-smoke"
    config_path = tmp_path / "evaluate_config.yaml"
    config_path.write_text(
        """\
data_source: toy
data_dir: data/toy_scale_easy
mu: 1.0
seed: 18
threshold: 0.5
how: abs
nonconf_target: n_pred
output_dir: eval-smoke
fit_parallel: false
valid_size: 0.2
calib_size: 0.5
n_test_experiments: 1
test_prefixes:
  - "7e39"
"""
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--config", str(config_path), "--model", "GLM"],
    )

    artifacts_dir = tmp_path / "results" / output_dir / "artifacts"
    stats_dir = tmp_path / "results" / output_dir / "stats"
    plots_dir = tmp_path / "results" / output_dir / "plots"
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
    monkeypatch.setattr(module, "load_model", lambda *_args, **_kwargs: _DummyModel())
    monkeypatch.setattr(
        module, "load_scaler", lambda *_args, **_kwargs: _IdentityScaler()
    )
    monkeypatch.setattr(
        module,
        "load_efficiencies",
        lambda *_args, **_kwargs: {"GLM": (1.0, 1.0)},
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
