# Conformal Predictions for Signal Strength Estimation

This repository is the official codebase accompanying the ISGC 2026 presentation
["Conformal Prediction for Reliable Uncertainty Quantification in Scientific AI Models"](https://indico4.twgrid.org/event/64/contributions/2877/).
A corresponding conference proceeding is forthcoming.

Conformal Prediction (CP) is compelling because it offers
*finite-sample, distribution-free coverage guarantees* while calibrating uncertainty on top
of trained models without retraining. 
In this project, we study that idea in the context of
scientific machine learning for signal-strength estimation, using both controlled toy
pseudo-experiments and a HiggsML-style benchmark pipeline.

More concretely, the repository implements an end-to-end workflow to generate or load data,
run exploratory data analysis, train classification models, calibrate conformal
nonconformity scores, and evaluate confidence intervals for the signal-strength parameter
$\mu$. 
The code is intended both as the reproducible companion to the academic presentation
and as the working research framework for the upcoming proceeding.

## Repository Structure

```text
conformal-predictions/
├── configs/
│   ├── toy_default*.yaml
│   ├── train_toy*.yaml
│   └── train_higgs*.yaml
├── data/
├── scripts/
│   ├── generate.py
│   ├── eda.py
│   ├── train.py
│   ├── reference.py
│   ├── calibrate.py
│   └── evaluate.py
├── src/conformal_predictions/
│   ├── calibration.py
│   ├── config.py
│   ├── data/
│   │   ├── toy.py
│   │   └── higgs.py
│   ├── data_viz.py
│   ├── evaluation.py
│   ├── models.py
│   ├── preprocessing.py
│   └── reference.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pyarrow` is required for the Higgs parquet workflow and is already listed in
`requirements.txt`.

## Pipeline Overview

The implemented workflow is:

1. generate toy pseudo-experiments, or point the config to Higgs parquet data
2. run EDA on the configured training and validation portion
3. train one classifier and save the scaler, model, and reference efficiencies
4. calibrate nonconformity scores on a disjoint calibration split
5. evaluate on a disjoint test split and compute confidence intervals and empirical coverage

All pipeline scripts consume a YAML config parsed through
`conformal_predictions.config.PipelineConfig`.

## Data Generation

### Toy pseudo-experiments

Toy data is generated from the counting model

```text
nu = mu * gamma + beta
N_signal ~ Poisson(mu * gamma)
N_background ~ Poisson(beta)
```

For each pseudo-experiment, signal and background features are sampled from
class-conditional multivariate Gaussians. The saved `.npz` artifact contains:

- `X` with dtype `float32`
- `y` binary labels
- per-event weights
- metadata including `mu_true`, `gamma_true`, `beta_true`, counts, and feature parameters

Output layout:

```text
<outdir>/mu=<mu_true>/experiment_<pseudo_experiment_id>.npz
```

To generate pseudo-experiments use:

```bash
python scripts/generate.py \
  --config configs/toy_default_easy.yaml \
  --outdir data/toy_scale_easy \
  --n-experiments 5000 \
  --n-workers 4 \
  --deterministic-ids
```

Notes:

- `--deterministic-ids` requires a non-null toy `seed`
- `--id` is only valid with `--n-experiments 1`
- batch generation uses `ProcessPoolExecutor`

### Higgs input data

The Higgs pipeline does not generate data, but it takes them from [FAIR HiggsML Uncertainty Challenge](https://fair-universe.lbl.gov/Higgs-Uncertainty-Challenge.html). It expects a parquet dataset plus a
flat labels file under:

```text
<data_dir>/
├── data/data.parquet
└── labels/data.labels
```

In the current configs, the Higgs pipeline points to `data/HiggsML/input_data/train`.

## EDA

Run:

```bash
python scripts/eda.py --config configs/train_toy.yaml
python scripts/eda.py --config configs/train_higgs.yaml
```

What the script does:

- loads training plus validation data
- for toy data, stacks pseudo-experiments selected by `list_split_files`
- for Higgs data, reads the first `train_size + valid_size` parquet row groups
- writes class-balance statistics to `results/<output_dir>/stats/class_balance.txt`
- writes a contour plot to `results/<output_dir>/plots/data_contour.png`
- writes per-feature histograms to `results/<output_dir>/plots/feature_distributions.png`

If the feature space has more than two dimensions, the contour plot uses PCA to
reduce the data to two dimensions for visualization.

## Train

Run:

```bash
python scripts/train.py --config configs/train_toy.yaml --model GLM
python scripts/train.py --config configs/train_toy.yaml --model RF
python scripts/train.py --config configs/train_higgs.yaml --model MLP
```

Training logic:

- load train and validation data
- fit a `StandardScaler` on training features only
- build one selected sklearn classifier
- fit the model
- compute validation accuracy, precision, recall, and F1
- count predicted signal events using `predict_proba > threshold`
- compute reference efficiencies unless `--no-compute-reference` is passed

Supported CLI model names:

- `GLM` -> logistic regression
- `RF` -> random forest
- `MLP` -> multilayer perceptron

Saved artifacts:

```text
results/<output_dir>/artifacts/
├── <ModelName>.joblib
├── scaler.joblib
└── reference_efficiencies.json
```

Reference efficiencies are:

- toy: always `(1.0, 1.0)`
- higgs: computed on the held-out reference split as `(eps_signal, eps_background)`

## Calibration

Run:

```bash
python scripts/calibrate.py --config configs/train_toy.yaml --model GLM
python scripts/calibrate.py --config configs/train_higgs.yaml --model MLP
```

Calibration logic:

- load the trained model, scaler, and saved reference efficiencies
- load a calibration split disjoint from train, validation, and reference data
- compute nonconformity scores per calibration block
- compute the calibration distribution of `mu_hat`
- save plots and summary statistics

Two config switches control the conformal target:

- `nonconf_target: mu_hat`
- `nonconf_target: n_pred`

`mu_hat` is computed in one of two modes:

- `raw`: `mu_hat = n_pred / gamma_true`
- `corrected`: `mu_hat = (n_pred - eps_background * beta_true) / (eps_signal * gamma_true)`

`interval_mode` can be:

- `symmetric`
- `free_form`

Calibration outputs are written under:

```text
results/<output_dir>/plots/<ModelName>/
results/<output_dir>/stats/<ModelName>/
```

These include:

- `<nonconf_target>_scores_distribution_<ModelName>.png`
- `mu_hat_distribution_<ModelName>.png`
- `<nonconf_target>_nonconf_scores.npz`
- `mu_hat_calib_distribution.npz`
- `mu_hat_calibration_stats.csv`

## Evaluation

Run:

```bash
python scripts/evaluate.py --config configs/train_toy.yaml --model GLM
python scripts/evaluate.py --config configs/train_higgs.yaml --model MLP
```

Evaluation logic:

- require the trained model, scaler, reference efficiencies, and saved calibration scores
- load the disjoint test split
- run inference block by block
- compute `mu_hat` on each test block
- convert calibration nonconformity scores into confidence intervals
- compute empirical coverage
- save per-experiment metrics, performance summaries, coverage summaries, and CI plots

For `nonconf_target: n_pred`, the code first builds an interval on predicted
signal counts and then converts that interval back to a `mu_hat` interval using
`gamma_true`.

Evaluation outputs are written under:

```text
results/<output_dir>/plots/<ModelName>/
results/<output_dir>/stats/<ModelName>/
```

These include:

- `test_CI_plots-*_<ModelName>.png`
- `test_coverage.csv`
- `test_experiment_metrics.csv`
- `test_performance_summary.csv`

## Split Logic

### Toy splits

Toy train, validation, calibration, and test splits are created from generated
`.npz` files in `data_dir/mu=<mu>/`.

`list_split_files(...)` applies the split in this order:

1. reserve test files by prefix match using `test_prefixes`
2. if no prefix matches, sample `n_test_experiments` test files at random
3. sample calibration files from the remaining files using `calib_size`
4. sample validation files from the remaining train and validation pool using `valid_size`
5. use the rest for training

The evaluation script explicitly checks that toy splits are disjoint.

### Higgs splits

Higgs splits are contiguous parquet row-group ranges:

1. training
2. validation
3. reference
4. calibration
5. test

Calibration and test row groups are then chunked into blocks using `block_size`.

## Configuration

Common config fields:

- `data_source`: `toy` or `higgs`
- `data_dir`
- `mu`
- `seed`
- `threshold`
- `mu_hat_mode`: `raw` or `corrected`
- `interval_mode`: `symmetric` or `free_form`
- `nonconf_target`: `mu_hat` or `n_pred`
- `output_dir`
- `fit_parallel`
- `valid_size`
- `calib_size`
- either `alpha` or `n_sigma`

Toy-specific fields:

- `n_test_experiments`
- `test_prefixes`

Higgs-specific fields:

- `train_size`
- `ref_size`
- `test_size`
- `block_size`

`output_dir` can be templated in YAML. For example:

```yaml
output_dir: higgs-{mu_hat_mode}-{interval_mode}-{train_size}train-{valid_size}valid-{ref_size}ref-{calib_size}calib-{test_size}test
```

If neither `alpha` nor `n_sigma` is provided, the config defaults to
`n_sigma: 1.0`, corresponding to about `68.27%` central coverage.

## Minimal Example

```bash
python scripts/generate.py --config configs/toy_default_easy.yaml --outdir data/toy_scale_easy --n-experiments 5000 --n-workers 4
python scripts/eda.py --config configs/train_toy.yaml
python scripts/train.py --config configs/train_toy.yaml --model GLM
python scripts/calibrate.py --config configs/train_toy.yaml --model GLM
python scripts/evaluate.py --config configs/train_toy.yaml --model GLM
```

## Tests

Run the smoke tests with:

```bash
python3 -m pytest tests/test_scripts_smoke.py
```

## License

This project is released under the MIT License.
