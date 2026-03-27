# Conformal Predictions for Signal Strength Estimation

This repository contains a proof-of-concept (PoC) and evolving framework for
**uncertainty quantification via Conformal Prediction (CP)** in scientific
machine learning applications, with a focus on **signal strength estimation**
in classification-based analyses.

The project is developed with two main goals:
1. **Methodological**: study how conformal prediction intervals at the event
   level can be propagated to global quantities such as signal strength, and
   assess their empirical coverage.
2. **Engineering / MLOps**: provide a clean, reproducible, and extensible
   pipeline suitable for scientific workflows and future large-scale studies
   (e.g. Higgs ML challenge–like datasets).

---

## Project status

🚧 **Work in progress**

The current focus is on a **toy-model PoC** used to validate the methodology
in a fully controlled setting.  
Extensions to realistic datasets (e.g. Higgs ML benchmarks) will follow.


## Repository structure

The repository is organized to clearly separate:
- reusable core code,
- experiment orchestration,
- configuration,
- data and results.

Intended initial structure is anticipated as follows (implementation ongoing):

```
conformal-predictions/
├── src/conformal_predictions/ # Core reusable library
│ ├── data/
│ │ ├── toy.py # Toy data generator
│ │ └── higgs.py # (stub) Higgs-specific loaders/preprocessing
│ ├── models.py # (stub) ML model wrappers
│ ├── conformal.py # (stub) conformal prediction logic
│ ├── signal_strength.py # (stub) aggregation to signal strength
│ ├── metrics.py # (stub) coverage and evaluation utilities
│ └── viz.py # (stub) plotting utilities
│
├── scripts/ # Experiment entry points
├── configs/ # YAML configuration files
├── notebooks/ # Exploratory notebooks
├── data/ # Small tracked datasets / examples
├── results/ # Experiment outputs (not tracked)
│
├── pyproject.toml # Packaging and tool configuration
├── requirements.txt
└── README.md
```


## Installation

The project uses a standard Python virtual environment.

```bash
python3 -m venv venv-cp
source venv-cp/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Quick sanity check
After installation, you can veify that the toy data generator works using:

```bash
python - <<EOF
from conformal_predictions.data.toy import make_default_toy_config, generate_toy_dataset

cfg = make_default_toy_config(random_state=42)
X, y, meta = generate_toy_dataset(cfg)

print(X.shape)
print(meta)
EOF
```

## Usage

Scripts in `scripts/` are the main entry points for data generation, training,
calibration, and evaluation. Run them from the repository root after
installing dependencies.

Generate a single pseudo-experiment from a YAML config:

```bash
python scripts/generate.py --config configs/toy_default.yaml --outdir data/toy_pseudo-experiment --n-experiments 1
```

Generate multiple pseudo-experiments (optionally in parallel):

```bash
python scripts/generate.py --config configs/toy_default.yaml --outdir data/toy_scale --n-experiments 10 --n-workers 4
```

Run the toy training and conformal workflow:

```bash
python scripts/train.py --config configs/train_toy.yaml --model GLM
python scripts/calibrate.py --config configs/train_toy.yaml --model GLM
python scripts/evaluate.py --config configs/train_toy.yaml --model GLM
```

## Planned extensions

The following components will be developed incrementally:

- support yaml train config

- add soft counting alternative: $n_{\text{pred}} = \sum_i p_{\text{pred}_i}$ instead of $\sum_i \mathbb{1}(p_{\text{pred}_i} > \text{threshold})$

- add plottings of CI VS mu_true values on test pseudo-experiments

- refactoring: modularize train.py offloading reusable utils into
  src/conformal_predictions submodules

- comparison with alternative uncertainty estimation methods

- integration with realistic physics datasets

- experiment tracking and MLOps tooling

## License

This project is released under the MIT License.
