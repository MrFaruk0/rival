# Rival

Config-driven ML benchmarking CLI. Compare multiple machine learning models on
the same dataset with a single command. Deterministic, minimal, no nonsense.

## Installation

```bash
pip install rivalml
```

For XGBoost support:

```bash
pip install rivalml[xgb]
```

## Usage

### YAML Config (recommended)

```yaml
# experiment.yaml
dataset:
  path: data.csv
  target: target_column

models:
  - logistic_regression
  - random_forest
  - xgboost

metrics:
  - accuracy
  - precision
  - recall
  - f1

training:
  test_size: 0.2
  random_seed: 42
```

```bash
rival run --config experiment.yaml
```

### CLI Args (legacy, still supported)

```bash
rival run --dataset data.csv --target target_column --models lr,rf,xgb
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` / `-c` | (none) | Path to YAML experiment config |
| `--dataset` / `-d` | (none) | Path to CSV file |
| `--target` / `-t` | (none) | Target column name |
| `--models` / `-m` | `lr,rf` | Comma-separated: `lr`, `rf`, `xgb` |
| `--missing` | `fill_mean` | Missing value strategy: `fill_mean` or `drop` |
| `--seed` / `-s` | `42` | Random seed for reproducibility |
| `--test-size` | `0.2` | Test split ratio |
| `--output` / `-o` | (none) | Export results as JSON |

### Example Output

```
------------------------------------------------------------
MODEL      ACCURACY   PRECISION   RECALL     F1         LATENCY
------------------------------------------------------------
LR         0.8523     0.8461      0.8300     0.8380     4.2ms
RF         0.8912     0.8850      0.8720     0.8784     11.8ms
XGB        0.9076     0.9031      0.8900     0.8965     17.3ms
------------------------------------------------------------
```

## Supported Models

| Alias | Full name | Backend |
|-------|-----------|---------|
| `lr` | `logistic_regression` | scikit-learn |
| `rf` | `random_forest` | scikit-learn |
| `xgb` | `xgboost` | XGBoost (optional) |

Both short (`lr`) and full (`logistic_regression`) names work everywhere.

## Motivation

Compare ML models easily. No boilerplate, no notebooks, no scattered scripts.
One command (or one YAML file), clean output, deterministic results.
