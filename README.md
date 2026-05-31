# Rival

Lightweight ML benchmarking CLI tool. Compare multiple machine learning models on
the same dataset with a single command. No deep learning, no cloud, no complexity
-- just clean and reproducible model comparison.

## Installation

```bash
pip install rival
```

For XGBoost support:

```bash
pip install rival[xgb]
```

## Usage

```bash
rival run --dataset data.csv --target target_column --models lr,rf,xgb
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` / `-d` | (required) | Path to CSV file |
| `--target` / `-t` | (required) | Target column name |
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

- **lr** -- Logistic Regression (scikit-learn)
- **rf** -- Random Forest (scikit-learn)
- **xgb** -- XGBoost (optional, requires `pip install rival[xgb]`)

## Motivation

Compare ML models easily. No boilerplate, no notebooks, no scattered scripts.
One command, clean output, deterministic results.
