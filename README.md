# HMM Market Regime Detection

A clean, extensible research codebase for detecting market regimes with Hidden Markov Models (HMMs).

## What This Repo Does

- Builds a feature matrix from market prices.
- Fits a Gaussian HMM to infer latent market states.
- Labels states by realized volatility (e.g., `calm`, `neutral`, `stress`).
- Produces regime diagnostics:
  - transition matrix
  - regime duration statistics
  - regime-conditional return summary
- Supports walk-forward retraining and regime-drift diagnostics.
- Exposes a CLI for reproducible batch runs from CSV or OpenBB.

## Project Layout

```text
hmm-market-regime-detection/
├─ pyproject.toml
├─ environment.yml
├─ README.md
├─ configs/
│  └─ example_daily_spy.yaml
├─ notebooks/
│  └─ README.md
├─ scripts/
│  └─ run_pipeline.py
├─ src/
│  └─ hmm_regime/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ data.py
│     ├─ features.py
│     ├─ labeling.py
│     ├─ metrics.py
│     ├─ model.py
│     ├─ pipeline.py
│     ├─ plotting.py
│     └─ walk_forward.py
└─ tests/
   ├─ test_features.py
   ├─ test_metrics.py
   ├─ test_pipeline.py
   └─ test_walk_forward.py
```

## Notebooks

Interactive examples live in `notebooks/`:

| # | Notebook | Topics |
|---|---|---|
| 01 | [`OpenBB to HMM Pipeline`](notebooks/01_openbb_hmm_pipeline_walkthrough.ipynb) | OpenBB download, Gaussian HMM fitting, state labeling (`calm`/`neutral`/`stress`), artifact export, regime visualization |
| 02 | [`Walk-Forward Drift Diagnostics`](notebooks/02_walk_forward_drift_diagnostics.ipynb) | Rolling retraining, transition-matrix drift, label-flip drift, log-likelihood drift, instability triage plots |
| 03 | [`Notebook Index`](notebooks/README.md) | Notebook usage notes and navigation |

## Quickstart

### 1) Create environment

```bash
conda env create -f environment.yml
conda activate hmm-regime
```

### 2) Install package

```bash
pip install -e .
```

Optional OpenBB support:

```bash
pip install -e .[openbb]
```

### 3) Run pipeline from CSV

Expected CSV columns by default:
- `datetime`
- `close`

```bash
hmm-regime \
  --input data/spy_daily.csv \
  --output-dir outputs/spy_daily \
  --n-states 3
```

### 4) Run pipeline from OpenBB

```bash
hmm-regime \
  --symbol SPY \
  --start-date 2015-01-01 \
  --end-date 2025-12-31 \
  --provider yfinance \
  --output-dir outputs/spy_daily_openbb \
  --n-states 3
```

### 5) Enable walk-forward diagnostics

```bash
hmm-regime \
  --symbol SPY \
  --start-date 2015-01-01 \
  --end-date 2025-12-31 \
  --provider yfinance \
  --output-dir outputs/spy_daily_wf \
  --n-states 3 \
  --wf-train-bars 504 \
  --wf-test-bars 126 \
  --wf-step-bars 63
```

Outputs:
- `states.csv` (state and regime label per timestamp)
- `transition_matrix.csv`
- `duration_stats.csv`
- `return_summary.csv`
- `walk_forward_folds.csv` (if walk-forward enabled)
- `drift_transition.csv` (if walk-forward enabled)
- `drift_label_flip.csv` (if walk-forward enabled)
- `drift_loglik.csv` (if walk-forward enabled)

## Design Notes

- The model wrapper is intentionally thin and can be swapped with alternative HMM implementations.
- The pipeline accepts `model_cls` injection to support robust testing and experimentation.
- Feature engineering is deterministic and index-safe for walk-forward workflows.

## Next Steps

- Add multiple asset classes and macro feature joins.
- Add posterior-based regime confidence filters for downstream strategies.
- Add online/streaming inference with live state-transition monitoring.
