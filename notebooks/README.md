# Notebooks

Use this folder for exploratory analysis and visual diagnostics.

## Included notebooks

1. `01_openbb_hmm_pipeline_walkthrough.ipynb`
   - End-to-end OpenBB download -> HMM fit -> regime labeling -> artifact export.
   - Includes markdown interpretation prompts and visualization.

2. `02_walk_forward_drift_diagnostics.ipynb`
   - Rolling retraining plus transition/label/log-likelihood drift diagnostics.
   - Includes fold-level plots and instability triage workflow.

## Suggested workflow

1. Run one of the notebooks with your symbol/date range.
2. Save artifacts to `outputs/` from notebook cells.
3. Iterate on HMM state count and feature windows.
