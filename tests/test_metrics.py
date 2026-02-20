import numpy as np
import pandas as pd

from hmm_regime.metrics import compute_transition_matrix, regime_duration_stats, regime_return_summary



def test_transition_matrix_rows_sum_to_one_when_observed():
    idx = pd.date_range("2021-01-01", periods=8, freq="B")
    states = pd.Series([0, 0, 1, 1, 1, 0, 2, 2], index=idx)
    tm = compute_transition_matrix(states, n_states=3)

    row_sums = tm.sum(axis=1)
    assert np.isclose(row_sums.loc[0], 1.0)
    assert np.isclose(row_sums.loc[1], 1.0)
    assert np.isclose(row_sums.loc[2], 1.0)



def test_regime_duration_stats_counts_episodes():
    idx = pd.date_range("2021-01-01", periods=9, freq="B")
    states = pd.Series([0, 0, 1, 1, 0, 0, 0, 2, 2], index=idx)
    stats = regime_duration_stats(states)

    assert stats.loc[0, "episodes"] == 2
    assert stats.loc[1, "episodes"] == 1
    assert stats.loc[2, "episodes"] == 1



def test_regime_return_summary_has_key_metrics():
    idx = pd.date_range("2021-01-01", periods=20, freq="B")
    returns = pd.Series(np.linspace(-0.01, 0.01, 20), index=idx)
    states = pd.Series(([0] * 10) + ([1] * 10), index=idx)

    summary = regime_return_summary(returns, states)
    for col in ["n_obs", "mean_log_return", "ann_return", "ann_vol", "sharpe", "hit_rate"]:
        assert col in summary.columns
