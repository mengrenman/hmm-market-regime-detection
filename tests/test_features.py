import numpy as np
import pandas as pd

from hmm_regime.features import build_feature_matrix



def _synthetic_close(n: int = 400, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    log_ret = rng.normal(loc=0.0003, scale=0.01, size=n)
    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx, name="close")
    return close



def test_build_feature_matrix_has_expected_columns():
    close = _synthetic_close()
    X = build_feature_matrix(close)
    expected = {
        "log_ret_1",
        "log_ret_5",
        "log_ret_20",
        "ann_vol",
        "ann_trend",
        "drawdown",
        "z_ret_1",
    }
    assert expected.issubset(set(X.columns))



def test_build_feature_matrix_no_infinite_values():
    close = _synthetic_close()
    X = build_feature_matrix(close)
    assert np.isfinite(X.to_numpy()).all()



def test_build_feature_matrix_preserves_datetime_index_type():
    close = _synthetic_close()
    X = build_feature_matrix(close)
    assert isinstance(X.index, pd.DatetimeIndex)
