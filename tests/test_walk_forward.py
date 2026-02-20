from __future__ import annotations

import numpy as np
import pandas as pd

from hmm_regime.model import HMMConfig
from hmm_regime.walk_forward import run_walk_forward_regime_analysis, walk_forward_splits


class DummyWFModel:
    """Deterministic model for walk-forward tests (no hmmlearn dependency)."""

    def __init__(self, config: HMMConfig):
        self.config = config
        self._fitted = False
        self.transmat_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "DummyWFModel":
        n = self.config.n_states
        self._fitted = True
        self.transmat_ = np.full((n, n), 1.0 / n, dtype=float)
        return self

    def predict_states(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("model not fitted")
        return np.arange(len(X), dtype=int) % self.config.n_states

    def predict_state_proba(self, X: pd.DataFrame) -> np.ndarray:
        states = self.predict_states(X)
        probs = np.zeros((len(X), self.config.n_states), dtype=float)
        probs[np.arange(len(X)), states] = 1.0
        return probs

    def score(self, X: pd.DataFrame) -> float:
        # Constant per-observation likelihood to make drift expectations simple.
        return float(-0.5 * len(X))



def _synthetic_prices(n: int = 900, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-01", periods=n, freq="B")
    ret = rng.normal(0.0001, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(ret))
    return pd.DataFrame({"close": close}, index=idx)



def test_walk_forward_splits_generates_expected_count():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    splits = walk_forward_splits(idx, train_bars=200, test_bars=50, step_bars=50)
    assert len(splits) == 2



def test_walk_forward_outputs_are_non_empty():
    prices = _synthetic_prices()
    cfg = HMMConfig(n_states=3, n_iter=5)

    result = run_walk_forward_regime_analysis(
        prices,
        hmm_config=cfg,
        model_cls=DummyWFModel,
        train_bars=252,
        test_bars=63,
        step_bars=63,
        min_test_bars=20,
    )

    assert not result.folds.empty
    assert not result.drift_transition.empty
    assert not result.drift_loglik.empty



def test_drift_rows_match_consecutive_fold_pairs():
    prices = _synthetic_prices()
    cfg = HMMConfig(n_states=3, n_iter=5)

    result = run_walk_forward_regime_analysis(
        prices,
        hmm_config=cfg,
        model_cls=DummyWFModel,
        train_bars=252,
        test_bars=63,
        step_bars=63,
        min_test_bars=20,
    )

    n_folds = len(result.folds)
    assert len(result.drift_transition) == max(0, n_folds - 1)
    assert len(result.drift_loglik) == max(0, n_folds - 1)
