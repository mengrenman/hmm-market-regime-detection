from __future__ import annotations

import numpy as np
import pandas as pd

from hmm_regime.model import HMMConfig
from hmm_regime.pipeline import run_regime_pipeline


class DummyModel:
    """Deterministic stand-in model for pipeline tests (no hmmlearn dependency)."""

    def __init__(self, config: HMMConfig):
        self.config = config
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "DummyModel":
        if X.empty:
            raise ValueError("empty feature matrix")
        self._fitted = True
        return self

    def predict_states(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("model not fitted")
        n = len(X)
        return np.arange(n) % self.config.n_states

    def predict_state_proba(self, X: pd.DataFrame) -> np.ndarray:
        states = self.predict_states(X)
        probs = np.zeros((len(X), self.config.n_states), dtype=float)
        probs[np.arange(len(X)), states] = 1.0
        return probs



def _synthetic_prices(n: int = 350, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    ret = rng.normal(0.0002, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(ret))
    return pd.DataFrame({"close": close}, index=idx)



def test_run_regime_pipeline_with_dummy_model():
    prices = _synthetic_prices()
    cfg = HMMConfig(n_states=3, n_iter=10)

    result = run_regime_pipeline(prices, hmm_config=cfg, model_cls=DummyModel)

    assert len(result.features) > 0
    assert len(result.states) == len(result.features)
    assert len(result.regime_labels) == len(result.features)
    assert result.state_probabilities.shape[0] == len(result.features)
    assert set(result.states.unique().tolist()).issubset({0, 1, 2})
    assert not result.transition_matrix.empty
    assert not result.duration_stats.empty
    assert not result.return_summary.empty
