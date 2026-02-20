from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 3
    covariance_type: str = "full"
    n_iter: int = 300
    tol: float = 1e-4
    random_state: int = 42


class GaussianRegimeHMM:
    """Thin wrapper around hmmlearn GaussianHMM with predictable interface."""

    def __init__(self, config: HMMConfig) -> None:
        if config.n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.config = config
        self._model: Any | None = None

    @staticmethod
    def _to_2d_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D feature array, got ndim={arr.ndim}")
        if len(arr) == 0:
            raise ValueError("Feature array is empty.")
        return arr

    def fit(self, X: pd.DataFrame | np.ndarray) -> "GaussianRegimeHMM":
        arr = self._to_2d_array(X)
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise ImportError(
                "hmmlearn is required for GaussianRegimeHMM. Install with `pip install hmmlearn`."
            ) from exc

        model = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            random_state=self.config.random_state,
        )
        model.fit(arr)
        self._model = model
        return self

    def _require_fitted(self) -> Any:
        if self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._model

    def predict_states(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        model = self._require_fitted()
        return model.predict(self._to_2d_array(X))

    def predict_state_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        model = self._require_fitted()
        return model.predict_proba(self._to_2d_array(X))

    def score(self, X: pd.DataFrame | np.ndarray) -> float:
        """Return total log-likelihood under the fitted HMM."""
        model = self._require_fitted()
        return float(model.score(self._to_2d_array(X)))

    @property
    def means_(self) -> np.ndarray:
        return self._require_fitted().means_

    @property
    def covars_(self) -> np.ndarray:
        return self._require_fitted().covars_

    @property
    def transmat_(self) -> np.ndarray:
        return self._require_fitted().transmat_
