from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

import numpy as np
import pandas as pd

from .data import prepare_close_series
from .features import build_feature_matrix
from .labeling import label_states_by_volatility, map_states_to_labels
from .metrics import compute_transition_matrix, regime_duration_stats, regime_return_summary
from .model import GaussianRegimeHMM, HMMConfig


@dataclass
class RegimePipelineResult:
    close: pd.Series
    features: pd.DataFrame
    log_returns: pd.Series
    states: pd.Series
    state_probabilities: pd.DataFrame
    regime_labels: pd.Series
    state_label_map: dict[int, str]
    transition_matrix: pd.DataFrame
    duration_stats: pd.DataFrame
    return_summary: pd.DataFrame
    model: Any



def run_regime_pipeline(
    prices: pd.Series | pd.DataFrame,
    *,
    close_col: str = "close",
    hmm_config: HMMConfig | None = None,
    model_cls: Type[GaussianRegimeHMM] = GaussianRegimeHMM,
) -> RegimePipelineResult:
    """Run a full HMM regime detection pipeline from close prices."""
    config = hmm_config or HMMConfig()
    close = prepare_close_series(prices, close_col=close_col)

    features = build_feature_matrix(close)
    log_returns = np.log(close).diff(1).reindex(features.index)
    log_returns.name = "log_ret_1"

    model = model_cls(config)
    model.fit(features)

    states_arr = model.predict_states(features)
    states = pd.Series(states_arr, index=features.index, name="state").astype(int)

    probs_arr = model.predict_state_proba(features)
    prob_cols = [f"state_{i}_prob" for i in range(probs_arr.shape[1])]
    probs = pd.DataFrame(probs_arr, index=features.index, columns=prob_cols)

    state_label_map = label_states_by_volatility(log_returns, states)
    regime_labels = map_states_to_labels(states, state_label_map, name="regime")

    transition = compute_transition_matrix(states, n_states=config.n_states)
    durations = regime_duration_stats(states)
    return_summary = regime_return_summary(log_returns, states)

    return RegimePipelineResult(
        close=close,
        features=features,
        log_returns=log_returns,
        states=states,
        state_probabilities=probs,
        regime_labels=regime_labels,
        state_label_map=state_label_map,
        transition_matrix=transition,
        duration_stats=durations,
        return_summary=return_summary,
        model=model,
    )
