from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

import numpy as np
import pandas as pd

from .data import prepare_close_series
from .features import build_feature_matrix
from .labeling import label_states_by_volatility, map_states_to_labels
from .model import GaussianRegimeHMM, HMMConfig


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    drift_transition: pd.DataFrame
    drift_label_flip: pd.DataFrame
    drift_loglik: pd.DataFrame


@dataclass
class _FoldArtifact:
    fold: int
    model: Any
    test_features: pd.DataFrame
    test_regimes: pd.Series
    transition_labeled: pd.DataFrame
    test_loglik_per_obs: float



def walk_forward_splits(
    index: pd.DatetimeIndex,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    min_test_bars: int = 1,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Generate rolling walk-forward train/test index splits."""
    if train_bars <= 0:
        raise ValueError(f"train_bars must be positive, got {train_bars}")
    if test_bars <= 0:
        raise ValueError(f"test_bars must be positive, got {test_bars}")
    if step_bars is None:
        step_bars = test_bars
    if step_bars <= 0:
        raise ValueError(f"step_bars must be positive, got {step_bars}")

    n = len(index)
    if train_bars + test_bars > n:
        return []

    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    start = 0
    while True:
        train_end = start + train_bars
        test_end = train_end + test_bars
        if train_end > n:
            break
        test_end = min(test_end, n)
        if (test_end - train_end) < min_test_bars:
            break
        splits.append((index[start:train_end], index[train_end:test_end]))
        start += step_bars
    return splits



def _score_per_obs(model: Any, X: pd.DataFrame) -> float:
    if X.empty or not hasattr(model, "score"):
        return float("nan")
    try:
        ll = float(model.score(X))
    except Exception:
        return float("nan")
    return ll / float(len(X))



def _extract_transition_matrix(model: Any, n_states: int) -> pd.DataFrame:
    tm = getattr(model, "transmat_", None)
    if tm is None:
        return pd.DataFrame(np.nan, index=range(n_states), columns=range(n_states))
    arr = np.asarray(tm, dtype=float)
    if arr.shape != (n_states, n_states):
        return pd.DataFrame(arr)
    return pd.DataFrame(arr, index=range(n_states), columns=range(n_states))



def _label_transition_matrix(trans_raw: pd.DataFrame, mapping: dict[int, str]) -> pd.DataFrame:
    labeled = trans_raw.copy()
    labeled.index = [mapping.get(int(i), f"state_{i}") for i in labeled.index]
    labeled.columns = [mapping.get(int(i), f"state_{i}") for i in labeled.columns]

    # In case multiple states map to the same label, aggregate deterministically.
    labeled = labeled.groupby(level=0).sum()
    labeled = labeled.T.groupby(level=0).sum().T
    return labeled



def _label_sort_key(label: str) -> tuple[int, int, str]:
    if label == "calm":
        return (0, 0, label)
    if label == "neutral":
        return (1, 0, label)
    if label == "stress":
        return (2, 0, label)
    if label.startswith("regime_"):
        try:
            return (3, int(label.split("_", 1)[1]), label)
        except (TypeError, ValueError):
            return (3, 0, label)
    return (4, 0, label)



def _empty_walk_forward_result() -> WalkForwardResult:
    return WalkForwardResult(
        folds=pd.DataFrame(
            columns=[
                "fold",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "n_train",
                "n_test",
                "train_loglik_per_obs",
                "test_loglik_per_obs",
                "share_calm",
                "share_neutral",
                "share_stress",
                "share_other",
            ]
        ),
        drift_transition=pd.DataFrame(
            columns=[
                "prev_fold",
                "curr_fold",
                "mean_abs_delta",
                "max_abs_delta",
                "fro_norm_delta",
            ]
        ),
        drift_label_flip=pd.DataFrame(
            columns=["prev_fold", "curr_fold", "n_overlap", "flip_rate"]
        ),
        drift_loglik=pd.DataFrame(
            columns=[
                "prev_fold",
                "curr_fold",
                "prev_model_loglik_per_obs_on_curr_test",
                "curr_model_loglik_per_obs_on_curr_test",
                "delta_loglik_per_obs",
            ]
        ),
    )



def run_walk_forward_regime_analysis(
    prices: pd.Series | pd.DataFrame,
    *,
    close_col: str = "close",
    hmm_config: HMMConfig | None = None,
    model_cls: Type[GaussianRegimeHMM] = GaussianRegimeHMM,
    train_bars: int = 504,
    test_bars: int = 126,
    step_bars: int | None = None,
    min_test_bars: int = 20,
) -> WalkForwardResult:
    """Run rolling retrain/inference and compute regime drift diagnostics."""
    config = hmm_config or HMMConfig()
    close = prepare_close_series(prices, close_col=close_col)
    splits = walk_forward_splits(
        close.index,
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        min_test_bars=min_test_bars,
    )
    if not splits:
        return _empty_walk_forward_result()

    fold_rows: list[dict[str, object]] = []
    artifacts: list[_FoldArtifact] = []

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        combined = close.loc[(close.index >= train_idx[0]) & (close.index <= test_idx[-1])]

        X_all = build_feature_matrix(combined)
        X_train = X_all.loc[(X_all.index >= train_idx[0]) & (X_all.index <= train_idx[-1])]
        X_test = X_all.loc[(X_all.index >= test_idx[0]) & (X_all.index <= test_idx[-1])]

        min_train = max(20, config.n_states * 5)
        if len(X_train) < min_train or X_test.empty:
            continue

        log_ret_all = np.log(combined).diff(1).reindex(X_all.index)
        log_ret_train = log_ret_all.reindex(X_train.index)

        model = model_cls(config)
        model.fit(X_train)

        states_train = pd.Series(model.predict_states(X_train), index=X_train.index, name="state").astype(int)
        states_test = pd.Series(model.predict_states(X_test), index=X_test.index, name="state").astype(int)

        mapping = label_states_by_volatility(log_ret_train, states_train)
        regimes_test = map_states_to_labels(states_test, mapping, name="regime")

        trans_raw = _extract_transition_matrix(model, config.n_states)
        trans_labeled = _label_transition_matrix(trans_raw, mapping)

        train_ll = _score_per_obs(model, X_train)
        test_ll = _score_per_obs(model, X_test)

        share_calm = float((regimes_test == "calm").mean())
        share_neutral = float((regimes_test == "neutral").mean())
        share_stress = float((regimes_test == "stress").mean())
        share_other = float(1.0 - (share_calm + share_neutral + share_stress))

        fold_rows.append(
            {
                "fold": fold_num,
                "train_start": train_idx[0],
                "train_end": train_idx[-1],
                "test_start": test_idx[0],
                "test_end": test_idx[-1],
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "train_loglik_per_obs": float(train_ll),
                "test_loglik_per_obs": float(test_ll),
                "share_calm": share_calm,
                "share_neutral": share_neutral,
                "share_stress": share_stress,
                "share_other": share_other,
            }
        )

        artifacts.append(
            _FoldArtifact(
                fold=fold_num,
                model=model,
                test_features=X_test,
                test_regimes=regimes_test,
                transition_labeled=trans_labeled,
                test_loglik_per_obs=float(test_ll),
            )
        )

    if not artifacts:
        return _empty_walk_forward_result()

    drift_transition_rows: list[dict[str, object]] = []
    drift_flip_rows: list[dict[str, object]] = []
    drift_loglik_rows: list[dict[str, object]] = []

    for prev, curr in zip(artifacts[:-1], artifacts[1:]):
        labels = sorted(
            set(prev.transition_labeled.index).union(curr.transition_labeled.index),
            key=_label_sort_key,
        )

        prev_t = prev.transition_labeled.reindex(index=labels, columns=labels, fill_value=0.0)
        curr_t = curr.transition_labeled.reindex(index=labels, columns=labels, fill_value=0.0)
        diff = curr_t - prev_t

        drift_transition_rows.append(
            {
                "prev_fold": prev.fold,
                "curr_fold": curr.fold,
                "mean_abs_delta": float(np.abs(diff.to_numpy()).mean()),
                "max_abs_delta": float(np.abs(diff.to_numpy()).max()),
                "fro_norm_delta": float(np.sqrt((diff.to_numpy() ** 2).sum())),
            }
        )

        overlap = prev.test_regimes.index.intersection(curr.test_regimes.index)
        if len(overlap) == 0:
            flip_rate = float("nan")
        else:
            flip_rate = float(
                (prev.test_regimes.reindex(overlap) != curr.test_regimes.reindex(overlap)).mean()
            )
        drift_flip_rows.append(
            {
                "prev_fold": prev.fold,
                "curr_fold": curr.fold,
                "n_overlap": int(len(overlap)),
                "flip_rate": flip_rate,
            }
        )

        prev_ll_on_curr = _score_per_obs(prev.model, curr.test_features)
        curr_ll_on_curr = curr.test_loglik_per_obs
        delta = (
            float(curr_ll_on_curr - prev_ll_on_curr)
            if np.isfinite(curr_ll_on_curr) and np.isfinite(prev_ll_on_curr)
            else float("nan")
        )
        drift_loglik_rows.append(
            {
                "prev_fold": prev.fold,
                "curr_fold": curr.fold,
                "prev_model_loglik_per_obs_on_curr_test": float(prev_ll_on_curr),
                "curr_model_loglik_per_obs_on_curr_test": float(curr_ll_on_curr),
                "delta_loglik_per_obs": delta,
            }
        )

    folds_df = pd.DataFrame(fold_rows).set_index("fold").sort_index()

    drift_transition_df = pd.DataFrame(drift_transition_rows)
    if not drift_transition_df.empty:
        drift_transition_df = drift_transition_df.set_index(["prev_fold", "curr_fold"]).sort_index()

    drift_flip_df = pd.DataFrame(drift_flip_rows)
    if not drift_flip_df.empty:
        drift_flip_df = drift_flip_df.set_index(["prev_fold", "curr_fold"]).sort_index()

    drift_loglik_df = pd.DataFrame(drift_loglik_rows)
    if not drift_loglik_df.empty:
        drift_loglik_df = drift_loglik_df.set_index(["prev_fold", "curr_fold"]).sort_index()

    return WalkForwardResult(
        folds=folds_df,
        drift_transition=drift_transition_df,
        drift_label_flip=drift_flip_df,
        drift_loglik=drift_loglik_df,
    )
