from __future__ import annotations

import math

import numpy as np
import pandas as pd



def compute_transition_matrix(states: pd.Series, *, n_states: int | None = None) -> pd.DataFrame:
    """Compute empirical one-step transition probabilities between latent states."""
    s = states.dropna().astype(int)
    if s.empty:
        raise ValueError("states is empty.")

    if n_states is None:
        ordered_states = sorted(s.unique().tolist())
    else:
        ordered_states = list(range(n_states))

    counts = pd.DataFrame(0.0, index=ordered_states, columns=ordered_states)
    arr = s.to_numpy()
    for i in range(len(arr) - 1):
        counts.loc[arr[i], arr[i + 1]] += 1.0

    row_sums = counts.sum(axis=1).replace(0.0, np.nan)
    probs = counts.div(row_sums, axis=0).fillna(0.0)
    probs.index.name = "from_state"
    probs.columns.name = "to_state"
    return probs



def regime_duration_stats(states: pd.Series) -> pd.DataFrame:
    """Compute contiguous run-length statistics for each regime state."""
    s = states.dropna().astype(int)
    if s.empty:
        raise ValueError("states is empty.")

    episodes: dict[int, list[int]] = {}
    prev = int(s.iloc[0])
    run = 1

    for cur in s.iloc[1:].astype(int):
        if cur == prev:
            run += 1
            continue
        episodes.setdefault(prev, []).append(run)
        prev = cur
        run = 1
    episodes.setdefault(prev, []).append(run)

    rows = []
    for state in sorted(episodes):
        durations = np.asarray(episodes[state], dtype=float)
        rows.append(
            {
                "state": state,
                "episodes": int(len(durations)),
                "total_bars": int(durations.sum()),
                "mean_duration": float(durations.mean()),
                "median_duration": float(np.median(durations)),
                "max_duration": int(durations.max()),
            }
        )

    out = pd.DataFrame(rows).set_index("state")
    return out.sort_index()



def _annualized_sharpe(mu: float, sigma: float, bars_per_year: float) -> float:
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float(mu / sigma * math.sqrt(bars_per_year))



def regime_return_summary(
    log_returns: pd.Series,
    states: pd.Series,
    *,
    bars_per_year: float = 252.0,
) -> pd.DataFrame:
    """Summarize return distribution and risk-adjusted stats by regime state."""
    r, s = log_returns.align(states, join="inner")
    r = r.dropna()
    s = s.reindex(r.index).dropna().astype(int)
    r = r.reindex(s.index)

    if r.empty:
        raise ValueError("No aligned returns/state observations.")

    rows = []
    for state in sorted(s.unique().tolist()):
        rs = r[s == state]
        mu = float(rs.mean())
        sigma = float(rs.std(ddof=1)) if len(rs) > 1 else float("nan")
        rows.append(
            {
                "state": state,
                "n_obs": int(len(rs)),
                "mean_log_return": mu,
                "std_log_return": sigma,
                "ann_return": float(mu * bars_per_year),
                "ann_vol": float(sigma * math.sqrt(bars_per_year)) if math.isfinite(sigma) else float("nan"),
                "sharpe": _annualized_sharpe(mu, sigma, bars_per_year),
                "hit_rate": float((rs > 0).mean()),
            }
        )

    out = pd.DataFrame(rows).set_index("state").sort_index()
    return out
