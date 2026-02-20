from __future__ import annotations

import pandas as pd



def label_states_by_volatility(log_returns: pd.Series, states: pd.Series) -> dict[int, str]:
    """Assign semantic labels to latent states using per-state return volatility ranking."""
    r, s = log_returns.align(states, join="inner")
    if r.empty:
        raise ValueError("No overlapping data between returns and states.")

    vol_by_state: dict[int, float] = {}
    for state in sorted(s.dropna().astype(int).unique().tolist()):
        state_vol = r[s == state].std(ddof=1)
        vol_by_state[state] = float(state_vol) if pd.notna(state_vol) else float("inf")

    ordered_states = [k for k, _ in sorted(vol_by_state.items(), key=lambda item: item[1])]

    n = len(ordered_states)
    if n == 0:
        raise ValueError("No states found for labeling.")

    if n == 1:
        return {ordered_states[0]: "single_regime"}

    mapping: dict[int, str] = {}
    for i, state in enumerate(ordered_states):
        if i == 0:
            mapping[state] = "calm"
        elif i == n - 1:
            mapping[state] = "stress"
        elif n == 3 and i == 1:
            mapping[state] = "neutral"
        else:
            mapping[state] = f"regime_{i}"
    return mapping



def map_states_to_labels(states: pd.Series, mapping: dict[int, str], *, name: str = "regime") -> pd.Series:
    """Map integer latent states to semantic labels."""
    labeled = states.astype(int).map(mapping)
    if labeled.isna().any():
        missing_states = sorted(set(states.astype(int).unique()) - set(mapping.keys()))
        raise KeyError(f"Missing labels for states: {missing_states}")
    labeled.name = name
    return labeled
