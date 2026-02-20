from __future__ import annotations

import pandas as pd



def plot_price_with_regimes(
    close: pd.Series,
    regimes: pd.Series,
    *,
    title: str = "Price with inferred regimes",
):
    """Plot close price with background shading by regime label."""
    import matplotlib.pyplot as plt

    c, r = close.align(regimes, join="inner")
    if c.empty:
        raise ValueError("No overlapping close/regime data to plot.")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(c.index, c.values, lw=1.5, color="black", label="Close")

    labels = sorted(r.dropna().astype(str).unique().tolist())
    colors = {
        label: color
        for label, color in zip(labels, ["#d9edf7", "#fcf8e3", "#f2dede", "#e8daef", "#d5f5e3"])
    }

    for label in labels:
        mask = r.astype(str) == label
        if not mask.any():
            continue
        y0, y1 = ax.get_ylim()
        ax.fill_between(c.index, y0, y1, where=mask, color=colors[label], alpha=0.25, label=label)

    ax.set_title(title)
    ax.set_xlabel("datetime")
    ax.set_ylabel("price")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, ax
