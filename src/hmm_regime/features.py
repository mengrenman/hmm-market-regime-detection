from __future__ import annotations

import numpy as np
import pandas as pd



def build_feature_matrix(
    close: pd.Series,
    *,
    return_lags: tuple[int, ...] = (1, 5, 20),
    vol_window: int = 20,
    trend_window: int = 63,
    drawdown_window: int = 126,
    zscore_window: int = 63,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build a deterministic feature matrix for regime modeling from a close-price series."""
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must have a DatetimeIndex.")

    c = pd.to_numeric(close, errors="coerce").sort_index()
    if c.isna().all():
        raise ValueError("close series contains only NaN values.")

    log_price = np.log(c)
    ret_1 = log_price.diff(1)

    features: dict[str, pd.Series] = {}
    for lag in return_lags:
        if lag <= 0:
            raise ValueError(f"return lag must be positive, got {lag}")
        features[f"log_ret_{lag}"] = log_price.diff(lag)

    vol = ret_1.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)
    trend = ret_1.rolling(trend_window, min_periods=trend_window).mean() * 252.0

    roll_max = c.rolling(drawdown_window, min_periods=drawdown_window).max()
    drawdown = (c / roll_max) - 1.0

    z_mean = ret_1.rolling(zscore_window, min_periods=zscore_window).mean()
    z_std = ret_1.rolling(zscore_window, min_periods=zscore_window).std(ddof=1)
    z_ret = (ret_1 - z_mean) / z_std

    features["ann_vol"] = vol
    features["ann_trend"] = trend
    features["drawdown"] = drawdown
    features["z_ret_1"] = z_ret

    X = pd.DataFrame(features, index=c.index)
    X = X.replace([np.inf, -np.inf], np.nan)

    if dropna:
        X = X.dropna(how="any")

    if X.empty:
        raise ValueError("Feature matrix is empty; increase sample size or adjust windows.")

    return X
