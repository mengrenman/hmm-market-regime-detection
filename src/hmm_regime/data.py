from __future__ import annotations

from pathlib import Path

import pandas as pd



def load_csv_prices(
    path: str | Path,
    *,
    datetime_col: str = "datetime",
    close_col: str = "close",
) -> pd.DataFrame:
    """Load a CSV price file and return a datetime-indexed close-price DataFrame."""
    df = pd.read_csv(path)
    if datetime_col not in df.columns:
        raise KeyError(f"Missing datetime column: {datetime_col!r}")
    if close_col not in df.columns:
        raise KeyError(f"Missing close column: {close_col!r}")

    out = df[[datetime_col, close_col]].copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")
    out[close_col] = pd.to_numeric(out[close_col], errors="coerce")
    out = out.dropna(subset=[datetime_col, close_col])
    out = out.sort_values(datetime_col)
    out = out.set_index(datetime_col)
    out.index.name = "datetime"

    if (out[close_col] <= 0).any():
        raise ValueError("Close prices must be positive for log-return calculations.")

    return out



def download_openbb_prices(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    provider: str = "yfinance",
    close_col: str = "close",
) -> pd.DataFrame:
    """Download historical prices from OpenBB and return datetime-indexed close prices."""
    try:
        from openbb import obb
    except ImportError as exc:
        raise ImportError(
            "OpenBB is not installed in this environment. Install with `pip install openbb` "
            "or use --input to load a local CSV."
        ) from exc

    obb.user.preferences.output_type = "dataframe"
    df = obb.equity.price.historical(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        provider=provider,
    )

    if df is None or len(df) == 0:
        raise ValueError(
            f"No rows returned from OpenBB for symbol={symbol!r}, "
            f"start_date={start_date!r}, end_date={end_date!r}, provider={provider!r}."
        )
    if close_col not in df.columns:
        raise KeyError(f"OpenBB response missing close column: {close_col!r}")

    out = df[[close_col]].copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.dropna().sort_index()
    out.index.name = "datetime"

    if out.empty:
        raise ValueError("Downloaded OpenBB data is empty after cleaning.")
    if (out[close_col] <= 0).any():
        raise ValueError("Close prices must be strictly positive.")

    return out



def prepare_close_series(prices: pd.Series | pd.DataFrame, *, close_col: str = "close") -> pd.Series:
    """Normalize input price object to a strictly positive datetime-indexed close series."""
    if isinstance(prices, pd.Series):
        close = pd.to_numeric(prices, errors="coerce").copy()
        close.name = close_col
    else:
        if close_col not in prices.columns:
            raise KeyError(f"Missing close column: {close_col!r}")
        close = pd.to_numeric(prices[close_col], errors="coerce").copy()

    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("Price index must be a DatetimeIndex.")

    close = close.sort_index().dropna()

    if close.empty:
        raise ValueError("Close series is empty after cleaning.")
    if (close <= 0).any():
        raise ValueError("Close prices must be strictly positive.")

    close.name = close_col
    return close
