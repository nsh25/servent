from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rolling_channel(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel_high": df["high"].rolling(lookback, min_periods=1).max(),
            "channel_low": df["low"].rolling(lookback, min_periods=1).min(),
        }
    )


def swing_points(df: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
    highs = pd.Series(False, index=df.index)
    lows = pd.Series(False, index=df.index)
    for idx in range(lookback, len(df) - lookback):
        highs.iloc[idx] = df["high"].iloc[idx] >= df["high"].iloc[idx - lookback : idx + lookback + 1].max()
        lows.iloc[idx] = df["low"].iloc[idx] <= df["low"].iloc[idx - lookback : idx + lookback + 1].min()
    return pd.DataFrame({"swing_high": highs, "swing_low": lows})


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    std = returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return np.sqrt(periods_per_year) * returns.mean() / std


def annualized_sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return np.sqrt(periods_per_year) * returns.mean() / downside_std
