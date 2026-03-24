from __future__ import annotations

import pandas as pd
import pytest

from sp2l_backtest.backtest_engine import BacktestEngine
from sp2l_backtest.config import SP2LConfig


@pytest.fixture
def base_config(tmp_path):
    config = SP2LConfig()
    config.reporting.output_dir = str(tmp_path / "output")
    config.reporting.save_plots = False
    config.reporting.verbose_debug = False
    config.filters.enable_htf_trend = False
    config.filters.enable_breakout_filter = False
    config.filters.enable_channel_edge_filter = False
    config.filters.enable_min_spike_strength_filter = False
    config.filters.enable_low_volatility_filter = False
    config.filters.enable_session_filter = False
    config.filters.enable_volume_filter = False
    config.entry.order_expiry_bars = 4
    config.correction.max_bars_after_spike = 5
    return config


@pytest.fixture
def engine(base_config):
    return BacktestEngine(base_config)


def make_df(rows):
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def bullish_walkthrough_rows() -> list[dict[str, float | int | str]]:
    """Synthetic long setup that walks through spike -> correction -> entry -> scale-in -> TP."""

    return [
        {"timestamp": "2024-01-01 09:30", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000},
        {"timestamp": "2024-01-01 09:31", "open": 101.2, "high": 103.6, "low": 101.1, "close": 103.3, "volume": 1200},
        {"timestamp": "2024-01-01 09:32", "open": 104.0, "high": 105.6, "low": 103.9, "close": 105.4, "volume": 1250},
        {"timestamp": "2024-01-01 09:33", "open": 105.1, "high": 105.3, "low": 103.5, "close": 103.9, "volume": 950},
        {"timestamp": "2024-01-01 09:34", "open": 104.0, "high": 105.8, "low": 104.0, "close": 105.6, "volume": 1400},
        {"timestamp": "2024-01-01 09:35", "open": 105.3, "high": 105.5, "low": 102.2, "close": 103.0, "volume": 1500},
        {"timestamp": "2024-01-01 09:36", "open": 103.0, "high": 109.2, "low": 102.8, "close": 108.8, "volume": 1500},
    ]
