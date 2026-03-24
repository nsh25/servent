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
