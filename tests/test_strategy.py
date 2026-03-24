from __future__ import annotations

from sp2l_backtest.backtest_engine import BacktestEngine

from .conftest import make_df


def test_bullish_spike_with_gap_and_continuation(base_config):
    df = make_df(
        [
            {"timestamp": "2024-01-01 09:30", "open": 100, "high": 101, "low": 99, "close": 100.2, "volume": 1000},
            {"timestamp": "2024-01-01 09:31", "open": 101, "high": 103, "low": 100.8, "close": 102.8, "volume": 1200},
            {"timestamp": "2024-01-01 09:32", "open": 103.4, "high": 105, "low": 103.2, "close": 104.8, "volume": 1300},
            {"timestamp": "2024-01-01 09:33", "open": 104.5, "high": 104.8, "low": 103.0, "close": 103.4, "volume": 1100},
            {"timestamp": "2024-01-01 09:34", "open": 103.5, "high": 105.5, "low": 103.4, "close": 105.2, "volume": 1400},
            {"timestamp": "2024-01-01 09:35", "open": 105.2, "high": 108.0, "low": 104.9, "close": 107.8, "volume": 1500},
        ]
    )
    results = BacktestEngine(base_config).run(df)
    assert len(results["trades"]) == 1
    assert results["trades"][0].direction == "long"
    assert results["trades"][0].exit_reason == "take_profit"


def test_bearish_spike_with_gap_and_continuation(base_config):
    df = make_df(
        [
            {"timestamp": "2024-01-01 09:30", "open": 110, "high": 111, "low": 109, "close": 109.8, "volume": 1000},
            {"timestamp": "2024-01-01 09:31", "open": 108.8, "high": 109.0, "low": 106.5, "close": 106.7, "volume": 1200},
            {"timestamp": "2024-01-01 09:32", "open": 106.0, "high": 106.2, "low": 103.5, "close": 103.8, "volume": 1250},
            {"timestamp": "2024-01-01 09:33", "open": 104.0, "high": 105.2, "low": 103.8, "close": 104.8, "volume": 900},
            {"timestamp": "2024-01-01 09:34", "open": 104.4, "high": 104.5, "low": 102.9, "close": 103.0, "volume": 1400},
            {"timestamp": "2024-01-01 09:35", "open": 102.8, "high": 103.0, "low": 100.2, "close": 100.5, "volume": 1600},
        ]
    )
    results = BacktestEngine(base_config).run(df)
    assert len(results["trades"]) == 1
    assert results["trades"][0].direction == "short"
    assert results["trades"][0].exit_reason == "take_profit"


def test_invalid_spike_without_gap(base_config):
    base_config.spike.require_at_least_one_gap = True
    df = make_df(
        [
            {"timestamp": "2024-01-01 09:30", "open": 100, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000},
            {"timestamp": "2024-01-01 09:31", "open": 100.0, "high": 101.2, "low": 99.8, "close": 101.0, "volume": 1200},
            {"timestamp": "2024-01-01 09:32", "open": 101.0, "high": 102.1, "low": 100.7, "close": 101.9, "volume": 1200},
            {"timestamp": "2024-01-01 09:33", "open": 101.8, "high": 102.0, "low": 101.0, "close": 101.2, "volume": 900},
            {"timestamp": "2024-01-01 09:34", "open": 101.2, "high": 101.6, "low": 101.1, "close": 101.4, "volume": 900},
        ]
    )
    results = BacktestEngine(base_config).run(df)
    assert len(results["trades"]) == 0


def test_valid_spike_but_no_second_leg(base_config):
    df = make_df(
        [
            {"timestamp": "2024-01-01 09:30", "open": 100, "high": 101, "low": 99, "close": 100.0, "volume": 1000},
            {"timestamp": "2024-01-01 09:31", "open": 101.3, "high": 103.5, "low": 101.2, "close": 103.2, "volume": 1200},
            {"timestamp": "2024-01-01 09:32", "open": 103.8, "high": 105.4, "low": 103.7, "close": 105.1, "volume": 1300},
            {"timestamp": "2024-01-01 09:33", "open": 105.0, "high": 105.5, "low": 104.8, "close": 105.3, "volume": 1300},
            {"timestamp": "2024-01-01 09:34", "open": 105.2, "high": 105.6, "low": 105.0, "close": 105.4, "volume": 1350},
        ]
    )
    results = BacktestEngine(base_config).run(df)
    assert len(results["trades"]) == 0
    assert any(candidate.rejection_reason == "setup_expired_without_correction" for candidate in results["candidates"])


def test_scale_in_triggered(base_config):
    df = make_df(
        [
            {"timestamp": "2024-01-01 09:30", "open": 100, "high": 101, "low": 99, "close": 100.0, "volume": 1000},
            {"timestamp": "2024-01-01 09:31", "open": 101.2, "high": 103.6, "low": 101.1, "close": 103.3, "volume": 1200},
            {"timestamp": "2024-01-01 09:32", "open": 104.0, "high": 105.6, "low": 103.9, "close": 105.4, "volume": 1250},
            {"timestamp": "2024-01-01 09:33", "open": 105.1, "high": 105.3, "low": 103.5, "close": 103.9, "volume": 950},
            {"timestamp": "2024-01-01 09:34", "open": 104.0, "high": 105.8, "low": 104.0, "close": 105.6, "volume": 1400},
            {"timestamp": "2024-01-01 09:35", "open": 105.3, "high": 105.5, "low": 102.2, "close": 103.0, "volume": 1500},
            {"timestamp": "2024-01-01 09:36", "open": 103.0, "high": 109.2, "low": 102.8, "close": 108.8, "volume": 1500},
        ]
    )
    results = BacktestEngine(base_config).run(df)
    assert len(results["trades"]) == 1
    assert results["trades"][0].scale_in_price is not None


def test_stop_loss_hit(base_config):
    df = make_df(
        [
            {"timestamp": "2024-01-01 09:30", "open": 100, "high": 101, "low": 99, "close": 100.1, "volume": 1000},
            {"timestamp": "2024-01-01 09:31", "open": 101.2, "high": 103.7, "low": 101.1, "close": 103.4, "volume": 1200},
            {"timestamp": "2024-01-01 09:32", "open": 104.0, "high": 105.5, "low": 103.9, "close": 105.2, "volume": 1250},
            {"timestamp": "2024-01-01 09:33", "open": 104.9, "high": 105.0, "low": 103.3, "close": 103.6, "volume": 900},
            {"timestamp": "2024-01-01 09:34", "open": 103.7, "high": 105.8, "low": 103.5, "close": 105.5, "volume": 1400},
            {"timestamp": "2024-01-01 09:35", "open": 105.0, "high": 105.2, "low": 98.8, "close": 99.0, "volume": 1600},
        ]
    )
    results = BacktestEngine(base_config).run(df)
    assert len(results["trades"]) == 1
    assert results["trades"][0].exit_reason in {"stop_loss", "stop_and_target_same_bar_stop_priority"}


def test_take_profit_hit(base_config):
    df = make_df(
        [
            {"timestamp": "2024-01-01 09:30", "open": 100, "high": 101, "low": 99, "close": 100.1, "volume": 1000},
            {"timestamp": "2024-01-01 09:31", "open": 101.4, "high": 103.9, "low": 101.3, "close": 103.5, "volume": 1200},
            {"timestamp": "2024-01-01 09:32", "open": 104.2, "high": 105.7, "low": 104.1, "close": 105.3, "volume": 1250},
            {"timestamp": "2024-01-01 09:33", "open": 104.8, "high": 105.0, "low": 103.4, "close": 103.7, "volume": 900},
            {"timestamp": "2024-01-01 09:34", "open": 103.8, "high": 105.9, "low": 103.7, "close": 105.7, "volume": 1400},
            {"timestamp": "2024-01-01 09:35", "open": 105.6, "high": 108.9, "low": 105.5, "close": 108.7, "volume": 1700},
        ]
    )
    results = BacktestEngine(base_config).run(df)
    assert len(results["trades"]) == 1
    assert results["trades"][0].exit_reason == "take_profit"
