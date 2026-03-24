from __future__ import annotations

import pandas as pd

from sp2l_backtest.config import SP2LConfig
from sp2l_backtest.optimizer import ParameterOptimizer
from sp2l_backtest.walk_forward import WalkForwardRunner


def _make_linear_df(n: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    base = pd.Series(range(n), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0 + base * 0.1,
            "high": 100.5 + base * 0.1,
            "low": 99.5 + base * 0.1,
            "close": 100.2 + base * 0.1,
            "volume": 1000.0,
            "atr": 1.0,
            "body": 0.2,
            "ema_fast": 100.0 + base * 0.1,
            "ema_slow": 99.8 + base * 0.1,
            "channel_high": 101.0 + base * 0.1,
            "channel_low": 99.0 + base * 0.1,
            "volume_ma": 1000.0,
            "htf_ema_fast": 100.0 + base * 0.1,
            "htf_ema_slow": 99.8 + base * 0.1,
        }
    )


def test_rolling_walk_forward_split_generation(tmp_path):
    cfg = SP2LConfig()
    cfg.walk_forward.walk_forward_mode = "rolling_window"
    cfg.walk_forward.train_bars = 20
    cfg.walk_forward.test_bars = 10
    cfg.walk_forward.step_bars = 10

    runner = WalkForwardRunner(cfg, _make_linear_df(60), tmp_path)
    folds = runner.generate_folds(60)

    assert len(folds) == 4
    assert folds[0].train_start == 0
    assert folds[1].train_start == 10
    assert folds[0].test_start == 20
    assert folds[0].test_end == 30


def test_expanding_walk_forward_split_generation(tmp_path):
    cfg = SP2LConfig()
    cfg.walk_forward.walk_forward_mode = "expanding_window"
    cfg.walk_forward.train_bars = 20
    cfg.walk_forward.test_bars = 10
    cfg.walk_forward.step_bars = 10

    runner = WalkForwardRunner(cfg, _make_linear_df(60), tmp_path)
    folds = runner.generate_folds(60)

    assert len(folds) == 4
    assert all(f.train_start == 0 for f in folds)
    assert [f.train_end for f in folds] == [20, 30, 40, 50]


def test_no_leakage_between_train_and_test(tmp_path):
    cfg = SP2LConfig()
    cfg.walk_forward.walk_forward_mode = "rolling_window"
    cfg.walk_forward.train_bars = 20
    cfg.walk_forward.test_bars = 10
    cfg.walk_forward.step_bars = 10

    runner = WalkForwardRunner(cfg, _make_linear_df(60), tmp_path)
    for fold in runner.generate_folds(60):
        assert fold.train_end <= fold.test_start


def test_optimizer_uses_training_data_only(monkeypatch):
    cfg = SP2LConfig()
    cfg.optimization.max_trials = 2

    seen_lengths: list[int] = []

    class DummyStrategy:
        def __init__(self, _cfg):
            pass

        def run(self, df):
            seen_lengths.append(len(df))
            return {
                "trades": [],
                "equity_curve": pd.DataFrame({"timestamp": df["timestamp"], "equity": [100000.0] * len(df)}),
            }

    def fake_summary(self, trades, equity_curve, df):
        return pd.DataFrame([{"total_trades": 0, "sharpe": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "average_r": 0.0, "total_return": 0.0}])

    monkeypatch.setattr("sp2l_backtest.optimizer.SP2LStrategy", DummyStrategy)
    monkeypatch.setattr("sp2l_backtest.optimizer.Reporter.summary", fake_summary)

    train_df = _make_linear_df(25)
    ParameterOptimizer(cfg).optimize(train_df)
    assert seen_lengths
    assert all(length == 25 for length in seen_lengths)


def test_combined_oos_metrics_generated(tmp_path):
    cfg = SP2LConfig()
    cfg.walk_forward.walk_forward_enabled = True
    cfg.walk_forward.walk_forward_mode = "rolling_window"
    cfg.walk_forward.train_bars = 20
    cfg.walk_forward.test_bars = 10
    cfg.walk_forward.step_bars = 10
    cfg.reporting.output_dir = str(tmp_path)

    runner = WalkForwardRunner(cfg, _make_linear_df(60), tmp_path)
    results = runner.run()

    assert "summary" in results
    assert not results["summary"].empty
    assert "total_return" in results["summary"].columns


def test_retune_false_reuses_frozen_params(tmp_path):
    cfg = SP2LConfig()
    cfg.walk_forward.walk_forward_enabled = True
    cfg.walk_forward.walk_forward_mode = "rolling_window"
    cfg.walk_forward.train_bars = 20
    cfg.walk_forward.test_bars = 10
    cfg.walk_forward.step_bars = 10
    cfg.walk_forward.retune_each_window = False
    cfg.optimization.optimization_enabled = True
    cfg.optimization.max_trials = 1

    runner = WalkForwardRunner(cfg, _make_linear_df(60), tmp_path)
    results = runner.run()

    assert "optimization_summary" in results
    assert (tmp_path / "walk_forward_optimization_summary.csv").exists()
