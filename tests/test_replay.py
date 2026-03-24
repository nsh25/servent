from __future__ import annotations

from sp2l_backtest.backtest_engine import BacktestEngine
from sp2l_backtest.replay import SetupReplay

from .conftest import bullish_walkthrough_rows, make_df


def test_walkthrough_replay_sequence(base_config, tmp_path):
    df = make_df(bullish_walkthrough_rows())
    results = BacktestEngine(base_config).run(df)

    replay = SetupReplay(base_config, tmp_path)
    output = replay.replay_setup(df, results, setup_id=0)

    events = output["events"]
    assert any("spike_detected" in event for event in events)
    assert any("correction_detected" in event for event in events)
    assert any("entry_triggered" in event for event in events)
    assert any("exit" in event for event in events)
