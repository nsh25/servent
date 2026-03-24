from __future__ import annotations

from sp2l_backtest.backtest_engine import BacktestEngine

from .conftest import bullish_walkthrough_rows, make_df


def test_bullish_sp2l_walkthrough(base_config):
    """Walk through the full SP2L lifecycle on deterministic synthetic data."""

    df = make_df(bullish_walkthrough_rows())
    results = BacktestEngine(base_config).run(df)

    # 1) A long candidate should be detected and survive filters.
    assert len(results["candidates"]) >= 1
    candidate = next(candidate for candidate in results["candidates"] if candidate.direction == "long")
    assert candidate.spike.gap_passed is True
    assert candidate.correction is not None
    assert candidate.correction.second_leg_valid is True
    assert candidate.status == "entered"
    assert candidate.filters_passed["htf_trend"] is True

    # 2) The signal table should show both the spike and the pending-entry phase.
    signal_statuses = set(results["signal_table"]["status"].tolist())
    assert "spike_detected" in signal_statuses
    assert "pending_entry" in signal_statuses

    # 3) The resulting trade should fill, scale in, and exit at the take-profit.
    assert len(results["trades"]) == 1
    trade = results["trades"][0]
    assert trade.direction == "long"
    assert trade.entry_idx > candidate.spike.end_idx
    assert trade.scale_in_price is not None
    assert trade.average_entry < trade.entry_price
    assert trade.exit_reason == "take_profit"
    assert trade.pnl > 0
    assert trade.r_multiple > 0
