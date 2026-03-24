from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import SP2LConfig
from .indicators import annualized_sharpe, annualized_sortino
from .models import SetupCandidate, TradeRecord


class Reporter:
    """Create CSV outputs and performance summary from engine results."""

    def __init__(self, config: SP2LConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir

    def write_outputs(self, df: pd.DataFrame, results: Dict[str, object]) -> None:
        trades = results["trades"]
        candidates = results["candidates"]
        signal_table = results["signal_table"]
        equity_curve = results["equity_curve"]

        trade_df = self._trade_log_frame(trades)
        trade_df.to_csv(self.output_dir / "trade_log.csv", index=False)

        if self.config.reporting.save_candidate_setups:
            candidate_df = self._candidate_frame(candidates)
            candidate_df.to_csv(self.output_dir / "candidate_setups.csv", index=False)

        signal_table.to_csv(self.output_dir / "signal_table.csv", index=False)
        equity_curve.to_csv(self.output_dir / "equity_curve.csv", index=False)
        self.summary(trades, equity_curve, df).to_csv(self.output_dir / "performance_summary.csv", index=False)

        debug_log = pd.DataFrame({"message": results["debug_log"]})
        debug_log.to_csv(self.output_dir / "debug_log.csv", index=False)

    def summary(self, trades: List[TradeRecord], equity_curve: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        if trades:
            trade_df = self._trade_log_frame(trades)
        else:
            trade_df = pd.DataFrame(columns=["pnl", "r_multiple", "direction", "duration_bars"])

        equity = equity_curve["equity"]
        returns = equity.pct_change().fillna(0.0)
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 and equity.iloc[0] else 0.0
        days = max((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days, 1)
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (365 / days) - 1 if len(equity) > 1 and equity.iloc[0] > 0 else 0.0
        gross_profit = trade_df[trade_df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trade_df[trade_df["pnl"] < 0]["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss else np.inf if gross_profit else 0.0
        expectancy = trade_df["pnl"].mean() if not trade_df.empty else 0.0
        average_r = trade_df["r_multiple"].mean() if not trade_df.empty else 0.0
        rolling_max = equity.cummax()
        drawdown = equity / rolling_max - 1.0
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0
        summary = {
            "total_return": total_return,
            "cagr": cagr,
            "win_rate": (trade_df["pnl"] > 0).mean() if not trade_df.empty else 0.0,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "average_r": average_r,
            "max_drawdown": max_drawdown,
            "sharpe": annualized_sharpe(returns),
            "sortino": annualized_sortino(returns),
            "total_trades": len(trade_df),
            "long_trades": int((trade_df["direction"] == "long").sum()) if not trade_df.empty else 0,
            "short_trades": int((trade_df["direction"] == "short").sum()) if not trade_df.empty else 0,
            "average_trade_duration": trade_df["duration_bars"].mean() if not trade_df.empty else 0.0,
        }
        return pd.DataFrame([summary])

    @staticmethod
    def _trade_log_frame(trades: List[TradeRecord]) -> pd.DataFrame:
        frame = pd.DataFrame([asdict(trade) for trade in trades])
        return frame

    @staticmethod
    def _candidate_frame(candidates: List[SetupCandidate]) -> pd.DataFrame:
        records = []
        for candidate in candidates:
            record = {
                "direction": candidate.direction,
                "spike_start_idx": candidate.spike.start_idx,
                "spike_end_idx": candidate.spike.end_idx,
                "origin_idx": candidate.spike.origin_idx,
                "gap_passed": candidate.spike.gap_passed,
                "status": candidate.status,
                "rejection_reason": candidate.rejection_reason,
                "entry_confirmed": candidate.entry_confirmed,
                "entry_idx": candidate.entry_idx,
                "entry_price": candidate.entry_price,
                "stop_price": candidate.stop_price,
                "take_profit": candidate.take_profit,
                "filters_passed": candidate.filters_passed,
                "debug": candidate.debug,
            }
            if candidate.correction:
                record.update(
                    {
                        "signal_idx": candidate.correction.signal_idx,
                        "correction_bars": candidate.correction.correction_bars,
                        "correction_depth": candidate.correction.correction_depth,
                        "second_leg_valid": candidate.correction.second_leg_valid,
                    }
                )
            records.append(record)
        return pd.DataFrame(records)
