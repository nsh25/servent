from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from .config import SP2LConfig


class PlotBuilder:
    """Generate simple matplotlib charts for equity, drawdown, and annotated candles."""

    def __init__(self, config: SP2LConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir

    def save_all(self, df: pd.DataFrame, results: Dict[str, object]) -> None:
        self._equity_chart(results["equity_curve"])
        self._drawdown_chart(results["equity_curve"])
        self._candlestick_chart(df, results)

    def _equity_chart(self, equity_curve: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(equity_curve["timestamp"], equity_curve["equity"], label="Equity")
        ax.set_title("Equity Curve")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self.output_dir / "equity_curve.png")
        plt.close(fig)

    def _drawdown_chart(self, equity_curve: pd.DataFrame) -> None:
        equity = equity_curve["equity"]
        drawdown = equity / equity.cummax() - 1.0
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(equity_curve["timestamp"], drawdown, 0.0, color="tab:red", alpha=0.3)
        ax.set_title("Drawdown Curve")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self.output_dir / "drawdown_curve.png")
        plt.close(fig)

    def _candlestick_chart(self, df: pd.DataFrame, results: Dict[str, object]) -> None:
        window = df.tail(self.config.reporting.candlestick_bars).reset_index(drop=False)
        fig, ax = plt.subplots(figsize=(14, 7))
        width = 0.6
        for plot_idx, (_, row) in enumerate(window.iterrows()):
            color = "tab:green" if row["close"] >= row["open"] else "tab:red"
            ax.plot([plot_idx, plot_idx], [row["low"], row["high"]], color="black", linewidth=1)
            body_low = min(row["open"], row["close"])
            body_height = max(abs(row["close"] - row["open"]), 1e-6)
            rect = patches.Rectangle((plot_idx - width / 2, body_low), width, body_height, color=color, alpha=0.7)
            ax.add_patch(rect)

        base_index_map = {int(row["index"]): plot_idx for plot_idx, (_, row) in enumerate(window.iterrows())}
        for candidate in results["candidates"]:
            if candidate.spike.start_idx in base_index_map and candidate.spike.end_idx in base_index_map:
                x0 = base_index_map[candidate.spike.start_idx]
                x1 = base_index_map[candidate.spike.end_idx]
                ax.axvspan(x0 - 0.5, x1 + 0.5, color="gold", alpha=0.12)
            if candidate.correction and candidate.correction.signal_idx in base_index_map:
                xs = base_index_map[candidate.correction.signal_idx]
                ax.scatter(xs, df.iloc[candidate.correction.signal_idx]["close"], color="blue", marker="o", label="Signal")

        for trade in results["trades"]:
            if trade.entry_idx in base_index_map:
                x = base_index_map[trade.entry_idx]
                ax.scatter(x, trade.entry_price, color="green", marker="^", s=80)
                ax.hlines([trade.stop_price, trade.take_profit], x - 0.4, x + 0.4, colors=["red", "green"], linestyles="--")
            if trade.exit_idx in base_index_map:
                ax.scatter(base_index_map[trade.exit_idx], trade.exit_price, color="black", marker="x", s=70)
            if trade.scale_in_price and trade.entry_idx in base_index_map:
                ax.scatter(base_index_map.get(trade.exit_idx, base_index_map[trade.entry_idx]), trade.scale_in_price, color="orange", marker="s", s=60)

        ax.set_title("SP2L Candlestick Annotations")
        ax.set_xlim(-1, len(window) + 1)
        fig.tight_layout()
        fig.savefig(self.output_dir / "candlestick_annotations.png")
        plt.close(fig)
