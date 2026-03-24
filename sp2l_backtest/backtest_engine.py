from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .config import SP2LConfig
from .data_loader import DataLoader
from .indicators import atr, ema, rolling_channel
from .plot_utils import PlotBuilder
from .reporting import Reporter
from .strategy import SP2LStrategy


class BacktestEngine:
    """High-level orchestrator for data preparation, strategy execution, and reporting."""

    def __init__(self, config: SP2LConfig):
        self.config = config

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = atr(df)
        df["body"] = (df["close"] - df["open"]).abs()
        df["ema_fast"] = ema(df["close"], 12)
        df["ema_slow"] = ema(df["close"], 26)
        channel = rolling_channel(df, self.config.filters.channel_lookback)
        df = pd.concat([df, channel], axis=1)
        df["volume_ma"] = df["volume"].rolling(self.config.filters.volume_lookback, min_periods=1).mean()

        if self.config.filters.enable_htf_trend:
            htf = (
                df.set_index("timestamp")["close"]
                .resample(self.config.filters.htf_resample_rule)
                .last()
                .dropna()
                .to_frame(name="close")
            )
            htf["htf_ema_fast"] = ema(htf["close"], self.config.filters.htf_fast_ema)
            htf["htf_ema_slow"] = ema(htf["close"], self.config.filters.htf_slow_ema)
            htf = htf[["htf_ema_fast", "htf_ema_slow"]]
            df = df.merge(htf, left_on="timestamp", right_index=True, how="left")
            df[["htf_ema_fast", "htf_ema_slow"]] = df[["htf_ema_fast", "htf_ema_slow"]].ffill()
        else:
            df["htf_ema_fast"] = df["ema_fast"]
            df["htf_ema_slow"] = df["ema_slow"]

        return df

    def run(self, df: pd.DataFrame | None = None) -> Dict[str, object]:
        if df is None:
            df = DataLoader.load_csv(self.config.data)
        prepared = self.prepare_data(df)
        strategy = SP2LStrategy(self.config)
        results = strategy.run(prepared)

        output_dir = Path(self.config.reporting.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        reporter = Reporter(self.config, output_dir)
        reporter.write_outputs(prepared, results)

        if self.config.reporting.save_plots:
            PlotBuilder(self.config, output_dir).save_all(prepared, results)
        return results
