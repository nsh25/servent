from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig


REQUIRED_COLUMNS = ["open", "high", "low", "close"]


class DataLoader:
    """Load and normalize OHLCV CSV data for the backtester."""

    @staticmethod
    def load_csv(config: DataConfig) -> pd.DataFrame:
        df = pd.read_csv(Path(config.csv_path))
        rename_map = {config.timestamp_col: "timestamp"}
        df = df.rename(columns=rename_map)

        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLC columns: {missing}")

        if "timestamp" not in df.columns:
            raise ValueError("CSV must include a timestamp column or configure timestamp_col.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if config.timezone and config.timezone.upper() != "UTC":
            df["timestamp"] = df["timestamp"].dt.tz_convert(config.timezone)

        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df["volume"] = df["volume"].fillna(0.0)

        if config.start:
            df = df[df["timestamp"] >= pd.Timestamp(config.start, tz=df["timestamp"].dt.tz)]
        if config.end:
            df = df[df["timestamp"] <= pd.Timestamp(config.end, tz=df["timestamp"].dt.tz)]

        numeric_columns = [column for column in ["open", "high", "low", "close", "volume"] if column in df.columns]
        df[numeric_columns] = df[numeric_columns].astype(float)
        return df.reset_index(drop=True)
