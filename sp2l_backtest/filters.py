from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from .config import FilterConfig


@dataclass
class FilterResult:
    passed: bool
    details: Dict[str, bool]
    reasons: list[str]


class SetupFilters:
    """Optional quality filters kept separate from the main strategy state machine."""

    def __init__(self, config: FilterConfig):
        self.config = config

    def evaluate(self, df: pd.DataFrame, idx: int, direction: str, spike_strength_atr: float) -> FilterResult:
        details: Dict[str, bool] = {}
        reasons: list[str] = []

        checks = [
            self._htf_trend(df, idx, direction),
            self._breakout(df, idx, direction),
            self._channel_edge(df, idx, direction),
            self._spike_strength(spike_strength_atr),
            self._volatility(df, idx),
            self._session(df, idx),
            self._volume(df, idx),
        ]

        for name, passed, reason in checks:
            details[name] = passed
            if not passed and reason:
                reasons.append(reason)

        return FilterResult(passed=all(details.values()), details=details, reasons=reasons)

    def _htf_trend(self, df: pd.DataFrame, idx: int, direction: str) -> Tuple[str, bool, str]:
        if not self.config.enable_htf_trend:
            return "htf_trend", True, ""
        row = df.iloc[idx]
        if direction == "long":
            passed = row["htf_ema_fast"] > row["htf_ema_slow"]
        else:
            passed = row["htf_ema_fast"] < row["htf_ema_slow"]
        return "htf_trend", bool(passed), "HTF trend alignment failed"

    def _breakout(self, df: pd.DataFrame, idx: int, direction: str) -> Tuple[str, bool, str]:
        if not self.config.enable_breakout_filter:
            return "breakout", True, ""
        start = max(0, idx - self.config.breakout_lookback)
        window = df.iloc[start:idx]
        if window.empty:
            return "breakout", False, "Breakout filter lacks lookback history"
        passed = df.iloc[idx]["close"] > window["high"].max() if direction == "long" else df.iloc[idx]["close"] < window["low"].min()
        return "breakout", bool(passed), "Breakout filter failed"

    def _channel_edge(self, df: pd.DataFrame, idx: int, direction: str) -> Tuple[str, bool, str]:
        if not self.config.enable_channel_edge_filter:
            return "channel_edge", True, ""
        row = df.iloc[idx]
        channel_range = row["channel_high"] - row["channel_low"]
        if channel_range <= 0:
            return "channel_edge", False, "Channel edge range was non-positive"
        if direction == "long":
            distance = (row["close"] - row["channel_low"]) / channel_range
            passed = distance <= self.config.channel_edge_threshold
        else:
            distance = (row["channel_high"] - row["close"]) / channel_range
            passed = distance <= self.config.channel_edge_threshold
        return "channel_edge", bool(passed), "Channel edge filter failed"

    def _spike_strength(self, spike_strength_atr: float) -> Tuple[str, bool, str]:
        if not self.config.enable_min_spike_strength_filter:
            return "spike_strength", True, ""
        passed = spike_strength_atr >= self.config.min_spike_strength_atr
        return "spike_strength", bool(passed), "Spike strength filter failed"

    def _volatility(self, df: pd.DataFrame, idx: int) -> Tuple[str, bool, str]:
        if not self.config.enable_low_volatility_filter:
            return "volatility", True, ""
        row = df.iloc[idx]
        atr_pct = row["atr"] / max(row["close"], 1e-9)
        passed = atr_pct >= self.config.min_atr_percent
        return "volatility", bool(passed), "Low volatility filter failed"

    def _session(self, df: pd.DataFrame, idx: int) -> Tuple[str, bool, str]:
        if not self.config.enable_session_filter:
            return "session", True, ""
        timestamp = pd.Timestamp(df.iloc[idx]["timestamp"])
        current_time = timestamp.strftime("%H:%M")
        passed = self.config.session_start <= current_time <= self.config.session_end
        return "session", bool(passed), "Session filter failed"

    def _volume(self, df: pd.DataFrame, idx: int) -> Tuple[str, bool, str]:
        if not self.config.enable_volume_filter:
            return "volume", True, ""
        row = df.iloc[idx]
        baseline = row.get("volume_ma", 0.0)
        if baseline <= 0:
            return "volume", False, "Volume filter enabled but baseline unavailable"
        passed = row["volume"] / baseline >= self.config.min_volume_ratio
        return "volume", bool(passed), "Volume filter failed"
