from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .config import SpikeConfig
from .models import SpikeEvent


class SpikeDetector:
    """Explicit spike detector using only historical candles through the evaluation bar."""

    def __init__(self, config: SpikeConfig):
        self.config = config

    def detect(self, df: pd.DataFrame, idx: int) -> List[SpikeEvent]:
        events: List[SpikeEvent] = []
        for direction in ("long", "short"):
            event = self._detect_direction(df, idx, direction)
            if event is not None:
                events.append(event)
        return events

    def _detect_direction(self, df: pd.DataFrame, idx: int, direction: str) -> Optional[SpikeEvent]:
        max_len = min(self.config.max_spike_candles, idx + 1)
        for window in range(max_len, self.config.min_spike_candles - 1, -1):
            start_idx = idx - window + 1
            if start_idx < 0:
                continue
            if self._has_existing_overlap(df, start_idx, idx):
                continue
            candidate = df.iloc[start_idx : idx + 1]
            diagnostics = self._evaluate_window(df, candidate, start_idx, idx, direction)
            if diagnostics["valid"]:
                origin_idx = max(0, start_idx - 1)
                return SpikeEvent(
                    direction=direction,
                    start_idx=start_idx,
                    end_idx=idx,
                    origin_idx=origin_idx,
                    spike_high=float(candidate["high"].max()),
                    spike_low=float(candidate["low"].min()),
                    gap_passed=diagnostics["gap_passed"],
                    directional_ratio=diagnostics["directional_ratio"],
                    average_body_to_atr=diagnostics["average_body_to_atr"],
                    candle_count=window,
                    gap_count=diagnostics["gap_count"],
                    diagnostics=diagnostics,
                )
        return None

    @staticmethod
    def _has_existing_overlap(df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        return start_idx == end_idx and len(df) > 0

    def _evaluate_window(
        self,
        df: pd.DataFrame,
        candidate: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        direction: str,
    ) -> Dict[str, float | bool | int | list]:
        bullish = direction == "long"
        bodies = (candidate["close"] - candidate["open"]).abs()
        candle_dirs = (candidate["close"] > candidate["open"]) if bullish else (candidate["close"] < candidate["open"])
        directional_ratio = float(candle_dirs.mean())
        atr_values = candidate["atr"].replace(0.0, pd.NA).fillna(method="bfill").fillna(method="ffill").fillna(1e-9)
        average_body_to_atr = float((bodies / atr_values).mean())

        gaps = []
        for local_idx in range(start_idx + 1, end_idx + 1):
            prev_row = df.iloc[local_idx - 1]
            row = df.iloc[local_idx]
            gap_size = self._gap_size(prev_row, row, direction)
            gaps.append(gap_size)
        gap_count = sum(gap > 0 for gap in gaps)
        gap_passed = gap_count > 0 or not self.config.require_at_least_one_gap

        valid = (
            directional_ratio >= self.config.min_directional_ratio
            and average_body_to_atr >= self.config.min_body_to_atr
            and gap_passed
        )
        return {
            "valid": valid,
            "directional_ratio": directional_ratio,
            "average_body_to_atr": average_body_to_atr,
            "gap_passed": gap_passed,
            "gap_count": int(gap_count),
            "gap_sizes": gaps,
            "window_start": start_idx,
            "window_end": end_idx,
        }

    def _gap_size(self, prev_row: pd.Series, row: pd.Series, direction: str) -> float:
        if self.config.gap_mode == "true_gap":
            raw_gap = row["low"] - prev_row["high"] if direction == "long" else prev_row["low"] - row["high"]
        elif self.config.gap_mode == "body_gap":
            raw_gap = min(row["open"], row["close"]) - max(prev_row["open"], prev_row["close"])
            if direction == "short":
                raw_gap = min(prev_row["open"], prev_row["close"]) - max(row["open"], row["close"])
        elif self.config.gap_mode == "atr_fraction":
            baseline = prev_row.get("atr", 0.0) or 1e-9
            raw_gap = row["low"] - prev_row["high"] if direction == "long" else prev_row["low"] - row["high"]
            raw_gap = raw_gap / baseline
            return raw_gap if raw_gap >= self.config.min_gap_atr_fraction else 0.0
        else:
            raise ValueError(f"Unsupported gap_mode: {self.config.gap_mode}")

        if raw_gap <= 0:
            return 0.0
        if self.config.min_gap_ticks and raw_gap < self.config.min_gap_ticks:
            return 0.0
        if self.config.min_gap_percent:
            reference = prev_row["close"] or 1e-9
            if (raw_gap / reference) < self.config.min_gap_percent:
                return 0.0
        if self.config.min_gap_atr_fraction:
            baseline = prev_row.get("atr", 0.0) or 1e-9
            if (raw_gap / baseline) < self.config.min_gap_atr_fraction:
                return 0.0
        return float(raw_gap)
