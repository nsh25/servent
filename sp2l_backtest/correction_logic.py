from __future__ import annotations

from typing import Optional

import pandas as pd

from .config import CorrectionConfig
from .models import CorrectionSignal, SpikeEvent


class CorrectionDetector:
    """Detect the post-spike correction and second-leg signal without forward peeking."""

    def __init__(self, config: CorrectionConfig):
        self.config = config

    def find_signal(self, df: pd.DataFrame, spike: SpikeEvent, idx: int) -> Optional[CorrectionSignal]:
        if idx <= spike.end_idx:
            return None
        if idx - spike.end_idx > self.config.max_bars_after_spike:
            return None
        if self.config.second_leg_mode == "simplified_trigger":
            return self._simplified_trigger(df, spike, idx)
        if self.config.second_leg_mode == "strict_two_leg":
            return self._strict_two_leg(df, spike, idx)
        raise ValueError(f"Unsupported second_leg_mode: {self.config.second_leg_mode}")

    def _simplified_trigger(self, df: pd.DataFrame, spike: SpikeEvent, idx: int) -> Optional[CorrectionSignal]:
        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]
        if spike.direction == "long":
            condition = row["low"] <= prev_row["low"]
            correction_depth = spike.spike_high - row["low"]
            rejection = row["close"] > row["open"]
        else:
            condition = row["high"] >= prev_row["high"]
            correction_depth = row["high"] - spike.spike_low
            rejection = row["close"] < row["open"]
        if correction_depth < self.config.minimum_correction_depth:
            return None
        if self.config.max_correction_depth is not None and correction_depth > self.config.max_correction_depth:
            return None
        if self.config.require_rejection_confirmation and not rejection:
            return None
        if not condition:
            return None
        return CorrectionSignal(
            spike=spike,
            signal_idx=idx,
            correction_start_idx=spike.end_idx + 1,
            correction_bars=idx - spike.end_idx,
            correction_depth=float(correction_depth),
            second_leg_valid=True,
            rejection_confirmed=rejection,
            diagnostics={"mode": "simplified_trigger", "previous_idx": idx - 1},
        )

    def _strict_two_leg(self, df: pd.DataFrame, spike: SpikeEvent, idx: int) -> Optional[CorrectionSignal]:
        correction = df.iloc[spike.end_idx + 1 : idx + 1]
        if len(correction) < 3:
            return None

        pushes = 0
        if spike.direction == "long":
            reference = correction.iloc[0]["low"]
            for local_idx in range(1, len(correction)):
                if correction.iloc[local_idx]["low"] < reference:
                    pushes += 1
                    reference = correction.iloc[local_idx]["low"]
            correction_depth = spike.spike_high - correction["low"].min()
            rejection = correction.iloc[-1]["close"] > correction.iloc[-1]["open"]
        else:
            reference = correction.iloc[0]["high"]
            for local_idx in range(1, len(correction)):
                if correction.iloc[local_idx]["high"] > reference:
                    pushes += 1
                    reference = correction.iloc[local_idx]["high"]
            correction_depth = correction["high"].max() - spike.spike_low
            rejection = correction.iloc[-1]["close"] < correction.iloc[-1]["open"]

        if pushes < 2 or correction_depth < self.config.minimum_correction_depth:
            return None
        if self.config.max_correction_depth is not None and correction_depth > self.config.max_correction_depth:
            return None
        if self.config.require_rejection_confirmation and not rejection:
            return None
        return CorrectionSignal(
            spike=spike,
            signal_idx=idx,
            correction_start_idx=spike.end_idx + 1,
            correction_bars=idx - spike.end_idx,
            correction_depth=float(correction_depth),
            second_leg_valid=True,
            rejection_confirmed=rejection,
            diagnostics={"mode": "strict_two_leg", "push_count": pushes},
        )
