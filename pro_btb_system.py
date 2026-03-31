"""
Production-style Pro BTB (Professional Back To Breakeven) strategy engine.

Designed for:
1) Backtesting first
2) Paper trading next
3) Optional live-trading integration (disabled by default)

Dependencies:
- pandas
- numpy
- matplotlib

The implementation intentionally uses bar-close decisions only and avoids look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Literal
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

EntryMethod = Literal[
    "market_entry",
    "limit_entry",
    "stop_entry",
    "confirmation_candle_entry",
    "pattern_entry",
    "zone_entry",
]

GapType = Literal["all", "significant", "structural", "major"]
DisplayMode = Literal["setup", "signal"]


@dataclass
class StrategyConfig:
    # Instrument / data settings
    symbol: str = "XAUUSD"
    timeframe: str = "M15"

    # A) Spike Filter | Movement
    minimum_spike_bars: int = 3
    use_movement_power: bool = True
    movement_power_level: float = 1.8

    # B) Spike Filter | Gap
    use_gap_filter: bool = True
    gap_type: GapType = "significant"

    # C) Spike Filter | Doji
    allow_doji_in_spike: bool = True
    max_doji_body_ratio: float = 0.2
    max_doji_in_spike_ratio: float = 0.34

    # D) Position Management
    use_stop_loss_threshold: bool = True
    stop_loss_threshold_value: float = 8.0  # max stop distance in price units
    risk_reward_ratio: float = 2.0
    include_sl_threshold_in_rr: bool = True
    risk_per_trade: float = 0.01
    allow_multiple_positions: bool = False

    # E) Display Settings
    display_mode: DisplayMode = "signal"
    show_entry_levels: bool = True
    only_display_last_position: bool = False
    setup_width_drawing: float = 40.0

    # F) Alerts / Signal Settings
    enable_alerts: bool = True
    alert_on_setup: bool = True
    alert_on_entry: bool = True

    # G) Structure / Level Detection
    swing_lookback: int = 4
    breakout_buffer: float = 0.20
    retest_tolerance: float = 0.35
    zone_width_factor: float = 0.5  # multiplied by ATR
    confirmation_bars: int = 2
    use_htf_trend_filter: bool = False
    htf_timeframe: str = "H1"
    trend_ma_period: int = 50

    # Entry and execution
    entry_method: EntryMethod = "confirmation_candle_entry"
    compare_entry_methods: Optional[List[EntryMethod]] = None

    # Execution assumptions
    commission_perc: float = 0.0002  # 2 bps
    spread: float = 0.05
    slippage: float = 0.02

    # Break-even management
    move_to_breakeven_enabled: bool = False
    move_to_breakeven_at_r: float = 1.0

    # Backtest account
    initial_equity: float = 10_000.0


# --------------------------------------------------------------------------------------
# Trade/setup domain models
# --------------------------------------------------------------------------------------


@dataclass
class BreakoutSetup:
    direction: Literal["long", "short"]
    setup_timestamp: pd.Timestamp
    breakout_index: int
    breakout_level: float
    zone_low: float
    zone_high: float
    breakout_candle_low: float
    breakout_candle_high: float
    spike_start_index: int
    spike_end_index: int
    movement_power: float
    first_retest_index: Optional[int] = None
    invalidated: bool = False
    notes: str = ""


@dataclass
class TradeSignal:
    direction: Literal["long", "short"]
    setup_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount_per_unit: float
    reward_target_per_unit: float
    entry_method_used: str
    notes: str


@dataclass
class Position:
    direction: Literal["long", "short"]
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    risk_per_unit: float
    setup_timestamp: pd.Timestamp
    entry_method: str
    notes: str
    moved_to_be: bool = False


@dataclass
class ClosedTrade:
    direction: str
    setup_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    gross_pnl: float
    net_pnl: float
    r_multiple: float
    exit_reason: str
    entry_method: str
    notes: str


# --------------------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------------------


class DataHandler:
    REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return DataHandler.validate_and_prepare(df)

    @staticmethod
    def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in DataHandler.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Force numeric fields and drop invalid rows
        for col in ["open", "high", "low", "close", "volume"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna(subset=["open", "high", "low", "close"])  # volume can be zero
        out = out[out["high"] >= out["low"]]
        out = out.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        return out

    @staticmethod
    def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV to a pandas-compatible timeframe string, e.g. '15T', '1H'."""
        temp = df.copy().set_index("timestamp")
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        out = temp.resample(timeframe).agg(agg).dropna(subset=["open", "high", "low", "close"])
        return out.reset_index()


# --------------------------------------------------------------------------------------
# Indicators and helpers
# --------------------------------------------------------------------------------------


class Indicators:
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close_prev = (df["high"] - df["close"].shift(1)).abs()
        low_close_prev = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=period).mean()

    @staticmethod
    def candle_body_ratio(df: pd.DataFrame) -> pd.Series:
        body = (df["close"] - df["open"]).abs()
        full = (df["high"] - df["low"]).replace(0, np.nan)
        return (body / full).fillna(0.0)

    @staticmethod
    def is_doji(df: pd.DataFrame, max_ratio: float) -> pd.Series:
        return Indicators.candle_body_ratio(df) <= max_ratio

    @staticmethod
    def swing_points(df: pd.DataFrame, lookback: int) -> Tuple[pd.Series, pd.Series]:
        """
        Confirmed swing points using only historical info at the time of confirmation.
        A swing at i is confirmed at i+lookback; this avoids repainting in live logic.
        """
        highs = pd.Series(False, index=df.index)
        lows = pd.Series(False, index=df.index)
        for i in range(lookback, len(df) - lookback):
            h = df.loc[i, "high"]
            l = df.loc[i, "low"]
            if h == df.loc[i - lookback : i + lookback, "high"].max():
                highs.iloc[i] = True
            if l == df.loc[i - lookback : i + lookback, "low"].min():
                lows.iloc[i] = True
        return highs, lows

    @staticmethod
    def fvg_imbalance_score(df: pd.DataFrame, idx: int, direction: str) -> float:
        """
        Approximate fair-value-gap/imbalance score.
        For bullish: if candle i low > candle i-2 high, positive gap.
        For bearish: if candle i high < candle i-2 low, positive gap.
        """
        if idx < 2:
            return 0.0
        if direction == "long":
            return max(0.0, df.loc[idx, "low"] - df.loc[idx - 2, "high"])
        return max(0.0, df.loc[idx - 2, "low"] - df.loc[idx, "high"])

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()


# --------------------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------------------


class AlertManager:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def notify(self, message: str) -> None:
        if self.enabled:
            logging.info(message)
            print(message)

    def webhook_placeholder(self, payload: Dict) -> None:
        # Replace with real webhook logic if needed.
        if self.enabled:
            logging.info("Webhook placeholder payload=%s", payload)


# --------------------------------------------------------------------------------------
# Pro BTB strategy engine
# --------------------------------------------------------------------------------------


class ProBTBStrategy:
    """
    Generates setup and entry signals using bar-by-bar logic.

    IMPORTANT assumptions where Pro BTB is qualitative:
    - Key levels from confirmed swing highs/lows.
    - Gap types are quantitative approximations:
      * all: no minimum gap, any FVG-like imbalance accepted.
      * significant: imbalance >= 0.15 * ATR.
      * structural: imbalance >= 0.30 * ATR and breakout candle body ratio >= 0.55.
      * major: imbalance >= 0.50 * ATR and breakout candle range >= 1.2 * ATR.
    - Spike movement power = abs(spike_close - spike_open) / avg_ATR_over_spike.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.alerts = AlertManager(enabled=config.enable_alerts)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["atr"] = Indicators.atr(out, 14)
        out["body_ratio"] = Indicators.candle_body_ratio(out)
        out["is_doji"] = Indicators.is_doji(out, self.config.max_doji_body_ratio)
        out["swing_high"], out["swing_low"] = Indicators.swing_points(out, self.config.swing_lookback)

        if self.config.use_htf_trend_filter:
            out = self._add_htf_trend(out)
        else:
            out["htf_trend_long"] = True
            out["htf_trend_short"] = True
        return out

    def _add_htf_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        rule_map = {"M1": "1T", "M5": "5T", "M15": "15T", "M30": "30T", "H1": "1H", "H4": "4H", "D1": "1D"}
        rule = rule_map.get(self.config.htf_timeframe, "1H")
        htf = DataHandler.resample(df, rule).set_index("timestamp")
        htf_ma = Indicators.ema(htf["close"], self.config.trend_ma_period)
        htf["trend_long"] = htf["close"] > htf_ma
        htf["trend_short"] = htf["close"] < htf_ma

        out = df.copy().set_index("timestamp")
        out = out.join(htf[["trend_long", "trend_short"]], how="left")
        out[["trend_long", "trend_short"]] = out[["trend_long", "trend_short"]].ffill().fillna(False)
        out = out.reset_index().rename(columns={"trend_long": "htf_trend_long", "trend_short": "htf_trend_short"})
        return out

    def generate_signals(self, df: pd.DataFrame) -> Tuple[List[BreakoutSetup], List[TradeSignal]]:
        data = self.prepare_features(df)
        setups: List[BreakoutSetup] = []
        signals: List[TradeSignal] = []
        active_setups: List[BreakoutSetup] = []

        resistance_levels: List[float] = []
        support_levels: List[float] = []

        for i in range(len(data)):
            if i < max(20, self.config.swing_lookback * 2):
                continue

            row = data.iloc[i]

            # Update key levels only from confirmed prior swings (no look-ahead).
            prev_idx = i - self.config.swing_lookback
            if prev_idx >= 0:
                if bool(data.iloc[prev_idx]["swing_high"]):
                    resistance_levels.append(float(data.iloc[prev_idx]["high"]))
                if bool(data.iloc[prev_idx]["swing_low"]):
                    support_levels.append(float(data.iloc[prev_idx]["low"]))

            resistance_levels = self._dedupe_levels(resistance_levels, tol=self.config.retest_tolerance)
            support_levels = self._dedupe_levels(support_levels, tol=self.config.retest_tolerance)

            # New breakout setup detection
            bullish_setup = self._detect_breakout_setup(data, i, "long", resistance_levels)
            bearish_setup = self._detect_breakout_setup(data, i, "short", support_levels)

            for setup in [bullish_setup, bearish_setup]:
                if setup is not None:
                    setups.append(setup)
                    active_setups.append(setup)
                    if self.config.alert_on_setup:
                        self.alerts.notify(
                            f"[SETUP] {setup.direction.upper()} {setup.setup_timestamp} "
                            f"level={setup.breakout_level:.3f} movement_power={setup.movement_power:.2f}"
                        )

            # Retest + confirmation on existing active setups
            still_active: List[BreakoutSetup] = []
            for setup in active_setups:
                updated_setup, maybe_signal = self._process_active_setup(data, i, setup)
                if maybe_signal is not None:
                    signals.append(maybe_signal)
                    if self.config.alert_on_entry:
                        self.alerts.notify(
                            f"[ENTRY] {maybe_signal.direction.upper()} {maybe_signal.entry_timestamp} "
                            f"entry={maybe_signal.entry_price:.3f} sl={maybe_signal.stop_loss:.3f} "
                            f"tp={maybe_signal.take_profit:.3f} method={maybe_signal.entry_method_used}"
                        )
                    continue

                if not updated_setup.invalidated:
                    still_active.append(updated_setup)
            active_setups = still_active

        return setups, signals

    def _dedupe_levels(self, levels: List[float], tol: float) -> List[float]:
        if not levels:
            return levels
        levels_sorted = sorted(levels)
        merged = [levels_sorted[0]]
        for lv in levels_sorted[1:]:
            if abs(lv - merged[-1]) <= tol:
                merged[-1] = (merged[-1] + lv) / 2.0
            else:
                merged.append(lv)
        return merged[-50:]  # keep memory bounded

    def _detect_breakout_setup(
        self, df: pd.DataFrame, idx: int, direction: Literal["long", "short"], key_levels: List[float]
    ) -> Optional[BreakoutSetup]:
        if not key_levels:
            return None

        row = df.iloc[idx]
        atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
        if atr <= 0:
            return None

        close = float(row["close"])
        low = float(row["low"])
        high = float(row["high"])

        if direction == "long":
            candidates = [lv for lv in key_levels if close > lv + self.config.breakout_buffer]
            if not candidates:
                return None
            breakout_level = max(candidates)
        else:
            candidates = [lv for lv in key_levels if close < lv - self.config.breakout_buffer]
            if not candidates:
                return None
            breakout_level = min(candidates)

        spike_ok, start_idx, movement_power = self._validate_spike(df, idx, direction)
        if not spike_ok:
            return None

        gap_ok = self._validate_gap_filter(df, idx, direction, atr)
        if self.config.use_gap_filter and not gap_ok:
            return None

        zone_width = self.config.zone_width_factor * atr
        zone_low = breakout_level - zone_width
        zone_high = breakout_level + zone_width

        return BreakoutSetup(
            direction=direction,
            setup_timestamp=row["timestamp"],
            breakout_index=idx,
            breakout_level=breakout_level,
            zone_low=zone_low,
            zone_high=zone_high,
            breakout_candle_low=low,
            breakout_candle_high=high,
            spike_start_index=start_idx,
            spike_end_index=idx,
            movement_power=movement_power,
            notes=f"gap_ok={gap_ok}",
        )

    def _validate_spike(self, df: pd.DataFrame, idx: int, direction: str) -> Tuple[bool, int, float]:
        min_bars = self.config.minimum_spike_bars
        start = idx - min_bars + 1
        if start < 0:
            return False, start, 0.0

        window = df.iloc[start : idx + 1]
        if direction == "long":
            directional = (window["close"] > window["open"]).sum() >= int(np.ceil(min_bars * 0.7))
            displacement = float(window.iloc[-1]["close"] - window.iloc[0]["open"])
        else:
            directional = (window["close"] < window["open"]).sum() >= int(np.ceil(min_bars * 0.7))
            displacement = float(window.iloc[0]["open"] - window.iloc[-1]["close"])

        if displacement <= 0 or not directional:
            return False, start, 0.0

        if not self.config.allow_doji_in_spike:
            if bool(window["is_doji"].any()):
                return False, start, 0.0
        else:
            doji_ratio = float(window["is_doji"].mean())
            if doji_ratio > self.config.max_doji_in_spike_ratio:
                return False, start, 0.0

        atr_avg = float(window["atr"].mean()) if pd.notna(window["atr"].mean()) else 0.0
        if atr_avg <= 0:
            return False, start, 0.0

        movement_power = displacement / atr_avg
        if self.config.use_movement_power and movement_power < self.config.movement_power_level:
            return False, start, movement_power

        return True, start, movement_power

    def _validate_gap_filter(self, df: pd.DataFrame, idx: int, direction: str, atr: float) -> bool:
        score = Indicators.fvg_imbalance_score(df, idx, direction)
        body_ratio = float(df.iloc[idx]["body_ratio"])
        rng = float(df.iloc[idx]["high"] - df.iloc[idx]["low"])

        gt = self.config.gap_type
        if gt == "all":
            return True
        if gt == "significant":
            return score >= 0.15 * atr
        if gt == "structural":
            return score >= 0.30 * atr and body_ratio >= 0.55
        if gt == "major":
            return score >= 0.50 * atr and rng >= 1.2 * atr
        return True

    def _process_active_setup(
        self, df: pd.DataFrame, idx: int, setup: BreakoutSetup
    ) -> Tuple[BreakoutSetup, Optional[TradeSignal]]:
        row = df.iloc[idx]
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        # Invalidation if continuation runs too far before retest (quantitative assumption)
        atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
        if setup.first_retest_index is None and atr > 0:
            if setup.direction == "long" and close > setup.breakout_level + 3.0 * atr:
                setup.invalidated = True
                setup.notes += "|invalid_no_retest"
                return setup, None
            if setup.direction == "short" and close < setup.breakout_level - 3.0 * atr:
                setup.invalidated = True
                setup.notes += "|invalid_no_retest"
                return setup, None

        # Detect first retest
        in_zone = low <= setup.zone_high and high >= setup.zone_low
        if setup.first_retest_index is None and idx > setup.breakout_index and in_zone:
            setup.first_retest_index = idx

        if setup.first_retest_index is None:
            return setup, None

        # Confirmations and entry method logic
        signal = self._entry_from_method(df, idx, setup)
        return setup, signal

    def _entry_from_method(self, df: pd.DataFrame, idx: int, setup: BreakoutSetup) -> Optional[TradeSignal]:
        row = df.iloc[idx]

        # Optional HTF trend filter
        if setup.direction == "long" and not bool(row["htf_trend_long"]):
            return None
        if setup.direction == "short" and not bool(row["htf_trend_short"]):
            return None

        method = self.config.entry_method
        level = setup.breakout_level
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        should_enter = False
        entry_price = close
        note = ""

        if method == "market_entry":
            if low <= setup.zone_high and high >= setup.zone_low:
                should_enter = True
                entry_price = close
                note = "market_on_retest"

        elif method == "limit_entry":
            if low <= level <= high:
                should_enter = True
                entry_price = level
                note = "limit_filled_at_level"

        elif method == "stop_entry":
            # continuation trigger beyond confirmation high/low after retest
            trig = self._continuation_trigger(df, setup.first_retest_index, idx, setup.direction)
            if trig is not None:
                if setup.direction == "long" and close > trig:
                    should_enter = True
                    entry_price = close
                    note = f"stop_continuation_above_{trig:.3f}"
                elif setup.direction == "short" and close < trig:
                    should_enter = True
                    entry_price = close
                    note = f"stop_continuation_below_{trig:.3f}"

        elif method == "confirmation_candle_entry":
            if self._candle_confirmation(df, idx, setup):
                should_enter = True
                entry_price = close
                note = "confirmation_candle"

        elif method == "pattern_entry":
            if self._pattern_confirmation(df, idx, setup):
                should_enter = True
                entry_price = close
                note = "candlestick_pattern"

        elif method == "zone_entry":
            # enter at center of zone once price closes back in breakout direction
            zone_mid = (setup.zone_low + setup.zone_high) / 2.0
            if setup.direction == "long" and close > level and low <= setup.zone_high:
                should_enter = True
                entry_price = zone_mid
                note = "zone_reclaim_long"
            elif setup.direction == "short" and close < level and high >= setup.zone_low:
                should_enter = True
                entry_price = zone_mid
                note = "zone_reclaim_short"

        if not should_enter:
            return None

        return self._build_signal(setup, row["timestamp"], entry_price, note)

    def _candle_confirmation(self, df: pd.DataFrame, idx: int, setup: BreakoutSetup) -> bool:
        n = self.config.confirmation_bars
        if idx - n + 1 < 0:
            return False
        window = df.iloc[idx - n + 1 : idx + 1]
        if setup.direction == "long":
            return bool((window["close"] > setup.breakout_level).all() and window.iloc[-1]["close"] > window.iloc[-1]["open"])
        return bool((window["close"] < setup.breakout_level).all() and window.iloc[-1]["close"] < window.iloc[-1]["open"])

    def _continuation_trigger(
        self, df: pd.DataFrame, retest_idx: int, current_idx: int, direction: str
    ) -> Optional[float]:
        if retest_idx is None or current_idx <= retest_idx:
            return None
        start = max(retest_idx, current_idx - self.config.confirmation_bars)
        window = df.iloc[start : current_idx + 1]
        if len(window) < 2:
            return None
        return float(window["high"].max()) if direction == "long" else float(window["low"].min())

    def _pattern_confirmation(self, df: pd.DataFrame, idx: int, setup: BreakoutSetup) -> bool:
        if idx < 1:
            return False
        c0 = df.iloc[idx - 1]
        c1 = df.iloc[idx]

        # Pin bar approximation
        body = abs(float(c1["close"] - c1["open"]))
        rng = float(c1["high"] - c1["low"])
        if rng <= 0:
            return False
        upper_wick = float(c1["high"] - max(c1["open"], c1["close"]))
        lower_wick = float(min(c1["open"], c1["close"]) - c1["low"])

        bullish_pin = lower_wick >= 2 * body and upper_wick <= body
        bearish_pin = upper_wick >= 2 * body and lower_wick <= body

        # Engulfing
        bullish_engulf = (
            c0["close"] < c0["open"]
            and c1["close"] > c1["open"]
            and c1["open"] <= c0["close"]
            and c1["close"] >= c0["open"]
        )
        bearish_engulf = (
            c0["close"] > c0["open"]
            and c1["close"] < c1["open"]
            and c1["open"] >= c0["close"]
            and c1["close"] <= c0["open"]
        )

        near_level = c1["low"] <= setup.zone_high and c1["high"] >= setup.zone_low

        if setup.direction == "long":
            return bool(near_level and (bullish_pin or bullish_engulf))
        return bool(near_level and (bearish_pin or bearish_engulf))

    def _build_signal(self, setup: BreakoutSetup, entry_ts: pd.Timestamp, entry_price: float, note: str) -> Optional[TradeSignal]:
        if setup.direction == "long":
            sl_candidates = [setup.breakout_candle_low, setup.zone_low]
            stop = min(sl_candidates)
            if self.config.use_stop_loss_threshold:
                stop = max(stop, entry_price - self.config.stop_loss_threshold_value)
            risk = entry_price - stop
            if risk <= 0:
                return None
            reward = self.config.risk_reward_ratio * risk
            tp = entry_price + reward
        else:
            sl_candidates = [setup.breakout_candle_high, setup.zone_high]
            stop = max(sl_candidates)
            if self.config.use_stop_loss_threshold:
                stop = min(stop, entry_price + self.config.stop_loss_threshold_value)
            risk = stop - entry_price
            if risk <= 0:
                return None
            reward = self.config.risk_reward_ratio * risk
            tp = entry_price - reward

        return TradeSignal(
            direction=setup.direction,
            setup_timestamp=setup.setup_timestamp,
            entry_timestamp=entry_ts,
            entry_price=float(entry_price),
            stop_loss=float(stop),
            take_profit=float(tp),
            risk_amount_per_unit=float(risk),
            reward_target_per_unit=float(reward),
            entry_method_used=self.config.entry_method,
            notes=note,
        )


# --------------------------------------------------------------------------------------
# Backtester
# --------------------------------------------------------------------------------------


class Backtester:
    def __init__(self, config: StrategyConfig):
        self.config = config

    def run(self, df: pd.DataFrame, signals: List[TradeSignal]) -> Dict[str, object]:
        data = df.copy().reset_index(drop=True)
        signal_map: Dict[pd.Timestamp, List[TradeSignal]] = {}
        for s in signals:
            signal_map.setdefault(s.entry_timestamp, []).append(s)

        open_positions: List[Position] = []
        closed_trades: List[ClosedTrade] = []

        equity = self.config.initial_equity
        equity_curve = []

        for i in range(len(data)):
            row = data.iloc[i]
            ts = row["timestamp"]
            high = float(row["high"])
            low = float(row["low"])

            # 1) Check exits first (intra-bar)
            survivors: List[Position] = []
            for pos in open_positions:
                exit_trade = self._maybe_exit_position(pos, row)
                if exit_trade is not None:
                    equity += exit_trade.net_pnl
                    closed_trades.append(exit_trade)
                else:
                    updated = self._maybe_move_to_breakeven(pos, high, low)
                    survivors.append(updated)
            open_positions = survivors

            # 2) New entries on current bar timestamp
            todays_signals = signal_map.get(ts, [])
            for sig in todays_signals:
                if (not self.config.allow_multiple_positions) and open_positions:
                    continue
                pos = self._signal_to_position(sig, equity)
                if pos is not None:
                    open_positions.append(pos)

            equity_curve.append({"timestamp": ts, "equity": equity})

        stats = self._compute_stats(closed_trades, pd.DataFrame(equity_curve))
        return {
            "closed_trades": closed_trades,
            "equity_curve": pd.DataFrame(equity_curve),
            "stats": stats,
        }

    def _signal_to_position(self, sig: TradeSignal, equity: float) -> Optional[Position]:
        risk_capital = equity * self.config.risk_per_trade
        if sig.risk_amount_per_unit <= 0:
            return None
        qty = risk_capital / sig.risk_amount_per_unit
        if qty <= 0:
            return None

        # Apply spread/slippage to entry
        if sig.direction == "long":
            entry = sig.entry_price + self.config.spread / 2 + self.config.slippage
        else:
            entry = sig.entry_price - self.config.spread / 2 - self.config.slippage

        return Position(
            direction=sig.direction,
            entry_time=sig.entry_timestamp,
            entry_price=entry,
            stop_loss=sig.stop_loss,
            take_profit=sig.take_profit,
            quantity=qty,
            risk_per_unit=sig.risk_amount_per_unit,
            setup_timestamp=sig.setup_timestamp,
            entry_method=sig.entry_method_used,
            notes=sig.notes,
        )

    def _maybe_move_to_breakeven(self, pos: Position, high: float, low: float) -> Position:
        if not self.config.move_to_breakeven_enabled or pos.moved_to_be:
            return pos

        r = pos.risk_per_unit
        if pos.direction == "long":
            if high >= pos.entry_price + self.config.move_to_breakeven_at_r * r:
                pos.stop_loss = pos.entry_price
                pos.moved_to_be = True
        else:
            if low <= pos.entry_price - self.config.move_to_breakeven_at_r * r:
                pos.stop_loss = pos.entry_price
                pos.moved_to_be = True
        return pos

    def _maybe_exit_position(self, pos: Position, row: pd.Series) -> Optional[ClosedTrade]:
        high = float(row["high"])
        low = float(row["low"])
        ts = row["timestamp"]

        # Conservative assumption: if both SL and TP touch in one bar, SL gets hit first.
        hit_sl = low <= pos.stop_loss <= high or (pos.direction == "long" and low <= pos.stop_loss) or (
            pos.direction == "short" and high >= pos.stop_loss
        )
        hit_tp = low <= pos.take_profit <= high or (pos.direction == "long" and high >= pos.take_profit) or (
            pos.direction == "short" and low <= pos.take_profit
        )

        if not hit_sl and not hit_tp:
            return None

        if hit_sl:
            exit_price = pos.stop_loss
            reason = "stop_loss"
        else:
            exit_price = pos.take_profit
            reason = "take_profit"

        if pos.direction == "long":
            gross = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross = (pos.entry_price - exit_price) * pos.quantity

        # Commission for entry + exit
        commission = self.config.commission_perc * (pos.entry_price * pos.quantity + exit_price * pos.quantity)
        net = gross - commission
        r_mult = (exit_price - pos.entry_price) / pos.risk_per_unit if pos.direction == "long" else (pos.entry_price - exit_price) / pos.risk_per_unit

        return ClosedTrade(
            direction=pos.direction,
            setup_timestamp=pos.setup_timestamp,
            entry_timestamp=pos.entry_time,
            exit_timestamp=ts,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            quantity=pos.quantity,
            gross_pnl=gross,
            net_pnl=net,
            r_multiple=r_mult,
            exit_reason=reason,
            entry_method=pos.entry_method,
            notes=pos.notes,
        )

    def _compute_stats(self, closed: List[ClosedTrade], equity_curve: pd.DataFrame) -> Dict[str, float]:
        if len(closed) == 0:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "expectancy": 0.0,
                "total_net_pnl": 0.0,
                "max_drawdown": 0.0,
                "sharpe_like": 0.0,
            }

        pnl = np.array([t.net_pnl for t in closed], dtype=float)
        wins = (pnl > 0).sum()
        win_rate = wins / len(pnl)
        expectancy = pnl.mean()
        total_pnl = pnl.sum()

        ec = equity_curve["equity"].values
        running_max = np.maximum.accumulate(ec)
        dd = (ec - running_max) / np.where(running_max == 0, 1, running_max)
        max_dd = float(dd.min())

        rets = pd.Series(ec).pct_change().dropna()
        sharpe_like = float((rets.mean() / (rets.std() + 1e-12)) * np.sqrt(252)) if len(rets) > 1 else 0.0

        return {
            "trades": int(len(pnl)),
            "win_rate": float(win_rate),
            "expectancy": float(expectancy),
            "total_net_pnl": float(total_pnl),
            "max_drawdown": float(max_dd),
            "sharpe_like": float(sharpe_like),
        }

    @staticmethod
    def trades_to_csv(closed_trades: List[ClosedTrade], path: str) -> None:
        df = pd.DataFrame([asdict(t) for t in closed_trades])
        df.to_csv(path, index=False)


# --------------------------------------------------------------------------------------
# Broker adapter abstraction (paper first)
# --------------------------------------------------------------------------------------


class BrokerAdapter:
    def place_order(self, signal: TradeSignal) -> Dict:
        raise NotImplementedError


class PaperBrokerAdapter(BrokerAdapter):
    def __init__(self):
        self.orders: List[Dict] = []

    def place_order(self, signal: TradeSignal) -> Dict:
        order = {
            "status": "accepted",
            "mode": "paper",
            "symbol": "N/A",
            "direction": signal.direction,
            "entry": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "timestamp": signal.entry_timestamp,
        }
        self.orders.append(order)
        return order


class LiveBrokerAdapter(BrokerAdapter):
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def place_order(self, signal: TradeSignal) -> Dict:
        if not self.enabled:
            raise RuntimeError("Live trading is disabled by default. Set enabled=True intentionally.")
        # Placeholder for MT5/ccxt implementation.
        raise NotImplementedError("Implement broker SDK integration here.")


# --------------------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------------------


class StrategyPlotter:
    @staticmethod
    def plot(
        df: pd.DataFrame,
        setups: List[BreakoutSetup],
        closed_trades: List[ClosedTrade],
        title: str = "Pro BTB Backtest",
        max_bars: int = 400,
    ) -> None:
        data = df.tail(max_bars).copy().reset_index(drop=True)
        ts_to_x = {t: i for i, t in enumerate(data["timestamp"])}

        fig, ax = plt.subplots(figsize=(16, 8))

        # Candles (manual matplotlib candlesticks)
        for i, r in data.iterrows():
            color = "green" if r["close"] >= r["open"] else "red"
            ax.plot([i, i], [r["low"], r["high"]], color=color, linewidth=1)
            body_low = min(r["open"], r["close"])
            body_h = max(0.001, abs(r["close"] - r["open"]))
            rect = Rectangle((i - 0.3, body_low), 0.6, body_h, facecolor=color, edgecolor=color, alpha=0.75)
            ax.add_patch(rect)

        # Setups
        for s in setups:
            if s.setup_timestamp not in ts_to_x:
                continue
            x = ts_to_x[s.setup_timestamp]
            clr = "blue" if s.direction == "long" else "purple"
            ax.axhline(s.breakout_level, linestyle="--", color=clr, alpha=0.20)
            ax.fill_between([x - 8, x + 8], s.zone_low, s.zone_high, color=clr, alpha=0.08)

        # Closed trades markers
        for t in closed_trades:
            if t.entry_timestamp in ts_to_x:
                x_e = ts_to_x[t.entry_timestamp]
                m = "^" if t.direction == "long" else "v"
                c = "lime" if t.net_pnl >= 0 else "black"
                ax.scatter([x_e], [t.entry_price], marker=m, s=70, color=c)
            if t.exit_timestamp in ts_to_x:
                x_x = ts_to_x[t.exit_timestamp]
                ax.scatter([x_x], [t.exit_price], marker="x", s=55, color="orange")

        ax.set_title(title)
        ax.set_xlabel("Bars")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()


# --------------------------------------------------------------------------------------
# Runner examples
# --------------------------------------------------------------------------------------


def sample_config() -> StrategyConfig:
    return StrategyConfig(
        symbol="XAUUSD",
        timeframe="M15",
        entry_method="confirmation_candle_entry",
        compare_entry_methods=[
            "market_entry",
            "limit_entry",
            "stop_entry",
            "confirmation_candle_entry",
            "pattern_entry",
            "zone_entry",
        ],
        use_htf_trend_filter=True,
        htf_timeframe="H1",
        trend_ma_period=50,
    )


def run_backtest_from_csv(csv_path: str, cfg: Optional[StrategyConfig] = None) -> Dict[str, object]:
    cfg = cfg or sample_config()
    df = DataHandler.load_csv(csv_path)

    # Example: uncomment if you want to resample source feed.
    # df = DataHandler.resample(df, "15T")

    strategy = ProBTBStrategy(cfg)
    setups, signals = strategy.generate_signals(df)

    bt = Backtester(cfg)
    result = bt.run(df, signals)

    # Export trade log
    Backtester.trades_to_csv(result["closed_trades"], "btb_trades.csv")

    print("Backtest stats:", result["stats"])
    StrategyPlotter.plot(df, setups, result["closed_trades"], title=f"{cfg.symbol} Pro BTB")
    return result


def compare_entry_methods(csv_path: str, cfg: Optional[StrategyConfig] = None) -> pd.DataFrame:
    cfg = cfg or sample_config()
    methods = cfg.compare_entry_methods or [cfg.entry_method]
    summary = []
    base_df = DataHandler.load_csv(csv_path)

    for method in methods:
        local_cfg = StrategyConfig(**{**asdict(cfg), "entry_method": method})
        strat = ProBTBStrategy(local_cfg)
        _, signals = strat.generate_signals(base_df)
        bt = Backtester(local_cfg)
        out = bt.run(base_df, signals)
        s = out["stats"]
        s["entry_method"] = method
        summary.append(s)

    return pd.DataFrame(summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Minimal runnable example:
    # 1) Put OHLCV CSV at ./xauusd_m15.csv
    # 2) Ensure columns: timestamp, open, high, low, close, volume
    # 3) Run: python pro_btb_system.py
    #
    # result = run_backtest_from_csv("xauusd_m15.csv")
    # print(result["stats"])

    print("Pro BTB module loaded. Use run_backtest_from_csv('xauusd_m15.csv').")
