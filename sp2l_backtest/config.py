from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class DataConfig:
    csv_path: str = "examples/sample_ohlcv.csv"
    timestamp_col: str = "timestamp"
    timezone: str = "UTC"
    session_timezone: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None


@dataclass
class SpikeConfig:
    min_spike_candles: int = 2
    max_spike_candles: int = 6
    min_body_to_atr: float = 0.9
    min_directional_ratio: float = 0.7
    require_at_least_one_gap: bool = True
    gap_mode: str = "true_gap"
    min_gap_ticks: float = 0.0
    min_gap_percent: float = 0.0
    min_gap_atr_fraction: float = 0.0
    allow_mixed_candles: bool = True


@dataclass
class CorrectionConfig:
    second_leg_mode: str = "simplified_trigger"
    max_bars_after_spike: int = 8
    minimum_correction_depth: float = 0.0
    require_rejection_confirmation: bool = False
    strict_swing_lookback: int = 1


@dataclass
class EntryConfig:
    entry_mode: str = "stop_break"
    stop_reference: str = "pre_spike_origin_candle"
    stop_buffer_ticks: float = 0.0
    stop_buffer_atr_fraction: float = 0.0
    rr_target: float = 1.0
    partial_exit_rr: list[float] = field(default_factory=list)
    trailing_after_tp1: bool = False
    order_expiry_bars: int = 3


@dataclass
class ScaleInConfig:
    enable_scale_in: bool = True
    scale_in_size: float = 1.0


@dataclass
class FilterConfig:
    enable_htf_trend: bool = False
    htf_fast_ema: int = 20
    htf_slow_ema: int = 50
    htf_resample_rule: str = "1D"
    enable_breakout_filter: bool = False
    breakout_lookback: int = 20
    enable_channel_edge_filter: bool = False
    channel_lookback: int = 20
    channel_edge_threshold: float = 0.2
    enable_min_spike_strength_filter: bool = False
    min_spike_strength_atr: float = 2.0
    enable_low_volatility_filter: bool = False
    volatility_lookback: int = 20
    min_atr_percent: float = 0.002
    enable_session_filter: bool = False
    session_start: str = "09:30"
    session_end: str = "16:00"
    enable_volume_filter: bool = False
    volume_lookback: int = 20
    min_volume_ratio: float = 1.0


@dataclass
class RiskConfig:
    sizing_mode: str = "fixed_fractional"
    risk_per_trade: float = 0.01
    fixed_quantity: float = 1.0
    initial_equity: float = 100000.0
    commission_per_unit: float = 0.0
    slippage_per_unit: float = 0.0
    tick_size: float = 0.01
    max_open_trades: int = 1
    allow_multiple_positions: bool = False


@dataclass
class ReportingConfig:
    output_dir: str = "output"
    save_candidate_setups: bool = True
    verbose_debug: bool = True
    save_plots: bool = True
    candlestick_bars: int = 120


@dataclass
class SP2LConfig:
    data: DataConfig = field(default_factory=DataConfig)
    spike: SpikeConfig = field(default_factory=SpikeConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    scale_in: ScaleInConfig = field(default_factory=ScaleInConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dataclass(cls: Type[T], overrides: Optional[Dict[str, Any]]) -> T:
    overrides = overrides or {}
    kwargs: Dict[str, Any] = {}
    for field_def in fields(cls):
        value = overrides.get(field_def.name)
        default_value = getattr(cls(), field_def.name)
        if is_dataclass(default_value):
            kwargs[field_def.name] = _merge_dataclass(type(default_value), value)
        else:
            kwargs[field_def.name] = default_value if value is None else value
    return cls(**kwargs)


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> SP2LConfig:
    payload: Dict[str, Any] = {}
    if config_path:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    if overrides:
        payload = {**payload, **overrides}
    return _merge_dataclass(SP2LConfig, payload)
