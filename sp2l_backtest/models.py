from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SpikeEvent:
    """Detected spike segment ending at a known historical bar."""

    direction: str
    start_idx: int
    end_idx: int
    origin_idx: int
    spike_high: float
    spike_low: float
    gap_passed: bool
    directional_ratio: float
    average_body_to_atr: float
    candle_count: int
    gap_count: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectionSignal:
    """Correction / second-leg signal that can seed an entry."""

    spike: SpikeEvent
    signal_idx: int
    correction_start_idx: int
    correction_bars: int
    correction_depth: float
    second_leg_valid: bool
    rejection_confirmed: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SetupCandidate:
    """Full setup candidate, whether traded or rejected."""

    direction: str
    spike: SpikeEvent
    correction: Optional[CorrectionSignal]
    status: str
    rejection_reason: Optional[str]
    filters_passed: Dict[str, bool]
    entry_confirmed: bool
    entry_idx: Optional[int]
    entry_price: Optional[float]
    stop_price: Optional[float]
    take_profit: Optional[float]
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScaleInFill:
    idx: int
    time: Any
    price: float
    quantity: float


@dataclass
class TradeRecord:
    """Completed trade for reporting and persistence."""

    entry_idx: int
    entry_time: Any
    exit_idx: int
    exit_time: Any
    direction: str
    spike_start_idx: int
    spike_end_idx: int
    signal_idx: int
    entry_price: float
    scale_in_price: Optional[float]
    average_entry: float
    stop_price: float
    take_profit: float
    quantity: float
    scale_in_quantity: float
    exit_price: float
    pnl: float
    pnl_pct: float
    r_multiple: float
    return_on_equity: float
    exit_reason: str
    filters_passed: Dict[str, bool]
    duration_bars: int


@dataclass
class PendingOrder:
    """Pending order awaiting confirmation after the signal candle is known."""

    setup: SetupCandidate
    direction: str
    created_idx: int
    expires_idx: int
    order_type: str
    trigger_price: float
    market_entry_idx: Optional[int]
    stop_price: float
    take_profit: float
    add_price: Optional[float]
    quantity: float
    risk_per_unit: float


@dataclass
class ActiveTrade:
    """Open position state tracked by the execution engine."""

    setup: SetupCandidate
    direction: str
    entry_idx: int
    entry_time: Any
    initial_entry_price: float
    average_entry_price: float
    stop_price: float
    take_profit: float
    quantity: float
    scale_in_size: float
    scale_in_price: Optional[float]
    risk_per_unit: float
    initial_risk_amount: float
    filters_passed: Dict[str, bool]
    signal_idx: int
    spike_start_idx: int
    spike_end_idx: int
    scaled_in: bool = False
    scale_in_fill: Optional[ScaleInFill] = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
