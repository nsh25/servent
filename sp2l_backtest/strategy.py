from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import pandas as pd

from .config import SP2LConfig
from .correction_logic import CorrectionDetector
from .filters import SetupFilters
from .models import ActiveTrade, PendingOrder, ScaleInFill, SetupCandidate, SpikeEvent, TradeRecord
from .risk import PositionSizingResult, RiskManager
from .spike_detector import SpikeDetector


class SP2LStrategy:
    """Sequential SP2L strategy that separates pattern detection from order execution."""

    def __init__(self, config: SP2LConfig):
        self.config = config
        self.spike_detector = SpikeDetector(config.spike)
        self.correction_detector = CorrectionDetector(config.correction)
        self.filters = SetupFilters(config.filters)
        self.risk = RiskManager(config.risk)
        self.debug_messages: List[str] = []
        self.candidates: List[SetupCandidate] = []
        self.trades: List[TradeRecord] = []
        self.pending_orders: List[PendingOrder] = []
        self.active_trades: List[ActiveTrade] = []

    def run(self, df: pd.DataFrame) -> Dict[str, object]:
        equity = self.config.risk.initial_equity
        equity_curve: List[float] = []
        signal_rows: List[Dict[str, object]] = []
        spikes_in_play: List[SpikeEvent] = []

        for idx in range(len(df)):
            row = df.iloc[idx]
            timestamp = row["timestamp"]
            equity = self._mark_to_market(equity, df, idx)
            equity_curve.append(equity)

            filled_today = self._process_pending_orders(df, idx, equity)
            closed_today = self._process_active_trades(df, idx)
            if closed_today:
                equity += sum(trade.pnl for trade in closed_today)
                equity_curve[-1] = equity

            new_spikes = self.spike_detector.detect(df, idx)
            for spike in new_spikes:
                spikes_in_play.append(spike)
                signal_rows.append(
                    {
                        "timestamp": timestamp,
                        "bar_index": idx,
                        "direction": spike.direction,
                        "spike_detected": True,
                        "gap_detected": spike.gap_passed,
                        "correction_detected": False,
                        "second_leg_valid": False,
                        "entry_confirmed": False,
                        "filters_passed": False,
                        "status": "spike_detected",
                    }
                )

            for spike in list(spikes_in_play):
                correction = self.correction_detector.find_signal(df, spike, idx)
                if correction is None:
                    if idx - spike.end_idx > self.config.correction.max_bars_after_spike:
                        self._reject_spike(spike, "setup_expired_without_correction")
                        spikes_in_play.remove(spike)
                    continue
                filters_result = self.filters.evaluate(
                    df,
                    idx,
                    correction.spike.direction,
                    spike_strength_atr=(spike.spike_high - spike.spike_low) / max(df.iloc[idx]["atr"], 1e-9),
                )
                entry_plan = self._build_entry_plan(df, correction, equity)
                setup = SetupCandidate(
                    direction=spike.direction,
                    spike=spike,
                    correction=correction,
                    status="candidate",
                    rejection_reason=None if filters_result.passed else "; ".join(filters_result.reasons),
                    filters_passed=filters_result.details,
                    entry_confirmed=False,
                    entry_idx=None,
                    entry_price=entry_plan["entry_price"],
                    stop_price=entry_plan["stop_price"],
                    take_profit=entry_plan["take_profit"],
                    debug={
                        "timestamp": timestamp,
                        "correction_depth": correction.correction_depth,
                        "entry_mode": self.config.entry.entry_mode,
                        "pending_order": entry_plan,
                    },
                )
                if not filters_result.passed:
                    setup.status = "rejected"
                    self.candidates.append(setup)
                    spikes_in_play.remove(spike)
                    self._debug(f"Rejected setup at {timestamp}: {setup.rejection_reason}")
                    signal_rows.append(
                        {
                            "timestamp": timestamp,
                            "bar_index": idx,
                            "direction": spike.direction,
                            "spike_detected": True,
                            "gap_detected": spike.gap_passed,
                            "correction_detected": True,
                            "second_leg_valid": correction.second_leg_valid,
                            "entry_confirmed": False,
                            "filters_passed": False,
                            "status": "rejected",
                        }
                    )
                    continue

                if not self.config.risk.allow_multiple_positions and (self.active_trades or self.pending_orders):
                    setup.status = "rejected"
                    setup.rejection_reason = "position_limit_reached"
                    self.candidates.append(setup)
                    spikes_in_play.remove(spike)
                    self._debug(f"Rejected setup at {timestamp}: position limit reached")
                    continue

                self.candidates.append(setup)
                self.pending_orders.append(
                    PendingOrder(
                        setup=setup,
                        direction=spike.direction,
                        created_idx=idx,
                        expires_idx=idx + self.config.entry.order_expiry_bars,
                        order_type=self.config.entry.entry_mode,
                        trigger_price=entry_plan["entry_price"],
                        market_entry_idx=entry_plan["market_entry_idx"],
                        stop_price=entry_plan["stop_price"],
                        take_profit=entry_plan["take_profit"],
                        add_price=entry_plan["add_price"],
                        quantity=entry_plan["quantity"],
                        risk_per_unit=entry_plan["risk_per_unit"],
                    )
                )
                spikes_in_play.remove(spike)
                signal_rows.append(
                    {
                        "timestamp": timestamp,
                        "bar_index": idx,
                        "direction": spike.direction,
                        "spike_detected": True,
                        "gap_detected": spike.gap_passed,
                        "correction_detected": True,
                        "second_leg_valid": correction.second_leg_valid,
                        "entry_confirmed": False,
                        "filters_passed": True,
                        "status": "pending_entry",
                    }
                )

        if self.active_trades:
            last_idx = len(df) - 1
            last_row = df.iloc[last_idx]
            for trade in list(self.active_trades):
                self._close_trade(last_idx, last_row["timestamp"], float(last_row["close"]), "forced_end_of_data", trade)

        results = {
            "equity_curve": pd.DataFrame({"timestamp": df["timestamp"], "equity": equity_curve}),
            "trades": self.trades,
            "candidates": self.candidates,
            "signal_table": pd.DataFrame(signal_rows),
            "debug_log": self.debug_messages,
        }
        return results

    def _build_entry_plan(self, df: pd.DataFrame, correction, equity: float) -> Dict[str, float | int | None]:
        signal_row = df.iloc[correction.signal_idx]
        entry_price: float
        market_entry_idx: Optional[int] = None
        if self.config.entry.entry_mode == "stop_break":
            entry_price = float(signal_row["high"] if correction.spike.direction == "long" else signal_row["low"])
        elif self.config.entry.entry_mode == "market_on_close_of_signal":
            entry_price = float(signal_row["close"])
            market_entry_idx = correction.signal_idx
        elif self.config.entry.entry_mode == "market_on_next_open":
            next_idx = correction.signal_idx + 1
            if next_idx >= len(df):
                raise ValueError("market_on_next_open requires a future bar.")
            entry_price = float(df.iloc[next_idx]["open"])
            market_entry_idx = next_idx
        else:
            raise ValueError(f"Unsupported entry_mode: {self.config.entry.entry_mode}")

        stop_price = self._stop_price(df, correction.spike, correction.signal_idx)
        sizing: PositionSizingResult = self.risk.position_size(equity, entry_price, stop_price)
        risk_amount = abs(entry_price - stop_price)
        take_profit = entry_price + risk_amount * self.config.entry.rr_target if correction.spike.direction == "long" else entry_price - risk_amount * self.config.entry.rr_target
        add_price = None
        if self.config.scale_in.enable_scale_in:
            if correction.spike.direction == "long":
                add_price = entry_price - 0.5 * (entry_price - stop_price)
            else:
                add_price = entry_price + 0.5 * (stop_price - entry_price)
        return {
            "entry_price": entry_price,
            "market_entry_idx": market_entry_idx,
            "stop_price": stop_price,
            "take_profit": float(take_profit),
            "add_price": None if add_price is None else float(add_price),
            "quantity": sizing.quantity,
            "risk_per_unit": sizing.risk_per_unit,
        }

    def _stop_price(self, df: pd.DataFrame, spike: SpikeEvent, signal_idx: int) -> float:
        buffer_value = self.config.entry.stop_buffer_ticks + self.config.entry.stop_buffer_atr_fraction * float(df.iloc[signal_idx]["atr"])
        if self.config.entry.stop_reference == "pre_spike_origin_candle":
            ref_idx = spike.origin_idx
            ref_low = df.iloc[ref_idx]["low"]
            ref_high = df.iloc[ref_idx]["high"]
        elif self.config.entry.stop_reference == "first_spike_candle":
            ref_low = df.iloc[spike.start_idx]["low"]
            ref_high = df.iloc[spike.start_idx]["high"]
        elif self.config.entry.stop_reference == "swing_origin_extreme":
            ref_low = float(df.iloc[spike.origin_idx : spike.end_idx + 1]["low"].min())
            ref_high = float(df.iloc[spike.origin_idx : spike.end_idx + 1]["high"].max())
        else:
            raise ValueError(f"Unsupported stop reference: {self.config.entry.stop_reference}")

        if spike.direction == "long":
            return float(ref_low - buffer_value)
        return float(ref_high + buffer_value)

    def _process_pending_orders(self, df: pd.DataFrame, idx: int, equity: float) -> List[ActiveTrade]:
        filled: List[ActiveTrade] = []
        row = df.iloc[idx]
        for order in list(self.pending_orders):
            if idx <= order.created_idx:
                continue
            if idx > order.expires_idx:
                order.setup.status = "expired"
                order.setup.rejection_reason = "entry_not_triggered"
                self.pending_orders.remove(order)
                self._debug(f"Pending order expired for signal {order.setup.correction.signal_idx}")
                continue

            fill_price: Optional[float] = None
            if order.order_type == "stop_break":
                if order.direction == "long" and row["high"] >= order.trigger_price:
                    fill_price = max(order.trigger_price, float(row["open"]))
                elif order.direction == "short" and row["low"] <= order.trigger_price:
                    fill_price = min(order.trigger_price, float(row["open"]))
            elif order.order_type == "market_on_close_of_signal" and idx == order.created_idx + 1:
                fill_price = float(df.iloc[order.created_idx]["close"])
            elif order.order_type == "market_on_next_open" and order.market_entry_idx == idx:
                fill_price = float(row["open"])

            if fill_price is None:
                continue

            fill_price += self.config.risk.slippage_per_unit if order.direction == "long" else -self.config.risk.slippage_per_unit
            active = ActiveTrade(
                setup=order.setup,
                direction=order.direction,
                entry_idx=idx,
                entry_time=row["timestamp"],
                initial_entry_price=fill_price,
                average_entry_price=fill_price,
                stop_price=order.stop_price,
                take_profit=order.take_profit,
                quantity=order.quantity,
                scale_in_size=order.quantity * self.config.scale_in.scale_in_size,
                scale_in_price=order.add_price,
                risk_per_unit=order.risk_per_unit,
                initial_risk_amount=abs(fill_price - order.stop_price) * order.quantity,
                filters_passed=order.setup.filters_passed,
                signal_idx=order.setup.correction.signal_idx if order.setup.correction else order.setup.spike.end_idx,
                spike_start_idx=order.setup.spike.start_idx,
                spike_end_idx=order.setup.spike.end_idx,
            )
            order.setup.status = "entered"
            order.setup.entry_confirmed = True
            order.setup.entry_idx = idx
            order.setup.entry_price = fill_price
            self.active_trades.append(active)
            self.pending_orders.remove(order)
            filled.append(active)
            self._debug(f"Entered {order.direction} trade at {row['timestamp']} price {fill_price:.4f}")
        return filled

    def _process_active_trades(self, df: pd.DataFrame, idx: int) -> List[TradeRecord]:
        closed: List[TradeRecord] = []
        row = df.iloc[idx]
        for trade in list(self.active_trades):
            price_low = float(row["low"])
            price_high = float(row["high"])
            if self._try_scale_in(trade, idx, row["timestamp"], price_low, price_high):
                self._debug(f"Scale-in filled for trade entered on {trade.entry_time}")

            stop_hit, target_hit = self._check_exit_hits(trade, price_low, price_high)
            if stop_hit and target_hit:
                exit_price, reason = trade.stop_price, "stop_and_target_same_bar_stop_priority"
            elif stop_hit:
                exit_price, reason = trade.stop_price, "stop_loss"
            elif target_hit:
                exit_price, reason = trade.take_profit, "take_profit"
            else:
                continue

            closed.append(self._close_trade(idx, row["timestamp"], exit_price, reason, trade))
        return closed

    def _try_scale_in(self, trade: ActiveTrade, idx: int, timestamp, price_low: float, price_high: float) -> bool:
        if trade.scaled_in or trade.scale_in_price is None:
            return False
        if trade.direction == "long" and price_low <= trade.scale_in_price:
            fill = trade.scale_in_price
        elif trade.direction == "short" and price_high >= trade.scale_in_price:
            fill = trade.scale_in_price
        else:
            return False

        total_cost = trade.average_entry_price * trade.quantity + fill * trade.scale_in_size
        total_qty = trade.quantity + trade.scale_in_size
        trade.average_entry_price = total_cost / total_qty
        trade.quantity = total_qty
        trade.scaled_in = True
        trade.scale_in_fill = ScaleInFill(idx=idx, time=timestamp, price=fill, quantity=trade.scale_in_size)
        return True

    def _check_exit_hits(self, trade: ActiveTrade, price_low: float, price_high: float) -> tuple[bool, bool]:
        if trade.direction == "long":
            return price_low <= trade.stop_price, price_high >= trade.take_profit
        return price_high >= trade.stop_price, price_low <= trade.take_profit

    def _close_trade(self, idx: int, timestamp, exit_price: float, reason: str, trade: ActiveTrade) -> TradeRecord:
        direction_sign = 1 if trade.direction == "long" else -1
        gross_pnl = (exit_price - trade.average_entry_price) * trade.quantity * direction_sign
        commission = 2 * self.risk.commission(trade.quantity)
        pnl = gross_pnl - commission
        risk_basis = abs(trade.initial_entry_price - trade.stop_price) * max(trade.quantity - (trade.scale_in_fill.quantity if trade.scale_in_fill else 0.0), 1e-9)
        r_multiple = pnl / risk_basis if risk_basis else 0.0
        trade_record = TradeRecord(
            entry_idx=trade.entry_idx,
            entry_time=trade.entry_time,
            exit_idx=idx,
            exit_time=timestamp,
            direction=trade.direction,
            spike_start_idx=trade.spike_start_idx,
            spike_end_idx=trade.spike_end_idx,
            signal_idx=trade.signal_idx,
            entry_price=trade.initial_entry_price,
            scale_in_price=None if not trade.scale_in_fill else trade.scale_in_fill.price,
            average_entry=trade.average_entry_price,
            stop_price=trade.stop_price,
            take_profit=trade.take_profit,
            quantity=trade.quantity - (trade.scale_in_fill.quantity if trade.scale_in_fill else 0.0),
            scale_in_quantity=0.0 if not trade.scale_in_fill else trade.scale_in_fill.quantity,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl / max(trade.initial_risk_amount, 1e-9),
            r_multiple=r_multiple,
            return_on_equity=pnl / self.config.risk.initial_equity,
            exit_reason=reason,
            filters_passed=trade.filters_passed,
            duration_bars=idx - trade.entry_idx,
        )
        self.trades.append(trade_record)
        self.active_trades.remove(trade)
        self._debug(f"Closed {trade.direction} trade at {timestamp} reason={reason} pnl={pnl:.2f}")
        return trade_record

    def _reject_spike(self, spike: SpikeEvent, reason: str) -> None:
        self.candidates.append(
            SetupCandidate(
                direction=spike.direction,
                spike=spike,
                correction=None,
                status="rejected",
                rejection_reason=reason,
                filters_passed={},
                entry_confirmed=False,
                entry_idx=None,
                entry_price=None,
                stop_price=None,
                take_profit=None,
                debug={},
            )
        )
        self._debug(f"Rejected spike {spike.direction} ending at {spike.end_idx}: {reason}")

    def _mark_to_market(self, equity: float, df: pd.DataFrame, idx: int) -> float:
        if not self.active_trades:
            return equity
        current_close = float(df.iloc[idx]["close"])
        mark_to_market = 0.0
        for trade in self.active_trades:
            sign = 1 if trade.direction == "long" else -1
            mark_to_market += (current_close - trade.average_entry_price) * trade.quantity * sign
        return self.config.risk.initial_equity + sum(t.pnl for t in self.trades) + mark_to_market

    def _debug(self, message: str) -> None:
        if self.config.reporting.verbose_debug:
            self.debug_messages.append(message)
