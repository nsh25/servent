from __future__ import annotations

from dataclasses import dataclass

from .config import RiskConfig


@dataclass
class PositionSizingResult:
    quantity: float
    risk_amount: float
    risk_per_unit: float


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config

    def position_size(self, equity: float, entry_price: float, stop_price: float) -> PositionSizingResult:
        risk_per_unit = abs(entry_price - stop_price) + self.config.slippage_per_unit
        if risk_per_unit <= 0:
            raise ValueError("Stop distance must be positive for position sizing.")
        if self.config.sizing_mode == "fixed_quantity":
            quantity = self.config.fixed_quantity
            risk_amount = quantity * risk_per_unit
        else:
            risk_amount = equity * self.config.risk_per_trade
            quantity = risk_amount / risk_per_unit
        return PositionSizingResult(quantity=float(quantity), risk_amount=float(risk_amount), risk_per_unit=float(risk_per_unit))

    def commission(self, quantity: float) -> float:
        return abs(quantity) * self.config.commission_per_unit
