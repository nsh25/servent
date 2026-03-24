from __future__ import annotations

import itertools
import random
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, List

import pandas as pd

from .config import SP2LConfig
from .reporting import Reporter
from .strategy import SP2LStrategy


class ParameterOptimizer:
    """Fold-local parameter search that evaluates candidates only on the train segment."""

    def __init__(self, config: SP2LConfig):
        self.config = config

    def optimize(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        grid = self.config.optimization.grid or {
            "spike.min_spike_candles": [2, 3],
            "spike.min_body_to_atr": [0.8, 1.0],
            "spike.min_directional_ratio": [0.6, 0.7],
            "entry.entry_mode": ["stop_break", "market_on_close_of_signal"],
            "entry.rr_target": [1.0, 1.5],
            "correction.second_leg_mode": ["simplified_trigger", "strict_two_leg"],
        }
        candidates = self._build_candidates(grid)
        scored: List[Dict[str, Any]] = []

        for params in candidates:
            cfg = deepcopy(self.config)
            self._apply_params(cfg, params)
            strategy = SP2LStrategy(cfg)
            results = strategy.run(train_df)
            summary = Reporter(cfg, output_dir=Path(".")).summary(results["trades"], results["equity_curve"], train_df).iloc[0]
            score = self._score(summary)
            scored.append(
                {
                    "params": params,
                    "score": score,
                    "train_total_trades": int(summary["total_trades"]),
                    "train_sharpe": float(summary["sharpe"]),
                    "train_profit_factor": float(summary["profit_factor"]),
                    "train_expectancy": float(summary["expectancy"]),
                    "train_average_r": float(summary["average_r"]),
                    "train_total_return": float(summary["total_return"]),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0] if scored else {"params": {}, "score": float("-inf")}
        return {
            "best_params": best["params"],
            "best_score": best["score"],
            "ranked": scored[: self.config.optimization.top_n],
        }

    def _build_candidates(self, grid: Dict[str, list[Any]]) -> List[Dict[str, Any]]:
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        all_candidates = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        if self.config.optimization.optimization_mode == "random":
            random.Random(self.config.optimization.random_seed).shuffle(all_candidates)
            all_candidates = all_candidates[: self.config.optimization.max_trials]
        elif self.config.optimization.max_trials > 0:
            all_candidates = all_candidates[: self.config.optimization.max_trials]
        return all_candidates

    def _apply_params(self, config: SP2LConfig, params: Dict[str, Any]) -> None:
        for path, value in params.items():
            current = config
            parts = path.split(".")
            for part in parts[:-1]:
                current = getattr(current, part)
            setattr(current, parts[-1], value)

    def _score(self, summary_row: pd.Series) -> float:
        metric = self.config.optimization.optimization_metric
        trades = int(summary_row["total_trades"])
        if trades < self.config.optimization.min_trades_constraint:
            return -1e9 + trades
        metric_value = float(summary_row.get(metric, 0.0))
        return metric_value

    @staticmethod
    def flatten_params(params: Dict[str, Any]) -> Dict[str, Any]:
        return params
