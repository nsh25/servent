from __future__ import annotations

import itertools
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import SP2LConfig
from .reporting import Reporter
from .strategy import SP2LStrategy


class ParameterOptimizer:
    """Train-window-only parameter search used inside each walk-forward fold."""

    METRIC_MAP = {
        "net_profit": "expectancy",
        "sharpe": "sharpe",
        "expectancy": "expectancy",
        "profit_factor": "profit_factor",
        "average_r": "average_r",
    }

    def __init__(self, config: SP2LConfig):
        self.config = config

    def optimize(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        candidates = self._build_candidates(self._grid())
        scored: List[Dict[str, Any]] = []

        for params in candidates:
            cfg = deepcopy(self.config)
            self._apply_params(cfg, params)
            results = SP2LStrategy(cfg).run(train_df)
            summary = Reporter(cfg, Path(".")).summary(results["trades"], results["equity_curve"], train_df).iloc[0]
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

        scored.sort(key=lambda item: item["score"], reverse=True)
        best = scored[0] if scored else {"params": {}, "score": float("-inf")}
        return {
            "best_params": best["params"],
            "best_score": best["score"],
            "ranked": scored[: self.config.optimization.top_n_results_to_save],
        }

    def _grid(self) -> Dict[str, list[Any]]:
        return self.config.optimization.grid or {
            "spike.min_spike_candles": [2, 3],
            "spike.min_body_to_atr": [0.8, 1.0],
            "spike.min_directional_ratio": [0.6, 0.7],
            "spike.gap_mode": ["true_gap", "body_gap"],
            "spike.gap_threshold": [0.0, 0.05],
            "correction.second_leg_mode": ["simplified_trigger", "strict_two_leg"],
            "correction.max_bars_after_spike": [6, 10],
            "correction.minimum_correction_depth": [0.0, 0.2],
            "correction.max_correction_depth": [None, 3.0],
            "entry.entry_mode": ["stop_break", "market_on_close_of_signal"],
            "entry.stop_buffer_ticks": [0.0, 0.1],
            "entry.rr_target": [1.0, 1.5],
            "filters.enable_htf_trend": [False, True],
            "filters.enable_session_filter": [False, True],
        }

    def _build_candidates(self, grid: Dict[str, list[Any]]) -> List[Dict[str, Any]]:
        keys = list(grid.keys())
        combinations = [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]
        if self.config.optimization.optimization_mode == "random":
            rng = random.Random(self.config.optimization.random_seed)
            rng.shuffle(combinations)
        if self.config.optimization.max_trials > 0:
            combinations = combinations[: self.config.optimization.max_trials]
        return combinations

    def _apply_params(self, config: SP2LConfig, params: Dict[str, Any]) -> None:
        for path, value in params.items():
            cursor: Any = config
            parts = path.split(".")
            for part in parts[:-1]:
                cursor = getattr(cursor, part)
            setattr(cursor, parts[-1], value)

    def _score(self, summary_row: pd.Series) -> float:
        metric_key = self.METRIC_MAP.get(self.config.optimization.optimization_metric.lower(), "sharpe")
        trades = int(summary_row.get("total_trades", 0))
        penalty = 0.0
        if trades < self.config.optimization.min_trades_constraint:
            penalty = 1_000_000.0

        if self.config.optimization.optimization_metric.lower() == "net_profit":
            metric_value = float(summary_row.get("expectancy", 0.0)) * max(trades, 1)
        else:
            metric_value = float(summary_row.get(metric_key, 0.0))
        return metric_value - penalty
