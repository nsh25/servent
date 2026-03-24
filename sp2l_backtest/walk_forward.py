from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import SP2LConfig
from .optimizer import ParameterOptimizer
from .reporting import Reporter
from .strategy import SP2LStrategy


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


class WalkForwardRunner:
    """Run rolling/expanding walk-forward evaluation with optional train-only optimization."""

    def __init__(self, config: SP2LConfig, prepared_df: pd.DataFrame, output_dir: Path):
        self.config = config
        self.df = prepared_df
        self.output_dir = output_dir
        self.reporter = Reporter(config, output_dir)

    def generate_folds(self, n_bars: int) -> List[WalkForwardFold]:
        cfg = self.config.walk_forward
        folds: List[WalkForwardFold] = []
        fold_id = 1
        train_end = cfg.train_bars
        while True:
            test_start = train_end
            test_end = test_start + cfg.test_bars
            if test_end > n_bars:
                break
            if cfg.walk_forward_mode == "expanding_window":
                train_start = 0
            else:
                train_start = max(0, train_end - cfg.train_bars)
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            fold_id += 1
            train_end += cfg.step_bars
        return folds

    def run(self) -> Dict[str, Any]:
        folds = self.generate_folds(len(self.df))
        fold_rows: List[Dict[str, Any]] = []
        params_rows: List[Dict[str, Any]] = []
        all_oos_trades = []
        oos_equity_parts = []
        equity_offset = self.config.risk.initial_equity

        for fold in folds:
            train_df = self.df.iloc[fold.train_start : fold.train_end].copy()
            warmup_start = max(0, fold.test_start - self.config.walk_forward.warmup_bars)
            test_eval_df = self.df.iloc[warmup_start : fold.test_end].copy()

            selected_params: Dict[str, Any] = {}
            ranked = []
            if self.config.optimization.optimization_enabled:
                optimizer = ParameterOptimizer(self.config)
                opt = optimizer.optimize(train_df)
                selected_params = opt["best_params"]
                ranked = opt["ranked"]

            fold_cfg = deepcopy(self.config)
            for path, value in selected_params.items():
                current = fold_cfg
                parts = path.split(".")
                for part in parts[:-1]:
                    current = getattr(current, part)
                setattr(current, parts[-1], value)

            train_results = SP2LStrategy(fold_cfg).run(train_df)
            train_summary = Reporter(fold_cfg, self.output_dir).summary(train_results["trades"], train_results["equity_curve"], train_df).iloc[0]

            test_results = SP2LStrategy(fold_cfg).run(test_eval_df)
            global_test_start_ts = self.df.iloc[fold.test_start]["timestamp"]
            global_test_end_ts = self.df.iloc[fold.test_end - 1]["timestamp"]
            oos_trades = [
                t for t in test_results["trades"] if global_test_start_ts <= t.entry_time <= global_test_end_ts
            ]
            all_oos_trades.extend(oos_trades)
            oos_equity = test_results["equity_curve"].copy()
            oos_equity = oos_equity[oos_equity["timestamp"] >= global_test_start_ts].copy()
            if not oos_equity.empty:
                base = float(oos_equity.iloc[0]["equity"])
                oos_equity["equity"] = oos_equity["equity"] - base + equity_offset
                equity_offset = float(oos_equity.iloc[-1]["equity"])
                oos_equity["fold_id"] = fold.fold_id
                oos_equity_parts.append(oos_equity)

            test_slice_df = self.df.iloc[fold.test_start : fold.test_end].copy()
            oos_summary = Reporter(fold_cfg, self.output_dir).summary(
                oos_trades,
                pd.DataFrame({"timestamp": test_slice_df["timestamp"], "equity": oos_equity["equity"].values if len(oos_equity)==len(test_slice_df) else [self.config.risk.initial_equity]*len(test_slice_df)}),
                test_slice_df,
            ).iloc[0]

            fold_row = {
                "fold_id": fold.fold_id,
                "train_start": self.df.iloc[fold.train_start]["timestamp"],
                "train_end": self.df.iloc[fold.train_end - 1]["timestamp"],
                "test_start": global_test_start_ts,
                "test_end": global_test_end_ts,
                "selected_params": selected_params,
                "train_total_return": float(train_summary["total_return"]),
                "train_sharpe": float(train_summary["sharpe"]),
                "oos_total_return": float(oos_summary["total_return"]),
                "oos_sharpe": float(oos_summary["sharpe"]),
                "oos_profit_factor": float(oos_summary["profit_factor"]),
                "oos_expectancy": float(oos_summary["expectancy"]),
                "oos_average_r": float(oos_summary["average_r"]),
                "oos_total_trades": int(oos_summary["total_trades"]),
            }
            fold_rows.append(fold_row)

            for rank, candidate in enumerate(ranked, start=1):
                params_rows.append({"fold_id": fold.fold_id, "rank": rank, **candidate})

        folds_df = pd.DataFrame(fold_rows)
        params_df = pd.DataFrame(params_rows)
        combined_equity = pd.concat(oos_equity_parts, ignore_index=True) if oos_equity_parts else pd.DataFrame(columns=["timestamp", "equity", "fold_id"])
        combined_df = self.df[self.df["timestamp"].isin(combined_equity["timestamp"])].copy() if not combined_equity.empty else self.df.iloc[:1].copy()
        if combined_equity.empty:
            combined_equity = pd.DataFrame({"timestamp": combined_df["timestamp"], "equity": [self.config.risk.initial_equity] * len(combined_df)})
        combined_summary = self.reporter.summary(all_oos_trades, combined_equity[["timestamp", "equity"]], combined_df)

        folds_df.to_csv(self.output_dir / "walk_forward_folds.csv", index=False)
        params_df.to_csv(self.output_dir / "walk_forward_selected_params.csv", index=False)
        combined_summary.to_csv(self.output_dir / "walk_forward_summary.csv", index=False)
        combined_equity.to_csv(self.output_dir / "walk_forward_combined_equity.csv", index=False)

        return {
            "folds": folds_df,
            "selected_params": params_df,
            "summary": combined_summary,
            "combined_equity": combined_equity,
            "trades": all_oos_trades,
        }
