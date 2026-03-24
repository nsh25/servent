from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
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
    """Institutional-style walk-forward runner with strict train/test separation."""

    def __init__(self, config: SP2LConfig, prepared_df: pd.DataFrame, output_dir: Path):
        self.config = config
        self.df = prepared_df.reset_index(drop=True)
        self.output_dir = output_dir

    def generate_folds(self, n_bars: int) -> List[WalkForwardFold]:
        cfg = self.config.walk_forward
        if cfg.train_bars <= 0 or cfg.test_bars <= 0:
            raise ValueError("train_bars and test_bars must be positive")

        folds: List[WalkForwardFold] = []
        fold_id = 1
        train_end = cfg.train_bars
        while train_end + cfg.test_bars <= n_bars:
            test_start = train_end
            test_end = test_start + cfg.test_bars
            if cfg.walk_forward_mode == "expanding_window":
                train_start = 0
            elif cfg.walk_forward_mode == "rolling_window":
                train_start = max(0, train_end - cfg.train_bars)
            else:
                raise ValueError("walk_forward_mode must be rolling_window or expanding_window")

            folds.append(WalkForwardFold(fold_id, train_start, train_end, test_start, test_end))
            fold_id += 1
            train_end += cfg.step_bars

        return folds

    def run(self) -> Dict[str, Any]:
        folds = self.generate_folds(len(self.df))
        fold_rows: List[Dict[str, Any]] = []
        param_rows: List[Dict[str, Any]] = []
        optimization_rows: List[Dict[str, Any]] = []
        combined_oos_trades = []
        combined_oos_equity_rows: List[pd.DataFrame] = []
        equity_anchor = self.config.risk.initial_equity
        frozen_params: Dict[str, Any] = {}

        for fold in folds:
            self._assert_no_overlap(fold)
            train_df = self.df.iloc[fold.train_start : fold.train_end].copy()
            test_df = self.df.iloc[fold.test_start : fold.test_end].copy()
            warmup_start = max(0, fold.test_start - self.config.walk_forward.warmup_bars)
            test_eval_df = self.df.iloc[warmup_start : fold.test_end].copy()

            selected_params: Dict[str, Any] = frozen_params.copy()
            ranked: List[Dict[str, Any]] = []
            if self.config.optimization.optimization_enabled:
                should_retune = self.config.walk_forward.retune_each_window or not frozen_params
                if should_retune:
                    tuned = ParameterOptimizer(self.config).optimize(train_df)
                    selected_params = tuned["best_params"]
                    ranked = tuned["ranked"]
                    frozen_params = selected_params.copy()
                else:
                    selected_params = frozen_params.copy()
            fold_config = self._apply_params_to_config(selected_params)

            train_results = SP2LStrategy(fold_config).run(train_df)
            train_summary = Reporter(fold_config, self.output_dir).summary(
                train_results["trades"], train_results["equity_curve"], train_df
            ).iloc[0]

            test_results = SP2LStrategy(fold_config).run(test_eval_df)
            test_start_ts = self.df.iloc[fold.test_start]["timestamp"]
            test_end_ts = self.df.iloc[fold.test_end - 1]["timestamp"]
            oos_trades = [t for t in test_results["trades"] if test_start_ts <= t.entry_time <= test_end_ts]
            combined_oos_trades.extend(oos_trades)

            fold_equity = test_results["equity_curve"].copy()
            fold_equity = fold_equity[(fold_equity["timestamp"] >= test_start_ts) & (fold_equity["timestamp"] <= test_end_ts)]
            if fold_equity.empty:
                fold_equity = pd.DataFrame({"timestamp": test_df["timestamp"], "equity": [equity_anchor] * len(test_df)})
            else:
                base = float(fold_equity.iloc[0]["equity"])
                fold_equity["equity"] = fold_equity["equity"] - base + equity_anchor
                equity_anchor = float(fold_equity.iloc[-1]["equity"])

            fold_equity["fold_id"] = fold.fold_id
            combined_oos_equity_rows.append(fold_equity)
            fold_equity.to_csv(self.output_dir / f"walk_forward_fold_{fold.fold_id}_equity.csv", index=False)
            self._save_fold_equity_plot(fold.fold_id, fold_equity)

            oos_summary = Reporter(fold_config, self.output_dir).summary(
                oos_trades, fold_equity[["timestamp", "equity"]], test_df
            ).iloc[0]

            optimization_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "selected_parameters": selected_params,
                    "train_objective_value": float(train_summary.get(self.config.optimization.optimization_metric.lower(), train_summary.get("sharpe", 0.0))),
                    "train_total_return": float(train_summary["total_return"]),
                    "train_total_trades": int(train_summary["total_trades"]),
                    "oos_total_return": float(oos_summary["total_return"]),
                    "oos_total_trades": int(oos_summary["total_trades"]),
                }
            )

            fold_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "train_start": self.df.iloc[fold.train_start]["timestamp"],
                    "train_end": self.df.iloc[fold.train_end - 1]["timestamp"],
                    "test_start": test_start_ts,
                    "test_end": test_end_ts,
                    "selected_parameters": selected_params,
                    "in_sample_metrics": {
                        "total_return": float(train_summary["total_return"]),
                        "sharpe": float(train_summary["sharpe"]),
                        "profit_factor": float(train_summary["profit_factor"]),
                        "expectancy": float(train_summary["expectancy"]),
                        "average_r": float(train_summary["average_r"]),
                        "total_trades": int(train_summary["total_trades"]),
                    },
                    "out_of_sample_metrics": {
                        "total_return": float(oos_summary["total_return"]),
                        "sharpe": float(oos_summary["sharpe"]),
                        "sortino": float(oos_summary["sortino"]),
                        "profit_factor": float(oos_summary["profit_factor"]),
                        "expectancy": float(oos_summary["expectancy"]),
                        "average_r": float(oos_summary["average_r"]),
                        "max_drawdown": float(oos_summary["max_drawdown"]),
                        "total_trades": int(oos_summary["total_trades"]),
                        "long_trades": int(oos_summary["long_trades"]),
                        "short_trades": int(oos_summary["short_trades"]),
                    },
                    "number_of_trades": int(oos_summary["total_trades"]),
                }
            )

            for rank, item in enumerate(ranked, start=1):
                param_rows.append({"fold_id": fold.fold_id, "rank": rank, **item})

        folds_df = pd.DataFrame(fold_rows)
        params_df = pd.DataFrame(param_rows)
        optimization_df = pd.DataFrame(optimization_rows)
        combined_oos_equity = (
            pd.concat(combined_oos_equity_rows, ignore_index=True)
            if combined_oos_equity_rows
            else pd.DataFrame(columns=["timestamp", "equity", "fold_id"])
        )
        summary_base_df = (
            self.df[self.df["timestamp"].isin(combined_oos_equity["timestamp"])].copy()
            if not combined_oos_equity.empty
            else self.df.iloc[:1].copy()
        )
        if combined_oos_equity.empty:
            combined_oos_equity = pd.DataFrame(
                {"timestamp": summary_base_df["timestamp"], "equity": [self.config.risk.initial_equity] * len(summary_base_df)}
            )
        combined_summary = Reporter(self.config, self.output_dir).summary(
            combined_oos_trades, combined_oos_equity[["timestamp", "equity"]], summary_base_df
        )

        folds_df.to_csv(self.output_dir / "walk_forward_folds.csv", index=False)
        params_df.to_csv(self.output_dir / "walk_forward_selected_params.csv", index=False)
        optimization_df.to_csv(self.output_dir / "walk_forward_optimization_summary.csv", index=False)
        combined_summary.to_csv(self.output_dir / "walk_forward_summary.csv", index=False)
        combined_oos_equity.to_csv(self.output_dir / "walk_forward_combined_equity.csv", index=False)
        self._save_combined_equity_plot(combined_oos_equity)

        return {
            "folds": folds_df,
            "selected_params": params_df,
            "optimization_summary": optimization_df,
            "summary": combined_summary,
            "combined_equity": combined_oos_equity,
            "trades": combined_oos_trades,
        }

    def _apply_params_to_config(self, selected_params: Dict[str, Any]) -> SP2LConfig:
        cfg = deepcopy(self.config)
        for path, value in selected_params.items():
            target: Any = cfg
            parts = path.split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], value)
        return cfg

    @staticmethod
    def _assert_no_overlap(fold: WalkForwardFold) -> None:
        if fold.train_end > fold.test_start:
            raise AssertionError("Leakage detected: train window overlaps test window")

    def _save_fold_equity_plot(self, fold_id: int, fold_equity: pd.DataFrame) -> None:
        if fold_equity.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(fold_equity["timestamp"], fold_equity["equity"], label=f"Fold {fold_id}")
        ax.set_title(f"Walk-forward Fold {fold_id} OOS Equity")
        ax.legend(loc="best")
        fig.autofmt_xdate()
        fig.savefig(self.output_dir / f"walk_forward_fold_{fold_id}_equity.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _save_combined_equity_plot(self, combined_oos_equity: pd.DataFrame) -> None:
        if combined_oos_equity.empty:
            return
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(combined_oos_equity["timestamp"], combined_oos_equity["equity"], color="black", label="Combined OOS")
        for fold_id, fold_df in combined_oos_equity.groupby("fold_id"):
            ax.axvline(fold_df["timestamp"].iloc[0], color="gray", linestyle="--", alpha=0.4)
            ax.text(fold_df["timestamp"].iloc[0], combined_oos_equity["equity"].min(), f"F{fold_id}", fontsize=8)
        ax.set_title("Combined Walk-forward OOS Equity (Fold Boundaries Marked)")
        ax.legend(loc="best")
        fig.autofmt_xdate()
        fig.savefig(self.output_dir / "walk_forward_combined_equity.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
