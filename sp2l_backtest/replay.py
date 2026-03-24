from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .config import SP2LConfig


class SetupReplay:
    """Replay one setup candle-by-candle with explainable event output."""

    def __init__(self, config: SP2LConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir

    def replay_setup(self, df: pd.DataFrame, results: Dict[str, Any], setup_id: int) -> Dict[str, Any]:
        candidates = results.get("candidates", [])
        if setup_id < 0 or setup_id >= len(candidates):
            raise IndexError(f"setup_id={setup_id} out of range")

        setup = candidates[setup_id]
        trade = next((t for t in results.get("trades", []) if t.signal_idx == (setup.correction.signal_idx if setup.correction else -1)), None)

        events: List[str] = [f"setup_id={setup_id} direction={setup.direction}"]
        events.append(f"spike_detection start={setup.spike.start_idx} end={setup.spike.end_idx} gap={setup.spike.gap_passed}")
        if setup.correction:
            events.append(
                f"correction_development signal_idx={setup.correction.signal_idx} "
                f"second_leg={setup.correction.second_leg_valid} depth={setup.correction.correction_depth:.4f}"
            )
        else:
            events.append("correction_development missing")

        if setup.rejection_reason:
            events.append(f"rejected reason={setup.rejection_reason}")
        else:
            events.append(f"entry_trigger planned_entry={setup.entry_price} stop={setup.stop_price} tp={setup.take_profit}")

        if trade:
            events.append(f"entry_filled idx={trade.entry_idx} price={trade.entry_price:.4f}")
            if trade.scale_in_price is not None:
                events.append(f"scale_in_trigger idx~{trade.entry_idx} price={trade.scale_in_price:.4f}")
            events.extend(self._price_progression(df, trade.entry_idx, trade.exit_idx, trade.stop_price, trade.take_profit))
            events.append(f"exit_condition {trade.exit_reason} idx={trade.exit_idx} price={trade.exit_price:.4f}")
        else:
            events.append("no_trade_executed")

        report_path = self.output_dir / f"walkthrough_setup_{setup_id}.md"
        report_path.write_text("\n".join(f"- {line}" for line in events), encoding="utf-8")

        chart_path: Optional[str] = None
        if self.config.walkthrough.walkthrough_save_chart:
            chart_path = str(self._save_chart(df, setup_id, setup))

        frame_paths: List[str] = []
        if self.config.walkthrough.walkthrough_save_frames:
            frame_paths = self._save_frames(df, setup_id, setup)

        return {
            "events": events,
            "report_path": str(report_path),
            "chart_path": chart_path,
            "frame_paths": frame_paths,
        }

    def _price_progression(self, df: pd.DataFrame, start_idx: int, end_idx: int, stop: float, target: float) -> List[str]:
        steps: List[str] = []
        for idx in range(start_idx, end_idx + 1):
            row = df.iloc[idx]
            steps.append(
                "sl_tp_progression "
                f"idx={idx} time={row['timestamp']} low={row['low']:.4f} high={row['high']:.4f} "
                f"stop={stop:.4f} target={target:.4f}"
            )
        return steps

    def _save_chart(self, df: pd.DataFrame, setup_id: int, setup: Any) -> Path:
        lo = max(0, setup.spike.start_idx - 5)
        hi = min(len(df), setup.spike.end_idx + 15)
        window = df.iloc[lo:hi]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(window["timestamp"], window["close"], label="close")
        ax.axvspan(
            df.iloc[setup.spike.start_idx]["timestamp"],
            df.iloc[setup.spike.end_idx]["timestamp"],
            alpha=0.2,
            color="orange",
            label="spike",
        )
        if setup.correction:
            ax.axvline(df.iloc[setup.correction.signal_idx]["timestamp"], color="blue", linestyle="--", label="signal")
        if setup.entry_price:
            ax.axhline(setup.entry_price, color="green", linestyle=":", label="entry")
        if setup.stop_price:
            ax.axhline(setup.stop_price, color="red", linestyle=":", label="stop")
        if setup.take_profit:
            ax.axhline(setup.take_profit, color="purple", linestyle=":", label="tp")
        ax.legend(loc="best")
        ax.set_title(f"SP2L Walkthrough Setup {setup_id}")
        fig.autofmt_xdate()

        out = self.output_dir / f"walkthrough_setup_{setup_id}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    def _save_frames(self, df: pd.DataFrame, setup_id: int, setup: Any) -> List[str]:
        frame_dir = self.output_dir / f"walkthrough_setup_{setup_id}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        start = max(0, setup.spike.start_idx - 2)
        end = min(len(df), (setup.correction.signal_idx if setup.correction else setup.spike.end_idx) + 5)
        paths: List[str] = []
        for idx in range(start, end):
            window = df.iloc[start : idx + 1]
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(window["timestamp"], window["close"], color="black")
            ax.set_title(f"Setup {setup_id} frame {idx}")
            fig.autofmt_xdate()
            path = frame_dir / f"frame_{idx:04d}.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(path))
        return paths
