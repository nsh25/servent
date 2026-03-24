from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .config import SP2LConfig


class SetupReplay:
    """Generate a step-by-step textual replay for one setup candidate."""

    def __init__(self, config: SP2LConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir

    def replay_setup(self, df: pd.DataFrame, results: Dict[str, Any], setup_id: int) -> Dict[str, Any]:
        candidates = results.get("candidates", [])
        if setup_id < 0 or setup_id >= len(candidates):
            raise IndexError(f"setup_id={setup_id} out of range")

        setup = candidates[setup_id]
        events: List[str] = []
        events.append(f"setup_id={setup_id} direction={setup.direction}")
        events.append(
            f"spike_detected start={setup.spike.start_idx} end={setup.spike.end_idx} gap_passed={setup.spike.gap_passed}"
        )
        if setup.correction:
            events.append(
                "correction_detected "
                f"signal_idx={setup.correction.signal_idx} second_leg_valid={setup.correction.second_leg_valid} "
                f"depth={setup.correction.correction_depth:.4f}"
            )
        else:
            events.append("correction_missing")

        if setup.rejection_reason:
            events.append(f"rejected reason={setup.rejection_reason}")
        else:
            events.append("filters_passed")
            events.append(f"entry_plan entry={setup.entry_price} stop={setup.stop_price} tp={setup.take_profit}")

        trade = next((t for t in results.get("trades", []) if t.signal_idx == (setup.correction.signal_idx if setup.correction else -1)), None)
        if trade:
            events.append(f"entry_triggered at_idx={trade.entry_idx} price={trade.entry_price:.4f}")
            if trade.scale_in_price is not None:
                events.append(f"scale_in_triggered price={trade.scale_in_price:.4f}")
            events.append(f"exit {trade.exit_reason} at_idx={trade.exit_idx} price={trade.exit_price:.4f}")
        else:
            events.append("no_trade_executed")

        report_path = self.output_dir / f"walkthrough_setup_{setup_id}.md"
        report_text = "\n".join(f"- {line}" for line in events)
        report_path.write_text(report_text, encoding="utf-8")

        chart_path: Optional[str] = None
        if self.config.walkthrough.walkthrough_save_chart:
            chart_path = str(self._save_chart(df, setup_id, setup))

        return {"events": events, "report_path": str(report_path), "chart_path": chart_path}

    def _save_chart(self, df: pd.DataFrame, setup_id: int, setup: Any) -> Path:
        lo = max(0, setup.spike.start_idx - 5)
        hi = min(len(df), setup.spike.end_idx + 15)
        window = df.iloc[lo:hi]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(window["timestamp"], window["close"], label="close")
        ax.axvspan(df.iloc[setup.spike.start_idx]["timestamp"], df.iloc[setup.spike.end_idx]["timestamp"], alpha=0.2, color="orange", label="spike")
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
