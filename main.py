from __future__ import annotations

import argparse

from sp2l_backtest import BacktestEngine, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SP2L backtesting project.")
    parser.add_argument("--config", default="example_config.yaml", help="Path to YAML configuration file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    engine = BacktestEngine(config)
    engine.run()
    print(f"Backtest complete. Outputs saved to {config.reporting.output_dir}")


if __name__ == "__main__":
    main()
