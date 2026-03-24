# SP2L Backtesting Project

This repository contains a full Python backtesting project for the **SP2L (Spike → 2nd Leg → Entry Level)** discretionary price-action concept. The implementation keeps the spike, correction, entry, stop, target, and scale-in logic explicit rather than hiding it inside a black-box framework.

## Project structure

```text
sp2l_backtest/
  __init__.py
  backtest_engine.py
  config.py
  correction_logic.py
  data_loader.py
  filters.py
  indicators.py
  models.py
  plot_utils.py
  reporting.py
  risk.py
  spike_detector.py
  strategy.py
  walk_forward.py
  optimizer.py
  replay.py
main.py
example_config.yaml
examples/sample_ohlcv.csv
requirements.txt
tests/
```

## How SP2L is translated into code

### 1. Spike
A spike is detected only from candles that are already known at the current bar close.

The baseline detector checks:
- a configurable minimum number of impulse candles;
- a minimum directional ratio in the spike window;
- a minimum average real body size relative to ATR;
- at least one qualifying gap inside the spike window when `require_at_least_one_gap` is enabled.

Both long and short spikes are evaluated explicitly and symmetrically.

### 2. Correction / second leg
Two interpretations are implemented:
- `simplified_trigger` (default): after the spike, a correction qualifies when the current bar reaches the previous bar's opposite extreme (`low <= previous low` for bullish continuation, `high >= previous high` for bearish continuation).
- `strict_two_leg`: a simple two-push approximation that requires two distinct corrective extensions before the signal is accepted.

### 3. Entry
Three entry modes are supported:
- `stop_break` (default): buy stop above the signal high / sell stop below the signal low;
- `market_on_close_of_signal`;
- `market_on_next_open`.

### 4. Stop and target
Stop placement is configurable:
- `pre_spike_origin_candle` (default);
- `first_spike_candle`;
- `swing_origin_extreme`.

The default take-profit is a single `1R` target, but the risk/reward target is configurable.

### 5. Scale-in
When enabled, the strategy places a secondary entry halfway between the initial entry and the stop. The average entry is recalculated when the scale-in fills, and PnL is computed on the combined position.

### 6. Reporting / explainability
The project writes:
- `trade_log.csv`;
- `candidate_setups.csv` for all accepted and rejected candidates;
- `signal_table.csv` with per-event diagnostics;
- `performance_summary.csv`;
- `debug_log.csv` for verbose rejection and lifecycle messages;
- PNG charts for equity, drawdown, and annotated candles.

## Explicit assumptions required by the discretionary nature of SP2L

The user explicitly requested that assumptions be called out clearly. The baseline implementation therefore makes these assumptions and exposes them as configuration wherever practical.

1. **Spike window selection**
   - Assumption: the spike ends at the current evaluation bar and is chosen using the longest valid recent window within `max_spike_candles`.
   - Alternative: use swing-segment or breakout-based spike segmentation.

2. **Directional ratio interpretation**
   - Assumption: directional ratio is the fraction of bullish or bearish candle bodies inside the candidate window.
   - Alternative: weight by body size or close location instead of simple candle counts.

3. **Gap interpretation**
   - Assumption: the default baseline uses a literal gap (`current low > previous high` for bullish, `current high < previous low` for bearish).
   - Alternative: use body-gap or fair-value-gap approximations. Both are supported/configurable.

4. **Second-leg simplification**
   - Assumption: in `simplified_trigger`, one corrective bar that breaches the previous candle extreme qualifies.
   - Alternative: stricter micro-structure or swing logic. A `strict_two_leg` approximation is included.

5. **Signal confirmation timing**
   - Assumption: the signal candle is only actionable after it closes, so stop-break entries become eligible starting on the next bar.
   - Alternative: intrabar live triggering with tick data.

6. **Intrabar ordering**
   - Assumption: if both stop and target are touched within the same OHLC bar, the engine uses **stop-first priority** as the conservative baseline.
   - Alternative: target-first, midpoint interpolation, or lower-timeframe replay.

7. **Scale-in ordering**
   - Assumption: the scale-in can occur before exit checks on the same bar if the bar reaches the add level.
   - Alternative: disable same-bar add fills or require a separate lower timeframe.

8. **HTF trend alignment**
   - Assumption: when enabled, higher timeframe direction is approximated by resampled close data and EMA fast/slow alignment.
   - Alternative: structure-based HTF trend states.

9. **Channel edge filter**
   - Assumption: “near edge” is normalized by the recent rolling high-low channel and compared with a configurable threshold.
   - Alternative: regression channels, Donchian channels, or manually marked context zones.

10. **Position sizing**
    - Assumption: fixed fractional risk sizing uses current equity and price stop distance only.
    - Alternative: volatility targeting, capped leverage, or portfolio-level exposure constraints.

## Running the backtest

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the example

```bash
python main.py --config example_config.yaml
```

Outputs are saved to `output/` by default.

## Changing parameters

Edit `example_config.yaml` or provide another YAML file. The nested configuration sections mirror the code modules:
- `spike`
- `correction`
- `entry`
- `scale_in`
- `filters`
- `risk`
- `reporting`


## Walkthrough test

For a step-by-step baseline validation, run the dedicated walkthrough-style unit test after installing dependencies:

```bash
pytest -q tests/test_walkthrough.py -k walkthrough
```

That test uses a deterministic synthetic bullish sequence and verifies the entire lifecycle in order:
1. bullish spike with an internal gap is detected;
2. the correction satisfies the configured second-leg rule;
3. the setup survives filters and becomes a pending entry;
4. the continuation entry fills;
5. the optional scale-in fills at the 50% add level;
6. the combined position reaches the take-profit.

This gives you a readable "research walkthrough" test before you branch into more complex parameter sweeps.

## Inspecting candidate setups

Inspect these files after a run:
- `output/candidate_setups.csv` for all candidates, including rejections;
- `output/signal_table.csv` for a compact signal-status table;
- `output/debug_log.csv` for verbose reasoning on lifecycle events and rejections.

## Interpreting charts and metrics

- **Equity curve**: cumulative strategy equity through time.
- **Drawdown curve**: percentage drop from the running equity peak.
- **Candlestick annotations**: highlights spike windows and plots signal / entry / exit markers.
- **Summary metrics**: include total return, CAGR, win rate, profit factor, expectancy, average R, max drawdown, Sharpe, Sortino, and trade counts.

## Sample optimization workflow without overfitting

To explore parameters responsibly:
1. Start with the provided defaults as a baseline.
2. Change only one parameter family at a time, such as spike detection or correction timing.
3. Use walk-forward splits rather than one in-sample period.
4. Prefer broad parameter zones that remain stable across timeframes and market regimes.
5. Track rejected setups alongside executed ones to understand *why* results change.
6. Validate any promising setting on unseen symbols and unseen periods.
7. If possible, replay signals on a lower timeframe to test the intrabar assumptions described above.

## Notes

This project is intentionally transparent and modular. For real trading research, consider enriching it with:
- lower timeframe replay for fill realism;
- partial exits / trailing logic beyond the baseline `1R` target;
- more advanced strict second-leg swing parsing;
- portfolio-level constraints and walk-forward optimization tooling.


## Walk-forward evaluation engine

Enable in config:

```yaml
walk_forward:
  walk_forward_enabled: true
  walk_forward_mode: rolling_window   # or expanding_window
  train_bars: 200
  test_bars: 80
  step_bars: 80
  warmup_bars: 50
  retune_each_window: true
optimization:
  optimization_enabled: true
  optimization_mode: grid             # or random
  optimization_metric: sharpe
  min_trades_constraint: 3
  max_trials: 30
```

When enabled, the engine creates chronological folds, runs optional parameter search on **train-only** data, freezes selected parameters, and evaluates only on each fold's out-of-sample test window.

Generated files:
- `walk_forward_folds.csv` (fold ranges + in-sample/out-of-sample metrics)
- `walk_forward_selected_params.csv` (top parameter sets per fold)
- `walk_forward_summary.csv` (combined out-of-sample summary)
- `walk_forward_combined_equity.csv`

Run:

```bash
python main.py --config example_config.yaml
```

## Trade walkthrough / replay mode

Enable replay for a specific setup candidate:

```yaml
walkthrough:
  walkthrough_enabled: true
  walkthrough_setup_id: 0
  walkthrough_save_chart: true
  walkthrough_verbose: true
```

Outputs:
- `walkthrough_setup_<id>.md` with candle-by-candle decision events
- optional `walkthrough_setup_<id>.png` annotated chart
