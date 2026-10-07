1. **Explore `src/trading_system/backtest/`**
   - Create `models.py` to hold Backtest models (e.g. BacktestResult, Position, Trade, etc)
   - Create `engine.py` for the Backtest Engine, handling OHLCV replay, order execution, fill models, PnL, risk, stats calculation.
2. **Implement Backtest Engine (`src/trading_system/backtest/engine.py`)**
   - Must use pure OHLCV data.
   - Strictly process sequentially to enforce "No look-ahead" (decisions on closed bars).
   - Simulates Limit entries, stop loss, and take profit using High/Low of the current bar. Gap-aware stop model (if open > stop, fill at open, etc).
   - "If stop and target hit in same bar, conservative assumption" -> assume stop loss is hit first.
   - Maintain track of initial capital, active positions, trades, exposure, etc. Calculate slippage and fees based on configuration.
   - Replay Oracle: Oracle is offline by default. If enabled, it only uses previously stored LLM decisions (no live calls).
   - Calculate metrics: total return, max drawdown, trade count, win rate, profit factor, exposure, etc.
   - Document corporate-action adjustment (e.g. assume data is already split/dividend adjusted) and survivorship bias (fixed watchlist) in docstrings and CLI output.
3. **Extend CLI (`src/trading_system/cli.py`)**
   - Add `backtest` command with `--help` and a local dataset example.
   - Backtest command loads a historical dataset (e.g., CSV/JSON mock or simple generator if file not found for tests) and runs the backtest engine, outputting the report.
4. **Tests (`tests/test_backtest.py`)**
   - Add unit tests for fill logic (limit, gap stops, conservative dual-hit).
   - Add e2e replay test running the whole backtest loop on mock data.
