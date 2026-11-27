# Rules and facts of the trading bot (alpaca-roundtable-trader), for a reviewer

## Facts: do NOT report these as defects

- Python 3.15 in Docker (`python:3.15-slim`) and in CI; `python = "^3.11"` stays in pyproject and ruff's
  `target-version` stays `py311` on purpose (with `py315`, `ruff format` would rewrite `except (A, B):`
  into syntax only 3.14+ reads). pandas 3.1, numpy 2.5, pydantic 2 with pydantic-settings, SQLAlchemy 2.1,
  alembic, alpaca-py 0.45, requests. mypy runs in strict mode on `src`. The ruff rules are pinned to
  `E4, E7, E9, F`: style that ruff does not check (f-strings inside logging calls, import order, line
  breaks) is not a defect.
- PAPER ONLY. `Settings` refuses `IS_PAPER=false` and every service is a paper account. The code is still
  built as if real money were at stake, so a bug on the order path counts as if it were real money.
- One owner, one Windows PC, Docker Desktop. The services in `compose.yaml` run one image:
  `trader` (the live loop in dry-run mode: real data, SIMULATED orders), `paper3` (the same code with
  `run --paper`: REAL paper orders on a third account, its own database `paper3.db`, 20 positions / 80 %
  gross, the Oracle in ACTIVE mode), `observe` (the Oracle observer: no risk or execution engine, so no
  order at all), `fractional` (a second paper account, fractional shares, a virtual budget of 500
  dollars), `forward` (weekly picks recorded forward in time, no orders), `data`, `news` and `backup`
  (no orders). They share one Docker volume; SQLite runs in WAL mode with a 30 s busy timeout, and only
  `paper3` has a database of its own.
- The flow is Strategy signal -> Oracle (APPROVE / VETO / HOLD) -> Risk -> Execution -> Alpaca, and each
  step can stop the signal. The Execution engine is the ONLY code that sends or cancels orders. The
  Oracle never proposes a symbol, a size or a price.
- The Oracle is reached only through the owner's Go backend (`POST /oracle/evaluate`, bearer token).
  A provider never raises: a timeout, a refusal or an unreadable reply is a `HOLD` with `failed=True`,
  never a `VETO` and never an `APPROVE`. It does NOT retry (the next tick asks again) and pauses 30 s
  after a failure: both deliberate. In `shadow` mode the real answer is recorded and the loop is told
  `APPROVE` (deliberate: the model is being measured, not obeyed). The backend caches answers by
  `signal_id` alone (HOLD 3 minutes, APPROVE / VETO 30 minutes), and `trader`, `observe` and `paper3`
  share that cache on purpose. A final `APPROVE` / `VETO` is read back from the record after a restart.
- A `HOLD` is asked again on later ticks until the signal expires (`expires_at`); an expired signal is
  refused by risk and is not asked about.
- An order is identified by `client_order_id = "exec_<signal_id>"`, made before anything is sent; the
  broker id stays empty until the broker answers. This is the idempotency key across restarts.
- Bracket orders are `GTC` on purpose (the legs inherit the entry's time in force; a `DAY` bracket loses
  its stop at the close). Alpaca takes no bracket and no GTC stop for FRACTIONAL orders, and a stop and a
  target cannot share one quantity: the fractional loop places a `DAY` stop after each fill and again
  whenever it is missing, and its take-profit is done in software. That is the design, not an omission.
- A position without an active stop halts trading and raises an alarm; it is NOT closed automatically
  unless `EMERGENCY_CLOSE_UNPROTECTED=true` (default false). Stopping the system does not pause entries
  unless `PAUSE_ON_SHUTDOWN=true`.
- A broker refusal (a 4xx) followed by a lookup that does not find the order marks the order REJECTED
  without a halt; a timeout, a 5xx, a 429 or a failed lookup still halts (2026-11-21, deliberate).
- Retries are only for transient failures (network, timeouts, HTTP 408 / 429 / 5xx); a 4xx and a bug
  fail at once.
- The daily loss guard measures from the broker's `last_equity` (the previous close) or, without it,
  from the equity at the first cycle of the New York trading day; the baseline never moves within the day.
- Time: timestamps are UTC; the trading day and the session (09:30 to 16:00, weekdays) are New York time
  (`America/New_York`). Holidays and early closes are not modelled: there are simply no bars then.
- Bars are 15 minutes and 1 hour, fixed (the settings refuse other sizes), adjusted for splits and
  dividends. The IEX feed includes extended-hours bars; the strategies that need the regular session
  filter them themselves. The entry look-up runs 30 s after a bar closes; an entry needs a last trade no
  older than 120 s and within 5 % of the last closed bar.
- Repositories run ONE committed transaction per call, with no long-lived session (deliberate: long
  sessions held SQLite's write lock).
- Research and the Oracle judge: every variant, number, threshold and criterion was registered in the docs
  BEFORE any result was seen. "Another number is another variant": module constants and version strings
  that carry the parameters are deliberate. Do NOT suggest tuning a parameter, widening a threshold or
  looking at the hold-out. Report a bug only when the code does not do what its own docstring or comment
  says the rule is (look-ahead, an off-by-one in a window, a wrong count, a wrong sign).
- The comments explain why, often with dates and measurements: do not report them as noise.
- The tests (about 1440) are in files that are NOT shown: never write that something has no test.
- Only the files of the group are reviewed. Files marked "context" are shown so that callers and callees
  can be read; do not report findings in them. About any other file, say that you cannot see it instead
  of guessing what it does.
- The threads: the loop is synchronous (no asyncio); the Execution engine holds a `threading.Lock`
  around submit, and the runner uses a `threading.Event` to stop. Different containers are different
  processes that meet only in the database and at the broker.
- Metrics listen on `0.0.0.0` inside the container on purpose; the port is published on the PC's
  loopback only (`127.0.0.1:9101` / `:9102`). The containers run as an unprivileged user.

## Rules of the project: report breaking them

- No order without: trading not halted, risk's approval on a fresh account snapshot, a live price check,
  `IS_PAPER`, and the deterministic `client_order_id`. A price sent to the broker respects the tick size
  (0.01 at or above 1 dollar, 0.0001 below). Whole shares, except in the fractional loop.
- Anything unclear goes to the safe side: no order, a refusal, a `HOLD`. Never an approval, a bigger
  size or a missing stop because a value was None, NaN, 0, empty or of an unexpected type.
- A write that must happen once (an order, a final Oracle verdict, a journal row that a judge counts)
  stays once across a retry and a restart.
- Secrets never reach a log line, an exception message, a journal row, a metric or the CLI output: the
  Alpaca keys, `ORACLE_BACKEND_TOKEN`, any `.env` value. Settings hold them as `SecretStr`; the config
  check prints names only; a backend failure is logged by its kind, never with the body or the token.
- A failure that stops nothing is logged (with the exception type), not swallowed silently; an
  `except Exception` that hides a bug on the order path is a defect.
- No naive datetime is compared with an aware one; a trading day is a New York date, not a UTC date.
- A signal uses only bars that had closed before its decision time (no look-ahead), in the live loop, in
  the backtest and in the research alike.
- Text from outside (news) reaches the model only through `oracle/news_input.py`, cleaned and
  length-limited (the Go backend, not shown, builds the prompt).
