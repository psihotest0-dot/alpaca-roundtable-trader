import argparse
import sys
from typing import Optional, Sequence

from pydantic import ValidationError

from trading_system.config import Settings


def inject_offline_dummy_keys() -> None:
    import os

    os.environ["ALPACA_API_KEY"] = os.environ.get("ALPACA_API_KEY", "dummy_offline_key")
    os.environ["ALPACA_SECRET_KEY"] = os.environ.get("ALPACA_SECRET_KEY", "dummy_offline_secret")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alpaca Roundtable Trader - Automated paper trading system."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate-config command
    subparsers.add_parser(
        "validate-config", help="Validates the local configuration without making network requests"
    )

    # doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Runs system checks (local by default).")
    doctor_parser.add_argument(
        "--network",
        action="store_true",
        help="Perform read-only network checks against Alpaca/Oracle.",
    )

    # backtest command
    backtest_parser = subparsers.add_parser(
        "backtest", help="Replays the stored candles (see fetch-history) through the strategy."
    )
    backtest_parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="How many days of stored candles to replay. The first days are only warm-up.",
    )
    backtest_parser.add_argument(
        "--replay-oracle",
        type=str,
        help="Path to JSON file containing historical Oracle decisions.",
    )
    backtest_parser.add_argument(
        "--strategy",
        default="trend_cross",
        help="Which strategy to replay (default trend_cross, the one the live loop runs). "
        "See compare-strategies for the choices.",
    )
    backtest_parser.add_argument(
        "--windows",
        type=int,
        default=4,
        help="Cut the replay into this many consecutive periods and report each (default 4).",
    )
    backtest_parser.add_argument(
        "--cost-sweep",
        action="store_true",
        help="Also replay at higher slippage (0 to 0.2 % a side) to see how fragile the result is.",
    )

    # compare-strategies command
    compare_parser = subparsers.add_parser(
        "compare-strategies",
        help="Replays several strategies on the same stored candles and costs, side by side.",
    )
    compare_parser.add_argument("--days", type=int, default=90, help="Days of candles to replay.")
    compare_parser.add_argument(
        "--strategies", type=str, help="Comma-separated names. Default: all of them."
    )

    # fetch-history command
    history_parser = subparsers.add_parser(
        "fetch-history", help="Downloads candles from Alpaca into the database."
    )
    history_parser.add_argument(
        "--days", type=int, default=180, help="How far back the stored history should reach."
    )
    history_parser.add_argument(
        "--symbols", type=str, help="Comma-separated symbols. Default: the WATCHLIST."
    )

    # fetch-observe command
    observe_fetch_parser = subparsers.add_parser(
        "fetch-observe",
        help="Downloads or updates the 15-minute and 1-hour bars of the observation universe "
        "(~155 symbols) into var/observe.db.",
    )
    observe_fetch_parser.add_argument(
        "--days", type=int, default=180, help="How far back the stored history should reach."
    )

    # fetch-daily command
    daily_parser = subparsers.add_parser(
        "fetch-daily",
        help="Downloads daily bars of the research universe (full-market feed) into research.db.",
    )
    daily_parser.add_argument(
        "--since",
        type=str,
        default="2016-01-01",
        help="First day of history, YYYY-MM-DD. The full-market feed starts on 2016-01-04.",
    )
    daily_parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated symbols. Default: the research universe (data/universe.py).",
    )

    # fetch-news command
    news_parser = subparsers.add_parser(
        "fetch-news", help="Brings the news cache (news.db) up to date from Alpaca's news API."
    )
    news_parser.add_argument(
        "--days",
        type=int,
        help="Fetch this many days back (a backfill). Default: from the newest stored article.",
    )
    news_parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated symbols. Default: the research universe and the WATCHLIST.",
    )

    # measure-signals command (the research modules are imported only when it runs: the
    # health check starts this program every minute and should not pay for pandas)
    measure_parser = subparsers.add_parser(
        "measure-signals",
        help="Measures what prices do after a signal (development part of the daily bars only).",
    )
    measure_parser.add_argument(
        "--signal",
        required=True,
        help="momentum_12_1, reversal_1w or random_tenth (a control that must show nothing).",
    )
    measure_parser.add_argument(
        "--horizons",
        type=str,
        help="Comma-separated trading days held. Default: the signal's own (research/signals.py).",
    )
    measure_parser.add_argument(
        "--regime-filter",
        help="Count the signal only on days the market is in this regime: spy200.",
    )
    measure_parser.add_argument(
        "--cost",
        type=float,
        default=None,
        help="Round-trip cost as a fraction (default 0.001 = 0.10 %%).",
    )
    measure_parser.add_argument(
        "--symbols", type=str, help="Comma-separated symbols. Default: the research universe."
    )

    # judge-oracle command
    subparsers.add_parser(
        "judge-oracle",
        help="Judges the Oracle on its recorded shadow answers (docs/oracle-provider-task.md, "
        "section 5). Shows counts only until the minimum is reached; one look, then a second at "
        "double the minimum.",
    )

    # judge-candidates command
    subparsers.add_parser(
        "judge-candidates",
        help="Judges the two strategy candidates the observer runs forward in time "
        "(docs/strategy-research-plan.md, section 12.1). Shows counts only until the minimum is "
        "reached; at most two looks.",
    )

    # measure-intraday command (the research modules are imported only when it runs)
    intraday_parser = subparsers.add_parser(
        "measure-intraday",
        help="Measures what prices do in the hours after an intraday signal (development part of "
        "the 15-minute bars only; needs `fetch-observe` first).",
    )
    intraday_parser.add_argument(
        "--signal",
        required=True,
        help="orb_30m_long, or random_entry (a control that must show nothing).",
    )
    intraday_parser.add_argument(
        "--cost",
        type=float,
        default=None,
        help="Round-trip cost as a fraction (default 0.001 = 0.10 %%).",
    )

    # forward-record / forward-report: the weekly signal tested forward in time (no orders)
    subparsers.add_parser(
        "forward-record",
        help="Records today's picks of the weekly signal (and its control) once the day's bar is "
        "final. Safe to run every hour; places no order. See docs/strategy-research-plan.md, "
        "section 10.",
    )
    subparsers.add_parser(
        "forward-report",
        help="Judges the recorded picks as their prices arrive (same measurement as "
        "measure-signals); a verdict only after the pre-registered minimum time.",
    )

    # run command
    run_parser = subparsers.add_parser("run", help="Runs the trading system.")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (no live broker execution).",
    )
    run_parser.add_argument("--paper", action="store_true", help="Run in paper trading mode.")
    run_parser.add_argument(
        "--pause-on-exit",
        action="store_true",
        help="When stopped, pause new entries, so the next start waits for resume-entries "
        "(the PAUSE_ON_SHUTDOWN setting does the same for every run).",
    )

    # observe command
    subparsers.add_parser(
        "observe",
        help="Runs the Oracle observer: the live strategy over the observation universe, each "
        "signal put to the Oracle, NO order ever placed (needs `fetch-observe` first).",
    )

    # pause-entries command
    subparsers.add_parser("pause-entries", help="Halts the trading system, preventing new entries.")

    # resume-entries command
    resume_parser = subparsers.add_parser(
        "resume-entries",
        help="Lifts a halt after looking at the account again (positions protected, orders "
        "accounted for).",
    )
    resume_parser.add_argument(
        "--force",
        action="store_true",
        help="Resume even if the check finds problems (you have seen them and go on).",
    )

    # reconcile command
    subparsers.add_parser(
        "reconcile",
        help="Compares the account with our records (orders, positions, stops). Exit 0 if "
        "they agree, 1 if not.",
    )

    # export-oracle-decisions command
    export_parser = subparsers.add_parser(
        "export-oracle-decisions",
        help="Writes the recorded Oracle decisions as a file the backtest can replay.",
    )
    export_parser.add_argument("--out", required=True, help="Path of the JSON file to write.")

    # oracle-probe command
    probe_parser = subparsers.add_parser(
        "oracle-probe",
        help="Asks the Oracle backend about one ARTIFICIAL signal, to prove the road (token, "
        "network, record). Places no order. Run it in the container, where the database is.",
    )
    probe_parser.add_argument(
        "--symbol", help="A symbol for the artificial signal. Default: the first of the watchlist."
    )

    # status command
    subparsers.add_parser("status", help="Prints the current status of the trading system.")

    # report command
    report_parser = subparsers.add_parser(
        "report", help="What the system did on one New York trading day (default: today)."
    )
    report_parser.add_argument("--date", help="The day, YYYY-MM-DD (New York).")

    # corporate-actions command
    corporate_parser = subparsers.add_parser(
        "corporate-actions",
        help="Splits, spin-offs, mergers and renames that hit (or are about to hit) our stocks.",
    )
    corporate_parser.add_argument(
        "--symbols",
        help="Comma-separated symbols (default: the open positions and orders and the watchlist).",
    )
    corporate_parser.add_argument(
        "--universe",
        action="store_true",
        help="Look at the whole research universe plus the watchlist instead.",
    )
    corporate_parser.add_argument(
        "--ahead",
        type=int,
        default=30,
        help="Days ahead to look (default 30; Alpaca knows ~4 weeks).",
    )
    corporate_parser.add_argument(
        "--back", type=int, default=0, help="Days back to look (default 0)."
    )

    # equity-check command
    equity_parser = subparsers.add_parser(
        "equity-check",
        help="Compares our journal of the account's equity with Alpaca's end-of-day history.",
    )
    equity_parser.add_argument(
        "--days", type=int, default=30, help="How many days of Alpaca's history (default 30)."
    )

    # health command
    subparsers.add_parser(
        "health",
        help="Exit 0 if the trading loop is doing its job, 1 if not (for Docker's healthcheck).",
    )

    # backup-db command
    backup_parser = subparsers.add_parser(
        "backup-db",
        help="Writes a checked, compressed copy of the SQLite database and removes old copies.",
    )
    backup_parser.add_argument(
        "--out", default="var/backups", help="Folder for the archives (default: var/backups)."
    )
    backup_parser.add_argument(
        "--keep", type=int, default=28, help="How many of the newest archives to keep."
    )

    return parser


LOG_FILE = "var/alpaca_trading.log"


def log_file_for(command: Optional[str]) -> Optional[str]:
    """Only the long-running loop writes the log file. A short command (a healthcheck every
    minute, `status`, `report`) that also opened it would add a handler for a few seconds to a
    file that is being rotated under it; its output goes to the console only."""
    return LOG_FILE if command == "run" else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    import logging
    from trading_system.observability.logging import setup_json_logger

    parser = create_parser()
    args = parser.parse_args(argv)

    setup_json_logger("", level=logging.INFO, log_file=log_file_for(args.command))

    if args.command == "validate-config":
        from trading_system.config_check import check_config

        config_report = check_config()
        for warning in config_report.warnings:
            print(f"WARNING: {warning}")
        if config_report.valid:
            print("Configuration is valid.")
            return 0
        print("Configuration is NOT valid:")
        for error in config_report.errors:
            print(f"  - {error}")
        return 1

    elif args.command == "doctor":
        try:
            from trading_system.config_check import unknown_key_warnings

            settings = Settings()
            for warning in unknown_key_warnings():
                print(f"WARNING: {warning}")
            print("Local configuration is valid.")

            if args.network:
                print("Running network checks...")
                try:
                    from trading_system.broker.alpaca_broker import AlpacaBroker

                    broker = AlpacaBroker(settings)
                    clock = broker.get_clock()
                    print(
                        f"Broker connection successful. Market is currently {'open' if clock.is_open else 'closed'}."
                    )
                except Exception as e:
                    print(f"Network check failed: {e}")
                    return 1
                # Advisory: an event on what we hold is a warning to read, not a broken system.
                try:
                    from datetime import datetime, timezone

                    from trading_system.data.corporate_actions import (
                        CorporateActionsFeed,
                        check_holdings,
                    )
                    from trading_system.risk.daily_equity import trading_day

                    for line in check_holdings(
                        CorporateActionsFeed(settings),
                        broker,
                        trading_day(datetime.now(timezone.utc)),
                    ):
                        print(line)
                except Exception as e:
                    print(f"Corporate actions could not be checked (advisory only): {e}")
                from trading_system.oracle.checks import ping_backend

                ping = ping_backend(settings)
                if ping is None:
                    print("Oracle: not using the backend (ORACLE_ENABLED / ORACLE_PROVIDER).")
                elif ping.ok:
                    print(
                        f"Oracle backend reachable at {settings.oracle_backend_url}; the token "
                        f"is accepted (mode: {settings.oracle_mode})."
                    )
                else:
                    print(f"Oracle backend check failed ({ping.kind}): {ping.detail}")
                    return 1
            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1

    elif args.command == "backtest":
        print("Starting offline backtest...")
        try:
            # For backtest, we don't strictly need real API keys if they are just validated.
            # We bypass the strict check by providing dummy values just for the engine init.
            import os

            os.environ["ALPACA_API_KEY"] = os.environ.get("ALPACA_API_KEY", "dummy")
            os.environ["ALPACA_SECRET_KEY"] = os.environ.get("ALPACA_SECRET_KEY", "dummy")
            settings = Settings()
            from trading_system.backtest.models import BacktestConfig
            from trading_system.backtest.engine import BacktestEngine
            from trading_system.backtest.loader import NoStoredCandles, load_candles
            from trading_system.strategy.registry import get_strategy

            try:
                strategy = get_strategy(args.strategy)
                df_15m, df_1h = load_candles(settings, args.days)
            except (ValueError, NoStoredCandles) as e:
                print(e)
                return 1
            symbols = list(df_15m)
            first = min(df["timestamp"].iloc[0] for df in df_15m.values())
            last = max(df["timestamp"].iloc[-1] for df in df_15m.values())
            print(f"Replaying {len(symbols)} symbols, {first:%Y-%m-%d} to {last:%Y-%m-%d}.")

            config = BacktestConfig(replay_oracle_path=args.replay_oracle)
            engine = BacktestEngine(settings, config, df_15m, df_1h, strategy=strategy)
            print(f"Strategy: {strategy.name} ({strategy.version})")

            result = engine.run()

            print("\n=== Backtest Report ===")
            print(f"Initial Capital: ${result.initial_capital:,.2f}")
            print(f"Final Capital:   ${result.final_capital:,.2f}")
            print(f"Total Return:    {result.total_return_pct * 100:.2f}%")
            print(f"Max Drawdown:    {result.max_drawdown_pct * 100:.2f}%")
            print(f"Signals:         {result.signals}")
            print(f"Trade Count:     {result.trade_count}")
            print(f"Win Rate:        {result.win_rate * 100:.2f}%")
            print(f"Profit Factor:   {result.profit_factor:.2f}")
            print(f"Max Exposure:    {result.max_exposure_pct * 100:.2f}%")
            print(f"Fees Paid:       ${result.total_fees:,.2f}")
            print(f"Slippage Paid:   ${result.total_slippage:,.2f} (already inside the prices)")

            days = result.daily_returns()
            if days:
                best = max(days, key=lambda d: d[1])
                worst = min(days, key=lambda d: d[1])
                down = sum(1 for _, r in days if r < 0)
                print(f"Days:            {len(days)} ({down} down)")
                print(f"Best Day:        {best[1] * 100:+.2f}% ({best[0]})")
                print(f"Worst Day:       {worst[1] * 100:+.2f}% ({worst[0]})")

            if result.not_traded:
                print("\n=== Signals that did not become a trade ===")
                for reason, count in sorted(result.not_traded.items(), key=lambda kv: -kv[1]):
                    print(f"- {reason}: {count}")

            from trading_system.backtest.analysis import (
                cost_sweep,
                format_analysis,
                format_cost_sweep,
                format_oracle_value,
                oracle_value,
            )

            print("\n" + format_analysis(result, periods=args.windows))
            value = oracle_value(result)
            if value is not None:
                print("\n" + format_oracle_value(value))
            if args.cost_sweep:
                print("\nRunning the replay again at higher costs...")
                print(
                    "\n" + format_cost_sweep(cost_sweep(settings, config, df_15m, df_1h, strategy))
                )

            print("\n=== Assumptions & Limitations ===")
            for a in result.assumptions_and_limitations:
                print(f"- {a}")

            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1
        except Exception as e:
            print(f"Backtest error: {e}")
            return 1

    elif args.command == "fetch-history":
        try:
            from datetime import datetime, timedelta, timezone

            from trading_system.data.alpaca_data import AlpacaData
            from trading_system.data.history import TIMEFRAMES, update_history
            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import MarketBarRepository

            settings = Settings()
            symbols = (
                [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
                if args.symbols
                else settings.watchlist_symbols
            )
            now = datetime.now(timezone.utc)
            repo = MarketBarRepository(get_session_factory(get_engine(settings.database_url)))
            print(f"Updating {len(symbols)} symbols, {args.days} days back...")
            report = update_history(
                AlpacaData(settings),
                repo,
                symbols,
                list(TIMEFRAMES),
                start=now - timedelta(days=args.days),
                now=now,
            )
            for name in TIMEFRAMES:
                written = sum(n for (_, tf), n in report.rows.items() if tf == name)
                again = [s for s, tf in report.downloaded_again if tf == name]
                line = f"{name}: {written:,} rows written"
                if again:
                    line += f"; prices out of date, downloaded again: {', '.join(again)}"
                print(line)
            if report.no_data:
                print(
                    "No data from Alpaca for: " + ", ".join(sorted({s for s, _ in report.no_data}))
                )
            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1
        except Exception as e:
            print(f"fetch-history error: {e}")
            return 1

    elif args.command == "fetch-observe":
        try:
            from datetime import datetime, timedelta, timezone

            from trading_system.data.alpaca_data import AlpacaData
            from trading_system.data.history import TIMEFRAMES, update_history
            from trading_system.data.universe import observation_universe
            from trading_system.storage.research import SameDatabaseError, open_observe_store

            settings = Settings()
            symbols = observation_universe(settings.watchlist_symbols)
            try:
                repo = open_observe_store(settings.observe_database_url, settings.database_url)
            except SameDatabaseError as e:
                print(f"fetch-observe error: {e}")
                return 1
            now = datetime.now(timezone.utc)
            print(f"Updating {len(symbols)} symbols, {args.days} days back...")
            report = update_history(
                AlpacaData(settings),
                repo,
                symbols,
                list(TIMEFRAMES),
                start=now - timedelta(days=args.days),
                now=now,
            )
            for name in TIMEFRAMES:
                written = sum(n for (_, tf), n in report.rows.items() if tf == name)
                print(f"{name}: {written:,} rows written")
            if report.no_data:
                print(
                    "No data from Alpaca for: " + ", ".join(sorted({s for s, _ in report.no_data}))
                )
            return 0
        except Exception as e:
            print(f"fetch-observe error: {e}")
            return 1

    elif args.command == "fetch-daily":
        try:
            from datetime import datetime, timedelta, timezone

            from trading_system.data.alpaca_data import AlpacaData
            from trading_system.data.history import (
                RESEARCH_TIMEFRAMES,
                stale_symbols,
                update_history,
            )
            from trading_system.data.universe import MARKET_SYMBOL, research_universe
            from trading_system.storage.research import SameDatabaseError, open_research_store

            settings = Settings()
            symbols = (
                [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
                if args.symbols
                else research_universe()
            )
            since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            try:
                repo = open_research_store(settings.research_database_url, settings.database_url)
            except SameDatabaseError as e:
                print(f"fetch-daily error: {e}")
                return 1
            print(f"Updating {len(symbols)} symbols, daily bars from {args.since}...")
            report = update_history(
                AlpacaData(settings),
                repo,
                symbols,
                list(RESEARCH_TIMEFRAMES),
                start=since,
                now=datetime.now(timezone.utc),
            )
            print(f"1d: {report.rows_written:,} rows written")
            if report.downloaded_again:
                again = sorted({s for s, _ in report.downloaded_again})
                print(
                    "Downloaded again from the start (prices re-adjusted, or the history is "
                    "shorter than asked for): " + ", ".join(again)
                )
            if report.no_data:
                print(
                    "No data from Alpaca for: " + ", ".join(sorted({s for s, _ in report.no_data}))
                )
            stale = [
                (s, last)
                for s, last in stale_symbols(repo, symbols, "1d", tolerance=timedelta(days=5))
                if (s, "1d") not in report.no_data
            ]
            if stale:
                print(
                    "WARNING, the newest bar is old (renamed, delisted or halted?): "
                    + ", ".join(f"{s} ({last:%Y-%m-%d})" for s, last in stale if last is not None)
                )
            market = repo.span(MARKET_SYMBOL, "1d")
            if market is not None:
                print(f"{MARKET_SYMBOL}: {market[0]:%Y-%m-%d} to {market[1]:%Y-%m-%d}")
            return 0
        except Exception as e:
            print(f"fetch-daily error: {e}")
            return 1

    elif args.command == "fetch-news":
        try:
            from datetime import datetime, timezone

            from trading_system.data.news import NewsFeed, update_news
            from trading_system.data.universe import research_universe
            from trading_system.storage.news import SameNewsDatabaseError, open_news_store

            settings = Settings()
            news_symbols = (
                [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
                if args.symbols
                else sorted(set(research_universe()) | set(settings.watchlist_symbols))
            )
            try:
                store = open_news_store(
                    settings.news_database_url,
                    settings.database_url,
                    settings.research_database_url,
                )
            except SameNewsDatabaseError as e:
                print(f"fetch-news error: {e}")
                return 1
            now = datetime.now(timezone.utc)
            news_report = update_news(
                NewsFeed(settings), store, news_symbols, now=now, days=args.days
            )
            newest = store.latest_created_at()
            print(
                f"News for {len(news_symbols)} symbols: {news_report.new} new, "
                f"{news_report.updated} changed, {news_report.unchanged} already stored "
                f"({news_report.windows} request window(s)); {store.count():,} articles in the "
                f"cache, newest published {newest:%Y-%m-%d %H:%M} UTC"
                if newest
                else "No news came back."
            )
            if news_report.full_windows:
                print(
                    "WARNING, a window reached the limit, some articles may be missing: "
                    + ", ".join(f"{w:%Y-%m-%d %H:%M}" for w in news_report.full_windows)
                )
            return 0
        except Exception as e:
            print(f"fetch-news error: {e}")
            return 1

    elif args.command == "measure-signals":
        try:
            from trading_system.research.measure import (
                MIN_UNIVERSE,
                ROUND_TRIP_COST,
                development_note,
                format_measurement,
                measure,
            )
            from trading_system.research.panel import big_moves, load_panel
            from trading_system.research.signals import SIGNALS, with_regime_filter
            from trading_system.research.trials import record_trial, trial_counts
            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import AuditRepository
            from trading_system.storage.research import SameDatabaseError, open_research_store

            settings = Settings()
            if args.signal not in SIGNALS:
                print(
                    f"measure-signals error: unknown signal {args.signal!r}; "
                    f"known: {', '.join(sorted(SIGNALS))}"
                )
                return 1
            signal = SIGNALS[args.signal]
            horizons = (
                tuple(int(h) for h in args.horizons.split(",") if h.strip())
                if args.horizons
                else signal.horizons
            )
            cost = ROUND_TRIP_COST if args.cost is None else args.cost
            only_symbols = (
                [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
                if args.symbols
                else None
            )
            try:
                repo = open_research_store(settings.research_database_url, settings.database_url)
            except SameDatabaseError as e:
                print(f"measure-signals error: {e}")
                return 1
            panel = load_panel(repo, only_symbols)
            if len(panel.symbols) < MIN_UNIVERSE:
                print(
                    f"measure-signals error: only {len(panel.symbols)} stocks have daily bars "
                    f"(need {MIN_UNIVERSE}); run `fetch-daily` first"
                )
                return 1
            print(development_note())
            print(
                f"{len(panel.symbols)} stocks, {panel.dates[0].date()} to {panel.dates[-1].date()}"
            )
            moves = big_moves(panel)
            if moves:
                print(
                    f"Daily moves beyond 30% still in the data ({len(moves)}): "
                    + ", ".join(f"{s} {d} {m * 100:+.0f}%" for s, d, m in moves[:6])
                )
            on = signal.build(panel)
            if args.regime_filter:
                on = with_regime_filter(on, panel, args.regime_filter)
            audit = AuditRepository(get_session_factory(get_engine(settings.database_url)))
            for horizon in horizons:
                measurement = measure(panel, on, signal=signal.name, horizon=horizon, cost=cost)
                print()
                print(format_measurement(measurement))
                if not signal.control:
                    record_trial(
                        audit,
                        measurement,
                        regime_filter=args.regime_filter,
                        universe_size=len(panel.symbols),
                    )
            runs, distinct = trial_counts(audit)
            print()
            print(
                "Control, not counted as a variant."
                if signal.control
                else f"Variants tried so far: {distinct} ({runs} runs)."
            )
            return 0
        except Exception as e:
            print(f"measure-signals error: {e}")
            return 1

    elif args.command == "judge-oracle":
        try:
            from datetime import datetime, timezone

            from trading_system.oracle.judge import format_judgement, judge
            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import (
                AuditRepository,
                OracleDecisionRepository,
            )
            from trading_system.storage.research import SameDatabaseError, open_observe_store

            settings = Settings()
            factory = get_session_factory(get_engine(settings.database_url))
            try:
                candle_store = open_observe_store(
                    settings.observe_database_url, settings.database_url
                )
            except SameDatabaseError as e:
                print(f"judge-oracle error: {e}")
                return 1
            judgement = judge(
                factory,
                OracleDecisionRepository(factory),
                candle_store,
                AuditRepository(factory),
                settings,
                datetime.now(timezone.utc),
            )
            print(format_judgement(judgement))
            return 0
        except Exception as e:
            print(f"judge-oracle error: {type(e).__name__}: {e}")
            return 1

    elif args.command == "judge-candidates":
        try:
            from datetime import datetime, timezone

            from trading_system.research.candidates import (
                format_candidates,
                judge_candidates,
            )
            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import AuditRepository
            from trading_system.storage.research import SameDatabaseError, open_observe_store

            settings = Settings()
            factory = get_session_factory(get_engine(settings.database_url))
            try:
                candle_store = open_observe_store(
                    settings.observe_database_url, settings.database_url
                )
            except SameDatabaseError as e:
                print(f"judge-candidates error: {e}")
                return 1
            print(
                format_candidates(
                    judge_candidates(
                        AuditRepository(factory),
                        candle_store,
                        settings,
                        datetime.now(timezone.utc),
                    )
                )
            )
            return 0
        except Exception as e:
            print(f"judge-candidates error: {type(e).__name__}: {e}")
            return 1

    elif args.command == "measure-intraday":
        try:
            import numpy as np

            from trading_system.data.universe import observation_universe
            from trading_system.research.intraday import (
                BUILDERS,
                CONTROLS,
                development_note as intraday_note,
                format_intraday,
                load_bars,
                measure_intraday,
                record_intraday_trial,
            )
            from trading_system.research.measure import MIN_UNIVERSE, ROUND_TRIP_COST
            from trading_system.research.trials import trial_counts
            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import AuditRepository
            from trading_system.storage.research import SameDatabaseError, open_observe_store

            settings = Settings()
            if args.signal not in BUILDERS:
                print(
                    f"measure-intraday error: unknown signal {args.signal!r}; "
                    f"known: {', '.join(sorted(BUILDERS))}"
                )
                return 1
            try:
                repo = open_observe_store(settings.observe_database_url, settings.database_url)
            except SameDatabaseError as e:
                print(f"measure-intraday error: {e}")
                return 1
            bars = load_bars(repo, observation_universe(settings.watchlist_symbols))
            stocks = int(np.isfinite(bars.close).any(axis=(0, 1)).sum())
            if stocks < MIN_UNIVERSE:
                print(
                    f"measure-intraday error: only {stocks} stocks have 15-minute bars "
                    f"(need {MIN_UNIVERSE}); run `fetch-observe` first"
                )
                return 1
            print(intraday_note())
            print(f"{stocks} stocks, {bars.days[0].date()} to {bars.days[-1].date()}")
            m = measure_intraday(
                bars, args.signal, cost=ROUND_TRIP_COST if args.cost is None else args.cost
            )
            print()
            print(format_intraday(m))
            audit = AuditRepository(get_session_factory(get_engine(settings.database_url)))
            if args.signal not in CONTROLS:
                record_intraday_trial(audit, m, universe_size=stocks)
            runs, distinct = trial_counts(audit)
            print()
            print(
                "Control, not counted as a variant."
                if args.signal in CONTROLS
                else f"Variants tried so far: {distinct} ({runs} runs)."
            )
            return 0
        except Exception as e:
            print(f"measure-intraday error: {e}")
            return 1

    elif args.command in ("forward-record", "forward-report"):
        try:
            from datetime import datetime, timezone

            from trading_system.research.forward import evaluate, format_report, record_picks
            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import AuditRepository
            from trading_system.storage.research import SameDatabaseError, open_research_store

            settings = Settings()
            try:
                repo = open_research_store(settings.research_database_url, settings.database_url)
            except SameDatabaseError as e:
                print(f"{args.command} error: {e}")
                return 1
            audit = AuditRepository(get_session_factory(get_engine(settings.database_url)))
            now = datetime.now(timezone.utc)
            if args.command == "forward-record":
                print(record_picks(audit, repo, now).line())
            else:
                print(format_report(evaluate(audit, repo, now)))
            return 0
        except Exception as e:
            print(f"{args.command} error: {type(e).__name__}: {e}")
            return 1

    elif args.command == "run":
        is_paper = getattr(args, "paper", False)
        mode = "Paper" if is_paper else "Dry-Run"
        print(f"Starting Orchestrator in {mode} mode...")
        try:
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()

            # Use appropriate repo logic here
            orchestrator = Orchestrator(settings)

            # Simple simulation for CLI demo
            print("Orchestrator running. (Press Ctrl+C to stop)")
            # None leaves it to the PAUSE_ON_SHUTDOWN setting; the flag can only add a pause.
            orchestrator.run(is_paper, pause_on_shutdown=True if args.pause_on_exit else None)
            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1
        except KeyboardInterrupt:
            print("Shutting down Orchestrator gracefully...")
            return 0
        except Exception as e:
            print(f"Run error: {e}")
            return 1

    elif args.command == "observe":
        print("Starting the Oracle observer (no orders)...")
        try:
            from trading_system.runtime.orchestrator import Orchestrator

            Orchestrator(Settings()).run_observation()
            return 0
        except ValidationError as e:
            print(f"Configuration validation failed: {e}")
            return 1
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            print(f"Observe error: {type(e).__name__}: {e}")
            return 1

    elif args.command == "pause-entries":
        print("Pausing entries...")
        try:
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()
            orchestrator = Orchestrator(settings)
            orchestrator.pause_entries()
            return 0
        except Exception as e:
            print(f"Pause error: {e}")
            return 1

    elif args.command == "reconcile":
        print("Comparing the account with our records...")
        try:
            from trading_system.runtime.account_check import format_account_check
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()
            check = Orchestrator(settings).reconcile_now()
            print(format_account_check(check))
            return 0 if check.ok else 1
        except Exception as e:
            print(f"Reconcile error: {e}")
            return 1

    elif args.command == "resume-entries":
        print("Resuming entries...")
        try:
            from trading_system.runtime.account_check import format_account_check
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()
            orchestrator = Orchestrator(settings)
            resumed, looked = orchestrator.resume_entries_checked(force=args.force)
            if looked is None:
                print("Entries are not halted; nothing to resume.")
                return 0
            print(format_account_check(looked))
            if resumed:
                print(
                    "Entries resumed."
                    + ("" if looked.ok else " (Forced: the problems above stand.)")
                )
                return 0
            print(
                "Entries stay halted. Put the problems right and run this again, or use "
                "--force if you have seen them and want to go on."
            )
            return 1
        except Exception as e:
            print(f"Resume error: {e}")
            return 1

    elif args.command == "export-oracle-decisions":
        try:
            import json

            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import OracleDecisionRepository

            settings = Settings()
            repository = OracleDecisionRepository(
                get_session_factory(get_engine(settings.database_url))
            )
            decisions = repository.replay_export()
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(decisions, f, indent=2, sort_keys=True)
            decided = sum(1 for d in decisions.values() if d["final"])
            print(
                f"Wrote {len(decisions)} signals to {args.out} "
                f"({decided} decided, {len(decisions) - decided} undecided: no trade)."
            )
            return 0
        except Exception as e:
            print(f"Export error: {e}")
            return 1

    elif args.command == "oracle-probe":
        try:
            from trading_system.oracle.checks import run_probe
            from trading_system.storage.database import get_engine, get_session_factory

            inject_offline_dummy_keys()
            settings = Settings()
            symbol = (args.symbol or (settings.watchlist_symbols or ["AAPL"])[0]).upper()
            probe = run_probe(
                settings, get_session_factory(get_engine(settings.database_url)), symbol
            )
        except Exception as e:
            print(f"Probe error: {type(e).__name__}: {e}")
            return 1
        for line in probe.lines:
            print(line)
        return 0 if probe.ok else 1

    elif args.command == "status":
        print("Fetching system status...")
        try:
            from trading_system.runtime.orchestrator import Orchestrator

            settings = Settings()
            orchestrator = Orchestrator(settings)
            orchestrator.status()
            return 0
        except Exception as e:
            print(f"Status error: {e}")
            return 1

    elif args.command == "compare-strategies":
        print("Comparing strategies on the stored candles...")
        try:
            import os

            os.environ["ALPACA_API_KEY"] = os.environ.get("ALPACA_API_KEY", "dummy")
            os.environ["ALPACA_SECRET_KEY"] = os.environ.get("ALPACA_SECRET_KEY", "dummy")
            settings = Settings()
            from trading_system.backtest.comparison import compare_strategies, format_comparison
            from trading_system.backtest.loader import NoStoredCandles, load_candles
            from trading_system.backtest.models import BacktestConfig
            from trading_system.strategy.registry import strategy_names

            names = (
                [n.strip() for n in args.strategies.split(",") if n.strip()]
                if args.strategies
                else strategy_names()
            )
            unknown = [n for n in names if n not in strategy_names()]
            if unknown:
                print(
                    f"unknown strategy {', '.join(unknown)}; "
                    f"the choices are: {', '.join(strategy_names())}"
                )
                return 1
            try:
                df_15m, df_1h = load_candles(settings, args.days)
                config = BacktestConfig()
                rows = compare_strategies(settings, config, df_15m, df_1h, names)
            except (ValueError, NoStoredCandles) as e:
                print(e)
                return 1
            first = min(df["timestamp"].iloc[0] for df in df_15m.values())
            last = max(df["timestamp"].iloc[-1] for df in df_15m.values())
            print(f"{len(df_15m)} symbols, {first:%Y-%m-%d} to {last:%Y-%m-%d}.\n")
            print(format_comparison(rows))
            print(
                "\nSame data, same risk engine and entry plan, same costs "
                f"(fee {config.fee_pct_per_trade * 100:.2f}%, "
                f"slippage {config.slippage_pct * 100:.2f}% a side)."
                "\nReturn, win rate and costs are for the whole replay; PF is the profit factor; "
                "t is the average trade over its standard error (below 2 in absolute value the "
                "average cannot be told from luck)."
                "\nWith more signals than room (the risk engine's position limit) the symbols "
                "first in the watchlist get the places: a signal count is not a trade count."
                "\nA few dozen trades in one market regime cannot tell a good strategy from a "
                "lucky one: read the differences with that in mind."
            )
            return 0
        except ValidationError as e:
            print(f"Configuration validation failed:\n{e}")
            return 1
        except Exception as e:
            print(f"compare-strategies error: {e}")
            return 1

    elif args.command == "report":
        try:
            from datetime import date, datetime, timezone

            from trading_system.observability.report import (
                build_daily_report,
                format_daily_report,
            )
            from trading_system.risk.daily_equity import trading_day
            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import StateRepository

            inject_offline_dummy_keys()
            settings = Settings()
            day = (
                date.fromisoformat(args.date)
                if args.date
                else trading_day(datetime.now(timezone.utc))
            )
            factory = get_session_factory(get_engine(settings.database_url))
            print(
                format_daily_report(
                    build_daily_report(factory, day, StateRepository(factory).get_state())
                )
            )
            return 0
        except ValueError as e:
            print(f"Report error: {e} (the date is YYYY-MM-DD)")
            return 1
        except Exception as e:
            print(f"Report error: {e}")
            return 1

    elif args.command == "corporate-actions":
        try:
            from datetime import datetime, timedelta, timezone

            from trading_system.broker.alpaca_broker import AlpacaBroker
            from trading_system.data.corporate_actions import (
                CorporateActionsFeed,
                format_events,
                held_symbols,
            )
            from trading_system.data.universe import research_universe
            from trading_system.risk.daily_equity import trading_day

            if args.ahead < 0 or args.back < 0:
                print("corporate-actions error: --ahead and --back cannot be negative")
                return 1
            settings = Settings()
            if args.symbols:
                names = sorted({s.strip().upper() for s in args.symbols.split(",") if s.strip()})
            elif args.universe:
                names = sorted(set(research_universe()) | set(settings.watchlist_symbols))
            else:
                names = sorted(
                    set(held_symbols(AlpacaBroker(settings))) | set(settings.watchlist_symbols)
                )
            today = trading_day(datetime.now(timezone.utc))
            first, last = today - timedelta(days=args.back), today + timedelta(days=args.ahead)
            found = CorporateActionsFeed(settings).events(names, first, last)
            print(f"Corporate actions for {len(names)} symbol(s), {first} to {last}:")
            print(format_events(found) if found else "  nothing found.")
            print(
                "Alpaca lists an event only a few weeks before it happens, so 'nothing found' is "
                "not a promise that nothing will happen. Dividends are left out (the prices are "
                "adjusted for them)."
            )
            return 0
        except Exception as e:
            print(f"corporate-actions error: {e}")
            return 1

    elif args.command == "equity-check":
        try:
            from trading_system.broker.alpaca_broker import AlpacaBroker
            from trading_system.observability.equity_check import (
                alpaca_equity_by_day,
                compare_equity,
                format_equity_check,
                journal_equity_by_day,
            )
            from trading_system.storage.database import get_engine, get_session_factory
            from trading_system.storage.repositories import AuditRepository

            if args.days < 1:
                print("equity-check error: --days must be at least 1")
                return 1
            settings = Settings()
            alpaca_days = alpaca_equity_by_day(
                AlpacaBroker(settings).get_portfolio_history(args.days)
            )
            if not alpaca_days:
                print("Alpaca returned no equity history for the period.")
                return 1
            journal = AuditRepository(get_session_factory(get_engine(settings.database_url)))
            equity_check = compare_equity(
                alpaca_days, journal_equity_by_day(journal, min(alpaca_days), max(alpaca_days))
            )
            print(format_equity_check(equity_check))
            return 0 if equity_check.ok else 1
        except Exception as e:
            print(f"equity-check error: {e}")
            return 1

    elif args.command == "health":
        try:
            from datetime import datetime, timezone
            from pathlib import Path

            from trading_system.observability.health import evaluate_heartbeat
            from trading_system.runtime.heartbeat import read_heartbeat

            from trading_system.oracle.checks import health_findings

            inject_offline_dummy_keys()
            settings = Settings()
            verdict = evaluate_heartbeat(
                read_heartbeat(Path(settings.heartbeat_path)), datetime.now(timezone.utc)
            )
            oracle_problems, oracle_warnings = health_findings(settings)
        except Exception as e:
            print(f"UNHEALTHY: could not judge the heartbeat: {e}")
            return 1
        problems = list(verdict.reasons) + oracle_problems
        if verdict.healthy and not oracle_problems:
            print("healthy" + "".join(f" (warning: {w})" for w in oracle_warnings))
            return 0
        print("UNHEALTHY: " + "; ".join(problems))
        return 1

    elif args.command == "backup-db":
        try:
            from pathlib import Path

            from trading_system.storage.backup import BackupError, backup_database

            # Only the database location is needed: no Alpaca keys, so the container that
            # takes the backups is not given any.
            inject_offline_dummy_keys()
            settings = Settings()
            try:
                archive = backup_database(settings.database_url, Path(args.out), keep=args.keep)
            except BackupError as e:
                print(f"Backup FAILED: {e}")
                return 1
            print(
                f"Backup written: {archive.path} ({archive.size_bytes / 1e6:.1f} MB); "
                f"{archive.kept} kept, {len(archive.removed)} old removed."
            )
            return 0
        except Exception as e:
            print(f"Backup FAILED: {e}")
            return 1

    # Normally we would start the app here
    print("Alpaca Roundtable Trader MVP Skeleton")
    if not args.command:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
