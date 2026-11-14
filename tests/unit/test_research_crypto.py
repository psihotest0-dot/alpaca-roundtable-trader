"""The crypto research (plan section 15): the universe rule, the panel, the signal, and the
measurement with the crypto rules. Nothing here is the stock market's."""

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_system.data.crypto import (
    CRYPTO_UNIVERSE,
    MARKET_COIN,
    CryptoDailyFeed,
    closed_only,
    eligible_pairs,
    update_crypto_daily,
)
from trading_system.research.crypto import (
    CRYPTO_RULES,
    CRYPTO_SIGNALS,
    TREND_DAYS,
    build_crypto_panel,
    crypto_trend_30,
    frames_from_stored,
    load_crypto_panel,
    random_half,
    zero_volume_share,
)
from trading_system.research.measure import STOCK_RULES, measure
from trading_system.research.panel import HOLDOUT_START
from trading_system.research.trials import CRYPTO, STOCKS, record_trial, trial_counts
from trading_system.storage.database import get_engine, get_session_factory
from trading_system.storage.repositories import AuditRepository
from trading_system.storage.research import SameDatabaseError, open_crypto_store

# ------------------------------------------------------------------ the universe rule (15.2)

# The history counts the probe of 2026-11-14 printed: (bars since 2021-01-01, zero-volume bars).
# Only the pairs that matter for the rule are listed.
PROBE_DAYS = 2143
PROBE = {
    "BAT/USD": (2143, 52),
    "BCH/USD": (2143, 15),
    "BTC/USD": (2143, 0),
    "DOGE/USD": (2143, 2),
    "ETH/USD": (2143, 0),
    "LINK/USD": (2143, 5),
    "LTC/USD": (2143, 5),
    "SOL/USD": (1725, 11),  # gaps: 80 % of the days
    "UNI/USD": (2143, 126),  # 5.9 % of the bars have no volume
    "XRP/USD": (1048, 0),  # starts in 2024
    "AVAX/USD": (1822, 33),
    "USDT/USD": (2142, 38),  # pegged: left out whatever its history
    "USDC/USD": (1180, 213),
    "PAXG/USD": (1172, 8),
}


def test_the_rule_keeps_exactly_the_seven_coins_that_were_written_down() -> None:
    assert eligible_pairs(PROBE, PROBE_DAYS) == sorted(CRYPTO_UNIVERSE)
    assert len(CRYPTO_UNIVERSE) == 7 and MARKET_COIN in CRYPTO_UNIVERSE


def test_the_rule_drops_a_pegged_coin_and_a_coin_with_gaps_or_no_volume() -> None:
    kept = eligible_pairs(PROBE, PROBE_DAYS)

    assert "USDT/USD" not in kept  # the history is fine, but it is pegged
    assert "SOL/USD" not in kept  # a bar on 80 % of the days
    assert "UNI/USD" not in kept  # 5.9 % zero-volume bars is above the 5 % ceiling
    assert "BAT/USD" in kept  # 2.4 % is below it


# ------------------------------------------------------------------------------- the panel


def _frames(days: int = 120, seed: int = 1) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=days, freq="D")
    frames = {}
    for symbol in CRYPTO_UNIVERSE:
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, days)))
        opens = np.concatenate([[100.0], closes[:-1]])
        frames[symbol] = pd.DataFrame(
            {"open": opens, "close": closes, "volume": np.full(days, 1000.0)}, index=index
        )
    return frames


def test_a_zero_volume_bar_is_missing_and_so_is_every_window_that_touches_it() -> None:
    frames = _frames()
    frames["BAT/USD"].loc[frames["BAT/USD"].index[50], "volume"] = 0.0

    panel = build_crypto_panel(frames)

    assert np.isnan(panel.close["BAT/USD"].iloc[50])
    assert np.isnan(panel.log_return["BAT/USD"].iloc[50])
    assert np.isnan(
        panel.log_return["BAT/USD"].iloc[51]
    )  # the return INTO the next day is gone too
    assert panel.close["BTC/USD"].notna().all()
    # a 10-day window that contains the day is missing, one that does not is not
    from trading_system.research.measure import forward_returns

    forward, _ = forward_returns(panel, 10)
    assert np.isnan(forward["BAT/USD"].iloc[45]) and np.isnan(forward["BAT/USD"].iloc[50])
    assert np.isfinite(forward["BAT/USD"].iloc[30]) and np.isfinite(forward["BAT/USD"].iloc[60])


def test_a_calendar_with_a_missing_day_is_refused() -> None:
    frames = _frames()
    frames[MARKET_COIN] = frames[MARKET_COIN].drop(frames[MARKET_COIN].index[40])

    with pytest.raises(ValueError, match="missing day"):
        build_crypto_panel(frames)


def test_the_panel_needs_btc() -> None:
    frames = _frames()
    del frames[MARKET_COIN]

    with pytest.raises(ValueError, match="BTC/USD"):
        build_crypto_panel(frames)


def test_zero_volume_share_counts_placeholders() -> None:
    frames = _frames(days=100)
    frames["ETH/USD"].loc[frames["ETH/USD"].index[:7], "volume"] = 0.0

    assert zero_volume_share(frames) == pytest.approx(7 / (100 * len(frames)))


# -------------------------------------------------------------------------------- the signal


def test_the_signal_is_on_when_the_close_is_above_the_close_thirty_days_earlier() -> None:
    frames = _frames(days=90)
    btc = frames[MARKET_COIN]
    btc["close"] = np.concatenate([np.full(45, 100.0), np.full(45, 120.0)])  # up at day 45
    panel = build_crypto_panel(frames)

    on = crypto_trend_30(panel)[MARKET_COIN]

    assert not on.iloc[:TREND_DAYS].any()  # no close 30 days earlier yet: off
    assert not on.iloc[44]  # still flat against 30 days before
    assert on.iloc[45] and on.iloc[60]  # 120 against 100 thirty days before
    assert not on.iloc[80]  # 120 against 120: not above


def test_a_missing_bar_on_either_date_means_off() -> None:
    frames = _frames(days=90)
    frames["LTC/USD"].loc[frames["LTC/USD"].index[40], "volume"] = 0.0
    panel = build_crypto_panel(frames)

    on = crypto_trend_30(panel)["LTC/USD"]

    assert not on.iloc[40]  # no close that day
    assert not on.iloc[70]  # the close 30 days earlier is the missing one


def test_the_signal_never_looks_ahead() -> None:
    """Change every price after day 80: no signal of day 80 or earlier moves."""
    frames = _frames(days=140, seed=4)
    before = crypto_trend_30(build_crypto_panel(frames)).iloc[:81].to_numpy()

    changed = {s: f.copy() for s, f in frames.items()}
    for frame in changed.values():
        frame.iloc[81:, frame.columns.get_loc("close")] *= 5.0
        frame.iloc[81:, frame.columns.get_loc("open")] *= 5.0
    after = crypto_trend_30(build_crypto_panel(changed)).iloc[:81].to_numpy()

    assert (before == after).all()


def test_the_control_is_about_half_and_has_the_shape_of_the_panel() -> None:
    panel = build_crypto_panel(_frames(days=400, seed=2))

    on = random_half(panel)

    assert on.shape == panel.close.shape
    assert 0.45 < on.to_numpy().mean() < 0.55
    assert CRYPTO_SIGNALS["random_half"].control and not CRYPTO_SIGNALS["crypto_trend_30"].control


# ---------------------------------------------------------------- the measurement, crypto rules


def _planted_panel(bonus: float, seed: int = 7, days: int = 1500) -> tuple[Any, pd.DataFrame]:
    """Noise, a random half of the coins on each day, and `bonus` added to their NEXT day's
    return (it opens where the last day closed)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=days, freq="D")
    coins = list(CRYPTO_UNIVERSE)
    returns = pd.DataFrame(rng.normal(0, 0.03, (days, len(coins))), index=index, columns=coins)
    draw = np.random.default_rng(seed + 1).random(returns.shape) < 0.5
    on = pd.DataFrame(draw, index=index, columns=coins)
    returns = returns + on.shift(1, fill_value=False).astype(float) * bonus
    closes = 100.0 * np.exp(returns.cumsum())
    opens = closes.shift(1).fillna(100.0)
    frames = {
        c: pd.DataFrame({"open": opens[c], "close": closes[c], "volume": 1000.0}) for c in coins
    }
    return build_crypto_panel(frames), on


def test_a_planted_effect_is_found_with_the_crypto_rules() -> None:
    panel, on = _planted_panel(bonus=0.01)

    m = measure(panel, on, signal="planted", horizon=1, cost=0.0, rules=CRYPTO_RULES)

    assert m.vs_universe.t > 5
    assert m.rules is CRYPTO_RULES and m.vs_universe.label == "vs all coins"
    assert m.vs_market.label == "vs BTC"


def test_noise_does_not_pass_with_the_crypto_rules() -> None:
    worst = 0.0
    for seed in range(6):
        panel, on = _planted_panel(bonus=0.0, seed=seed)
        for horizon in (10, 20):
            m = measure(panel, on, signal="noise", horizon=horizon, cost=0.005, rules=CRYPTO_RULES)
            worst = max(worst, abs(m.vs_universe.t))
            assert not m.criteria.all_met
    assert worst < 3.5


def test_a_day_on_which_every_coin_is_on_is_not_an_observation() -> None:
    panel, _ = _planted_panel(bonus=0.0, days=300)
    everything = pd.DataFrame(True, index=panel.close.index, columns=panel.close.columns)

    crypto = measure(panel, everything, signal="all", horizon=1, rules=CRYPTO_RULES)
    stocks_like = measure(
        panel,
        everything,
        signal="all",
        horizon=1,
        rules=CRYPTO_RULES.__class__(**{**CRYPTO_RULES.__dict__, "need_one_off": False}),
    )

    assert crypto.vs_universe.dates == 0 and not crypto.usable
    assert stocks_like.vs_universe.dates > 200  # without the rule, those days count as zeros


def test_the_event_minimum_of_the_crypto_rules_is_300_and_the_stocks_keep_1000() -> None:
    assert CRYPTO_RULES.min_events == 300 and STOCK_RULES.min_events == 1000
    assert CRYPTO_RULES.min_signal == 2 and CRYPTO_RULES.min_universe == 6


# ---------------------------------------------------------------------- store and hold-out


def _stamp(frame: pd.DataFrame) -> pd.DataFrame:
    stamped = frame.copy()
    stamped.index = pd.DatetimeIndex(stamped.index).tz_localize("UTC").rename("timestamp")
    stamped["high"] = stamped[["open", "close"]].max(axis=1) * 1.001
    stamped["low"] = stamped[["open", "close"]].min(axis=1) * 0.999
    return stamped


def test_the_hold_out_is_never_loaded(tmp_path: Any) -> None:
    url = f"sqlite:///{(tmp_path / 'crypto.db').as_posix()}"
    store = open_crypto_store(url, f"sqlite:///{(tmp_path / 'trading.db').as_posix()}")
    now = datetime(2026, 11, 14, tzinfo=timezone.utc)
    for symbol, frame in _frames(days=2000).items():  # 2021-01-01 .. 2026-06: past the hold-out
        store.replace(symbol, "1d", _stamp(frame), adjustment="none", fetched_at=now)

    panel = load_crypto_panel(store)

    assert panel.dates[-1].date() < HOLDOUT_START
    assert panel.dates[-1] >= pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=2)
    with pytest.raises(ValueError, match="hold-out"):
        load_crypto_panel(store, end=datetime(2026, 7, 1).date())


def test_utc_days_are_kept_as_they_are(tmp_path: Any) -> None:
    url = f"sqlite:///{(tmp_path / 'crypto.db').as_posix()}"
    store = open_crypto_store(url, f"sqlite:///{(tmp_path / 'trading.db').as_posix()}")
    now = datetime(2026, 11, 14, tzinfo=timezone.utc)
    for symbol, frame in _frames(days=10).items():
        store.replace(symbol, "1d", _stamp(frame), adjustment="none", fetched_at=now)

    stored = store.load(list(CRYPTO_UNIVERSE), "1d")
    frames = frames_from_stored(stored)

    assert frames[MARKET_COIN].index[0] == pd.Timestamp("2021-01-01")  # not shifted to New York


def test_the_crypto_file_may_not_be_the_trading_or_the_research_file(tmp_path: Any) -> None:
    trading = f"sqlite:///{(tmp_path / 'trading.db').as_posix()}"
    research = f"sqlite:///{(tmp_path / 'research.db').as_posix()}"

    with pytest.raises(SameDatabaseError):
        open_crypto_store(trading, trading)
    with pytest.raises(SameDatabaseError, match="stock measurements"):
        open_crypto_store(research, trading, research)
    open_crypto_store(f"sqlite:///{(tmp_path / 'crypto.db').as_posix()}", trading, research)


# --------------------------------------------------------------------- counter C is apart


def test_the_crypto_variants_are_counted_apart_from_the_stock_variants(tmp_path: Any) -> None:
    engine = get_engine(f"sqlite:///{(tmp_path / 'trading.db').as_posix()}")
    from trading_system.storage.models import Base

    Base.metadata.create_all(engine)
    audit = AuditRepository(get_session_factory(engine))
    panel, on = _planted_panel(bonus=0.0, days=300)
    m = measure(panel, on, signal="crypto_trend_30", horizon=10, rules=CRYPTO_RULES)

    record_trial(audit, m, regime_filter=None, universe_size=7, market=CRYPTO)
    record_trial(audit, m, regime_filter=None, universe_size=7, market=CRYPTO)  # a re-run
    m20 = measure(panel, on, signal="crypto_trend_30", horizon=20, rules=CRYPTO_RULES)
    record_trial(audit, m20, regime_filter=None, universe_size=7, market=CRYPTO)

    assert trial_counts(audit, CRYPTO) == (3, 2)
    assert trial_counts(audit, STOCKS) == (0, 0)  # the stock count did not move
    assert trial_counts(audit) == (0, 0)  # and it is the default


# ------------------------------------------------------------------------------ the download


class _FakeClient:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def get_crypto_bars(self, request: Any) -> Any:
        class Response:
            df = self._frame

        return Response()


def _alpaca_shaped(days: int) -> pd.DataFrame:
    parts = []
    for symbol, frame in _frames(days=days).items():
        stamped = _stamp(frame)
        stamped["trade_count"] = 10.0
        stamped["vwap"] = stamped["close"]
        stamped["symbol"] = symbol
        parts.append(stamped.reset_index().set_index(["symbol", "timestamp"]))
    return pd.concat(parts)


def test_only_closed_days_are_stored_and_the_history_is_replaced(tmp_path: Any) -> None:
    url = f"sqlite:///{(tmp_path / 'crypto.db').as_posix()}"
    store = open_crypto_store(url, f"sqlite:///{(tmp_path / 'trading.db').as_posix()}")
    feed = CryptoDailyFeed(_FakeClient(_alpaca_shaped(10)))  # 2021-01-01 .. 2021-01-10
    noon = datetime(2021, 1, 10, 12, 0, tzinfo=timezone.utc)  # the 10th is still going on

    first = update_crypto_daily(feed, store, list(CRYPTO_UNIVERSE), now=noon)
    again = update_crypto_daily(feed, store, list(CRYPTO_UNIVERSE), now=noon)

    assert first.rows_written == again.rows_written == 9 * len(CRYPTO_UNIVERSE)
    stored = store.load([MARKET_COIN], "1d")[MARKET_COIN]
    assert len(stored) == 9 and stored.index[-1] == pd.Timestamp("2021-01-09", tz="UTC")


def test_a_pair_with_no_bars_is_reported_not_silent(tmp_path: Any) -> None:
    url = f"sqlite:///{(tmp_path / 'crypto.db').as_posix()}"
    store = open_crypto_store(url, f"sqlite:///{(tmp_path / 'trading.db').as_posix()}")
    shaped = _alpaca_shaped(10)
    shaped = shaped.drop("LTC/USD", level="symbol")
    feed = CryptoDailyFeed(_FakeClient(shaped))

    report = update_crypto_daily(
        feed, store, list(CRYPTO_UNIVERSE), now=datetime(2021, 2, 1, tzinfo=timezone.utc)
    )

    assert report.no_data == ["LTC/USD"]


def test_closed_only_cuts_the_running_day() -> None:
    index = pd.DatetimeIndex(pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"], utc=True))
    bars = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index)

    kept = closed_only(bars, datetime(2021, 1, 3, 0, 0, tzinfo=timezone.utc))  # exactly midnight

    assert list(kept["close"]) == [1.0, 2.0]  # Jan 2 is over at that instant, Jan 3 has begun


def test_a_stale_row_outside_the_new_download_does_not_stay(tmp_path: Any) -> None:
    """The history is replaced, not extended: a bar that is no longer in what Alpaca returns
    (a wrong print that was removed) must not stay behind in the store."""
    url = f"sqlite:///{(tmp_path / 'crypto.db').as_posix()}"
    store = open_crypto_store(url, f"sqlite:///{(tmp_path / 'trading.db').as_posix()}")
    old = _stamp(_frames(days=3)[MARKET_COIN])  # 2021-01-01 .. 01-03
    old.index = old.index - pd.Timedelta(days=30)  # 2020-12-02 .. 12-04, before the new range
    store.replace(
        MARKET_COIN,
        "1d",
        old,
        adjustment="none",
        fetched_at=datetime(2021, 1, 5, tzinfo=timezone.utc),
    )
    feed = CryptoDailyFeed(_FakeClient(_alpaca_shaped(10)))

    update_crypto_daily(
        feed, store, list(CRYPTO_UNIVERSE), now=datetime(2021, 2, 1, tzinfo=timezone.utc)
    )

    stored = store.load([MARKET_COIN], "1d")[MARKET_COIN]
    assert stored.index.min() == pd.Timestamp("2021-01-01", tz="UTC")
    assert len(stored) == 10
