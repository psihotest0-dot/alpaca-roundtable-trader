"""The judge of the Oracle (oracle/judge.py): counts only until the minimum, one look, a control, the
rules for "helps" and "hurts" -- on signals, answers and bars made up with a known truth."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from tests.fixtures.research import stock_names
from trading_system.cli import main
from trading_system.domain.enums import OracleVerdict, SignalDirection
from trading_system.domain.models import OracleDecision, StrategySignal
from trading_system.oracle import judge as judge_module
from trading_system.oracle.judge import (
    JUDGEMENT_EVENT,
    MIN_PER_GROUP,
    PROBE_VERSION,
    START,
    _daily_t,
    format_judgement,
    format_not_approved,
    judge,
    judge_not_approved,
)
from trading_system.storage.repositories import (
    AuditRepository,
    OracleDecisionRepository,
    SignalRepository,
)
from trading_system.storage.research import open_observe_store

UTC = timezone.utc
NOW = datetime(2026, 12, 20, tzinfo=UTC)
FIRST_SIGNAL = datetime(2026, 11, 9, 15, 0, tzinfo=UTC)  # a Monday, after START


@pytest.fixture
def observe_store(tmp_path: Path, db_url: str) -> Any:
    return open_observe_store(f"sqlite:///{(tmp_path / 'observe.db').as_posix()}", db_url)


class World:
    """Signals, the model's answers and the bars that follow them."""

    def __init__(self, session_factory: Any, observe_store: Any, settings: Any) -> None:
        self.factory, self.store, self.settings = session_factory, observe_store, settings
        self.signals = SignalRepository(session_factory)
        self.answers = OracleDecisionRepository(session_factory)
        self.audit = AuditRepository(session_factory)
        self.names = iter(stock_names(140) * 3)
        self.count = 0
        # the model and the backend's prompt label of the answers added next (None: unlabelled)
        self.model: str | None = None
        self.label: str | None = None
        # how many different days the signals added next are spread over (question B needs days)
        self.spread = 1

    def add(
        self,
        verdict: str,
        *,
        growth: float | None = None,
        failed: bool = False,
        mode: str = "shadow",
        version: str = "trend_cross-1",
        answered_at: datetime | None = None,
        day: int | None = None,
    ) -> StrategySignal:
        """One signal with one answer. `growth`: the price moves by this much each bar after the
        entry (up reaches the target, down the stop); None = no bars at all after it."""
        self.count += 1
        symbol = next(self.names)
        offset = (self.count % self.spread) if day is None else day
        # one day after the other, in a few hours each, when signals are spread over days (so that a
        # day stays a day); a single run of 15-minute steps otherwise, as before
        steps = self.count if (day is None and self.spread == 1) else self.count % 40
        closed = FIRST_SIGNAL + timedelta(days=offset, minutes=15 * steps)
        signal = StrategySignal(
            version=version,
            symbol=symbol,
            timeframe="15m",
            closed_bar_timestamp=closed,
            direction=SignalDirection.LONG,
            signal_id=f"sig-{self.count}-{symbol}",
        )
        self.signals.save_signal(signal)
        self.answers.save(
            OracleDecision(
                signal_id=signal.signal_id or "",
                verdict=OracleVerdict.HOLD if failed else OracleVerdict(verdict),
                failed=failed,
                provider="backend",
                mode=mode,
                model=None if failed else self.model,
                prompt_version=None if failed else self.label,
                decision_timestamp=answered_at or (closed + timedelta(minutes=1)),
            )
        )
        if growth is not None:
            self._bars(signal, growth)
        return signal

    def _bars(self, signal: StrategySignal, growth: float) -> None:
        entry = pd.Timestamp(signal.closed_bar_timestamp) + pd.Timedelta(minutes=15)
        index = pd.date_range(
            entry - pd.Timedelta(hours=3), periods=12 + 120, freq="15min", tz="UTC"
        )
        closes = [100.0] * 12
        for _ in range(120):
            closes.append(closes[-1] * (1 + growth))
        closes_arr = pd.Series(closes, index=index)
        opens = closes_arr.shift(1).fillna(100.0)
        frame = pd.DataFrame(
            {
                "open": opens,
                "high": pd.concat([opens, closes_arr], axis=1).max(axis=1) * 1.0001,
                "low": pd.concat([opens, closes_arr], axis=1).min(axis=1) * 0.9999,
                "close": closes_arr,
                "volume": 1000.0,
            },
            index=index.rename("timestamp"),
        )
        self.store.replace(str(signal.symbol), "15m", frame, adjustment="all", fetched_at=NOW)

    def fill(self, approved: int, vetoed: int, *, up: float = 0.006, down: float = -0.006) -> None:
        """Approved and vetoed signals interleaved in time, the approved ones going up (they reach
        the target) and the vetoed ones going down (the stop)."""
        for i in range(max(approved, vetoed)):
            if i < approved:
                self.add("APPROVE", growth=up)
            if i < vetoed:
                self.add("VETO", growth=down)

    def judge(self, when: datetime = NOW) -> Any:
        return judge(self.factory, self.answers, self.store, self.audit, self.settings, when)


@pytest.fixture
def world(session_factory: Any, observe_store: Any, make_settings: Any) -> World:
    return World(session_factory, observe_store, make_settings())


def _looks(world: World) -> int:
    return len(
        world.audit.events(
            JUDGEMENT_EVENT, datetime(2000, 1, 1, tzinfo=UTC), NOW + timedelta(days=1)
        )
    )


# ---------------------------------------------------------------------------- no peeking


def test_below_the_minimum_only_counts_are_shown_and_no_return_is_computed(world: World) -> None:
    world.fill(MIN_PER_GROUP - 1, MIN_PER_GROUP)

    j = world.judge()

    assert j.verdict == "no verdict" and j.detail is None
    assert (j.counts.approve, j.counts.veto, j.counts.asked) == (29, 30, 59)
    text = format_judgement(j)
    assert "need 30 of each" in text
    assert "Welch" not in text and "per trade" not in text  # no return was computed or shown
    assert _looks(world) == 0  # nothing was used


def test_without_any_answers_it_says_so(world: World) -> None:
    j = world.judge()

    assert j.counts.asked == 0 and j.counts.availability == 0 and j.detail is None


# ------------------------------------------------------------------ what counts as a sample


def test_probes_old_answers_other_modes_holds_and_outages_are_not_in_the_sample(
    world: World,
) -> None:
    world.add("APPROVE", growth=0.006)
    world.add("VETO", growth=-0.006, version=PROBE_VERSION)  # the probe of the road
    world.add("VETO", growth=-0.006, answered_at=START - timedelta(hours=1))  # before the switch-on
    world.add("VETO", growth=-0.006, mode="active")  # not shadow
    world.add("VETO", growth=-0.006, mode="")  # from before there was a mode
    world.add("HOLD", growth=0.0)  # no final word
    world.add("HOLD", growth=0.0, failed=True)  # an outage

    c = world.judge().counts

    assert (c.approve, c.veto) == (1, 0)
    assert c.asked == 3  # the approve, the hold and the outage were asked in shadow mode
    # replies / asked: the approve and the model's own HOLD are replies, the outage is not
    assert c.availability == pytest.approx(2 / 3) and (c.held, c.no_reply) == (1, 1)


def test_availability_under_eighty_percent_gives_no_verdict_and_no_returns(world: World) -> None:
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)
    for _ in range(31):  # 60 answered of 91 asked: 66 %
        world.add("HOLD", failed=True)

    j = world.judge()

    assert j.verdict == "no verdict" and "availability 66%" in j.reason and j.detail is None
    assert _looks(world) == 0


def test_the_models_own_holds_are_not_an_outage_and_do_not_block_a_verdict(world: World) -> None:
    """On the first real day the model held on 3 of 7 signals. A HOLD is a reply ("not clear"): it is no
    final answer and stays out of both groups, but it must not count against the availability, or a
    cautious model would never be judged."""
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)
    for _ in range(40):
        world.add("HOLD")

    j = world.judge()

    assert j.counts.held == 40 and j.counts.no_reply == 0 and j.counts.availability == 1.0
    assert j.verdict == "helps" and j.detail is not None  # judged on the 60 final answers
    assert "the model held on 40" in format_judgement(j)


def test_holds_and_outages_together_only_the_outages_count_against_availability(
    world: World,
) -> None:
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)
    for _ in range(30):
        world.add("HOLD")  # 30 holds: fine
    for _ in range(14):
        world.add("HOLD", failed=True)  # 14 of 104 asked got no reply: 87 %
    assert world.judge().verdict == "helps"

    for _ in range(40):  # now 54 of 144 asked got no reply: 62 %
        world.add("HOLD", failed=True)
    # the look of the first call is used; availability is what is under test
    worse = world.judge()
    assert worse.counts.availability == pytest.approx(90 / 144) and worse.counts.no_reply == 54


# ------------------------------------------------------------------------- the verdicts


def test_vetoes_that_remove_worse_trades_help(world: World) -> None:
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP, up=0.006, down=-0.006)

    j = world.judge()

    assert j.verdict == "helps" and j.detail is not None
    d = j.detail
    assert d.approved == d.vetoed == 30 and d.approved_mean > 0.03 and d.vetoed_mean < -0.03
    assert d.t > 10 and d.first_half_edge > 0 and d.second_half_edge > 0
    assert abs(d.control_t) < 2  # a random relabelling finds nothing
    assert _looks(world) == 1
    assert "This was a LOOK" in format_judgement(j)


def test_vetoes_that_remove_better_trades_hurt(world: World) -> None:
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP, up=-0.006, down=0.006)  # the approved lose

    j = world.judge()

    assert j.verdict == "hurts" and j.detail is not None and j.detail.t < -10


def test_equal_groups_cannot_be_told_from_noise(world: World) -> None:
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP, up=0.006, down=0.006)

    j = world.judge()

    assert j.verdict == "cannot be told from noise" and j.detail is not None
    assert abs(j.detail.t) < 2


def test_an_edge_that_changes_sign_between_the_halves_is_not_a_help(world: World) -> None:
    """The first twenty pairs go as the model would like (the approved reach the target, the vetoed
    the stop); in the last twenty, fifteen go the other way. Pooled, the approved still win (t about
    2.3, approved mean above zero) -- but the second half of the period is the wrong way round."""
    for i in range(40):
        flipped = i >= 25
        world.add("APPROVE", growth=-0.006 if flipped else 0.006)
        world.add("VETO", growth=0.006 if flipped else -0.006)

    j = world.judge()

    assert j.detail is not None
    assert j.detail.t >= 2 and j.detail.approved_mean > 0  # everything else would say "helps"
    assert j.detail.first_half_edge > 0 > j.detail.second_half_edge
    assert j.verdict == "cannot be told from noise"


def test_the_first_look_passes_at_a_t_between_two_and_two_and_a_half(world: World) -> None:
    """A real but modest edge: 19 of 30 approved and 11 of 30 vetoed reach the target, spread evenly
    over the period. The bar of the first look is 2, of the second 2.5."""
    approved_down = {i for i in range(30) if i % 3 == 2} | {0}
    vetoed_up = {i for i in range(30) if i % 3 == 1} | {3}
    for i in range(30):
        world.add("APPROVE", growth=-0.006 if i in approved_down else 0.006)
        world.add("VETO", growth=0.006 if i in vetoed_up else -0.006)

    j = world.judge()

    assert j.detail is not None and 2.0 <= j.detail.t < 2.5
    assert j.verdict == "helps"


def test_an_edge_over_the_vetoed_is_not_a_help_if_the_approved_still_lose_money(
    world: World,
) -> None:
    """8 of 30 approved reach the target (+5 %), 22 the stop (-4 %): about -1.7 % a trade. All 30
    vetoed reach the stop. The approved are far better than the vetoed and still lose."""
    for i in range(MIN_PER_GROUP):
        world.add("APPROVE", growth=0.006 if i % 4 == 0 else -0.006)
        world.add("VETO", growth=-0.006)

    j = world.judge()

    assert j.detail is not None
    assert j.detail.edge > 0.01 and j.detail.t > 3 and j.detail.approved_mean < 0
    assert j.verdict == "cannot be told from noise"  # not "helps": the money is lost either way


def test_a_suspect_control_overrides_the_verdict(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)
    monkeypatch.setattr(judge_module, "CONTROL_MAX_T", -1.0)  # any control counts as suspect

    j = world.judge()

    assert j.verdict == "control suspect" and "instrument" in j.reason


# ---------------------------------------------------------------- unresolved, and the looks


def test_trades_that_have_not_reached_a_level_do_not_use_up_the_look(world: World) -> None:
    world.fill(
        MIN_PER_GROUP, MIN_PER_GROUP, up=0.0001, down=-0.0001
    )  # a tenth of a percent in 30 hours

    j = world.judge()

    assert j.verdict == "no verdict" and "resolved" in j.reason and "No look was used" in j.reason
    assert _looks(world) == 0


def test_signals_without_a_bar_to_enter_on_are_counted_and_do_not_count_as_trades(
    world: World,
) -> None:
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)
    world.add("APPROVE")  # no bars at all
    j = world.judge()

    assert j.detail is not None and j.detail.no_bars == 1 and j.detail.approved == 30


def test_one_look_then_a_second_only_at_double_the_minimum_with_a_higher_bar(world: World) -> None:
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)
    assert world.judge().verdict == "helps"

    again = world.judge()
    assert again.verdict == "no verdict" and "need 60 of each" in again.reason
    assert again.looks_before == 1 and _looks(world) == 1  # asking again changed nothing

    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)  # 60 and 60
    second = world.judge()
    assert second.verdict == "helps" and _looks(world) == 2

    third = world.judge()
    assert third.verdict == "no more looks" and third.detail is None and _looks(world) == 2


# --------------------------------------------------------------------------- the command


def test_the_command_prints_the_counts(
    world: World, make_settings: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world.fill(5, 4)
    settings = make_settings(
        observe_database_url=f"sqlite:///{(tmp_path / 'observe.db').as_posix()}"
    )

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["judge-oracle"]) == 0

    out = capsys.readouterr().out
    assert "asked about 9 signals; final answers 9 (5 approve, 4 veto)" in out
    assert "looks taken in this period: 0 of 2" in out and "verdict: no verdict" in out


# ------------------------------------------------------------------------------------ periods

A, B = "a" * 12, "b" * 12


def _journal(world: World) -> list[dict[str, Any]]:
    return [
        details
        for _, details in world.audit.events(
            JUDGEMENT_EVENT, datetime(2000, 1, 1, tzinfo=UTC), NOW + timedelta(days=1)
        )
    ]


def test_answers_under_another_prompt_label_are_another_period_and_only_the_newest_is_judged(
    world: World,
) -> None:
    world.label = A
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)  # a full sample, all under the old label
    world.label = B
    world.add("APPROVE", growth=0.006)  # the prompt changed; one answer under the new label

    j = world.judge()

    assert j.verdict == "no verdict" and j.detail is None  # the old sample is NOT pooled in
    assert j.counts.period == ("", B) and (j.counts.approve, j.counts.veto) == (1, 0)
    (old,) = j.other_periods
    assert old.period == ("", A) and (old.approve, old.veto) == (30, 30)
    text = format_judgement(j)
    assert f"unknown model|{B}" in text and f"other period unknown model|{A}" in text
    assert "counts only, not judged" in text
    assert _journal(world) == []  # no look was used on the old sample either


def test_a_changed_model_is_another_period_even_under_the_same_label(world: World) -> None:
    world.model, world.label = "gemma", A
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)
    world.model = "qwen"
    world.add("VETO", growth=-0.006)

    j = world.judge()

    assert j.counts.period == ("qwen", A) and j.counts.veto == 1 and j.verdict == "no verdict"
    assert [p.period for p in j.other_periods] == [("gemma", A)]


def test_answers_from_before_the_label_existed_are_their_own_period(world: World) -> None:
    world.label = None  # made before 22:11 on 2026-11-06
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)
    world.label = A
    world.add("APPROVE", growth=0.006)

    j = world.judge()

    assert j.counts.period == ("", A) and j.counts.answered == 1
    assert [p.period for p in j.other_periods] == [("", "")]
    assert "unlabelled" in format_judgement(j)


def test_an_outage_counts_against_the_period_that_was_running_when_it_happened(
    world: World,
) -> None:
    world.label = A
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)
    world.label = B
    world.add("APPROVE", growth=0.006)
    for _ in range(9):  # nine asked after the new label began, none answered
        world.add("HOLD", failed=True)

    j = world.judge()

    assert j.counts.period == ("", B) and j.counts.asked == 10 and j.counts.answered == 1
    assert j.counts.availability == pytest.approx(0.1)
    (old,) = j.other_periods
    assert old.asked == 60 and old.availability == 1.0  # the old period is not charged for them


def test_looks_are_counted_per_period_and_each_is_written_with_its_period(world: World) -> None:
    world.label = A
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)
    first = world.judge()
    assert first.verdict == "helps" and first.looks_before == 0

    world.label = B  # a new prompt: a new sample, its own first look
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)
    second = world.judge()

    assert second.counts.period == ("", B)
    assert second.looks_before == 0 and second.verdict == "helps"
    journal = _journal(world)
    assert [e["period"] for e in journal] == [f"unknown model|{A}", f"unknown model|{B}"]
    assert [e["periods_so_far"] for e in journal] == [
        1,
        2,
    ]  # how many periods were judged is on record
    again = world.judge()
    assert again.looks_before == 1 and again.verdict == "no verdict"  # B's own second look needs 60


def test_before_any_answer_there_is_no_period(world: World) -> None:
    j = world.judge()

    assert j.counts.period is None and j.other_periods == ()
    assert "period judged (model | prompt label): none" in format_judgement(j)


# ------------------------------------------------------------- question B (docs section 5.1)
# "Would skipping everything the model did not approve (a VETO, or a HOLD it never lifted) have been
# better?" The approved are the same group as in question A; the not approved are vetoes AND holds.

DAYS = 21  # signals spread over this many different days (the minimum is 20)


def _b_judge(world: World, when: datetime = NOW) -> Any:
    return judge_not_approved(
        world.factory, world.answers, world.store, world.audit, world.settings, when
    )


def _fill_b(
    world: World,
    approved: int,
    vetoed: int,
    held: int,
    *,
    up: float = 0.006,
    down: float = -0.006,
    held_growth: float | None = None,
) -> None:
    """Approved, vetoed and held signals interleaved over DAYS days. The approved go `up`, the vetoed
    go `down`, the held go `held_growth` (default: down, like the vetoed)."""
    world.spread = DAYS
    held_growth = down if held_growth is None else held_growth
    for i in range(max(approved, vetoed, held)):
        if i < approved:
            world.add("APPROVE", growth=up)
        if i < vetoed:
            world.add("VETO", growth=down)
        if i < held:
            world.add("HOLD", growth=held_growth)


def _b_journal(world: World) -> list[dict[str, Any]]:
    return [d for d in _journal(world) if d.get("question") == "B"]


def test_b_below_the_minimum_shows_counts_only_and_computes_no_return(world: World) -> None:
    _fill_b(world, 60, 20, 39)  # 59 not approved: one short
    world.add("HOLD", failed=True)  # an outage is in neither group

    j = _b_judge(world)

    assert j.question == "B" and j.verdict == "no verdict" and j.detail is None
    assert "60 approved and 59 not approved (20 vetoes, 39 holds), need 60 of each" in j.reason
    text = format_not_approved(j)
    assert "Welch" not in text and "per trade" not in text  # no return was computed or shown
    assert _looks(world) == 0


def test_b_needs_twenty_different_trading_days_before_any_return_is_computed(world: World) -> None:
    world.spread = 1  # all the signals in one or two days
    for _ in range(60):
        world.add("APPROVE", growth=0.006)
        world.add("VETO", growth=-0.006)

    j = _b_judge(world)

    assert j.verdict == "no verdict" and j.detail is None
    assert "trading days of answers in this period, need 20" in j.reason
    assert _looks(world) == 0


def test_b_holds_count_as_not_approved_and_skipping_them_can_help(world: World) -> None:
    _fill_b(world, 80, 20, 60)  # 80 approved, 20 vetoed + 60 held = 80 not approved

    j = _b_judge(world)

    assert j.verdict == "helps" and j.detail is not None
    assert j.detail.approved >= 60 and j.detail.vetoed >= 60  # the not approved, vetoes and holds
    assert j.detail.t > 2.4 and j.detail.day_t >= 1.5 and j.detail.paired_days >= 10
    assert "Question B" in format_not_approved(j)
    row = _b_journal(world)[0]
    assert row["question"] == "B" and row["look"] == 1 and row["verdict"] == "helps"


def test_b_a_hold_that_was_lifted_counts_by_its_final_word(world: World) -> None:
    """A signal that was HOLD and later APPROVE is an approved signal (the last word counts)."""
    _fill_b(world, 60, 60, 0)
    first = world.add("HOLD", growth=0.006, day=3)
    world.answers.save(
        OracleDecision(
            signal_id=first.signal_id or "",
            verdict=OracleVerdict.APPROVE,
            provider="backend",
            mode="shadow",
            decision_timestamp=first.closed_bar_timestamp + timedelta(minutes=20),  # still valid
        )
    )

    j = _b_judge(world)

    assert (j.counts.approve, j.counts.veto, j.counts.held) == (61, 60, 0)


def test_a_signal_put_to_the_model_after_it_expired_is_not_in_the_sample(world: World) -> None:
    """Until 2026-11-19 the observer's first pass of a session asked about the cross on the last bar of
    the day before, 18 to 66 hours late (docs section 5.3). A signal is valid for 30 minutes from its
    bar's open time (`expires_at`: the bar's close plus 15 minutes); an answer later than that says
    nothing about a signal that could have been traded."""
    on_time = world.add("APPROVE", growth=0.006)  # a minute after the bar
    at_the_limit = world.add("APPROVE", growth=0.006)
    one_late = world.add("APPROVE", growth=0.006)
    next_day = world.add("HOLD", growth=0.006)
    for signal, minutes in ((at_the_limit, 30), (one_late, 31), (next_day, 18 * 60)):
        world.answers.save(
            OracleDecision(
                signal_id=signal.signal_id or "",
                verdict=OracleVerdict.APPROVE if signal is not next_day else OracleVerdict.HOLD,
                provider="backend",
                mode="shadow",
                model=world.model,
                prompt_version=world.label,
                decision_timestamp=signal.closed_bar_timestamp + timedelta(minutes=minutes),
            )
        )

    counts = world.judge().counts

    # the latest answer of each signal decides: on time and exactly at the limit stay; the other two go
    assert on_time.signal_id and counts.asked == 2 and counts.approve == 2 and counts.held == 0


def test_b_skipping_what_was_not_approved_can_hurt(world: World) -> None:
    _fill_b(world, 80, 20, 60, up=-0.006, down=0.006)  # the approved lose, the rest win

    j = _b_judge(world)

    assert j.verdict == "hurts" and j.detail is not None and j.detail.t < -2.4


def test_b_equal_groups_cannot_be_told_from_noise(world: World) -> None:
    _fill_b(world, 80, 20, 60, up=0.006, down=0.006, held_growth=0.006)

    j = _b_judge(world)

    assert j.verdict == "cannot be told from noise"
    assert len(_b_journal(world)) == 1  # a look was used, as the rules say


def test_b_two_strong_days_are_not_enough_even_when_the_t_over_signals_is_high(
    world: World,
) -> None:
    """The edge exists on the first and the last day only. Over the signals it looks certain, but the
    signals of a day move together: the per-day t is under the bar, so there is no verdict."""
    world.spread = DAYS
    for index in range(DAYS):
        for _ in range(3):  # three approved and three vetoed on each day
            for verdict in ("APPROVE", "VETO"):
                strong = index in (0, DAYS - 1)
                growth = 0.006 if verdict == "APPROVE" or not strong else -0.006
                world.add(verdict, growth=growth, day=index)
    for index in range(DAYS):
        world.add("HOLD", growth=0.006, day=index)  # held ones that go up on every day

    j = _b_judge(world)

    assert j.detail is not None
    assert j.detail.t >= 2.4  # sure over the signals ...
    assert j.detail.day_t < 1.5  # ... but not over the days
    assert j.verdict == "cannot be told from noise"


def test_b_has_its_own_looks_and_a_second_look_needs_double_and_a_higher_bar(
    world: World,
) -> None:
    _fill_b(world, 80, 20, 60)
    assert _b_judge(world).verdict == "helps"

    again = _b_judge(world)
    assert again.verdict == "no verdict" and "need 120 of each" in again.reason
    assert again.looks_before == 1 and len(_b_journal(world)) == 1  # asking again changed nothing

    # question A has used none of its own looks (3 vetoes are not 30)
    a = world.judge()
    assert a.looks_before == 0 and a.verdict == "no verdict"


def test_b_a_look_at_a_does_not_use_up_the_looks_at_b(world: World) -> None:
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP)  # question A can be judged at once
    assert world.judge().verdict == "helps"
    assert world.judge().looks_before == 1

    assert _b_judge(world).looks_before == 0  # B starts with none
    rows = _journal(world)
    assert len(rows) == 1 and rows[0]["question"] == "A"


def test_b_a_new_model_is_a_new_period_with_its_own_counts(world: World) -> None:
    world.label = A
    _fill_b(world, 80, 20, 60)
    assert _b_judge(world).verdict == "helps"

    world.label = B
    for _ in range(5):  # answered after all the others: the newest answer decides the period
        world.add("APPROVE", growth=0.006, day=60)
    j = _b_judge(world)

    assert j.counts.period is not None and j.counts.period[1] == B
    assert j.verdict == "no verdict" and j.looks_before == 0  # the new period starts again
    assert len(j.other_periods) >= 1  # the old one is shown with counts only


def test_the_daily_t_needs_ten_days_with_both_groups() -> None:
    day = pd.Timestamp("2026-11-09 15:00", tz="UTC")
    few = [("APPROVE", 0.01, day), ("NOT_APPROVED", 0.0, day)]

    t, days = _daily_t(few)

    assert days == 1 and t != t  # nan: too few days to say anything

    many = []
    for i in range(12):
        when = day + pd.Timedelta(days=i)
        many += [("APPROVE", 0.01 + 0.001 * (i % 3), when), ("NOT_APPROVED", 0.0, when)]
    t, days = _daily_t(many)

    assert days == 12 and t > 5


def test_the_command_prints_both_questions(
    world: World, make_settings: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world.fill(5, 4)
    settings = make_settings(
        observe_database_url=f"sqlite:///{(tmp_path / 'observe.db').as_posix()}"
    )

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["judge-oracle"]) == 0

    out = capsys.readouterr().out
    assert "verdict: no verdict" in out  # question A, as before
    assert "Question B: would skipping every signal the model did not approve" in out
    assert "not approved: 4 (4 vetoes, 0 holds), approved: 5" in out


def test_b_outages_are_in_neither_group_not_even_among_the_trades(world: World) -> None:
    """Twenty outages with bars that go up: an outage is not an opinion, so none of them may reach the
    shadow trades of either group (140 signals, every symbol used once)."""
    world.spread = DAYS
    for _ in range(60):
        world.add("APPROVE", growth=0.006)
    for _ in range(30):
        world.add("VETO", growth=-0.006)
        world.add("HOLD", growth=-0.006)
    for _ in range(20):
        world.add("HOLD", growth=0.006, failed=True)

    j = _b_judge(world)

    assert j.detail is not None
    assert (j.detail.approved, j.detail.vetoed) == (60, 60)
    assert j.counts.no_reply == 20


def test_b_a_t_between_two_and_two_point_four_is_not_a_pass_at_the_first_look(
    world: World,
) -> None:
    """The bar for B is higher than for A (2.4 against 2.0), because it is a second question on the
    same signals."""
    world.spread = DAYS
    approved_down = {i for i in range(60) if i % 5 in (0, 3)}  # 24 of 60 reach the stop
    rest_up = {i for i in range(60) if i % 5 in (1, 4)}  # 24 of 60 reach the target
    for i in range(60):
        world.add("APPROVE", growth=-0.006 if i in approved_down else 0.006)
        world.add("VETO" if i % 2 else "HOLD", growth=0.006 if i in rest_up else -0.006)

    j = _b_judge(world)

    assert j.detail is not None and 2.0 <= j.detail.t < 2.4, j.detail.t
    assert j.verdict == "cannot be told from noise"


def test_b_the_approved_must_still_make_money_for_a_help(world: World) -> None:
    world.spread = DAYS
    for i in range(60):
        world.add("APPROVE", growth=0.006 if i % 4 == 0 else -0.006)  # most of them lose
        world.add("VETO" if i % 2 else "HOLD", growth=-0.006)  # all the rest lose

    j = _b_judge(world)

    assert j.detail is not None
    assert j.detail.edge > 0.01 and j.detail.t > 3 and j.detail.approved_mean < 0
    assert j.verdict == "cannot be told from noise"


def test_b_a_suspect_control_overrides_the_verdict(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fill_b(world, 80, 20, 60)
    monkeypatch.setattr(judge_module, "CONTROL_MAX_T", -1.0)  # any control counts as suspect

    j = _b_judge(world)

    assert j.verdict == "control suspect"


def test_b_an_edge_that_changes_sign_between_the_halves_is_not_a_help(world: World) -> None:
    """Strongly in favour of the approved in the first days, mildly against them in the last: the
    total looks significant, the halves disagree."""
    world.spread = DAYS
    for index in range(DAYS):
        early = index < DAYS // 2
        for _ in range(3):
            world.add("APPROVE", growth=0.006 if early else -0.006, day=index)
            world.add("VETO", growth=-0.006 if early else 0.006, day=index)
            world.add("HOLD", growth=-0.006 if early else 0.006, day=index)
    for index in range(DAYS // 2):  # more weight on the early days
        for _ in range(3):
            world.add("APPROVE", growth=0.006, day=index)
            world.add("VETO", growth=-0.006, day=index)

    j = _b_judge(world)

    assert j.detail is not None
    assert j.detail.first_half_edge > 0 > j.detail.second_half_edge
    assert j.verdict == "cannot be told from noise"


# ----------------------------------------------- the registered numbers, and A is not changed


def test_the_registered_numbers_of_both_questions_are_pinned() -> None:
    """The numbers are written in docs/oracle-provider-task.md (sections 5 and 5.1) before any result.
    Changing one is a change of the registration and must be written there first: this test is the
    reminder."""
    assert (judge_module.MIN_PER_GROUP, judge_module.MIN_AVAILABILITY) == (30, 0.80)
    assert (judge_module.PASS_T, judge_module.SECOND_LOOK_T, judge_module.MAX_LOOKS) == (
        2.0,
        2.5,
        2,
    )
    assert judge_module.CONTROL_MAX_T == 2.0
    assert (judge_module.B_MIN_PER_GROUP, judge_module.B_MIN_DAYS) == (60, 20)
    assert (judge_module.B_PASS_T, judge_module.B_SECOND_LOOK_T) == (2.4, 2.7)
    assert (judge_module.B_DAY_T, judge_module.B_MIN_PAIRED_DAYS) == (1.5, 10)


def test_question_a_still_needs_30_and_30_and_a_t_of_two(world: World) -> None:
    """Question A is not changed by the arrival of B: 29 vetoes are still no verdict whatever B has,
    and a t under 2.0 is still no pass at the first look."""
    for _ in range(40):
        world.add("HOLD", growth=-0.006)  # plenty of holds: B's group of the not approved is large
    world.fill(MIN_PER_GROUP, MIN_PER_GROUP - 1)

    short = world.judge()

    assert short.verdict == "no verdict" and "need 30 of each" in short.reason
    assert short.detail is None and short.counts.veto == 29


def test_question_a_a_t_under_two_is_not_a_pass(world: World) -> None:
    approved_down = {i for i in range(30) if i % 5 in (0, 3)}  # 12 of 30 reach the stop
    vetoed_up = {i for i in range(30) if i % 5 in (1, 4)}  # 12 of 30 reach the target
    for i in range(30):
        world.add("APPROVE", growth=-0.006 if i in approved_down else 0.006)
        world.add("VETO", growth=0.006 if i in vetoed_up else -0.006)

    j = world.judge()

    assert j.detail is not None and 1.0 <= j.detail.t < 2.0, j.detail.t
    assert j.verdict == "cannot be told from noise"


def test_a_journal_row_from_before_question_b_is_a_look_at_a(world: World) -> None:
    """Rows written before B existed have no `question`; they are looks at A and never at B."""
    world.fill(5, 5)
    world.audit.record(
        JUDGEMENT_EVENT,
        {"look": 1, "period": judge_module.period_key(("", "")), "verdict": "helps"},
        when=NOW - timedelta(days=1),
    )

    assert world.judge().looks_before == 1
    assert _b_judge(world).looks_before == 0


def test_b_the_resolved_trades_must_come_from_twenty_days_too(world: World) -> None:
    """The answers are spread over 21 days, but the shadow trades that reached a level are on 11:
    one strong week is not twenty days. No look is used."""
    world.spread = DAYS
    for index in range(11):
        for _ in range(6):
            world.add("APPROVE", growth=0.006, day=index)
            world.add("VETO", growth=-0.006, day=index)
    for index in range(11, DAYS):  # answered, but nothing to play out
        world.add("APPROVE", day=index)
        world.add("VETO", day=index)

    j = _b_judge(world)

    assert j.verdict == "no verdict" and j.detail is None
    assert "on 11 trading days" in j.reason and "No look was used" in j.reason
    assert _looks(world) == 0


def test_b_an_edge_that_changes_sign_between_the_halves_is_not_a_help_whatever_the_t(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trades made up: strongly for the approved in the first days, a little against them in the last.
    The total t and the per-day t are both high, the approved make money, and still the halves
    disagree: that is not a help."""
    world.spread = DAYS
    for index in range(DAYS):
        for _ in range(3):
            world.add("APPROVE", day=index)
            world.add("VETO", day=index)
    start = pd.Timestamp("2026-11-09 15:00", tz="UTC")
    made_up = []
    for index in range(DAYS):
        for n in range(3):
            early = index <= 10
            jitter = 0.0001 * (n + index)
            made_up.append(
                ("APPROVE", (0.04 if early else 0.0) + jitter, start + pd.Timedelta(days=index))
            )
            made_up.append(
                (
                    judge_module.NOT_APPROVED,
                    (0.01 if early else 0.005) + jitter,
                    start + pd.Timedelta(days=index),
                )
            )
    monkeypatch.setattr(judge_module, "_not_approved_trades", lambda s, b, st: (made_up, 0, 0))

    j = _b_judge(world)

    assert j.detail is not None
    assert j.detail.t >= 2.4 and j.detail.day_t >= 1.5 and j.detail.approved_mean > 0
    assert j.detail.first_half_edge > 0 > j.detail.second_half_edge
    assert j.verdict == "cannot be told from noise"


def test_the_daily_t_is_nan_below_ten_days_with_both_groups_and_a_number_at_ten() -> None:
    day = pd.Timestamp("2026-11-09 15:00", tz="UTC")

    def trades(days: int) -> list[tuple[str, float, pd.Timestamp]]:
        out = []
        for i in range(days):
            when = day + pd.Timedelta(days=i)
            out += [("APPROVE", 0.01 + 0.001 * (i % 3), when), ("NOT_APPROVED", 0.0, when)]
        return out

    nine, days9 = _daily_t(trades(9))
    ten, days10 = _daily_t(trades(10))

    assert days9 == 9 and nine != nine  # nan
    assert days10 == 10 and ten > 5
