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
    format_judgement,
    judge,
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

    def add(
        self,
        verdict: str,
        *,
        growth: float | None = None,
        failed: bool = False,
        mode: str = "shadow",
        version: str = "trend_cross-1",
        answered_at: datetime | None = None,
    ) -> StrategySignal:
        """One signal with one answer. `growth`: the price moves by this much each bar after the
        entry (up reaches the target, down the stop); None = no bars at all after it."""
        self.count += 1
        symbol = next(self.names)
        closed = FIRST_SIGNAL + timedelta(minutes=15 * self.count)
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
