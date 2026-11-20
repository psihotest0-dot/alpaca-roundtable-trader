"""The paper3 portfolio (docs/paper-third-account.md, section 6): the frozen list, the freeze
procedure and the setting that points the live loop at it."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from trading_system.config import Settings
from trading_system.data.paper3_portfolio import (
    MAX_CORRELATION,
    PAPER3_FACTS,
    PAPER3_NORMAL,
    PAPER3_PASSED_OVER,
    PAPER3_RISKY,
    PAPER3_SYMBOLS,
)

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "docs" / "paper3-portfolio"


def _freeze_module():
    spec = importlib.util.spec_from_file_location(
        "freeze_paper3_portfolio", ROOT / "scripts" / "freeze_paper3_portfolio.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_frozen_list_has_the_registered_shape():
    assert len(PAPER3_RISKY) == 30
    assert len(PAPER3_NORMAL) == 25
    assert len(PAPER3_SYMBOLS) == 55
    assert len(set(PAPER3_SYMBOLS)) == 55  # no stock in both groups, none twice
    assert all(s.isalpha() and s.isupper() and 1 <= len(s) <= 5 for s in PAPER3_SYMBOLS)
    assert set(PAPER3_FACTS) == set(PAPER3_SYMBOLS)


def test_no_sector_has_more_than_six_in_a_group():
    for group in ("risky", "normal"):
        sectors = Counter(sector for g, sector, _, _ in PAPER3_FACTS.values() if g == group)
        assert max(sectors.values()) <= 6, (group, sectors)


def test_the_groups_are_what_they_are_called():
    risky_ranges = [PAPER3_FACTS[s][2] for s in PAPER3_RISKY]
    normal_ranges = [PAPER3_FACTS[s][2] for s in PAPER3_NORMAL]
    assert risky_ranges == sorted(risky_ranges, reverse=True)  # the most volatile first
    assert min(risky_ranges) > 1.4 * max(normal_ranges)  # two groups far apart, not neighbours


def test_a_vetoed_stock_is_not_in_the_list():
    review = json.loads((INPUTS / "review.json").read_text(encoding="utf-8"))
    assert review["vetoes"], "the review file records its vetoes with a reason, a source and a date"
    for symbol, why in review["vetoes"].items():
        assert symbol not in PAPER3_SYMBOLS
        assert why["reason"] and why["source"] and why["date"]
    assert {s for _, s, w in PAPER3_PASSED_OVER if w == "veto"} <= set(review["vetoes"])


def test_the_committed_inputs_reproduce_the_frozen_list():
    freeze = _freeze_module()
    candidates = json.loads((INPUTS / "candidates.json").read_text(encoding="utf-8"))
    review = json.loads((INPUTS / "review.json").read_text(encoding="utf-8"))
    risky, _, normal, _ = freeze.choose(candidates, review)
    assert tuple(risky) == PAPER3_RISKY
    assert tuple(normal) == PAPER3_NORMAL


def test_no_two_stocks_of_a_group_move_together_more_than_the_cap():
    freeze = _freeze_module()
    candidates = json.loads((INPUTS / "candidates.json").read_text(encoding="utf-8"))
    corr = freeze.correlations(candidates)
    for group in (PAPER3_RISKY, PAPER3_NORMAL):
        block = corr.loc[list(group), list(group)].to_numpy(dtype=float).copy()
        np.fill_diagonal(block, 0.0)
        assert block.max() <= MAX_CORRELATION, (group, block.max())
    # the reason for the cap (section 6.3): left alone the risky group was clumped at 0.54, and a
    # random draw of 30 from the 100 most volatile stocks has 0.37; the cap brings it to that level
    assert freeze.mean_pair_correlation(corr, PAPER3_RISKY) <= 0.37


def test_pick_skips_vetoes_and_enforces_the_sector_cap_with_the_next_one_as_replacement():
    freeze = _freeze_module()
    order = ["A", "B", "C", "D", "E", "F"]
    sectors = {"A": "x", "B": "x", "C": "x", "D": "y", "E": "y", "F": "z"}
    chosen, dropped = freeze.pick(order, 3, sectors, {"A": {}}, max_per_sector=1)
    # A is vetoed; B takes the one place of sector x; C is over the cap; D takes y; E is over the
    # cap; F (the next in order) takes z and fills the group.
    assert chosen == ["B", "D", "F"]
    assert dropped == [
        ("A", "veto"),
        ("C", "sector cap (x)"),
        ("E", "sector cap (y)"),
    ]


def test_pick_skips_a_stock_that_moves_with_one_already_chosen_and_takes_the_next():
    freeze = _freeze_module()
    corr = pd.DataFrame(
        [[1.0, 0.9, 0.2, 0.6], [0.9, 1.0, 0.1, 0.5], [0.2, 0.1, 1.0, 0.3], [0.6, 0.5, 0.3, 1.0]],
        index=list("ABCD"),
        columns=list("ABCD"),
    )
    sectors = dict.fromkeys("ABCD", "x")
    chosen, dropped = freeze.pick(list("ABCD"), 3, sectors, {}, corr=corr, max_corr=0.75)
    # B is a twin of A (0.9) and drops out; C and D (0.6 with A) are far enough
    assert chosen == ["A", "C", "D"]
    assert dropped == [("B", "correlation 0.90 with A")]
    # without the cap, the order is kept
    assert freeze.pick(list("ABCD"), 3, sectors, {})[0] == ["A", "B", "C"]


def test_pick_refuses_to_fill_a_group_by_hand_when_the_order_runs_out():
    freeze = _freeze_module()
    with pytest.raises(SystemExit):
        freeze.pick(["A", "B"], 3, {"A": "x", "B": "x"}, {}, max_per_sector=6)


def test_the_preset_replaces_the_watchlist_and_is_off_by_default():
    base = {"alpaca_api_key": "k", "alpaca_secret_key": "s", "_env_file": None}
    plain = Settings(**base)
    assert plain.watchlist_preset == ""
    assert "AAPL" in plain.watchlist_symbols and len(plain.watchlist_symbols) == 22

    preset = Settings(watchlist_preset="paper3_portfolio", **base)
    assert preset.watchlist_symbols == list(PAPER3_SYMBOLS)
    assert "AAPL" not in preset.watchlist_symbols  # the preset replaces the list, not adds to it

    with pytest.raises(ValidationError):
        Settings(watchlist_preset="some_other_list", **base)
