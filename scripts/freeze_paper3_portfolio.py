"""Steps 2 and 3 of the paper3 portfolio (docs/paper-third-account.md, section 6): the review and the freeze.

    python scripts/freeze_paper3_portfolio.py CANDIDATES.json REVIEW.json > paper3_portfolio.py

CANDIDATES.json is the output of `scripts/select_paper3_portfolio.py` (step 1). REVIEW.json holds
what the review (step 2) found, and ONLY that:

    {"sector_source": "...", "sectors": {"AAPL": "Technology", ...},
     "vetoes": {"XYZ": {"reason": "...", "source": "...", "date": "2026-11-20"}}}

The procedure is mechanical and written in the document before any number: walk each group in its
order (risk order for the risky group, nearest the median risk for the normal one), skip a vetoed stock,
skip a stock whose sector already has MAX_PER_SECTOR in the group (that is "the lowest-ranked of the
excess drops out and the replacement is the next one by the rule"), and stop when the group is full.
Nobody picks a name. The result is written as a Python module (stdout), never edited by hand.
"""

import json
import sys
import textwrap
from typing import Mapping, Sequence

RISKY_SIZE = 30
NORMAL_SIZE = 25
MAX_PER_SECTOR = 6


def pick(
    order: Sequence[str],
    size: int,
    sectors: Mapping[str, str],
    vetoes: Mapping[str, object],
    max_per_sector: int = MAX_PER_SECTOR,
) -> tuple[list[str], list[tuple[str, str]]]:
    """The first `size` of `order` that survive the vetoes and the sector cap, and why the others
    that were passed over on the way dropped out."""
    chosen: list[str] = []
    dropped: list[tuple[str, str]] = []
    count: dict[str, int] = {}
    for symbol in order:
        if len(chosen) == size:
            break
        if symbol in vetoes:
            dropped.append((symbol, "veto"))
            continue
        sector = sectors[symbol]
        if count.get(sector, 0) >= max_per_sector:
            dropped.append((symbol, f"sector cap ({sector})"))
            continue
        count[sector] = count.get(sector, 0) + 1
        chosen.append(symbol)
    if len(chosen) < size:
        raise SystemExit(f"only {len(chosen)} of {size} found; the order ran out")
    return chosen, dropped


def wrapped(symbols: Sequence[str], indent: str) -> str:
    text = textwrap.fill(" ".join(symbols), width=92, break_long_words=False)
    return "\n".join(f'{indent}"{line} "' for line in text.splitlines())


def build(candidates: Mapping, review: Mapping) -> str:
    sectors: Mapping[str, str] = review["sectors"]
    vetoes: Mapping[str, Mapping] = review.get("vetoes", {})
    risky, risky_dropped = pick(candidates["risky_order"], RISKY_SIZE, sectors, vetoes)
    normal, normal_dropped = pick(candidates["normal_order"], NORMAL_SIZE, sectors, vetoes)
    both = set(risky) & set(normal)
    if both:
        raise SystemExit(f"a stock is in both groups: {sorted(both)}")

    rows = {r["symbol"]: r for r in candidates["eligible_rows"]}
    chosen = {**{s: "risky" for s in risky}, **{s: "normal" for s in normal}}
    lines = [
        '"""The paper3 portfolio: 30 risky and 25 normal stocks '
        "(docs/paper-third-account.md, section 6).",
        "",
        f"Made from `scripts/select_paper3_portfolio.py` (made at {candidates['made_at']}, last completed",
        f"session {candidates['last_session']}, {candidates['eligible']} eligible stocks, median risk "
        f"{candidates['median_range']}) and",
        "`scripts/freeze_paper3_portfolio.py` (the review of docs/paper-third-account.md, step 2). "
        "The rule looks at",
        "no return of any kind. Risk = the median daily range over 45 sessions. Sectors: "
        + str(review["sector_source"])
        + ".",
        "",
        "**This file is never edited by hand and does not change during the 20 days.** Another list is a",
        "new entry in the document first. A test pins the count and the shape.",
        '"""',
        "",
        f"MADE_AT = {candidates['made_at']!r}",
        f"LAST_SESSION = {candidates['last_session']!r}",
        "",
        "# In order of risk, the most volatile first.",
        "PAPER3_RISKY: tuple[str, ...] = tuple(",
        "    (",
        wrapped(risky, "        "),
        "    ).split()",
        ")",
        "",
        "# In order of closeness to the median risk of the eligible, the nearest first.",
        "PAPER3_NORMAL: tuple[str, ...] = tuple(",
        "    (",
        wrapped(normal, "        "),
        "    ).split()",
        ")",
        "",
        "# symbol -> (group, sector, median daily range over 45 sessions, median dollar volume in millions)",
        "PAPER3_FACTS: dict[str, tuple[str, str, float, int]] = {",
    ]
    for symbol in risky + normal:
        r = rows[symbol]
        lines.append(
            f'    "{symbol}": ("{chosen[symbol]}", "{sectors[symbol]}", '
            f"{r['median_range']}, {round(r['median_dollar_volume'] / 1e6)}),"
        )
    lines.append("}")
    lines.append("")
    lines.append("# Passed over on the way to a full group: (group, symbol, why).")
    lines.append("PAPER3_PASSED_OVER: tuple[tuple[str, str, str], ...] = (")
    for group, dropped in (("risky", risky_dropped), ("normal", normal_dropped)):
        for symbol, why in dropped:
            lines.append(f'    ("{group}", "{symbol}", "{why}"),')
    lines.append(")")
    lines.append("")
    lines.append("PAPER3_SYMBOLS: tuple[str, ...] = PAPER3_RISKY + PAPER3_NORMAL")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    candidates = json.load(open(sys.argv[1], encoding="utf-8"))
    review = json.load(open(sys.argv[2], encoding="utf-8"))
    sys.stdout.write(build(candidates, review))


if __name__ == "__main__":
    main()
