"""The stocks the strategy research (Phase 3b, docs/strategy-research-plan.md) is run on.

Chosen with the owner on 2026-11-05: about 145 large, liquid US stocks across the sectors, plus
SPY, which stands for the market (the regime filter reads it). On that day every one of them had
daily bars on the full-market feed from January 2016 (LIN from October 2018, after a merger), and
the least traded had a median of 235 million dollars a day over the last 60 sessions.

**Known limitation, to be written under every result.** The list is of the big companies of
today. Companies that went bankrupt, were bought or fell out of the large-company group since 2016
are not in it, so the past looks a little better than it was (survivorship bias).

**A known artifact in the prices.** The adjusted prices cover splits and dividends, not spin-offs:
RTX on 2020-04-03 (the merger of United Technologies with Raytheon and the spin-off of Carrier and
Otis) shows -71 % in one day, which shareholders did not lose. Measurements must leave that day
out (docs/strategy-research-plan.md, section 8).

**Renames.** A ticker that changes (Marsh McLennan, MMC to MRSH in January 2026) leaves the old
name with a history that stops, while the new name carries all of it. Use the current name and
let `fetch-daily` say which symbols have stopped (`data.history.stale_symbols`).
"""

MARKET_SYMBOL = "SPY"

SECTORS: dict[str, tuple[str, ...]] = {
    "technology": tuple(
        "AAPL MSFT NVDA GOOGL AMZN META AVGO ORCL CRM ADBE CSCO ACN IBM INTC AMD QCOM TXN AMAT "
        "MU LRCX ADI NOW INTU NFLX".split()
    ),
    "communication": tuple("TMUS VZ T CMCSA DIS CHTR".split()),
    "financials": tuple(
        "JPM BAC WFC C GS MS SCHW BLK AXP USB PNC TFC COF V MA PYPL SPGI MRSH CB PGR MET AIG".split()
    ),
    "health": tuple(
        "UNH JNJ LLY PFE MRK ABBV ABT TMO DHR BMY AMGN GILD MDT CVS ISRG CI ELV VRTX REGN ZTS "
        "SYK BSX".split()
    ),
    "consumer": tuple(
        "WMT COST HD LOW TGT NKE SBUX MCD KO PEP PG PM MO CL MDLZ TJX BKNG F GM TSLA ORLY "
        "MAR".split()
    ),
    "industrials": tuple(
        "CAT DE BA HON UPS FDX RTX LMT GE MMM UNP CSX NSC ETN EMR ITW GD NOC ADP WM".split()
    ),
    "energy": tuple("XOM CVX COP SLB EOG OXY PSX MPC VLO KMI WMB".split()),
    "utilities_property_materials": tuple(
        "NEE DUK SO D AEP EXC AMT PLD CCI EQIX PSA SPG APD SHW FCX NEM LIN".split()
    ),
    "market": (MARKET_SYMBOL,),
}


def research_universe() -> list[str]:
    """Every symbol of the research universe, once, in the order of SECTORS."""
    seen: dict[str, None] = {}
    for symbols in SECTORS.values():
        for symbol in symbols:
            seen.setdefault(symbol)
    return list(seen)
