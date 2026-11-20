"""The paper3 portfolio: 30 risky and 25 normal stocks (docs/paper-third-account.md, section 6).

Made from `scripts/select_paper3_portfolio.py` (made at 2026-11-20T16:32:39+00:00, last completed
session 2026-11-19, 450 eligible stocks, median risk 0.02757) and
`scripts/freeze_paper3_portfolio.py` (the review of docs/paper-third-account.md, step 2). The rule picks nothing by the size or sign of a return (the daily returns only measure how much
two stocks move together). Risk = the median daily range over 45 sessions. Sectors: the sector column of the public Nasdaq stock screener (https://api.nasdaq.com/api/screener/stocks?download=true), fetched 2026-11-20, used as published and not corrected by hand.

Correlation cap (section 6.3): no stock has a daily-return correlation above 0.75 with
an earlier stock of its group; mean pair correlation: risky 0.33, normal 0.15.

**This file is never edited by hand and does not change during the 20 days.** Another list is a
new entry in the document first. A test pins the count and the shape.
"""

MADE_AT = "2026-11-20T16:32:39+00:00"
LAST_SESSION = "2026-11-19"
MAX_CORRELATION = 0.75

# In order of risk, the most volatile first.
PAPER3_RISKY: tuple[str, ...] = tuple(
    (
        "AXTI CIFR TWST MRNA ASTS CRWV TEM RIOT FPS DOCN ASST CBRS RKLB HIMS CLS SMR MOD CRCL GH FTAI "
        "QXO AKAM ILMN TLN MTZ MP ETSY DKNG CDE AFRM "
    ).split()
)

# In order of closeness to the median risk of the eligible, the nearest first.
PAPER3_NORMAL: tuple[str, ...] = tuple(
    (
        "NKE ISRG TMUS DE DVN DHR ADSK CMG DAL TSM ABNB BX TSLA SLB MELI GM CSCO HWM MRK PBR LNG NEM "
        "GE URI BSX "
    ).split()
)

# symbol -> (group, sector, median daily range over 45 sessions, median dollar volume in millions)
PAPER3_FACTS: dict[str, tuple[str, str, float, int]] = {
    "AXTI": ("risky", "Technology", 0.09208, 549),
    "CIFR": ("risky", "Finance", 0.09019, 464),
    "TWST": ("risky", "Health Care", 0.08457, 312),
    "MRNA": ("risky", "Health Care", 0.07743, 2887),
    "ASTS": ("risky", "Consumer Discretionary", 0.07682, 580),
    "CRWV": ("risky", "Technology", 0.07624, 2713),
    "TEM": ("risky", "Technology", 0.07532, 425),
    "RIOT": ("risky", "Finance", 0.07528, 328),
    "FPS": ("risky", "Energy", 0.0706, 259),
    "DOCN": ("risky", "Technology", 0.06919, 258),
    "ASST": ("risky", "Finance", 0.06848, 222),
    "CBRS": ("risky", "Technology", 0.06664, 1190),
    "RKLB": ("risky", "Industrials", 0.06539, 1119),
    "HIMS": ("risky", "Health Care", 0.06512, 253),
    "CLS": ("risky", "Technology", 0.06506, 761),
    "SMR": ("risky", "Industrials", 0.05677, 214),
    "MOD": ("risky", "Consumer Discretionary", 0.0531, 228),
    "CRCL": ("risky", "Finance", 0.05261, 858),
    "GH": ("risky", "Health Care", 0.05114, 294),
    "FTAI": ("risky", "Industrials", 0.0495, 281),
    "QXO": ("risky", "Consumer Discretionary", 0.04926, 221),
    "AKAM": ("risky", "Consumer Discretionary", 0.04919, 461),
    "ILMN": ("risky", "Health Care", 0.04856, 523),
    "TLN": ("risky", "Utilities", 0.04749, 288),
    "MTZ": ("risky", "Industrials", 0.04738, 294),
    "MP": ("risky", "Basic Materials", 0.04711, 228),
    "ETSY": ("risky", "Consumer Discretionary", 0.04538, 229),
    "DKNG": ("risky", "Consumer Discretionary", 0.04537, 257),
    "CDE": ("risky", "Basic Materials", 0.04433, 552),
    "AFRM": ("risky", "Finance", 0.04327, 282),
    "NKE": ("normal", "Consumer Discretionary", 0.02754, 1071),
    "ISRG": ("normal", "Health Care", 0.02754, 819),
    "TMUS": ("normal", "Telecommunications", 0.02753, 843),
    "DE": ("normal", "Industrials", 0.02749, 780),
    "DVN": ("normal", "Energy", 0.02747, 493),
    "DHR": ("normal", "Industrials", 0.02721, 794),
    "ADSK": ("normal", "Technology", 0.02792, 445),
    "CMG": ("normal", "Consumer Discretionary", 0.02806, 512),
    "DAL": ("normal", "Consumer Discretionary", 0.02813, 512),
    "TSM": ("normal", "Technology", 0.02685, 4576),
    "ABNB": ("normal", "Consumer Discretionary", 0.02834, 662),
    "BX": ("normal", "Finance", 0.02838, 528),
    "TSLA": ("normal", "Industrials", 0.02843, 12475),
    "SLB": ("normal", "Energy", 0.02666, 610),
    "MELI": ("normal", "Consumer Discretionary", 0.02665, 866),
    "GM": ("normal", "Industrials", 0.02855, 574),
    "CSCO": ("normal", "Telecommunications", 0.0265, 1827),
    "HWM": ("normal", "Industrials", 0.02865, 737),
    "MRK": ("normal", "Health Care", 0.02642, 1277),
    "PBR": ("normal", "Energy", 0.02615, 452),
    "LNG": ("normal", "Utilities", 0.02607, 512),
    "NEM": ("normal", "Basic Materials", 0.02601, 722),
    "GE": ("normal", "Technology", 0.02593, 1389),
    "URI": ("normal", "Consumer Discretionary", 0.02925, 501),
    "BSX": ("normal", "Health Care", 0.02586, 757),
}

# Passed over on the way to a full group: (group, symbol, why).
PAPER3_PASSED_OVER: tuple[tuple[str, str, str], ...] = (
    ("risky", "HUT", "correlation 0.89 with CIFR"),
    ("risky", "AAOI", "correlation 0.86 with AXTI"),
    ("risky", "TXG", "correlation 0.88 with TWST"),
    ("risky", "WULF", "correlation 0.85 with CIFR"),
    ("risky", "BE", "correlation 0.76 with AXTI"),
    ("risky", "APLD", "correlation 0.83 with CIFR"),
    ("risky", "COHR", "correlation 0.81 with AXTI"),
    ("risky", "LITE", "correlation 0.85 with AXTI"),
    ("risky", "SMTC", "correlation 0.83 with AXTI"),
    ("risky", "ALAB", "correlation 0.75 with AXTI"),
    ("risky", "TTMI", "correlation 0.81 with FPS"),
    ("risky", "MARA", "correlation 0.80 with RIOT"),
    ("risky", "IONQ", "sector cap (Technology)"),
    ("risky", "MRVL", "sector cap (Technology)"),
    ("risky", "STRL", "correlation 0.85 with FPS"),
    ("risky", "CIEN", "correlation 0.76 with AXTI"),
    ("risky", "MTSI", "sector cap (Technology)"),
    ("risky", "FN", "correlation 0.80 with FPS"),
    ("risky", "ONDS", "sector cap (Technology)"),
    ("risky", "OKLO", "correlation 0.75 with RKLB"),
    ("risky", "WDC", "sector cap (Technology)"),
    ("risky", "MDB", "sector cap (Technology)"),
    ("risky", "SNDK", "veto"),
    ("risky", "OKTA", "sector cap (Technology)"),
    ("risky", "TWLO", "sector cap (Technology)"),
    ("risky", "NET", "sector cap (Technology)"),
    ("risky", "P", "sector cap (Technology)"),
    ("risky", "SMCI", "sector cap (Technology)"),
    ("risky", "BMNR", "correlation 0.80 with ASST"),
    ("risky", "MSTR", "sector cap (Technology)"),
    ("risky", "ENTG", "correlation 0.81 with FPS"),
    ("risky", "RBRK", "sector cap (Technology)"),
    ("risky", "DELL", "sector cap (Technology)"),
    ("risky", "S", "sector cap (Technology)"),
    ("risky", "RBLX", "sector cap (Technology)"),
    ("risky", "U", "sector cap (Technology)"),
    ("risky", "ONTO", "correlation 0.82 with AXTI"),
    ("risky", "FLUT", "sector cap (Technology)"),
    ("risky", "DDOG", "sector cap (Technology)"),
    ("risky", "TER", "correlation 0.80 with AXTI"),
    ("risky", "GLW", "correlation 0.82 with AXTI"),
    ("risky", "HPE", "sector cap (Technology)"),
    ("risky", "INTC", "sector cap (Technology)"),
    ("risky", "ZS", "sector cap (Technology)"),
    ("risky", "FIX", "correlation 0.83 with FPS"),
    ("risky", "VRT", "sector cap (Technology)"),
    ("risky", "COIN", "correlation 0.92 with CRCL"),
    ("risky", "RDDT", "sector cap (Technology)"),
    ("risky", "AMD", "sector cap (Technology)"),
    ("risky", "CRWD", "sector cap (Technology)"),
    ("risky", "FIG", "sector cap (Technology)"),
    ("risky", "GEV", "sector cap (Technology)"),
    ("risky", "LRCX", "sector cap (Technology)"),
    ("risky", "ORCL", "sector cap (Technology)"),
    ("risky", "APP", "sector cap (Technology)"),
    ("risky", "CVNA", "sector cap (Consumer Discretionary)"),
    ("risky", "SWKS", "sector cap (Technology)"),
    ("risky", "EAT", "sector cap (Consumer Discretionary)"),
)

PAPER3_SYMBOLS: tuple[str, ...] = PAPER3_RISKY + PAPER3_NORMAL
