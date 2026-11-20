"""The paper3 portfolio: 30 risky and 25 normal stocks (docs/paper-third-account.md, section 6).

Made from `scripts/select_paper3_portfolio.py` (made at 2026-11-20T15:53:01+00:00, last completed
session 2026-11-19, 450 eligible stocks, median risk 0.02757) and
`scripts/freeze_paper3_portfolio.py` (the review of docs/paper-third-account.md, step 2). The rule looks at
no return of any kind. Risk = the median daily range over 45 sessions. Sectors: the sector column of the public Nasdaq stock screener (https://api.nasdaq.com/api/screener/stocks?download=true), fetched 2026-11-20, used as published and not corrected by hand.

**This file is never edited by hand and does not change during the 20 days.** Another list is a
new entry in the document first. A test pins the count and the shape.
"""

MADE_AT = "2026-11-20T15:53:01+00:00"
LAST_SESSION = "2026-11-19"

# In order of risk, the most volatile first.
PAPER3_RISKY: tuple[str, ...] = tuple(
    (
        "AXTI CIFR HUT TWST AAOI TXG MRNA ASTS WULF CRWV BE TEM RIOT APLD COHR LITE FPS MARA ASST "
        "RKLB HIMS STRL CIEN FN OKLO SMR ENTG MOD GH ONTO "
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
    "HUT": ("risky", "Finance", 0.08765, 356),
    "TWST": ("risky", "Health Care", 0.08457, 312),
    "AAOI": ("risky", "Technology", 0.08057, 797),
    "TXG": ("risky", "Industrials", 0.08044, 222),
    "MRNA": ("risky", "Health Care", 0.07743, 2887),
    "ASTS": ("risky", "Consumer Discretionary", 0.07682, 580),
    "WULF": ("risky", "Finance", 0.07673, 482),
    "CRWV": ("risky", "Technology", 0.07624, 2713),
    "BE": ("risky", "Energy", 0.07567, 3310),
    "TEM": ("risky", "Technology", 0.07532, 425),
    "RIOT": ("risky", "Finance", 0.07528, 328),
    "APLD": ("risky", "Technology", 0.07283, 416),
    "COHR": ("risky", "Technology", 0.07187, 1857),
    "LITE": ("risky", "Telecommunications", 0.0715, 3951),
    "FPS": ("risky", "Energy", 0.0706, 259),
    "MARA": ("risky", "Finance", 0.06892, 325),
    "ASST": ("risky", "Finance", 0.06848, 222),
    "RKLB": ("risky", "Industrials", 0.06539, 1119),
    "HIMS": ("risky", "Health Care", 0.06512, 253),
    "STRL": ("risky", "Industrials", 0.06125, 269),
    "CIEN": ("risky", "Utilities", 0.06066, 970),
    "FN": ("risky", "Utilities", 0.06017, 333),
    "OKLO": ("risky", "Utilities", 0.0579, 281),
    "SMR": ("risky", "Industrials", 0.05677, 214),
    "ENTG": ("risky", "Industrials", 0.05318, 363),
    "MOD": ("risky", "Consumer Discretionary", 0.0531, 228),
    "GH": ("risky", "Health Care", 0.05114, 294),
    "ONTO": ("risky", "Industrials", 0.04991, 224),
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
    ("risky", "SMTC", "sector cap (Technology)"),
    ("risky", "ALAB", "sector cap (Technology)"),
    ("risky", "TTMI", "sector cap (Technology)"),
    ("risky", "DOCN", "sector cap (Technology)"),
    ("risky", "CBRS", "sector cap (Technology)"),
    ("risky", "CLS", "sector cap (Technology)"),
    ("risky", "IONQ", "sector cap (Technology)"),
    ("risky", "MRVL", "sector cap (Technology)"),
    ("risky", "MTSI", "sector cap (Technology)"),
    ("risky", "ONDS", "sector cap (Technology)"),
    ("risky", "WDC", "sector cap (Technology)"),
    ("risky", "MDB", "sector cap (Technology)"),
    ("risky", "SNDK", "veto"),
    ("risky", "OKTA", "sector cap (Technology)"),
    ("risky", "TWLO", "sector cap (Technology)"),
    ("risky", "NET", "sector cap (Technology)"),
    ("risky", "P", "sector cap (Technology)"),
    ("risky", "SMCI", "sector cap (Technology)"),
    ("risky", "BMNR", "sector cap (Finance)"),
    ("risky", "MSTR", "sector cap (Technology)"),
    ("risky", "CRCL", "sector cap (Finance)"),
    ("risky", "RBRK", "sector cap (Technology)"),
    ("risky", "DELL", "sector cap (Technology)"),
    ("risky", "S", "sector cap (Technology)"),
    ("risky", "RBLX", "sector cap (Technology)"),
    ("risky", "U", "sector cap (Technology)"),
)

PAPER3_SYMBOLS: tuple[str, ...] = PAPER3_RISKY + PAPER3_NORMAL
