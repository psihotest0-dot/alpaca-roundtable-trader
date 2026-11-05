"""Strategy research (Phase 3b, docs/strategy-research-plan.md): measure what prices do after a
signal, not what an account would do.

    panel.py    the daily bars of the research universe as aligned tables, development part only
    signals.py  the signals under test, each a table of "on" flags that uses only the past
    stats.py    the standard error of a mean of overlapping observations (Newey-West)
    measure.py  forward returns after a signal, against all stocks and against SPY
    trials.py   the count of variants tried, kept in the journal

The part of the history from HOLDOUT_START on is the hold-out. Nothing here loads it.
"""
