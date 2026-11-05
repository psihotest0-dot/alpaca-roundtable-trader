"""The research universe (data/universe.py) is a list the owner approved: these tests keep it a
list the rest of the system can use."""

from trading_system.config import Settings
from trading_system.data.universe import MARKET_SYMBOL, SECTORS, research_universe


def test_the_universe_is_the_size_the_owner_asked_for_and_has_no_repeats() -> None:
    symbols = research_universe()

    assert 100 <= len(symbols) <= 200  # "about 100 to 200 large stocks plus SPY"
    assert len(symbols) == len(set(symbols))
    assert len(symbols) == sum(len(group) for group in SECTORS.values())  # no stock in two sectors


def test_every_symbol_is_a_plain_upper_case_ticker() -> None:
    for symbol in research_universe():
        assert symbol == symbol.upper() and symbol.isalnum(), symbol


def test_the_market_is_in_it_for_the_regime_filter() -> None:
    assert MARKET_SYMBOL == "SPY" and MARKET_SYMBOL in research_universe()


def test_a_renamed_company_is_listed_under_its_current_name() -> None:
    symbols = research_universe()

    assert "MRSH" in symbols and "MMC" not in symbols  # Marsh McLennan, January 2026


def test_the_symbols_pass_the_same_check_as_the_watchlist() -> None:
    watchlist = ",".join(research_universe())

    settings = Settings(
        _env_file=None, alpaca_api_key="k", alpaca_secret_key="s", watchlist=watchlist
    )

    assert settings.watchlist_symbols == research_universe()
