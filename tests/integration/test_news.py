"""The news cache: it stores, notices changes, answers "what about this symbol", and is kept up to
date from a fake Alpaca that answers in the real client's shape."""

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest
from alpaca.data.models.news import News

from tests.fixtures.fakes import asked
from tests.fixtures.news import FakeNewsClient, raw_news
from trading_system.cli import main
from trading_system.data import news as news_module
from trading_system.data.news import (
    MAX_CATCH_UP,
    NewsFeed,
    first_start,
    to_item,
    update_news,
)
from trading_system.storage.news import (
    NewsItem,
    NewsRepository,
    SameNewsDatabaseError,
    open_news_store,
)

UTC = timezone.utc
NOW = datetime(2026, 11, 5, 18, 0, tzinfo=UTC)


def item(
    article_id: int,
    symbols: list[str],
    created_at: datetime,
    *,
    headline: str = "A headline",
    summary: str = "",
    updated_at: datetime | None = None,
) -> NewsItem:
    return to_item(
        News(
            raw_data=raw_news(
                article_id,
                symbols,
                created_at,
                headline=headline,
                summary=summary,
                updated_at=updated_at,
            )
        )
    )


@pytest.fixture
def news_url(tmp_path: Any) -> str:
    return f"sqlite:///{(tmp_path / 'news.db').as_posix()}"


@pytest.fixture
def repo(news_url: str, db_url: str) -> NewsRepository:
    return open_news_store(news_url, db_url)


# ----------------------------------------------------------------------------- the store


def test_a_new_article_is_stored_with_its_symbols_and_the_time_it_was_first_seen(
    repo: NewsRepository,
) -> None:
    published = NOW - timedelta(hours=2)

    report = repo.save([item(1, ["AMZN", "AAPL"], published, headline="Beat")], seen_at=NOW)

    assert (report.new, report.updated, report.unchanged) == (1, 0, 0)
    (stored,) = repo.for_symbol("AMZN", before=NOW, max_age=timedelta(days=1), limit=5)
    assert stored.headline == "Beat" and stored.symbols == ("AAPL", "AMZN")
    assert stored.created_at == published and stored.first_seen_at == NOW


def test_the_same_article_again_changes_nothing(repo: NewsRepository) -> None:
    repo.save([item(1, ["AMZN"], NOW - timedelta(hours=2))], seen_at=NOW - timedelta(hours=1))

    report = repo.save([item(1, ["AMZN"], NOW - timedelta(hours=2))], seen_at=NOW)

    assert (report.new, report.updated, report.unchanged) == (0, 0, 1)
    (stored,) = repo.for_symbol("AMZN", before=NOW, max_age=timedelta(days=1), limit=5)
    assert stored.first_seen_at == NOW - timedelta(hours=1)  # the first sight stays


def test_a_changed_article_is_taken_in_place_and_keeps_its_first_sight(
    repo: NewsRepository,
) -> None:
    published = NOW - timedelta(hours=3)
    first_seen = NOW - timedelta(hours=2)
    repo.save([item(1, ["AMZN"], published, summary="old")], seen_at=first_seen)

    report = repo.save(
        [item(1, ["AMZN", "MSFT"], published, summary="new", updated_at=NOW - timedelta(hours=1))],
        seen_at=NOW,
    )

    assert (report.new, report.updated, report.unchanged) == (0, 1, 0)
    (stored,) = repo.for_symbol("MSFT", before=NOW, max_age=timedelta(days=1), limit=5)
    assert stored.summary == "new" and stored.symbols == ("AMZN", "MSFT")  # the new tag is there
    assert stored.first_seen_at == first_seen
    assert repo.count() == 1


def test_one_article_twice_in_one_batch_counts_once_and_the_later_copy_wins(
    repo: NewsRepository,
) -> None:
    published = NOW - timedelta(hours=3)
    older = item(1, ["AMZN"], published, headline="first")
    newer = item(1, ["AMZN"], published, headline="second", updated_at=NOW)

    report = repo.save([older, newer], seen_at=NOW)

    assert report.new == 1 and repo.count() == 1
    (stored,) = repo.for_symbol("AMZN", before=NOW, max_age=timedelta(days=1), limit=5)
    assert stored.headline == "second"


def test_nothing_in_nothing_out(repo: NewsRepository) -> None:
    assert repo.save([], seen_at=NOW).new == 0
    assert repo.count() == 0 and repo.latest_created_at() is None


def test_for_symbol_returns_the_newest_first_inside_the_window_and_for_that_symbol_only(
    repo: NewsRepository,
) -> None:
    repo.save(
        [
            item(1, ["AMZN"], NOW - timedelta(hours=30)),  # too old
            item(2, ["AMZN"], NOW - timedelta(hours=5)),
            item(3, ["AMZN"], NOW - timedelta(hours=1)),
            item(4, ["MSFT"], NOW - timedelta(hours=1)),  # another symbol
            item(5, ["AMZN"], NOW + timedelta(minutes=5)),  # after the decision
        ],
        seen_at=NOW,
    )

    found = repo.for_symbol("AMZN", before=NOW, max_age=timedelta(hours=24), limit=10)
    limited = repo.for_symbol("AMZN", before=NOW, max_age=timedelta(hours=24), limit=1)

    assert [n.id for n in found] == [3, 2]
    assert [n.id for n in limited] == [3]


def test_a_round_up_can_be_left_out_by_the_number_of_symbols_it_names(
    repo: NewsRepository,
) -> None:
    crowd = [f"S{n:02d}" for n in range(12)]
    repo.save(
        [
            item(1, ["AMZN"], NOW - timedelta(hours=2)),
            item(2, ["AMZN", *crowd], NOW - timedelta(hours=1)),
        ],
        seen_at=NOW,
    )

    everything = repo.for_symbol("AMZN", before=NOW, max_age=timedelta(days=1), limit=10)
    focused = repo.for_symbol(
        "AMZN", before=NOW, max_age=timedelta(days=1), limit=10, max_symbols=3
    )

    assert [n.id for n in everything] == [2, 1] and [n.id for n in focused] == [1]


def test_what_was_known_at_a_moment_is_not_what_was_published_before_it(
    repo: NewsRepository,
) -> None:
    published = NOW - timedelta(days=3)
    repo.save([item(1, ["AMZN"], published)], seen_at=NOW)  # filled in afterwards
    then = NOW - timedelta(days=2)

    published_by_then = repo.for_symbol("AMZN", before=then, max_age=timedelta(days=5), limit=5)
    known_by_then = repo.for_symbol(
        "AMZN", before=then, max_age=timedelta(days=5), limit=5, seen_by=True
    )

    assert len(published_by_then) == 1 and known_by_then == []


def test_the_newest_published_time_is_what_an_update_starts_from(repo: NewsRepository) -> None:
    repo.save(
        [item(1, ["AMZN"], NOW - timedelta(hours=9)), item(2, ["AAPL"], NOW - timedelta(hours=2))],
        seen_at=NOW,
    )

    assert repo.latest_created_at() == NOW - timedelta(hours=2)


def test_the_news_file_refuses_to_be_another_database_of_the_system(
    news_url: str, db_url: str
) -> None:
    with pytest.raises(SameNewsDatabaseError, match="NEWS_DATABASE_URL"):
        open_news_store(db_url, db_url)
    with pytest.raises(SameNewsDatabaseError):
        open_news_store(news_url, db_url, news_url)  # e.g. the research file named the same


def test_the_news_tables_are_not_in_the_trading_database(
    repo: NewsRepository, session_factory: Any
) -> None:
    with session_factory() as session:
        names = {
            row[0]
            for row in session.connection().exec_driver_sql(
                "select name from sqlite_master where type = 'table'"
            )
        }

    assert "news_articles" not in names and "news_symbols" not in names


# ----------------------------------------------------------------------------- the update


def _feed(client: FakeNewsClient, settings: Any) -> NewsFeed:
    feed = NewsFeed(settings)
    feed.client = client
    return feed


def test_the_feed_asks_for_the_symbols_with_the_text_oldest_first(settings: Any) -> None:
    client = FakeNewsClient([raw_news(1, ["AMZN"], NOW - timedelta(hours=1), summary="s")])

    items = _feed(client, settings).fetch(["AMZN", "AAPL"], NOW - timedelta(days=1), NOW)

    (request,) = client.requests
    assert request.symbols == "AMZN,AAPL" and request.include_content is True
    assert request.sort == "asc" and request.limit == news_module.FETCH_LIMIT
    assert [i.id for i in items] == [1]
    assert _feed(client, settings).fetch([], NOW, NOW) == []  # no symbols, no request
    assert len(client.requests) == 1


def test_the_first_update_takes_a_week_a_day_at_a_time(repo: NewsRepository, settings: Any) -> None:
    articles = [raw_news(n, ["AMZN"], NOW - timedelta(hours=7 + 24 * n)) for n in range(1, 6)]
    client = FakeNewsClient(articles)

    report = update_news(_feed(client, settings), repo, ["AMZN"], now=NOW)

    assert report.new == 5 and repo.count() == 5
    assert report.windows == len(client.requests) == 8  # seven days, a second of overlap each
    assert asked(client.requests[0].start) == NOW - timedelta(days=7)
    assert asked(client.requests[-1].end) == NOW


def test_the_next_update_starts_a_little_before_the_newest_stored_article(
    repo: NewsRepository, settings: Any
) -> None:
    newest = NOW - timedelta(minutes=5)
    client = FakeNewsClient(
        [
            raw_news(1, ["AMZN"], NOW - timedelta(hours=9)),
            raw_news(2, ["AMZN"], newest),
        ]
    )
    update_news(_feed(client, settings), repo, ["AMZN"], now=NOW - timedelta(minutes=4))
    client.requests.clear()
    client.articles.append(raw_news(3, ["AMZN"], NOW - timedelta(minutes=1)))

    report = update_news(_feed(client, settings), repo, ["AMZN"], now=NOW)

    assert report.new == 1 and report.unchanged == 2 and repo.count() == 3
    assert asked(client.requests[0].start) == newest - timedelta(days=2)


def test_an_article_the_provider_changed_after_we_stored_it_is_taken_again(
    repo: NewsRepository, settings: Any
) -> None:
    published = NOW - timedelta(hours=6)
    client = FakeNewsClient([raw_news(1, ["AMZN"], published, headline="before")])
    update_news(_feed(client, settings), repo, ["AMZN"], now=NOW - timedelta(hours=1))
    client.articles[0] = raw_news(1, ["AMZN"], published, headline="after", updated_at=NOW)

    report = update_news(_feed(client, settings), repo, ["AMZN"], now=NOW)

    assert report.updated == 1 and report.new == 0
    (stored,) = repo.for_symbol("AMZN", before=NOW, max_age=timedelta(days=1), limit=5)
    assert stored.headline == "after"


def test_after_a_long_stop_the_catch_up_goes_back_a_month_and_no_further() -> None:
    long_ago = NOW - timedelta(days=200)

    assert first_start(long_ago, NOW) == NOW - MAX_CATCH_UP
    assert first_start(None, NOW) == NOW - timedelta(days=7)
    assert first_start(NOW - timedelta(hours=1), NOW) == NOW - timedelta(hours=1, days=2)
    assert first_start(NOW, NOW, days=90) == NOW - timedelta(days=90)  # asked for, so given


def test_a_window_that_reaches_the_limit_is_reported_as_possibly_incomplete(
    repo: NewsRepository, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(news_module, "FETCH_LIMIT", 3)
    client = FakeNewsClient(
        [raw_news(n, ["AMZN"], NOW - timedelta(hours=1, minutes=n)) for n in range(1, 6)]
    )

    report = update_news(_feed(client, settings), repo, ["AMZN"], now=NOW, days=1)

    assert report.full_windows  # the day held more than it would give


# ------------------------------------------------------------------------------- the command


def _fetch(settings: Any, *argv: str) -> int:
    with patch("trading_system.cli.Settings", return_value=settings):
        return main(["fetch-news", *argv])


@pytest.fixture
def fake_alpaca(monkeypatch: pytest.MonkeyPatch, settings: Any) -> Any:
    def install(client: FakeNewsClient) -> None:
        def build(_settings: Any) -> NewsFeed:
            feed = NewsFeed(settings)
            feed.client = client
            return feed

        monkeypatch.setattr("trading_system.data.news.NewsFeed", build)

    return install


def test_the_command_fills_the_cache_and_says_what_it_did(
    fake_alpaca: Any,
    make_settings: Any,
    news_url: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = datetime.now(UTC)
    fake_alpaca(FakeNewsClient([raw_news(1, ["AMZN"], now - timedelta(hours=2), headline="Beat")]))
    settings = make_settings(news_database_url=news_url)

    assert _fetch(settings, "--symbols", "amzn,aapl") == 0
    assert _fetch(settings, "--symbols", "amzn,aapl") == 0

    out = capsys.readouterr().out.splitlines()
    assert "News for 2 symbols: 1 new, 0 changed, 0 already stored" in out[0]
    assert "1 articles in the cache" in out[0]
    assert "0 new, 0 changed, 1 already stored" in out[1]  # the second run found nothing new


def test_the_command_says_so_when_no_news_comes_back(
    fake_alpaca: Any,
    make_settings: Any,
    news_url: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_alpaca(FakeNewsClient([]))

    assert _fetch(make_settings(news_database_url=news_url)) == 0

    assert "No news came back." in capsys.readouterr().out


def test_the_command_refuses_the_trading_or_the_research_file_as_the_news_file(
    make_settings: Any,
    db_url: str,
    tmp_path: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    research = f"sqlite:///{(tmp_path / 'research.db').as_posix()}"

    assert _fetch(make_settings(news_database_url=db_url)) == 1
    assert "has a file of its own" in capsys.readouterr().out
    assert _fetch(make_settings(news_database_url=research, research_database_url=research)) == 1
    assert "has a file of its own" in capsys.readouterr().out


def test_the_command_fails_cleanly_when_alpaca_does(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: Any,
    news_url: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(_settings: Any) -> Any:
        raise RuntimeError("no network")

    monkeypatch.setattr("trading_system.data.news.NewsFeed", broken)

    assert _fetch(make_settings(news_database_url=news_url)) == 1

    assert "fetch-news error: no network" in capsys.readouterr().out
