"""From Alpaca's article to what the cache keeps: plain text, cut to size."""

from datetime import datetime, timezone

from alpaca.data.models.news import News

from tests.fixtures.news import raw_news
from trading_system.data.news import MAX_CONTENT_CHARS, strip_html, to_item

T0 = datetime(2026, 11, 5, 14, 30, tzinfo=timezone.utc)


def test_html_becomes_one_paragraph_per_line_without_tags_or_link_addresses() -> None:
    html = (
        '<p class="x"><strong>The Cigna Group</strong> <a href="https://www.benzinga.com/quote/CI" '
        'target="_blank">(NYSE: CI)</a> beat estimates.</p><p>Second&nbsp;paragraph &amp; more.</p>'
    )

    assert (
        strip_html(html) == "The Cigna Group (NYSE: CI) beat estimates.\nSecond paragraph & more."
    )


def test_scripts_and_styles_are_dropped_and_lists_keep_their_items_apart() -> None:
    html = (
        "<style>p{color:red}</style><ul><li>one</li><li>two</li></ul><script>alert(1)</script>end"
    )

    assert strip_html(html) == "one\ntwo\nend"


def test_plain_text_passes_through_only_tidied() -> None:
    assert strip_html("  Cigna   beat\n\n estimates.  ") == "Cigna beat\nestimates."
    assert strip_html("") == "" and strip_html("   ") == ""


def test_an_article_becomes_an_item_with_clean_symbols_and_text() -> None:
    article = News(
        raw_data=raw_news(
            7,
            ["amzn", "AAPL", "AMZN", " msft "],
            T0,
            headline="  Needham   Maintains Buy ",
            summary="<p>Short <b>summary</b>.</p>",
            content="<p>Body.</p>",
        )
    )

    item = to_item(article)

    assert item.symbols == ("AMZN", "AAPL", "MSFT")  # upper case, no repeats, order kept
    assert item.headline == "Needham Maintains Buy"
    assert item.summary == "Short summary." and item.content == "Body."
    assert item.content_truncated is False
    assert item.created_at == T0 and item.url == "https://example.com/news/7"


def test_a_long_body_is_cut_and_marked() -> None:
    long_body = "<p>" + "word " * 2000 + "</p>"

    item = to_item(News(raw_data=raw_news(8, ["AAPL"], T0, content=long_body)))

    assert len(item.content) == MAX_CONTENT_CHARS and item.content_truncated is True


def test_a_headline_only_article_keeps_empty_text_not_none() -> None:
    item = to_item(News(raw_data=raw_news(9, ["AAPL"], T0, headline="Upgrade")))

    assert item.summary == "" and item.content == "" and item.content_truncated is False


def test_an_article_without_symbols_has_an_empty_tuple() -> None:
    assert to_item(News(raw_data=raw_news(10, [], T0))).symbols == ()
