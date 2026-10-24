"""Product finder page: answers, ranked products, links to their pages, packaging."""

import json

from spricer.finder import CARDS, Objective, Protection, View, VolView
from web import finder_page as page
from web import registry
from web.market import default_data


def text_of(component) -> str:
    return json.dumps(component.to_plotly_json(), default=lambda o: o.to_plotly_json())


def test_every_card_links_to_a_page_of_the_site():
    for card in CARDS:
        assert registry.find(registry.slugify(card.name)) is not None, card.name


def test_the_page_asks_every_question():
    text = text_of(page.finder_page())
    for piece in ("finder-view", "finder-vol", "finder-horizon", "finder-protection"):
        assert piece in text


def test_results_rank_explain_and_link():
    top, packaging = page.results(
        View.MILD_BULL.value,
        VolView.NEUTRAL.value,
        3.0,
        Protection.FULL.value,
        Objective.GROWTH.value,
        default_data(),
    )
    text = text_of(top)
    assert "1. Shark note (up)" in text
    assert "/product/shark-note-up" in text
    assert f"Every product, ranked ({len(CARDS)})" in text
    assert "Same payoff, different packaging" in text_of(packaging)
    assert "Collateral" in text_of(packaging)


def test_defaults():
    top, packaging = page.results(None, None, None, None, None, default_data())
    assert "Products that fit" in text_of(top)
    assert "packaging" in text_of(packaging)


def test_invalid_market_data_are_reported():
    _, packaging = page.results(
        View.BULLISH.value, None, 1.0, None, None, default_data() | {"spot": -1.0}
    )
    assert "market data are invalid" in text_of(packaging)


def test_a_card_without_a_page_has_no_thumbnail(monkeypatch):
    from spricer.finder import Recommendation

    card = CARDS[0]
    monkeypatch.setattr(registry, "find", lambda slug: None)
    rendered = page._recommendation(Recommendation(card, 3.0, ("ok",), ("careful",)), 1)
    assert "Img" not in text_of(rendered)
