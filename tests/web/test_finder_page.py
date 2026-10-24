"""Product finder page: answers, ranked products, links to their pages."""

import json

from spricer.finder import CARDS, Objective, Recommendation, View, VolView
from web import finder_page as page
from web import registry


def text_of(component) -> str:
    return json.dumps(component.to_plotly_json(), default=lambda o: o.to_plotly_json())


def test_every_card_links_to_a_page_of_the_site():
    for card in CARDS:
        assert registry.find(registry.slugify(card.name)) is not None, card.name


def test_the_page_asks_every_question():
    text = text_of(page.finder_page())
    for piece in (
        "finder-view",
        "finder-vol",
        "finder-horizon",
        "finder-max-loss",
        "finder-barrier",
    ):
        assert piece in text
    assert "packaging" not in text


def test_results_rank_explain_and_link():
    text = text_of(
        page.results(
            View.MILD_BULL.value, VolView.NEUTRAL.value, 3.0, 0, False, Objective.GROWTH.value
        )
    )
    assert "1. Shark note (up)" in text
    assert "/product/shark-note-up" in text
    assert f"Every product, ranked ({len(CARDS)})" in text
    assert "ranking-table" in text
    assert "ranking-out" in text  # products that can lose more than accepted are greyed out
    assert "packaging" not in text


def test_a_larger_loss_and_a_barrier_change_the_ranking():
    buffer = text_of(page.results(View.MILD_BULL.value, None, 3.0, 90, False, None))
    assert "Loss limited to 90 %" in buffer
    barrier = text_of(page.results(View.RANGE.value, None, 3.0, 0, True, Objective.INCOME.value))
    assert "1. Phoenix autocall" in barrier


def test_defaults():
    assert "Products that fit" in text_of(page.results(None, None, None, None, None, None))


def test_a_card_without_a_page_has_no_thumbnail(monkeypatch):
    card = CARDS[0]
    monkeypatch.setattr(registry, "find", lambda slug: None)
    rendered = page._recommendation(Recommendation(card, 3.0, ("ok",), ("careful",)), 1)
    assert "Img" not in text_of(rendered)
