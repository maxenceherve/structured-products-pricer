"""Dash application: registry, thumbnails, market store and callbacks, without a browser."""

import base64
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from dash import html

from spricer.market import InterpolatedVolSurface
from spricer.products.capital_protection import NOTE_CATALOG
from spricer.products.coupon_notes import COUPON_NOTE_CATALOG
from spricer.products.participation import (
    PARTICIPATION_CATALOG,
    SHARK_WARRANTS,
    YIELD_CERTIFICATES,
)
from spricer.strategies import CATALOG
from web import layout, registry
from web.app import (
    apply_edits,
    create_app,
    market_view,
    panel_values,
    reset,
    route,
    sync_panel,
    toggle,
)
from web.market import MATURITY, default_data, market_from_data, strike_field
from web.thumbnails import payoff_data_uri, payoff_svg


def text_of(component) -> str:
    """Every string in a Dash component tree."""
    return json.dumps(component.to_plotly_json(), default=lambda o: o.to_plotly_json())


# --- Registry and thumbnails -----------------------------------------------------------------


def test_every_product_of_the_library_is_in_the_gallery_once():
    names = [p.name for p in registry.PRODUCTS]
    expected = [s.name for s in CATALOG] + [s.name for s in NOTE_CATALOG]
    expected += [s.name for s in COUPON_NOTE_CATALOG + PARTICIPATION_CATALOG]
    expected += [s.name for s in YIELD_CERTIFICATES + SHARK_WARRANTS]
    for name in expected:
        assert names.count(name) == 1, name
    assert len({p.slug for p in registry.PRODUCTS}) == len(registry.PRODUCTS)
    assert len(registry.PRODUCTS) == 56  # the count given in the README
    assert sum(p.family == registry.BARRIERS for p in registry.PRODUCTS) == 10
    assert sum(p.family == registry.CERTIFICATES for p in registry.PRODUCTS) == 8
    assert {p.family for p in registry.PRODUCTS} == set(registry.FAMILIES)
    assert registry.find("bull-call-spread") is not None
    assert registry.find("nothing") is None


def test_search_matches_every_word():
    assert [p.name for p in registry.matching("call spread")] == [
        "Bull call spread",
        "Bear call spread",
        "Digital call by call spread",
    ]
    assert {p.family for p in registry.matching("autocall")} == {registry.AUTOCALLS}
    assert registry.matching("") == list(registry.PRODUCTS)
    assert registry.matching("zzz") == []
    assert registry.slugify("Shark note (up)") == "shark-note-up"


@pytest.mark.parametrize("product", registry.PRODUCTS, ids=lambda p: p.slug)
def test_thumbnails_are_valid_svg_inside_their_box(product):
    svg = payoff_svg(product)
    root = ET.fromstring(svg)
    lines = root.findall("{http://www.w3.org/2000/svg}polyline")
    assert len(lines) == 2
    for line in lines:
        xy = np.array([[float(v) for v in p.split(",")] for p in line.attrib["points"].split()])
        assert np.all((xy[:, 0] >= 0) & (xy[:, 0] <= 240))
        assert np.all((xy[:, 1] >= 0) & (xy[:, 1] <= 140))
    markers = root.findall("{http://www.w3.org/2000/svg}line")
    assert len(markers) == (product.marker is not None)
    decoded = base64.b64decode(payoff_data_uri(product).split(",", 1)[1]).decode()
    assert decoded == svg


def test_a_flat_payoff_is_drawn_in_the_middle():
    flat = registry.Product("x", "Flat", registry.OPTIONS, "", lambda s: np.full_like(s, 3.0))
    root = ET.fromstring(payoff_svg(flat))
    line = root.findall("{http://www.w3.org/2000/svg}polyline")[1]
    ys = {float(p.split(",")[1]) for p in line.attrib["points"].split()}
    assert ys == {70.0}


# --- Market store ------------------------------------------------------------------------------


def test_default_market_round_trips():
    data = default_data()
    market = market_from_data(data)
    assert market.spot == 100.0
    assert isinstance(market.vol, InterpolatedVolSurface)
    assert data["rows"][0][MATURITY] == 0.25
    assert strike_field(100.0) in data["rows"][0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"spot": None}, "spot must be a number"),
        ({"spot": -1.0}, "spot must be positive"),
        ({"rate_pct": "abc"}, "rate must be a number"),
        ({"dividend_pct": float("nan")}, "dividend yield must be a number"),
        ({"rows": []}, "at least one maturity"),
    ],
)
def test_invalid_market_data_is_explained(change, message):
    with pytest.raises(ValueError, match=message):
        market_from_data(default_data() | change)


def test_a_missing_volatility_is_named():
    data = default_data()
    data["rows"][1][strike_field(100.0)] = None
    with pytest.raises(ValueError, match="volatility at 100%"):
        market_from_data(data)


def test_empty_rows_are_ignored_and_rows_are_sorted():
    data = default_data()
    data["rows"] = [*reversed(data["rows"]), {MATURITY: None}]
    market = market_from_data(data)
    assert isinstance(market.vol, InterpolatedVolSurface)
    assert list(market.vol.maturities) == sorted(market.vol.maturities)


# --- Callbacks ---------------------------------------------------------------------------------


def test_routes():
    assert "Options & strategies" in text_of(route("/", None))
    assert "Bull call spread" in text_of(route("/product/bull-call-spread/", None))
    assert "Monte Carlo" in text_of(route("/product/phoenix-autocall", None))
    assert "Monte Carlo" not in text_of(route("/product/call", None))
    assert "Monte Carlo" in text_of(route("/product/shark-warrant-call", None))
    assert "Monte Carlo" in text_of(route("/product/bonus-certificate", None))
    assert "Find a product" in text_of(route("/finder", None))
    assert "Page not found" in text_of(route("/product/nothing", None))
    assert "Page not found" in text_of(route("/elsewhere", None))
    searched = text_of(route("/product/call", "phoenix"))
    assert "Phoenix autocall" in searched
    assert "Bull call spread" not in searched
    assert "No product matches" in text_of(route(None, "zzz"))


def test_the_panel_opens_and_closes():
    assert toggle(1, [], False) is True
    assert toggle(2, None, True) is False
    assert toggle(None, [None], True) is True
    assert toggle(None, [1], False) is True  # the link on a product page


def test_edits_update_a_copy_of_the_market():
    data = default_data()
    field = strike_field(100.0)
    edits = [
        {"rowIndex": 0, "colId": field, "value": 25.0},
        {"rowIndex": 99, "colId": field, "value": 1.0},
        {"rowIndex": "x", "colId": field, "value": 1.0},
    ]
    out = apply_edits(110.0, None, 1.5, 25.0, edits, data)
    assert out["spot"] == 110.0
    assert out["rate_pct"] == data["rate_pct"]
    assert out["dividend_pct"] == 1.5
    assert out["rows"][0][field] == 25.0
    assert data["rows"][0][field] != 25.0  # the stored data is not mutated
    assert out["funding_bp"] == 25.0
    assert apply_edits(None, None, None, None, None, data) == data
    assert panel_values(out)[0] == 110.0
    assert sync_panel(True, out)[0] == 110.0
    fresh = reset(1)
    assert fresh[0] == default_data()
    assert fresh[1] == 100.0
    assert fresh[4] == 0.0


def test_the_market_summary_and_the_funding_spread():
    from web.app import market_summary
    from web.market import funding_bp

    text = text_of(html.Div(market_summary(default_data() | {"funding_bp": 40.0})))
    assert "Funding spread" in text
    assert "40 bp" in text
    assert "ATM volatility, 1 year" in text
    assert "invalid" in text_of(market_summary(default_data() | {"spot": None}))
    assert funding_bp({"funding_bp": "x"}) == 0.0
    assert funding_bp({"funding_bp": float("inf")}) == 0.0
    assert funding_bp({}) == 0.0


def test_market_view_reports_status_and_smile():
    status, figure = market_view(default_data())
    assert "No static arbitrage" in text_of(status)
    assert len(figure.data) == 3
    data = default_data()
    data["rows"][0][strike_field(100.0)] = 60.0
    status, _ = market_view(data)
    assert "static arbitrage" in text_of(status)
    assert "warning" in text_of(status)
    status, figure = market_view(default_data() | {"spot": -5.0})
    assert "positive" in text_of(status)
    assert len(figure.data) == 0


def test_the_app_builds_with_every_piece():
    app = create_app()
    text = text_of(app.layout)
    for piece in ("market-panel", "vol-grid", "search", "market-button", "page"):
        assert piece in text
    assert len(app.callback_map) == 38
    assert isinstance(layout.not_found("/x"), html.Div)


def test_a_flat_market_has_no_smile_to_draw():
    from spricer.market import FlatVol, Market

    figure = layout.smile_figure(Market(100.0, 0.03, 0.02, FlatVol(0.2)))
    assert len(figure.data) == 0
    status = layout.market_status(Market(100.0, 0.03, 0.02, FlatVol(0.2)), None)
    assert "No static arbitrage" in text_of(status)


def test_the_finder_answers_prefill_the_product_pages():
    from web import ui

    assert ui.prefill("?maturity=2&protection=80&x=1&buffer=abc&notional=-5") == {
        "maturity": 2.0,
        "protection": 80.0,
    }
    assert ui.prefill(None) == {}
    note = text_of(route("/product/participation-note", None, "?maturity=2&protection=80"))
    assert "Set from your answers in the finder: maturity 2 years, capital protection 80 %" in note
    buffer = text_of(route("/product/buffer-certificate", None, "?buffer=75&protection=25"))
    assert "buffer level 75 %" in buffer
    assert "capital protection" not in buffer  # the buffer certificate has no protection term
    option = text_of(route("/product/call", None, "?maturity=2"))
    assert "maturity 2 years" in option
    coupon = text_of(route("/product/phoenix-autocall", None, "?maturity=2"))
    assert "maturity 2 years" in coupon
    assert "Set from your answers" not in text_of(route("/product/call", None, None))
