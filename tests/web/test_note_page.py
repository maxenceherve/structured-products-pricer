"""Note pages (capital protection, certificates, shark warrants): terms, tabs, risk, sheet."""

import json
from datetime import date

import dash
import numpy as np
import pytest

from spricer.market import FlatVol, Market
from spricer.monitoring import Monitoring
from web import note_page as page
from web import registry, risk_panel, sheet, sheet_route, ui
from web.app import kpis
from web.market import default_data, market_from_data
from web.notes import CATALOG, build_note, default_terms, note_spec, shown_params, spot_profile

NOTE_PRODUCTS = [p for p in registry.PRODUCTS if note_spec(p) is not None]
MARKET = market_from_data(default_data())


def text_of(component) -> str:
    return json.dumps(component.to_plotly_json(), default=lambda o: o.to_plotly_json())


def store(name: str, **terms):
    spec = CATALOG[name]
    return {"slug": registry.slugify(name), "terms": default_terms(spec) | terms}


def position(name: str, **terms):
    return build_note(CATALOG[name], store(name, **terms)["terms"], MARKET)


# --- Model -------------------------------------------------------------------------------------


def test_every_note_of_the_gallery_has_a_page():
    names = {p.name for p in NOTE_PRODUCTS}
    assert names == set(CATALOG)
    assert note_spec(registry.BY_SLUG["call"]) is None


@pytest.mark.parametrize("product", NOTE_PRODUCTS, ids=lambda p: p.slug)
def test_every_note_page_prices(product):
    s = store(product.name)
    pos = build_note(CATALOG[product.name], s["terms"], MARKET)
    assert np.isfinite(pos.value())
    assert page.note_page(product) is not None
    assert "Price" in text_of(kpis(s, default_data()))


def test_initial_level_spread_and_maturity():
    today = position("Participation note")
    assert today.note.initial == MARKET.spot
    assert today.performance_pct == pytest.approx(100.0)
    alive = position("Participation note", initial=80.0)
    assert alive.performance_pct == pytest.approx(125.0)
    spread = position("Participation note", spread_bp=100.0)
    assert spread.value() < today.value()
    assert spread.bond_pct() == pytest.approx(100.0 * np.exp(-(MARKET.rate + 0.01) * 3.0))
    shorter = today.at(1.0)
    assert shorter.maturity == 1.0
    assert today.value(maturity=1.0) == pytest.approx(float(shorter.value(MARKET)))


def test_toggles_and_monitoring():
    capped = position("Bonus certificate", capped=True)
    assert "Cap" in [lv.name for lv in capped.note.levels]
    assert [p.key for p in shown_params(CATALOG["Bonus certificate"], {"capped": False})] == [
        "bonus",
        "barrier",
        "participation",
        "strike",
    ]
    daily = position("Shark note (up)", monitoring="daily")
    assert daily.monitoring is Monitoring.DISCRETE
    assert daily.at(1.0).legs[1].brick.observations == 252
    assert daily.warning() is None
    short = position("Shark note (up)", monitoring="daily", maturity=0.02)
    assert "approximate" in short.warning()
    assert position("Participation note", monitoring="daily").monitoring is Monitoring.EUROPEAN


@pytest.mark.parametrize(
    ("terms", "message"),
    [
        ({"maturity": "x"}, "maturity must be a number"),
        ({"notional": 0.0}, "notional must be positive"),
        ({"initial": -5.0}, "initial level must be positive"),
        ({"spread_bp": "x"}, "funding spread must be a number"),
        ({"bonus": None}, "Bonus level must be a number"),
    ],
)
def test_invalid_terms_are_explained(terms, message):
    with pytest.raises(ValueError, match=message):
        position("Bonus certificate", **terms)


def test_spot_profile_matches_the_closed_form_greeks():
    pos = position("Participation note")
    profile = spot_profile(pos, np.array([100.0]))
    from web.notes import desk_greeks

    desk = desk_greeks(pos.greeks(), MARKET.spot)
    assert profile["Price (%)"][0] == pytest.approx(pos.value())
    assert profile["Delta (%)"][0] == pytest.approx(desk["Delta (%)"], rel=1e-3)
    assert profile["Gamma (%)"][0] == pytest.approx(desk["Gamma (%)"], rel=1e-2)
    assert profile["Vega (%)"][0] == pytest.approx(desk["Vega (%)"], rel=1e-2)
    # one calendar day on the surface: the closed-form theta holds each brick's volatility
    assert profile["Theta (%)"][0] == pytest.approx(desk["Theta (%)"], rel=0.1)
    assert profile["Rho (%)"][0] == pytest.approx(desk["Rho (%)"], rel=1e-2)


# --- Tabs ----------------------------------------------------------------------------------------


def test_kpis_of_a_note():
    text = text_of(page.kpis(store("Participation note"), default_data()))
    assert "Bond " in text
    for greek in ("Delta", "Gamma", "Vega", "Theta", "Rho"):
        assert greek in text
    assert "Bond " not in text_of(page.kpis(store("Bonus certificate"), default_data()))
    path = text_of(page.kpis(store("Bonus certificate", monitoring="continuous"), default_data()))
    assert "observed along the path" in path
    short = text_of(
        page.kpis(store("Shark note (up)", monitoring="daily", maturity=0.02), default_data())
    )
    assert "approximate" in short
    option = {
        "slug": "call",
        "terms": {"side": "Long", "maturity": 1.0, "notional": 1.0, "strike_0": 100.0},
    }
    assert "no note to price" in text_of(page.kpis(option, default_data()))


def test_app_kpis_dispatch_and_errors():
    assert kpis(None, default_data()) is dash.no_update  # nothing before the Price button
    option = {
        "slug": "call",
        "terms": {"side": "Long", "maturity": 1.0, "notional": 1.0, "strike_0": 100.0},
    }
    assert "Delta" in text_of(kpis(option, default_data()))
    assert "no product to price" in text_of(kpis({"slug": "nothing", "terms": {}}, default_data()))


def test_payoff_tab():
    data = default_data()
    assert page.payoff_view(store("Bonus certificate"), data, "greeks") == (dash.no_update,) * 2
    fig, notes = page.payoff_view(store("Bonus certificate"), data, "payoff")
    assert fig.data[-1].name == "Paid at maturity"
    assert not any(str(t.meta).startswith("leg-") for t in fig.data)
    assert "never touched" not in text_of(notes)
    _, notes = page.payoff_view(store("Bonus certificate", monitoring="continuous"), data, "payoff")
    assert "never touched" in text_of(notes)
    fig, notes = page.payoff_view(None, data, "payoff")
    assert "Set the terms" in text_of(notes)


def test_decomposition_tab():
    data = default_data()
    assert page.decomposition_view(store("Bonus certificate"), data, "payoff") is dash.no_update
    text = text_of(page.decomposition_view(store("Digital coupon note"), data, "decomposition"))
    for column in ("Brick", "Terms", "Price (%)", "Theta (%)", "Rho (%)"):
        assert column in text
    assert "skew" not in text
    assert "Set the terms" in text_of(page.decomposition_view(None, data, "decomposition"))


def test_brick_terms_are_named():
    shark = text_of(
        page.decomposition_view(store("Shark warrant (call)"), default_data(), "decomposition")
    )
    assert "H 130%" in shark
    assert "pays 5%" in shark
    half = text_of(
        page.decomposition_view(
            store("Participation note", participation=60.0), default_data(), "decomposition"
        )
    )
    assert r"\u00d7 60%" in half  # json escapes the multiplication sign


def test_greeks_tab():
    data = default_data()
    assert page.greeks_view(store("Bonus certificate"), data, "payoff") is dash.no_update
    text = text_of(page.greeks_view(store("Bonus certificate"), data, "greeks"))
    for greek in ("Delta", "Gamma", "Vega", "Theta", "Rho"):
        assert f'"children": "{greek}"' in text
    assert "costs the note" in text
    text = text_of(page.greeks_view(store("Participation note"), data, "greeks"))
    assert "helps the note" in text
    assert "Set the terms" in text_of(page.greeks_view(None, data, "greeks"))


def test_greeks_chart_shows_every_greek():
    children = page.greeks_view(store("Bonus note"), default_data(), "greeks").children
    assert len(children[1].figure.data) == 12  # price and five Greeks, each a curve and a dot


def sheet_of(s):
    """The product sheet a note page links to."""
    return sheet_route.sheet_html({"store": s, "market": default_data()})


def test_product_sheet():
    content = sheet_of(store("Participation note"))
    assert "participation-note-product-sheet.html" in content
    assert "Brick decomposition" in content
    assert "Download the sheet" in content


def test_sheet_renders_every_part():
    chart = sheet.Chart(
        [0.0, 1.0, 2.0],
        [
            ("A <line>", [0.0, float("nan"), 2.0], sheet.PRIMARY),
            ("Guide", [float("nan"), 1.0, float("nan")], sheet.GUIDE),
        ],
        [(1.0, "K"), (9.0, "out of range")],
        "x",
        "y",
    )
    page_html = sheet.render(
        sheet.Sheet(
            "Family",
            "A <note>",
            "sub",
            [("Price", "99 %", "of notional")],
            chart,
            ("Level", [1.0, 2.0], [("Paid", [1.0, float("nan")]), ("P&L", [0.5, -0.5])]),
            [1.5],
            [("Term", 1234.5)],
            [("Delta", 0.001234)],
            [("Table", (["a", "b"], [[True, None]]))],
            ["Risk"],
        ),
        valuation_date=date(2026, 1, 2),
    )
    assert "A &lt;note&gt;" in page_html
    assert "1,234" in page_html
    assert "0.0012" in page_html
    assert "Breakeven at 1.5 %" in page_html
    assert "n/a" in page_html
    assert 'class="num neg"' in page_html
    assert "2026-01-02" in page_html
    assert "out of range" not in page_html
    bare = sheet.render(sheet.Sheet("F", "T", "d", [], chart, ("L", [1.0], [("Paid", [1.0])])))
    assert "Key features" not in bare
    assert "Breakeven" not in bare
    assert "toolbar" not in bare.split("<body>", 1)[1]
    assert sheet.file_name("Shark note (up)") == "shark-note-up-product-sheet.html"


# --- Hedging tab on a note --------------------------------------------------------------------


def test_the_issuer_hedges_the_note_it_sold():
    out, _, _ = risk_panel.static_view(
        store("Participation note"),
        default_data(),
        None,
        risk_panel.TAB,
        "Delta + gamma",
        "call",
        100.0,
        0.25,
        "put",
        100.0,
        2.0,
    )
    text = text_of(out)
    assert "Issuer, short the participation note" in text
    assert "market maker" in text
    item = position("Participation note")
    subject = risk_panel.subject_of(item)
    assert float(subject.greeks.delta) == pytest.approx(-float(item.greeks().delta))


def test_resolve_errors():
    with pytest.raises(ValueError, match="no product"):
        risk_panel.resolve({"slug": "nothing", "terms": {}}, default_data())
    with pytest.raises(ValueError, match="no product"):
        risk_panel.resolve({"slug": "phoenix-autocall", "terms": {}}, default_data())
    with pytest.raises(ValueError, match="market data"):
        risk_panel.resolve(store("Bonus certificate"), default_data() | {"spot": -1})


def test_ui_side_classes_and_flat_market():
    assert ui.side_class("Buy 3 calls") == "pos fw-semibold"
    assert ui.side_class("Sell 3 calls") == "neg fw-semibold"
    flat = Market(100.0, 0.03, 0.02, FlatVol(0.2))
    pos = build_note(
        CATALOG["Bonus certificate"], default_terms(CATALOG["Bonus certificate"]), flat
    )
    assert pos.value() > 0.0


@pytest.mark.parametrize(
    ("name", "terms", "phrases", "absent"),
    [
        ("Participation note", {}, ["Bond part", "capital protection applies"], ["Barrier:"]),
        ("Bonus certificate", {}, ["capital is at risk", "Barrier:", "Breakeven at 70.0 %"], []),
        ("Shark warrant (call)", {}, ["Knock-out barrier"], ["Underlying</text>"]),
        ("Outperformance certificate", {}, ["Breakeven at"], ["Bond part"]),
    ],
)
def test_note_sheets(name, terms, phrases, absent):
    content = sheet_of(store(name, **terms))
    for phrase in phrases:
        assert phrase in content, phrase
    for phrase in absent:
        assert phrase not in content, phrase


def test_a_shark_warrant_payoff_has_no_underlying_line():
    fig, _ = page.payoff_view(store("Shark warrant (call)"), default_data(), "payoff")
    assert "Underlying" not in [t.name for t in fig.data]
    fig, _ = page.payoff_view(store("Shark note (up)"), default_data(), "payoff")
    assert "Underlying" in [t.name for t in fig.data]
