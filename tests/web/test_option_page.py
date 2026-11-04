"""Option product pages: terms, position, tabs and readings, without a browser."""

import json

import dash
import numpy as np
import pytest

from spricer.bricks import BarrierOption
from spricer.hedging import static_hedge
from spricer.monitoring import Monitoring
from web import explain as ex
from web import figures, registry, risk_panel, sheet_route, ui
from web import option_page as page
from web.market import default_data, market_from_data
from web.notes import CATALOG, default_terms
from web.options import (
    OptionSetup,
    breakevens,
    build_position,
    decomposition,
    display_greeks,
    leg_name,
    setup_of,
)

OPTION_PRODUCTS = [p for p in registry.PRODUCTS if setup_of(p) is not None]
MARKET = market_from_data(default_data())


def text_of(component) -> str:
    """Every string in a Dash component tree (or figure)."""
    return json.dumps(component.to_plotly_json(), default=lambda o: o.to_plotly_json())


def setup(slug: str) -> OptionSetup:
    found = setup_of(registry.BY_SLUG[slug])
    assert found is not None
    return found


def maybe(slug: str) -> OptionSetup | None:
    return setup_of(registry.BY_SLUG[slug])


def store(slug: str, **terms):
    return {"slug": slug, "terms": setup(slug).default_terms() | terms}


def position(slug: str, **terms):
    return build_position(setup(slug), store(slug, **terms)["terms"], MARKET)


# --- Setups and positions ---------------------------------------------------------------------


def test_every_option_and_barrier_option_has_a_page():
    assert len(OPTION_PRODUCTS) == sum(
        p.family in (registry.OPTIONS, registry.BARRIERS) and p.group != "Sharks"
        for p in registry.PRODUCTS
    )
    assert maybe("phoenix-autocall") is None
    assert maybe("shark-warrant-call") is None
    barrier = setup("down-and-in-put")
    assert barrier.barrier_type == "down-and-in"
    assert barrier.kind == "put"
    assert "comes alive" in barrier.description
    assert setup("up-and-out-call").default_terms()["barrier"] == 130.0


@pytest.mark.parametrize("product", OPTION_PRODUCTS, ids=lambda p: p.slug)
def test_every_page_prices_its_position(product):
    s = store(product.slug)
    pos = build_position(setup(product.slug), s["terms"], MARKET)
    assert np.isfinite(pos.price_pct())
    assert pos.name.startswith("Long ")
    page.option_page(product)  # builds
    rows = decomposition(pos)
    assert rows[-1]["Price (%)"] == pytest.approx(pos.price_pct())


def test_a_short_position_flips_price_and_greeks():
    long, short = position("call"), position("call", side="Short")
    assert short.price_pct() == pytest.approx(-long.price_pct())
    assert display_greeks(short.strategy, MARKET)["Delta (%)"] == pytest.approx(
        -display_greeks(long.strategy, MARKET)["Delta (%)"]
    )


@pytest.mark.parametrize(
    ("terms", "message"),
    [
        ({"maturity": None}, "maturity must be a number"),
        ({"maturity": -1.0}, "maturity must be positive"),
        ({"notional": float("inf")}, "notional must be positive"),
        ({"strike_0": "x"}, "strike must be a number"),
    ],
)
def test_invalid_terms_are_explained(terms, message):
    with pytest.raises(ValueError, match=message):
        position("call", **terms)


def test_barrier_options_by_monitoring():
    continuous = position("down-and-out-call", barrier=80.0, monitoring="continuous")
    daily = position("down-and-out-call", barrier=80.0, monitoring="daily")
    european = position("down-and-out-call", barrier=80.0, monitoring="european")
    # observed more often, a knock-out is touched more often and worth less
    assert continuous.price_pct() < daily.price_pct() < european.price_pct()
    brick = daily.strategy.legs[0].brick
    assert isinstance(brick, BarrierOption)
    assert brick.observations == 252
    assert not continuous.barrier_crossed
    assert position("down-and-out-call", barrier=110.0).barrier_crossed
    assert position("up-and-in-put", barrier=95.0).barrier_crossed
    assert not position("call").barrier_crossed


def test_leg_names():
    rows = decomposition(position("call-butterfly"))
    assert [r["Brick"] for r in rows] == ["Long call", "Short 2 calls", "Long call", "Total"]
    assert leg_name(position("covered-call").strategy.legs[0]) == "Long underlying"
    assert leg_name(position("box-spread").strategy.legs[0]).startswith("Long")
    digital = decomposition(position("digital-call"))
    assert "pays 10%" in digital[0]["Terms"]
    barrier = decomposition(position("up-and-out-call"))
    assert barrier[0]["Brick"].startswith("Long ")
    assert "H 130%" in barrier[0]["Terms"]
    assert len(barrier) == 1


def test_leg_names_of_cash_and_bond_legs():
    from spricer.bricks import Underlying, ZeroCoupon
    from spricer.strategies import Leg

    assert leg_name(Leg(ZeroCoupon(1.0), -3.0)) == "Short zero-coupon"
    assert leg_name(Leg(Underlying(1.0), 2.0)) == "Long 2 underlying"


def test_breakevens():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    assert breakevens(x, np.array([-1.0, 1.0, 1.0, -1.0])) == [0.5, 2.5]
    assert breakevens(x, np.array([0.0, 1.0, 0.0, 0.0])) == [0.0, 2.0]
    assert breakevens(x, np.array([1e-15, -1e-15, 0.0, 1e-16])) == []


# --- Callbacks ---------------------------------------------------------------------------------


def test_collect_reads_the_terms_and_the_product():
    ids = [{"type": "term", "key": "side"}, {"type": "term", "key": "maturity"}]
    out = ui.collect(["Short", 2.0], ids, "/product/call/")
    assert out == {"slug": "call", "terms": {"side": "Short", "maturity": 2.0}}


def test_kpis():
    text = text_of(page.kpis(store("call"), default_data()))
    assert "Long call" in text
    for greek in ("Delta", "Gamma", "Vega", "Theta", "Rho"):
        assert greek in text
    assert "paid" in text
    assert "received" in text_of(page.kpis(store("call", side="Short"), default_data()))
    assert "Set the terms" in text_of(page.kpis(None, default_data()))
    assert "no product to price" in text_of(
        page.kpis(store("call") | {"slug": "x"}, default_data())
    )
    terms = default_terms(CATALOG["Bonus certificate"])
    note = {"slug": "bonus-certificate", "terms": terms}
    assert "no option to price" in text_of(page.kpis(note, default_data()))
    bad = text_of(page.kpis(store("call"), default_data() | {"spot": None}))
    assert "market data are invalid" in bad
    crossed = text_of(page.kpis(store("down-and-out-call", barrier=120.0), default_data()))
    assert "already crossed" in crossed


def test_tabs_show_one_block():
    tabs = [{"type": "tab-block", "tab": t} for t in ("payoff", "decomposition", "greeks")]
    assert [s["display"] for s in ui.show_tab("greeks", tabs)] == ["none", "none", "block"]


def test_payoff_tab():
    data = default_data()
    assert page.payoff_view(store("call"), data, "greeks") == (dash.no_update, dash.no_update)
    fig, notes = page.payoff_view(store("bull-call-spread"), data, "payoff")
    # the payoff and its levels only: no legs, no underlying
    assert [t.name for t in fig.data] == [
        "Long call strike 100%",
        "Short call strike 110%",
        "Paid at maturity",
    ]
    assert fig.data[-1].y.max() == pytest.approx(10.0)
    assert "104.5 %" in text_of(notes)  # the breakeven, above the table
    assert "Gain on the price paid" in text_of(notes)
    _, notes = page.payoff_view(store("digital-call"), data, "payoff")
    assert "Breakeven" in text_of(notes)
    _, notes = page.payoff_view(store("synthetic-forward", strike_0=150.0), data, "payoff")
    assert "Breakeven" in text_of(notes)
    _, notes = page.payoff_view(store("box-spread"), data, "payoff")
    assert "No breakeven" in text_of(notes)
    fig, notes = page.payoff_view(None, data, "payoff")
    assert "Set the terms" in text_of(notes)
    short, _ = page.payoff_view(store("call", side="Short"), data, "payoff")
    assert short.layout.yaxis.range[0] < 0.0  # room for what a short position pays


@pytest.mark.parametrize("slug", ["up-and-out-call", "down-and-in-put"])
def test_a_path_barrier_is_drawn_touched_and_never_touched(slug):
    fig, _ = page.payoff_view(store(slug, monitoring="continuous"), default_data(), "payoff")
    names = [t.name for t in fig.data]
    assert "Barrier never touched" in names
    assert "Barrier touched" in names
    fig, _ = page.payoff_view(store(slug, monitoring="european"), default_data(), "payoff")
    assert fig.data[-1].name == "Paid at maturity"


def test_decomposition_tab():
    data = default_data()
    assert page.decomposition_view(store("call"), data, "payoff") is dash.no_update
    text = text_of(page.decomposition_view(store("collar"), data, "decomposition"))
    for column in ("Brick", "Terms", "Price (%)", "Theta (%)", "Rho (%)"):
        assert column in text
    assert "skew" not in text
    assert "total-row" in text
    assert "Set the terms" in text_of(page.decomposition_view(None, data, "decomposition"))


def test_greeks_tab():
    data = default_data()
    assert page.greeks_view(store("call"), data, "payoff") is dash.no_update
    text = text_of(page.greeks_view(store("up-and-out-call"), data, "greeks"))
    for greek in ("Delta", "Gamma", "Vega", "Theta", "Rho"):
        assert f'"children": "{greek}"' in text
    assert "Set the terms" in text_of(page.greeks_view(None, data, "greeks"))


def test_greeks_chart_shows_every_greek():
    children = page.greeks_view(store("bull-call-spread"), default_data(), "greeks").children
    fig = children[1].figure
    assert len(fig.data) == 12  # price and five Greeks, each a curve and today's dot
    assert len(fig.layout.shapes) == 6 * 2  # the two strikes on each panel


HEDGE_ARGS = ("call", 100.0, 0.25, "put", 100.0, 2.0)


@pytest.mark.parametrize("target", risk_panel.HEDGES)
def test_static_hedges_neutralise_their_greeks(target):
    out, row1, row2 = risk_panel.static_view(
        store("call", side="Short"), default_data(), None, risk_panel.TAB, target, *HEDGE_ARGS
    )
    text = text_of(out)
    assert "market maker" in text
    assert "shares" in text
    needed = risk_panel.HEDGES.index(target)
    assert row1["display"] == ("block" if needed >= 1 else "none")
    assert row2["display"] == ("block" if needed == 2 else "none")


def view(s, target, *args):
    return risk_panel.static_view(s, default_data(), None, risk_panel.TAB, target, *args)


def test_static_hedge_readings_follow_the_buyer_s_gamma():
    assert "gamma scalping" in text_of(view(store("call"), "Delta", *HEDGE_ARGS)[0])
    spread = text_of(view(store("bull-call-spread"), "Delta", *HEDGE_ARGS)[0])
    assert "short gamma at today's spot" in spread


def test_static_hedge_quantities_are_for_the_notional():
    out = text_of(view(store("call", notional=2_000_000.0), "Delta", *HEDGE_ARGS)[0])
    delta = display_greeks(position("call").strategy, MARKET)["Delta (%)"]
    assert f"{delta / 100.0 * 2_000_000.0 / MARKET.spot:,.0f} shares" in out


def test_static_hedge_errors_and_hidden_tab():
    data = default_data()
    hidden = risk_panel.static_view(store("call"), data, None, "payoff", "Delta", *HEDGE_ARGS)
    assert hidden[0] is dash.no_update
    no_strike = view(store("call"), risk_panel.HEDGES[1], "call", None, 0.25, "put", 1, 1)
    assert "positive strike" in text_of(no_strike[0])
    same = view(store("call"), risk_panel.HEDGES[2], "call", 100, 1, "call", 100, 1)
    assert "proportional" in text_of(same[0])
    assert view(store("call"), None, *HEDGE_ARGS)[1]["display"] == "none"


# --- Readings ----------------------------------------------------------------------------------


def test_static_reading_of_a_flat_gamma_position():
    pos = position("synthetic-forward")
    hedge = static_hedge(pos.strategy.greeks(MARKET))
    lines = ex.static_reading(hedge, 10, short=False)
    assert not any("gamma**" in line for line in lines)
    assert "short gamma at today's spot" in lines[-1]


def test_european_barrier_payoff_has_no_two_scenarios():
    pos = position("up-and-out-call", monitoring="european")
    assert pos.strategy.legs[0].brick.monitoring is Monitoring.EUROPEAN
    fig = figures.option_payoff_figure(pos)
    assert fig.data[-1].name == "Paid at maturity"


@pytest.mark.parametrize(
    ("slug", "terms", "phrases"),
    [
        (
            "bull-call-spread",
            {},
            ["Breakeven at 104.5 %", "most you can lose", "Brick decomposition"],
        ),
        ("covered-call", {"side": "Short"}, ["Short options", "Delta one"]),
        (
            "up-and-out-call",
            {"monitoring": "continuous"},
            ["American (continuous)", "Paid if the barrier was never touched"],
        ),
        ("up-and-out-call", {"monitoring": "european"}, ["European (at maturity)"]),
        ("digital-call", {"maturity": 1.0}, ["Breakeven at"]),
    ],
)
def test_option_sheets(slug, terms, phrases):
    content = sheet_route.sheet_html({"store": store(slug, **terms), "market": default_data()})
    assert "-product-sheet.html" in content
    for phrase in phrases:
        assert phrase in content, phrase
    assert ">1 year<" in content
