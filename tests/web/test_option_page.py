"""Option product pages: terms, position, tabs and readings, without a browser."""

import json

import dash
import numpy as np
import pytest

from spricer.bricks import BarrierOption
from spricer.greeks_lab import Moneyness
from spricer.hedging import static_hedge
from spricer.market import FlatVol, Market
from spricer.monitoring import Monitoring
from spricer.risk import Shock, explain
from web import explain as ex
from web import figures, registry
from web import option_page as page
from web.market import default_data, market_from_data
from web.options import (
    OptionSetup,
    breakevens,
    build_position,
    decomposition,
    display_greeks,
    leg_name,
    setup_of,
    smile_vols,
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


def store(slug: str, vols=None, **terms):
    found = setup(slug)
    full = found.default_terms() | terms
    return {"slug": slug, "terms": full, "vols": vols or [None] * len(found.strike_labels)}


def position(slug: str, vols=None, **terms):
    s = store(slug, vols, **terms)
    return build_position(setup(slug), s["terms"], s["vols"], MARKET)


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
    pos = build_position(setup(product.slug), s["terms"], s["vols"], MARKET)
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
    assert short.situation.short


def test_moneyness_of_single_strikes_only():
    assert position("call", strike_0=80.0).moneyness is Moneyness.ITM
    assert position("put", strike_0=80.0).moneyness is Moneyness.OTM
    assert position("call").moneyness is Moneyness.ATM
    assert position("bull-call-spread").moneyness is None
    assert position("straddle").moneyness is None  # one strike, but no direction


def test_volatility_overrides_reprice_the_leg():
    smile = position("call")
    typed = position("call", vols=[30.0])
    assert typed.price_pct() > smile.price_pct()
    assert position("call", vols=[""]).price_pct() == smile.price_pct()
    vols = smile_vols(setup("call"), store("call")["terms"], MARKET)
    assert vols[0] == pytest.approx(100 * float(MARKET.vol.vol(100.0, 1.0)), abs=0.01)


@pytest.mark.parametrize(
    ("terms", "vols", "message"),
    [
        ({"maturity": None}, None, "maturity must be a number"),
        ({"maturity": -1.0}, None, "maturity must be positive"),
        ({"notional": float("inf")}, None, "notional must be positive"),
        ({"strike_0": "x"}, None, "strike must be a number"),
        ({}, ["abc"], "volatility at 100% must be a number"),
        ({}, [-5.0], "volatility at 100% must be positive"),
    ],
)
def test_invalid_terms_are_explained(terms, vols, message):
    with pytest.raises(ValueError, match=message):
        position("call", vols=vols, **terms)


def test_invalid_terms_leave_the_smile_placeholders_empty():
    spread = setup("bull-call-spread")
    terms = spread.default_terms()
    assert smile_vols(spread, terms | {"maturity": None}, MARKET) == [None, None]
    assert smile_vols(spread, terms | {"strike_1": None}, MARKET)[1] is None


def test_barrier_options_by_monitoring():
    continuous = position("down-and-out-call", barrier=80.0)
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
    assert continuous.situation.knock_out
    assert len(continuous.atm_bands()) == 1


def test_leg_names():
    rows = decomposition(position("call-butterfly"))
    assert [r["Leg"] for r in rows] == ["Long call", "Short 2 calls", "Long call", "Total"]
    assert leg_name(position("covered-call").strategy.legs[0]) == "Long underlying"
    assert leg_name(position("box-spread").strategy.legs[0]).startswith("Long")
    digital = decomposition(position("digital-call"))
    assert "pays 10%" in digital[0]["Terms"]
    assert digital[0]["Skew (%)"] != 0.0
    barrier = decomposition(position("up-and-out-call"))
    assert barrier[0]["Leg"].startswith("Long ")
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
    out = page.collect(["Short", 2.0], ids, [None], "/product/call/")
    assert out == {"slug": "call", "terms": {"side": "Short", "maturity": 2.0}, "vols": [None]}


def test_placeholders_show_the_smile():
    assert page.placeholders(None, default_data()) == []
    assert page.placeholders(store("bull-call-spread"), default_data())[0] == "18.70"
    other = {"slug": "phoenix-autocall", "terms": {}, "vols": [None]}
    assert page.placeholders(other, default_data()) == [""]
    bad_market = default_data() | {"spot": -1.0}
    assert page.placeholders(store("call"), bad_market) == [""]
    assert page.placeholders(store("call", maturity=None), default_data()) == [""]


def test_kpis():
    text = text_of(page.kpis(store("call"), default_data()))
    assert "Long call" in text
    assert "At the money" in text
    assert "paid" in text
    assert "received" in text_of(page.kpis(store("call", side="Short"), default_data()))
    assert "Set the terms" in text_of(page.kpis(None, default_data()))
    assert "no option to price" in text_of(page.kpis(store("call") | {"slug": "x"}, default_data()))
    bad = text_of(page.kpis(store("call"), default_data() | {"spot": None}))
    assert "market data are invalid" in bad
    crossed = text_of(page.kpis(store("down-and-out-call", barrier=120.0), default_data()))
    assert "already crossed" in crossed
    assert "Badge" not in text_of(page.kpis(store("bull-call-spread"), default_data()))


def test_tabs_and_risk_views_show_one_block():
    styles = page.show_tab("greeks")
    assert [s["display"] for s in styles] == ["none", "none", "block", "none"]
    assert [s["display"] for s in page.risk_blocks("P&L explain")] == ["none", "none", "block"]


def test_payoff_tab():
    data = default_data()
    assert page.payoff_view(store("call"), data, False, "greeks") == (
        dash.no_update,
        dash.no_update,
    )
    fig, notes = page.payoff_view(store("bull-call-spread"), data, False, "payoff")
    assert len(fig.data) == 4  # total, two legs, delta one
    assert "compounded" in text_of(notes)
    fig, notes = page.payoff_view(store("bull-call-spread"), data, True, "payoff")
    assert "Breakeven at" in text_of(notes)
    assert fig.data[-1].name == "Breakeven"
    _, notes = page.payoff_view(store("digital-call"), data, True, "payoff")
    assert "Breakeven at" in text_of(notes)
    _, notes = page.payoff_view(store("synthetic-forward", strike_0=150.0), data, True, "payoff")
    assert "Breakeven" in text_of(notes)
    _, notes = page.payoff_view(store("box-spread"), data, True, "payoff")
    assert "No breakeven" in text_of(notes)
    fig, notes = page.payoff_view(None, data, False, "payoff")
    assert "Set the terms" in text_of(notes)


@pytest.mark.parametrize("slug", ["up-and-out-call", "down-and-in-put"])
def test_a_path_barrier_is_drawn_touched_and_never_touched(slug):
    fig, _ = page.payoff_view(store(slug), default_data(), False, "payoff")
    names = [t.name for t in fig.data]
    assert names[:2] == ["Barrier never touched", "Barrier touched"]
    fig, _ = page.payoff_view(store(slug, monitoring="european"), default_data(), True, "payoff")
    assert fig.data[0].name == "P&L at maturity"


def test_covered_call_payoff_draws_the_underlying_as_delta_one():
    pos = position("covered-call")
    fig, _ = figures.payoff_figure(pos, net_of_premium=False)
    delta_one = next(t for t in fig.data if t.name == "Delta one")
    assert delta_one.y[0] == pytest.approx(delta_one.x[0])


def test_decomposition_tab():
    data = default_data()
    assert page.decomposition_view(store("call"), data, "payoff") is dash.no_update
    text = text_of(page.decomposition_view(store("digital-call"), data, "decomposition"))
    assert "of which skew" in text
    assert "smile slope adds" in text
    text = text_of(page.decomposition_view(store("collar"), data, "decomposition"))
    assert "of which skew" not in text
    assert "total-row" in text
    assert "Set the terms" in text_of(page.decomposition_view(None, data, "decomposition"))


def test_greeks_tab_views():
    data = default_data()
    assert page.greeks_view(store("call"), data, None, None, "payoff") == (dash.no_update,) * 4
    reading, fig, caption, picker = page.greeks_view(
        store("call"), data, page.GREEK_VIEWS[0], "Gamma (%)", "greeks"
    )
    assert "Delta" in text_of(reading)
    assert len(fig.data) == 12  # six Greeks, each a curve and today's dot
    assert picker["display"] == "none"
    assert "Blue band" in caption
    reading, fig, _, _ = page.greeks_view(store("put"), data, page.GREEK_VIEWS[2], None, "greeks")
    assert "Vanna" in text_of(reading)
    assert len(fig.data) == 8
    _, fig, caption, picker = page.greeks_view(
        store("put"), data, page.GREEK_VIEWS[1], "Vega (%)", "greeks"
    )
    assert picker["display"] == "block"
    assert len(fig.data) == 3
    assert "Vega" in fig.layout.title.text
    assert "expiry" in caption
    _, fig, _, _ = page.greeks_view(store("call"), data, page.GREEK_VIEWS[1], "?", "greeks")
    assert "Gamma" in fig.layout.title.text
    reading, *_ = page.greeks_view(None, data, page.GREEK_VIEWS[0], None, "greeks")
    assert "Set the terms" in text_of(reading)
    # a short-dated position has no quarter-time gamma comparison
    page.greeks_view(store("call", maturity=0.05), data, page.GREEK_VIEWS[0], None, "greeks")
    _, fig, _, _ = page.greeks_view(
        store("bull-call-spread"), data, page.GREEK_VIEWS[0], None, "greeks"
    )
    assert len(fig.layout.shapes) == 6 * 4  # per Greek: two ATM bands, two strike lines


@pytest.mark.parametrize(
    ("slug", "terms", "phrase"),
    [
        ("call", {"strike_0": 150.0}, "Gamma"),  # out of the money: no 1/sqrt(T) warning
        ("covered-call", {}, "largest at"),  # no vega sign change, not a vanilla
        ("put", {"strike_0": 160.0, "maturity": 3.0}, "same sign"),  # deep in the money put
        ("synthetic-forward", {}, "Without convexity"),
        ("call", {"side": "Short"}, "Short position"),
    ],
)
def test_first_order_reading_cases(slug, terms, phrase):
    reading, *_ = page.greeks_view(
        store(slug, **terms), default_data(), page.GREEK_VIEWS[0], None, "greeks"
    )
    assert phrase in text_of(reading)


HEDGE_ARGS = ("call", 100.0, 0.25, "put", 100.0, 2.0)


@pytest.mark.parametrize("target", page.HEDGES)
def test_static_hedges_neutralise_their_greeks(target):
    out, row1, row2 = page.static_view(
        store("call", side="Short"),
        default_data(),
        "risk-hedging",
        page.RISK_VIEWS[0],
        10_000,
        target,
        *HEDGE_ARGS,
    )
    text = text_of(out)
    assert "market maker" in text
    assert "shares" in text
    needed = page.HEDGES.index(target)
    assert row1["display"] == ("block" if needed >= 1 else "none")
    assert row2["display"] == ("block" if needed == 2 else "none")


def test_static_hedge_readings_follow_the_buyer_s_gamma():
    data, view = default_data(), page.RISK_VIEWS[0]
    long_call = text_of(
        page.static_view(store("call"), data, "risk-hedging", view, 100, "Delta", *HEDGE_ARGS)[0]
    )
    assert "gamma scalping" in long_call
    spread = text_of(
        page.static_view(
            store("bull-call-spread"), data, "risk-hedging", view, 100, "Delta", *HEDGE_ARGS
        )[0]
    )
    assert "short gamma at today's spot" in spread


def test_static_hedge_errors_and_hidden_tab():
    data, view = default_data(), page.RISK_VIEWS[0]
    hidden = page.static_view(store("call"), data, "payoff", view, 1, "Delta", *HEDGE_ARGS)
    assert hidden[0] is dash.no_update
    bad = page.static_view(store("call"), data, "risk-hedging", view, 0, "Delta", *HEDGE_ARGS)
    assert "at least 1" in text_of(bad[0])
    no_strike = page.static_view(
        store("call"),
        data,
        "risk-hedging",
        view,
        1,
        page.HEDGES[1],
        "call",
        None,
        0.25,
        "put",
        1,
        1,
    )
    assert "positive strike" in text_of(no_strike[0])
    same = page.static_view(
        store("call"), data, "risk-hedging", view, 1, page.HEDGES[2], "call", 100, 1, "call", 100, 1
    )
    assert "proportional" in text_of(same[0])
    unknown = page.static_view(store("call"), data, "risk-hedging", view, 1, None, *HEDGE_ARGS)
    assert unknown[1]["display"] == "none"


def test_the_simulator_matches_its_theory():
    data = default_data()
    assert "Click Simulate" in text_of(
        page.simulate_view(0, store("call"), data, None, "Daily", 0, 2000)
    )
    out = page.simulate_view(1, store("call"), data, None, "Weekly", 0.0, 2000)
    text = text_of(out)
    assert "Realised volatility equals the implied one" in text
    assert "Derman-Kamal" in text
    out = page.simulate_view(1, store("call"), data, 30.0, "Monthly", 2.0, 2000)
    text = text_of(out)
    assert "earns" in text
    assert "Transaction costs" in text
    smile = text_of(page.simulate_view(1, store("bull-call-spread"), data, None, None, None, None))
    assert "priced on the smile" in smile
    assert "Set the terms" in text_of(page.simulate_view(1, None, data, None, "Daily", 0, 2000))
    assert "positive" in text_of(page.simulate_view(1, store("call"), data, -5.0, "Daily", 0, 2000))
    path = text_of(page.simulate_view(1, store("up-and-out-call"), data, None, "Daily", 0, 2000))
    assert "barriers observed at maturity" in path
    european = store("up-and-out-call", monitoring="european")
    assert "Mean P&L" in text_of(page.simulate_view(1, european, data, None, "Monthly", 0, 2000))


def test_simulated_mean_is_the_theory_within_its_error():
    """Long call, realised 25 % against the smile: mean hedged P&L = value at 25 % - price."""
    pos = position("call")
    from spricer.hedging import simulate_delta_hedge

    sim = simulate_delta_hedge(
        pos.instrument, pos.market, 1.0, realised_vol=0.25, rebalancings=252, n_paths=4000
    )
    flat = Market(100.0, MARKET.rate, MARKET.dividend_yield, FlatVol(0.25))
    theory = float(pos.strategy.value(flat)) - pos.price_pct()
    assert abs(sim.mean.value - theory) < 4 * sim.mean.std_error + 0.02


def test_pnl_explain_tab():
    data, view = default_data(), page.RISK_VIEWS[2]
    assert page.pnl_view(store("call"), data, "payoff", view, -10, 2, 7, 0) is dash.no_update
    text = text_of(page.pnl_view(store("call"), data, "risk-hedging", view, -10, 2, 7, 0))
    assert "Full revaluation" in text
    assert "Stress test" in text
    empty = text_of(
        page.pnl_view(store("call"), data, "risk-hedging", view, None, None, None, None)
    )
    assert "Taylor total" in empty
    late = text_of(page.pnl_view(store("call"), data, "risk-hedging", view, 0, 0, 400, 0))
    assert "before maturity" in late


# --- Readings ----------------------------------------------------------------------------------


def test_static_reading_of_a_flat_gamma_position():
    pos = position("synthetic-forward")
    hedge = static_hedge(pos.strategy.greeks(MARKET))
    lines = ex.static_reading(hedge, 10, short=False)
    assert not any("gamma**" in line for line in lines)
    assert "short gamma at today's spot" in lines[-1]


def test_simulation_readings_for_one_volatility():
    lines = ex.simulation_reading(-0.5, -0.4, 0.3, 0.3, 15.0, (20.0, 20.0), True, 0.0)
    assert "loses" in lines[0]
    assert "Derman-Kamal formula:" in lines[1]


def test_attribution_reading_flags_large_residuals():
    pos = position("digital-call")

    def value(m, t):
        return 100.0 * float(pos.instrument(t).value(m)) / m.spot

    small = ex.attribution_reading(explain(value, MARKET, 1.0, Shock(0.5)))
    big = ex.attribution_reading(explain(value, MARKET, 1.0, Shock(-30.0, 10.0, 30.0)))
    assert "residual is small" in small[1]
    assert "too big" in big[1]
    flat = ex.attribution_reading(explain(value, MARKET, 1.0, Shock()))
    assert "(100 %)" in flat[0]


def test_first_order_readings_cover_every_family():
    for slug in ("call", "digital-call", "straddle", "up-and-out-call", "bull-call-spread"):
        pos = position(slug)
        spots = page.SPOTS_PCT / 100.0 * MARKET.spot
        from spricer.greeks_lab import greeks_at, second_order_at, spot_profile

        profile = spot_profile(pos.instrument, pos.market, 1.0, spots)
        now = {
            k: float(v)
            for k, v in (
                greeks_at(pos.strategy, pos.market)
                | second_order_at(pos.instrument, pos.market, 1.0)
            ).items()
        }
        assert ex.first_order(pos.situation, now, page.SPOTS_PCT, profile, None)
    dead = {
        k: 0.0 for k in ("Price (%)", "Delta (%)", "Gamma (%)", "Vega (%)", "Theta (%)", "Rho (%)")
    }
    assert (
        "worth nothing"
        in ex.first_order(position("call").situation, dead, page.SPOTS_PCT, {}, None)[0]
    )


def test_european_barrier_payoff_has_no_two_scenarios():
    pos = position("up-and-out-call", monitoring="european")
    assert pos.strategy.legs[0].brick.monitoring is Monitoring.EUROPEAN
    fig, _ = figures.payoff_figure(pos, net_of_premium=False)
    assert fig.data[0].name == "Payoff at maturity"
