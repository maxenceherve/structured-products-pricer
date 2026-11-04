"""Coupon note pages: terms, underlyings, correlation, Monte Carlo results, solver, sheet."""

import json

import dash
import numpy as np
import pytest

from spricer.conventions import to_desk
from web import coupon_page as page
from web import registry, risk_panel, sheet_route, ui
from web.app import kpis
from web.coupons import (
    CATALOG,
    MAX_UNDERLYINGS,
    add_underlying,
    basket_of,
    brick_rows,
    build_coupon,
    correlation_matrix,
    correlation_rows,
    coupon_spec,
    default_terms,
    default_underlyings,
    nearest_rows,
    underlying_market,
)
from web.market import default_data

N = 2_000  # paths: enough to exercise the pages


def text_of(component) -> str:
    return json.dumps(component.to_plotly_json(), default=lambda o: o.to_plotly_json())


def store(name: str, **terms):
    return {"slug": registry.slugify(name), "terms": default_terms(CATALOG[name]) | terms}


def two_underlyings():
    data = default_data()
    rows = add_underlying(default_underlyings(data), data)
    rows[1]["ATM vol 1y (%)"] = 30.0
    return rows, correlation_rows([r["Name"] for r in rows], None)


ONE = default_underlyings(default_data())


# --- Model -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "product", [p for p in registry.PRODUCTS if coupon_spec(p)], ids=lambda p: p.slug
)
def test_every_coupon_note_page_builds(product):
    assert page.coupon_page(product) is not None
    assert build_coupon(CATALOG[product.name], store(product.name)["terms"]).note.maturity == 3.0


def test_terms_build_the_note():
    terms = build_coupon(
        CATALOG["Phoenix autocall"],
        store(
            "Phoenix autocall",
            frequency="Monthly",
            monitoring="daily",
            elapsed=0.5,
            memory_accrued=1.0,
            breached=True,
            memory=True,
        )["terms"],
    )
    assert terms.frequency == 12
    assert terms.note.monitoring.name == "DISCRETE"
    assert terms.note.barrier_breached
    assert terms.note.memory_accrued == 1.0
    assert terms.build({**terms.values, "coupon_pa": 4.0}).observations[0].coupon == pytest.approx(
        4.0 / 12
    )
    rc = build_coupon(
        CATALOG["Reverse convertible"],
        store("Reverse convertible", monitoring="continuous")["terms"],
    )
    assert rc.monitoring.name == "EUROPEAN"  # no barrier to observe


@pytest.mark.parametrize(
    ("terms", "message"),
    [
        ({"maturity": None}, "maturity must be a number"),
        ({"issue_price": 0.0}, "issue price must be positive"),
        ({"spread_bp": -1.0}, "funding spread must be non-negative"),
        ({"coupon_pa": "x"}, "Coupon"),
        ({"elapsed": -1.0}, "years since issue"),
    ],
)
def test_invalid_terms_are_explained(terms, message):
    with pytest.raises(ValueError, match=message):
        build_coupon(CATALOG["Phoenix autocall"], store("Phoenix autocall", **terms)["terms"])


def test_underlyings_and_their_markets():
    data = default_data()
    rows = default_underlyings(data)
    assert rows[0]["Spot"] == 100.0
    for _ in range(MAX_UNDERLYINGS + 2):
        rows = add_underlying(rows, data)
    assert len(rows) == MAX_UNDERLYINGS
    assert len({r["Name"] for r in rows}) == MAX_UNDERLYINGS
    renamed = [rows[0] | {"Name": "Underlying 2"}]
    assert add_underlying(renamed, data)[1]["Name"] == "Underlying 3"
    market = underlying_market(rows[0] | {"ATM vol 1y (%)": 30.0, "Spot": 50.0}, data)
    assert float(market.vol.vol(50.0, 1.0)) == pytest.approx(0.30, abs=1e-6)
    assert market.spot == 50.0
    with pytest.raises(ValueError, match="needs a spot"):
        underlying_market({"Spot": None}, data)
    with pytest.raises(ValueError, match="positive"):
        underlying_market(rows[0] | {"Spot": 0.0}, data)
    bad_market = default_data() | {"spot": -1.0}
    assert default_underlyings(bad_market)[0]["ATM vol 1y (%)"] == 20.0


def test_correlation_grid():
    rows = correlation_rows(["A", "B"], None)
    assert rows == [{"Name": "A", "A": 1.0, "B": 0.6}, {"Name": "B", "A": 0.6, "B": 1.0}]
    kept = correlation_rows(["A", "B", "C"], [{"Name": "A", "A": 1.0, "B": 0.1}])
    assert kept[0]["B"] == 0.1
    assert kept[0]["C"] == 0.6
    with pytest.raises(ValueError, match="number"):
        correlation_matrix(["A", "B"], [{"Name": "A", "A": 1.0, "B": "x"}, {"Name": "B"}])
    bad = correlation_rows(["A", "B", "C"], None)
    bad[0]["B"] = bad[1]["A"] = 0.99
    bad[0]["C"] = bad[2]["A"] = 0.99
    bad[1]["C"] = bad[2]["B"] = -0.99
    fixed = nearest_rows(["A", "B", "C"], bad)
    assert np.all(np.linalg.eigvalsh(correlation_matrix(["A", "B", "C"], fixed)) > -1e-6)


@pytest.mark.parametrize(
    ("rows", "corr", "message"),
    [
        ([], None, "at least one"),
        (ONE * (MAX_UNDERLYINGS + 1), None, "At most"),
        (ONE * 2, None, "distinct names"),
        ([ONE[0] | {"Initial level": None}], None, "initial level"),
    ],
)
def test_invalid_baskets_are_explained(rows, corr, message):
    with pytest.raises(ValueError, match=message):
        basket_of(rows, corr, default_data())


def test_a_worst_of_needs_a_valid_correlation():
    rows, corr = two_underlyings()
    assert basket_of(rows, corr, default_data()).size == 2
    corr[0]["Underlying 2"] = 1.5
    with pytest.raises(ValueError, match="between -1 and 1"):
        basket_of(rows, corr, default_data())


# --- Grids -------------------------------------------------------------------------------------


def test_editing_the_underlyings():
    data = default_data()
    assert len(page.edit_underlyings("cn-add", ONE, data)) == 2
    assert page.edit_underlyings("cn-remove", ONE, data) is dash.no_update
    rows, _ = two_underlyings()
    assert len(page.edit_underlyings("cn-remove", rows, data)) == 1
    edited = page.edit_underlyings(
        "cn-underlyings",
        rows,
        data,
        [{"rowIndex": 1, "colId": "Spot", "value": 80.0}, {"rowIndex": 9, "colId": "Spot"}],
    )
    assert edited[1]["Spot"] == 80.0
    assert rows[1]["Spot"] == 100.0  # not mutated
    assert page.edit_underlyings(None, rows, data) is dash.no_update


def test_correlation_view():
    rows, corr = two_underlyings()
    grid, columns, shown, status = page.correlation_view(
        "cn-correlation", rows, corr, [{"rowIndex": 0, "colId": "Underlying 2", "value": 0.2}]
    )
    assert grid[1]["Underlying 1"] == 0.2  # mirrored
    assert [c["field"] for c in columns] == ["Name", "Underlying 1", "Underlying 2"]
    assert shown["display"] == "block"
    assert status == ""
    _, _, hidden, _ = page.correlation_view(None, ONE, None)
    assert hidden["display"] == "none"
    grid, _, _, status = page.correlation_view("cn-nearest", rows, corr)
    assert "nearest valid" in text_of(status)
    corr[0]["Underlying 2"] = "x"
    _, _, _, status = page.correlation_view("cn-nearest", rows, corr)
    assert "number" in text_of(status)
    # edits outside the grid, or a grid whose rows lag behind the names, are ignored
    partial = [{"Name": "Underlying 1", "Underlying 1": 1.0, "Underlying 2": 0.6}]
    grid, *_ = page.correlation_view(
        "cn-correlation",
        rows,
        partial,
        [
            {"rowIndex": 5, "colId": "Underlying 2", "value": 0.1},
            {"rowIndex": 0, "colId": "Underlying 2", "value": 0.3},
        ],
    )
    assert grid[0]["Underlying 2"] == 0.3


# --- Pricing and tabs --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def priced():
    s, data = store("Phoenix autocall"), default_data()
    result = page.price_compute(s, ONE, [], data, N)
    greeks = page.greeks_compute(result, s, ONE, [], data, N)
    return s, data, result, greeks


def test_the_price_follows_the_button(priced):
    s, data, result, _ = priced
    assert result["fingerprint"] == page.fingerprint(s, ONE, [], data, N)
    assert page.price_compute(None, ONE, [], data, N) is None  # nothing before the Price button
    assert ui.NOT_PRICED in text_of(page.kpis_view(None, None, None, ONE, [], data, N))
    refused = page.price_compute({"slug": "call"}, ONE, [], data, N)
    assert refused is not None
    assert "no coupon note" in refused["error"]
    assert kpis(s, data) is dash.no_update


def test_headline_is_the_same_as_on_every_product(priced):
    s, data, result, greeks = priced
    text = text_of(page.kpis_view(result, greeks, s, ONE, [], data, N))
    for piece in (
        "Price",
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
        "Issuer margin",
        "product sheet",
    ):
        assert piece in text
    assert "..." in text_of(page.price_headline(result, s, ONE, [], data, N))
    assert page.PRICING in text_of(page.kpis_view(None, None, s, ONE, [], data, N))
    assert page.PRICING in text_of(page.kpis_view(result, None, s, ONE, [], data, 5_000))
    error = {"error": "The spot must be positive"}
    assert "spot" in text_of(page.kpis_view(error, None, s, ONE, [], data, N))


def test_payoff_tab(priced):
    s, data, result, _ = priced
    assert page.payoff_view(result, s, ONE, [], data, N, "coupon") is dash.no_update
    out = page.payoff_view(result, s, ONE, [], data, N, "payoff")
    text = text_of(out)
    assert "With memory" in text
    assert "hence the step" in text
    assert "Gain on the price paid" in text
    assert "Breakeven" in text
    # before the price, what is paid only
    assert "Gain" not in text_of(page.payoff_view(None, s, ONE, [], data, N, "payoff")).replace(
        "the gain", ""
    )
    path = store("Barrier reverse convertible", monitoring="continuous")
    priced_path = page.price_compute(path, ONE, [], data, N)
    text = text_of(page.payoff_view(priced_path, path, ONE, [], data, N, "payoff"))
    assert "Capital, barrier breached during the life" in text
    assert "Gain if breached" in text
    assert "With memory" not in text
    assert "hence the step" not in text  # a certain coupon: no step
    no_rows = text_of(page.payoff_view(None, path, [], [], data, N, "payoff"))  # today 100 %
    assert "today 100.0%" in no_rows
    assert "no coupon note" in text_of(page.payoff_view(None, None, ONE, [], data, N, "payoff"))
    assert "Set the terms" in text_of(
        page.payoff_view(None, {"slug": "phoenix-autocall"}, ONE, [], data, N, "payoff")
    )


@pytest.mark.parametrize("name", list(CATALOG))
def test_the_capital_is_flat_above_the_barrier_on_every_coupon_note(name):
    terms = build_coupon(CATALOG[name], store(name)["terms"])
    note = terms.note
    above = np.linspace(note.protection_barrier + 0.5, 190.0, 60)
    capital = note.redemption(above, False)
    assert np.all(capital[above >= note.put_strike] == 100.0)
    # the chart draws that line first, the final coupon apart
    out = page.payoff_view(None, store(name), ONE, [], default_data(), N, "payoff")
    fig = out.children[0].figure
    assert fig.data[-1].name == "Capital paid back"
    assert fig.data[-2].name == "With the final coupon"


def test_coupon_tab_shows_only_the_observation_schedule(priced):
    s, data, result, _ = priced
    assert page.coupon_view(result, s, ONE, [], data, N, "payoff") is dash.no_update
    out = page.coupon_view(result, s, ONE, [], data, N, "coupon")
    text = text_of(out)
    assert "Observation schedule" in text
    assert "Call probability" in text
    assert "Graph" not in text
    assert "no coupon note" in text_of(page.coupon_view(None, None, ONE, [], data, N, "coupon"))
    assert "Set the terms" in text_of(
        page.coupon_view(None, {"slug": "phoenix-autocall"}, ONE, [], data, N, "coupon")
    )
    # before the price: the dates, no probability
    assert text_of(page.coupon_view(None, s, ONE, [], data, N, "coupon")).count("0.25") > 0


def test_tabs_are_those_of_every_product(priced):
    text = text_of(page.coupon_page(registry.BY_SLUG["phoenix-autocall"]))
    for tab in ("Payoff", "Coupon", "Decomposition", "Greeks", "Hedging"):
        assert f'"label": "{tab}"' in text
    assert "Monte Carlo paths" in text
    assert '"id": "cn-price"' not in text


def test_decomposition_tab_has_the_columns_of_every_product(priced):
    s, data, result, greeks = priced
    assert page.decomposition_view(result, greeks, s, ONE, [], data, N, "payoff") is dash.no_update
    text = text_of(page.decomposition_view(result, greeks, s, ONE, [], data, N, "decomposition"))
    for column in ("Brick", "Terms", "Price (%)", "Delta (%)", "Theta (%)", "Rho (%)"):
        assert column in text
    assert "Short down-and-in put (capital at risk)" in text
    assert "par if at or above 100 %" in text
    assert "memory" in text
    no_greeks = text_of(page.decomposition_view(result, None, s, ONE, [], data, N, "decomposition"))
    assert page.COMPUTING in no_greeks
    assert page.PRICING in text_of(
        page.decomposition_view(None, None, s, ONE, [], data, N, "decomposition")
    )
    error = {"error": "The spot must be positive"}
    assert "spot" in text_of(
        page.decomposition_view(error, None, s, ONE, [], data, N, "decomposition")
    )
    rc = store("Reverse convertible")
    rc_result = page.price_compute(rc, ONE, [], data, N)
    assert "Redemption at call" not in text_of(
        page.decomposition_view(rc_result, None, rc, ONE, [], data, N, "decomposition")
    )


def test_bricks_add_up_to_the_note(priced):
    s, _, result, greeks = priced
    terms = build_coupon(CATALOG["Phoenix autocall"], s["terms"])
    rows = brick_rows(terms.note, result, greeks)
    assert sum(r["Price (%)"] for r in rows[:-1]) == pytest.approx(rows[-1]["Price (%)"])
    for key in ("Delta (%)", "Gamma (%)", "Vega (%)", "Theta (%)", "Rho (%)"):
        assert sum(r[key] for r in rows[:-1]) == pytest.approx(rows[-1][key], abs=1e-9)


def test_greeks_tab():
    s, data = store("Reverse convertible"), default_data()
    rows, corr = two_underlyings()
    assert page.greeks_compute(None, s, rows, corr, data, N) is None
    result = page.price_compute(s, rows, corr, data, N)
    assert page.greeks_compute(result, s, rows, corr, data, 5_000) is None  # not these terms
    greeks = page.greeks_compute(result, s, rows, corr, data, N)
    assert greeks is not None
    assert len(greeks["assets"]) == 2
    view = page.greeks_view
    assert view(greeks, None, s, rows, corr, data, N, "payoff") is dash.no_update
    assert page.COMPUTING in text_of(view(None, None, s, rows, corr, data, N, "greeks"))
    # the Greeks first, the chart once the curves are there
    text = text_of(view(greeks, None, s, rows, corr, data, N, "greeks"))
    assert "Per underlying" in text
    assert page.COMPUTING in text
    compute = page.profile_compute
    assert compute(greeks, None, s, rows, corr, data, "payoff") is dash.no_update
    assert compute(None, None, s, rows, corr, data, "greeks") is dash.no_update
    profile = compute(greeks, None, s, rows, corr, data, "greeks")
    assert compute(greeks, profile, s, rows, corr, data, "greeks") is dash.no_update
    out = view(greeks, profile, s, rows, corr, data, N, "greeks")
    text = text_of(out)
    for greek in ("Delta", "Gamma", "Vega", "Theta", "Rho"):
        assert f'"children": "{greek}"' in text
    assert "standard error" in text
    assert page.COMPUTING not in text


def test_a_theta_that_cannot_be_computed_says_so():
    s = store("Reverse convertible", maturity=1.0, elapsed=0.25 - 0.5 / 365)
    data = default_data()
    # issued a quarter ago, minus half a day: the next observation is within a day
    result = page.price_compute(s, ONE, [], data, N)
    greeks = page.greeks_compute(result, s, ONE, [], data, N)
    assert greeks is not None
    assert greeks["theta"] is None
    text = text_of(page.kpis_view(result, greeks, s, ONE, [], data, N))
    assert "n/a" in text
    profile = page.profile_compute(greeks, None, s, ONE, [], data, "greeks")
    assert "falls within the next day" in text_of(
        page.greeks_view(greeks, profile, s, ONE, [], data, N, "greeks")
    )
    text = text_of(page.decomposition_view(result, greeks, s, ONE, [], data, N, "decomposition"))
    assert "Total" in text


def test_greeks_errors_after_a_price():
    s, data = store("Reverse convertible"), default_data()
    result = page.price_compute(s, ONE, [], data, N)
    assert result is not None
    bad = default_data() | {"spot": None}
    broken = result | {"fingerprint": page.fingerprint(s, ONE, [], bad, N)}
    assert page.greeks_compute(broken, s, ONE, [], bad, N) is None


def test_the_issuer_hedges_each_underlying():
    s, data = store("Barrier reverse convertible"), default_data()
    rows, corr = two_underlyings()
    result = page.price_compute(s, rows, corr, data, N)
    greeks = page.greeks_compute(result, s, rows, corr, data, N)
    assert greeks is not None
    args = ("call", 100.0, 0.25, "put", 100.0, 2.0)
    out, _, _ = risk_panel.static_view(s, data, greeks, risk_panel.TAB, "Delta + gamma", *args)
    text = text_of(out)
    assert "Issuer, short the note on Underlying 1" in text
    assert "Issuer, short the note on Underlying 2" in text
    assert "shares" in text
    waiting, _, _ = risk_panel.static_view(s, data, None, risk_panel.TAB, "Delta", *args)
    assert "being computed" in text_of(waiting)
    subject = risk_panel.coupon_subjects(greeks, data)[0]
    desk = to_desk(subject.greeks, spot=subject.market.spot, notional=100.0)
    # the issuer is short: the holder's desk Greeks, with the opposite sign
    assert 100.0 * float(desk.delta) == pytest.approx(-greeks["assets"][0]["delta"][0])
    assert 100.0 * float(desk.gamma) == pytest.approx(-greeks["assets"][0]["gamma"][0])
    assert float(desk.vega) == pytest.approx(-greeks["assets"][0]["vega"][0])


def link(result, greeks, s, rows, corr, data):
    """The token carried by the sheet link of a priced note."""
    href = find_sheet_link(page.kpis_view(result, greeks, s, rows, corr, data, N))
    return href.split("?s=", 1)[1]


def find_sheet_link(component):
    for child in component.children:
        if getattr(child, "id", None) == "sheet-link":
            return child.href
    raise AssertionError("no link to the sheet")


def test_product_sheet(priced):
    s, data, result, greeks = priced
    content = sheet_route.sheet_html(sheet_route.decode(link(result, None, s, ONE, [], data)))
    assert "phoenix-autocall-product-sheet.html" in content
    assert "Observation schedule" in content
    assert "Early redemption" in content
    assert "Delta Underlying 1" not in content
    state = sheet_route.decode(link(result, greeks, s, ONE, [], data))
    content = sheet_route.sheet_html(state)
    assert "Delta Underlying 1" in content
    assert "Theta (per day)" in content
    assert "Rho (per bp)" in content
    assert "Brick decomposition" in content


def test_coupon_sheets_name_their_risks():
    data = default_data()
    rows, corr = two_underlyings()
    s = store("Barrier reverse convertible", monitoring="continuous")
    result = page.price_compute(s, rows, corr, data, N)
    content = sheet_route.sheet_html(sheet_route.decode(link(result, None, s, rows, corr, data)))
    assert "Paid, barrier breached" in content
    assert "Worst-of" in content
    assert "Early redemption" not in content
    assert "Yield enhancement" in content


def test_the_price_button_fixes_the_underlyings_priced():
    assert page.priced_underlyings(0, ONE, [], N) == (dash.no_update,) * 3
    assert page.priced_underlyings(1, ONE, [], N) == (ONE, [], N)
    s = store("Phoenix autocall")
    status = page.underlyings_status
    assert status(ONE, [], N, None, None, None, None) == ""  # not priced yet
    assert status(ONE, [], N, s, ONE, [], N) == ""
    assert ui.STALE in text_of(status(ONE, [], 5_000, s, ONE, [], N))
