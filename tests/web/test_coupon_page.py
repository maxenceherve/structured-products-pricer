"""Coupon note pages: terms, underlyings, correlation, Monte Carlo results, solver, sheet."""

import json

import dash
import numpy as np
import pytest

from web import coupon_page as page
from web import registry
from web.app import kpis
from web.coupons import (
    CATALOG,
    MAX_UNDERLYINGS,
    add_underlying,
    basket_of,
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
    return {
        "slug": registry.slugify(name),
        "terms": default_terms(CATALOG[name]) | terms,
        "vols": [],
    }


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
        ({"spread_bp": -1.0}, "credit spread must be non-negative"),
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
    result, headline = page.price_view(1, s, ONE, [], data, N)
    return s, data, result, headline


def test_price_view(priced):
    s, data, result, headline = priced
    text = text_of(headline)
    assert "Issuer margin" in text
    assert "Probability of loss" in text
    assert result["fingerprint"] == page.fingerprint(s, ONE, [], data, N)
    assert page.price_view(0, s, ONE, [], data, N) == (dash.no_update, dash.no_update)
    result, headline = page.price_view(1, None, ONE, [], data, N)
    assert result is None
    assert "Set the terms" in text_of(headline)
    _, headline = page.price_view(1, {"slug": "call", "terms": {}}, ONE, [], data, N)
    assert "no coupon note" in text_of(headline)
    assert kpis(s, data) is dash.no_update


def test_payoff_tab(priced):
    s, data, _, _ = priced
    assert page.payoff_view(s, ONE, [], data, "legs") is dash.no_update
    text = text_of(page.payoff_view(s, ONE, [], data, "payoff"))
    assert "With memory" in text
    path = store("Barrier reverse convertible", monitoring="continuous")
    text = text_of(page.payoff_view(path, [], [], data, "payoff"))  # no underlying: today 100 %
    assert "Barrier breached during the life" in text
    assert "With memory" not in text
    assert "no coupon note" in text_of(page.payoff_view(None, ONE, [], data, "payoff"))
    assert "Set the terms" in text_of(
        page.payoff_view({"slug": "phoenix-autocall"}, ONE, [], data, "payoff")
    )


def test_legs_and_life_tabs(priced):
    s, data, result, _ = priced
    assert page.legs_view(result, s, ONE, [], data, N, "payoff") is dash.no_update
    assert "Click Price" in text_of(page.legs_view(None, s, ONE, [], data, N, "legs"))
    assert "changed" in text_of(page.legs_view(result, s, ONE, [], data, 5_000, "legs"))
    text = text_of(page.legs_view(result, s, ONE, [], data, N, "legs"))
    assert "Capital at risk" in text
    assert "Closed form" not in text  # an autocall has no closed form
    assert page.life_view(result, s, ONE, [], data, N, "payoff") is dash.no_update
    text = text_of(page.life_view(result, s, ONE, [], data, N, "life"))
    assert "When does the note end" in text
    assert "Observation schedule" in text
    text = text_of(page.life_view(None, s, ONE, [], data, N, "life"))
    assert "Click Price" in text
    assert "no coupon note" in text_of(page.life_view(None, None, ONE, [], data, N, "life"))
    assert "Set the terms" in text_of(
        page.life_view(None, {"slug": "phoenix-autocall"}, ONE, [], data, N, "life")
    )


@pytest.mark.parametrize(
    ("monitoring", "phrase"), [("european", "noise"), ("continuous", "effect on the barrier")]
)
def test_a_brc_on_one_underlying_is_checked_against_its_closed_form(monitoring, phrase):
    s, data = store("Barrier reverse convertible", monitoring=monitoring), default_data()
    result, _ = page.price_view(1, s, ONE, [], data, 20_000)
    text = text_of(page.legs_view(result, s, ONE, [], data, 20_000, "legs"))
    assert "Monte Carlo \\u2212 closed form" in text
    assert phrase in text


def test_greeks_after_the_price():
    s, data = store("Reverse convertible"), default_data()
    rows, corr = two_underlyings()
    assert page.greeks_compute(None, s, rows, corr, data, N) == (None, "")
    result, _ = page.price_view(1, s, rows, corr, data, N)
    stale = page.greeks_compute(result, s, rows, corr, data, 5_000)
    assert stale == (None, "")
    greeks, headline = page.greeks_compute(result, s, rows, corr, data, N)
    assert "Delta (all underlyings)" in text_of(headline)
    assert len(greeks["assets"]) == 2
    assert page.greeks_view(greeks, s, rows, corr, data, N, "payoff") is dash.no_update
    text = text_of(page.greeks_view(greeks, s, rows, corr, data, N, "greeks-hedging"))
    assert "most exposed" in text
    assert "delta hedge" in text
    assert "Buy" in text
    assert "Click Price" in text_of(
        page.greeks_view(None, s, rows, corr, data, N, "greeks-hedging")
    )
    # issued a quarter ago, minus half a day: the next observation is within a day
    alive = store("Reverse convertible", maturity=1.0, elapsed=0.25 - 0.5 / 365)
    result, _ = page.price_view(1, alive, ONE, [], data, N)
    greeks, headline = page.greeks_compute(result, alive, ONE, [], data, N)
    assert "observation within a day" in text_of(headline)
    text = text_of(page.greeks_view(greeks, alive, ONE, [], data, N, "greeks-hedging"))
    assert "not computed" in text
    # a short note: the issuer sells shares
    short_delta = {
        "fingerprint": "x",
        "assets": [{"name": "A", "delta": [-10.0, 1.0], "gamma": [0.0, 0.1], "vega": [0.0, 0.1]}],
        "theta": [0.0, 0.0],
        "rho": [0.0, 0.0],
        "spots": [100.0],
        "notional": 1e6,
    }
    current = page.fingerprint(alive, ONE, [], data, N)
    text = text_of(
        page.greeks_view(
            short_delta | {"fingerprint": current}, alive, ONE, [], data, N, "greeks-hedging"
        )
    )
    assert "Sell" in text


def test_greeks_errors_after_a_price():
    s, data = store("Reverse convertible"), default_data()
    result, _ = page.price_view(1, s, ONE, [], data, N)
    bad = default_data() | {"spot": None}
    broken = result | {"fingerprint": page.fingerprint(s, ONE, [], bad, N)}
    greeks, headline = page.greeks_compute(broken, s, ONE, [], bad, N)
    assert greeks is None
    assert "spot" in text_of(headline)


def test_coupon_greeks_reading_cases():
    from web import explain as ex

    lines = ex.coupon_greeks(["A"], [-5.0], [0.2])
    assert "loses" in lines[0]
    assert "helps" in lines[0]
    assert len(lines) == 2  # one underlying: no worst-of sentence
    assert "hurts" not in ex.coupon_greeks(["A"], [5.0], [0.0])[0]


def test_solve_tab(priced):
    s, data, _, _ = priced
    out, solution, style = page.solve_view(0, s, ONE, [], data, N, "coupon_pa", 2.0)
    assert solution is None
    assert style["display"] == "none"
    out, solution, style = page.solve_view(1, s, ONE, [], data, N, "coupon_pa", 2.0)
    assert solution["key"] == "coupon_pa"
    assert "target 98.00" in text_of(out)
    assert style["display"] == "block"
    out, solution, _ = page.solve_view(1, s, ONE, [], data, N, "strike", 2.0)
    assert "Pick a term" in text_of(out)
    ids = [{"type": "term", "key": "coupon_pa"}, {"type": "term", "key": "maturity"}]
    assert page.apply_solution(1, {"key": "coupon_pa", "value": 7.5}, ids) == [7.5, dash.no_update]
    assert page.apply_solution(None, None, ids) == [dash.no_update] * 2


def test_product_sheet(priced):
    s, data, result, _ = priced
    assert page.download_sheet(0, result, None, s, ONE, [], data, N) is dash.no_update
    assert page.download_sheet(1, result, None, s, ONE, [], data, 5_000) is dash.no_update
    out = page.download_sheet(1, result, None, s, ONE, [], data, N)
    assert out["filename"] == "phoenix-autocall-product-sheet.html"
    assert "Observation schedule" in out["content"]
    assert "Early redemption" in out["content"]
    greeks, _ = page.greeks_compute(result, s, ONE, [], data, N)
    out = page.download_sheet(1, result, greeks, s, ONE, [], data, N)
    assert "Delta Underlying 1" in out["content"]


def test_coupon_sheets_name_their_risks():
    data = default_data()
    rows, corr = two_underlyings()
    s = store("Barrier reverse convertible", monitoring="continuous")
    result, _ = page.price_view(1, s, rows, corr, data, N)
    content = page.download_sheet(1, result, None, s, rows, corr, data, N)["content"]
    assert "Paid, barrier breached" in content
    assert "Worst-of" in content
    assert "Early redemption" not in content
    assert "Yield enhancement" in content
