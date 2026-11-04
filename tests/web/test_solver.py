"""Solving a term from the terms form, on every family of products."""

import json

import dash
import pytest

from web import layout, registry, solver
from web.coupons import CATALOG as COUPONS
from web.coupons import build_coupon, default_underlyings
from web.coupons import default_terms as coupon_terms
from web.market import default_data, market_from_data
from web.notes import CATALOG, build_note, default_terms
from web.options import build_position, setup_of

DATA = default_data()
MARKET = market_from_data(DATA)
N = 2_000


def text_of(component) -> str:
    return json.dumps(component.to_plotly_json(), default=lambda o: o.to_plotly_json())


def ids(*keys):
    return [{"type": "term", "key": k} for k in keys]


def note_store(name, **terms):
    return {"slug": registry.slugify(name), "terms": default_terms(CATALOG[name]) | terms}


def option_setup(slug):
    setup = setup_of(registry.BY_SLUG[slug])
    assert setup is not None
    return setup


def option_store(slug, **terms):
    return {"slug": slug, "terms": option_setup(slug).default_terms() | terms}


def coupon_store(name, **terms):
    return {"slug": registry.slugify(name), "terms": coupon_terms(COUPONS[name]) | terms}


def written(values, all_ids, key):
    assert len(values) == len(all_ids)
    return next(v for v, i in zip(values, all_ids, strict=True) if i["key"] == key)


@pytest.mark.parametrize("key", ["participation", "strike", "maturity", "spread_bp"])
def test_every_term_of_a_note_reaches_the_target(key):
    store = note_store("Participation note")
    all_ids = ids("participation", "strike", "maturity", "spread_bp", "notional")
    out, values = solver.solve_view(1, key, 97.0, store, DATA, all_ids)
    assert "gives a price of" in text_of(out)
    value = written(values, all_ids, key)
    assert values[-1] is dash.no_update
    solved = build_note(CATALOG["Participation note"], store["terms"] | {key: value}, MARKET)
    assert solved.value() == pytest.approx(97.0, abs=0.05)


def test_a_note_explains_what_it_cannot_solve():
    all_ids = ids("participation")
    store = note_store("Participation note")
    for key, target, s, message in [
        ("participation", None, store, "Type a target"),
        ("nothing", 98.0, store, "cannot be solved"),
        ("participation", 500.0, store, "no participation gives"),
        ("cap", 98.0, note_store("Participation note", capped=False), "Switch the cap"),
        ("capped", 98.0, store, "cannot be solved"),
    ]:
        out, values = solver.solve_view(1, key, target, s, DATA, all_ids)
        assert message in text_of(out)
        assert values == [dash.no_update]
    assert "Set the terms" in text_of(
        solver.solve_view(1, "maturity", 98.0, None, DATA, all_ids)[0]
    )


def test_no_click_no_solve():
    for clicks, key in ((0, "maturity"), (None, "maturity"), (1, None)):
        assert solver.solve_view(clicks, key, 98.0, None, DATA, ids("a")) == (
            dash.no_update,
            [dash.no_update],
        )
        assert solver.solve_coupon_view(clicks, key, 98.0, None, [], [], DATA, N, ids("a")) == (
            dash.no_update,
            [dash.no_update],
        )


@pytest.mark.parametrize(
    ("slug", "button", "absent"),
    [
        ("call", "solve-run", "cn-solve-run"),
        ("digital-call", "solve-run", "cn-solve-run"),
        ("up-and-out-call", "solve-run", "cn-solve-run"),
        ("bonus-note", "solve-run", "cn-solve-run"),
        ("phoenix-autocall", "cn-solve-run", '"id": "solve-run"'),
    ],
)
def test_the_solver_sits_at_the_bottom_of_the_form(slug, button, absent):
    """One solver in the form, the other inputs being plain terms."""
    text = text_of(layout.product_page(registry.BY_SLUG[slug]))
    assert button in text
    assert absent not in text
    assert text.count("bi-bullseye") == 1
    assert text.index("Terms") < text.index("Term to find")
    assert text.index("Term to find") < text.index("product-kpis")


@pytest.mark.parametrize(
    ("slug", "key", "target"),
    [
        ("call", "strike_0", 5.0),
        ("call", "maturity", 5.0),
        ("digital-call", "payout", 3.0),
        ("up-and-out-call", "barrier", 4.0),
        ("bull-call-spread", "strike_1", 3.0),
    ],
)
def test_every_term_of_an_option_reaches_the_target(slug, key, target):
    store = option_store(slug)
    all_ids = ids(key)
    out, values = solver.solve_view(1, key, target, store, DATA, all_ids)
    assert "gives a price of" in text_of(out)
    solved = build_position(option_setup(slug), store["terms"] | {key: values[0]}, MARKET)
    assert solved.price_pct() == pytest.approx(target, abs=0.05)


def test_an_option_explains_what_it_cannot_solve():
    out, _ = solver.solve_view(1, "notional", 5.0, option_store("call"), DATA, ids("notional"))
    assert "cannot be solved" in text_of(out)


@pytest.mark.parametrize("key", ["coupon_pa", "barrier", "coupon_level", "strike", "spread_bp"])
def test_every_term_of_a_coupon_note_reaches_the_target(key):
    store = coupon_store("Phoenix autocall")
    rows = default_underlyings(DATA)
    all_ids = ids(key, "maturity")
    out, values = solver.solve_coupon_view(1, key, 99.0, store, rows, [], DATA, N, all_ids)
    assert "gives a price of **9" in text_of(out)
    assert "\\u00b1" in text_of(out)  # with its Monte Carlo error
    assert values[1] is dash.no_update
    assert isinstance(values[0], float)


def test_a_coupon_note_explains_what_it_cannot_solve():
    store = coupon_store("Phoenix autocall")
    rows = default_underlyings(DATA)
    all_ids = ids("coupon_pa")
    out, values = solver.solve_coupon_view(1, "coupon_pa", None, store, rows, [], DATA, N, all_ids)
    assert "Type a target" in text_of(out)
    assert values == [dash.no_update]
    out, _ = solver.solve_coupon_view(1, "coupon_pa", 98.0, None, rows, [], DATA, N, all_ids)
    assert "Set the terms" in text_of(out)
    out, _ = solver.solve_coupon_view(1, "barrier", 500.0, store, rows, [], DATA, N, all_ids)
    assert "no protection barrier gives" in text_of(out)
    built = build_coupon(COUPONS["Phoenix autocall"], store["terms"])
    assert built.spread == 0.0


def test_the_solver_reads_the_form_as_it_stands():
    terms = note_store("Participation note")["terms"]
    all_ids = ids(*terms)
    values = list(terms.values())
    path = "/product/participation-note"
    out, solved = solver.solve_form(1, "participation", 97.0, values, all_ids, path, DATA)
    assert "Click Price" in text_of(out)
    assert written(solved, all_ids, "participation") != terms["participation"]
    terms = coupon_store("Phoenix autocall")["terms"]
    all_ids = ids(*terms)
    rows = default_underlyings(DATA)
    out, solved = solver.solve_coupon_form(
        1,
        "coupon_pa",
        99.0,
        list(terms.values()),
        all_ids,
        "/product/phoenix-autocall",
        rows,
        [],
        DATA,
        N,
    )
    assert "Click Price" in text_of(out)
    assert isinstance(written(solved, all_ids, "coupon_pa"), float)
