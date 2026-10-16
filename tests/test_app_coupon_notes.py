"""End-to-end tests of the coupon note pages (Yield enhancement, Autocallables)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spricer.market import Basket
from spricer.market.defaults import default_market
from spricer.mc.coupon_notes import price_coupon_note
from spricer.products.coupon_notes import COUPON_NOTE_CATALOG

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ENTRY = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGES = {
    "Yield enhancement": "views/yield_enhancement.py",
    "Autocallables": "views/autocallables.py",
}
PREFIX = {"Yield enhancement": "cn_yield", "Autocallables": "cn_autocallables"}


@pytest.fixture
def app():
    at = AppTest.from_file(str(ENTRY), default_timeout=120)
    at.run()
    assert not at.exception
    return at


def open_family(at, family: str, product: str | None = None):
    at.switch_page(PAGES[family]).run()
    assert not at.exception
    prefix = PREFIX[family]
    if product is not None:
        at.pills(key=f"widget_{prefix}_product").set_value(product).run()
    at.radio(key=f"widget_{prefix}_paths").set_value("20,000").run()
    return prefix


def metric(at, label: str):
    return next(m for m in at.metric if m.label == label)


@pytest.mark.parametrize("spec", COUPON_NOTE_CATALOG, ids=lambda s: s.name)
def test_every_coupon_note_prices_like_the_library(app, spec):
    prefix = open_family(app, spec.family, spec.name)
    app.button(key=f"{prefix}_price").click().run()
    assert not app.exception
    note = spec.build({}, 3.0, 4)
    expected = price_coupon_note(note, Basket.single(default_market()), n_paths=20_000)
    assert metric(app, "Price").value == f"{expected.total.value:.2f} %"
    assert metric(app, "Expected life").value == f"{expected.expected_life.value:.2f} y"
    assert len(app.tabs) >= 3


def test_a_barrier_reverse_convertible_is_checked_against_its_bricks(app):
    prefix = open_family(app, "Yield enhancement", "Barrier reverse convertible")
    app.button(key=f"{prefix}_price").click().run()
    assert metric(app, "Closed form (bricks)") is not None
    assert any("Monte Carlo noise" in c.value for c in app.caption)
    app.radio(key=f"widget_{prefix}_monitoring").set_value("Continuously").run()
    app.button(key=f"{prefix}_price").click().run()
    assert any("smile's effect on the barrier" in i.value for i in app.info)


def two_underlyings(at, prefix: str) -> None:
    table = pd.DataFrame(
        [["Stock A", 100.0, 100.0, 2.0, 20.0], ["Stock B", 50.0, 55.0, 1.0, 28.0]],
        columns=["Name", "Spot", "Initial level", "Dividend yield (%)", "ATM vol 1y (%)"],
    )
    at.session_state[f"{prefix}_underlyings_last"] = table
    at.run()
    assert not at.exception


def test_a_worst_of_needs_a_valid_correlation(app):
    prefix = open_family(app, "Autocallables", "Phoenix autocall")
    two_underlyings(app, prefix)
    corr = app.session_state[f"{prefix}_correlation"]
    assert list(corr.index) == ["Stock A", "Stock B"]
    app.button(key=f"{prefix}_price").click().run()
    assert not app.exception
    worst = float(metric(app, "Price").value.split()[0])
    single_prefix = PREFIX["Autocallables"]
    assert single_prefix == prefix
    assert worst < 99.0  # the worst-of pays the same coupon for more risk

    bad = pd.DataFrame([[1.0, 1.5], [1.5, 1.0]], index=corr.index, columns=corr.columns)
    app.session_state[f"{prefix}_correlation_editor_last"] = bad
    app.run()
    assert any("between -1 and 1" in e.value for e in app.error)


def test_an_inconsistent_correlation_can_be_replaced_by_the_nearest_valid_one(app):
    prefix = open_family(app, "Autocallables", "Athena autocall")
    table = pd.DataFrame(
        [
            ["A", 100.0, 100.0, 2.0, 20.0],
            ["B", 100.0, 100.0, 2.0, 20.0],
            ["C", 100.0, 100.0, 2.0, 20.0],
        ],
        columns=["Name", "Spot", "Initial level", "Dividend yield (%)", "ATM vol 1y (%)"],
    )
    app.session_state[f"{prefix}_underlyings_last"] = table
    app.run()
    names = ["A", "B", "C"]
    bad = pd.DataFrame(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.8], [0.1, 0.8, 1.0]], index=names, columns=names
    )
    app.session_state[f"{prefix}_correlation_editor_last"] = bad
    app.run()
    assert any("not positive definite" in e.value for e in app.error)
    app.button(key=f"{prefix}_correlation_fix").click().run()
    assert not app.exception
    fixed = app.session_state[f"{prefix}_correlation"].to_numpy(dtype=float)
    assert np.linalg.eigvalsh(fixed)[0] > 0.0
    assert not any("positive definite" in e.value for e in app.error)


def test_invalid_inputs_show_errors(app):
    prefix = open_family(app, "Autocallables")
    app.number_input(key=f"widget_{prefix}_maturity").set_value(1.1).run()
    assert any("whole number" in e.value for e in app.error)
    app.number_input(key=f"widget_{prefix}_maturity").set_value(3.0).run()
    table = pd.DataFrame(
        [["A", 100.0, 100.0, 2.0, 20.0], ["A", 100.0, 100.0, 2.0, 20.0]],
        columns=["Name", "Spot", "Initial level", "Dividend yield (%)", "ATM vol 1y (%)"],
    )
    app.session_state[f"{prefix}_underlyings_last"] = table
    app.run()
    assert any("distinct names" in e.value for e in app.error)
    app.session_state[f"{prefix}_underlyings_last"] = table.iloc[:0]
    app.run()
    assert any("at least one underlying" in e.value for e in app.error)
    missing = table.iloc[:1].copy()
    missing.loc[0, "Spot"] = np.nan
    app.session_state[f"{prefix}_underlyings_last"] = missing
    app.run()
    assert any("Every underlying needs" in e.value for e in app.error)
    many = pd.concat([table.iloc[:1]] * 6, ignore_index=True)
    many["Name"] = [f"S{i}" for i in range(6)]
    app.session_state[f"{prefix}_underlyings_last"] = many
    app.run()
    assert any("At most 5" in e.value for e in app.error)


def test_solver_finds_the_coupon_and_applies_it(app):
    prefix = open_family(app, "Autocallables", "Phoenix autocall")
    app.button(key=f"{prefix}_price").click().run()
    app.button(key=f"{prefix}_solve_button").click().run()
    assert not app.exception
    assert app.success, "the solution is shown"
    app.button(key=f"{prefix}_solve_apply").click().run()
    assert not app.exception
    applied = app.session_state[f"{prefix}_Phoenix autocall_coupon_pa"]
    assert 0.0 < applied < 60.0
    assert applied != 8.0


def test_unreachable_targets_are_explained(app):
    prefix = open_family(app, "Yield enhancement", "Reverse convertible")
    app.button(key=f"{prefix}_price").click().run()
    app.number_input(key=f"widget_{prefix}_solve_margin").set_value(-400.0).run()
    app.button(key=f"{prefix}_solve_button").click().run()
    assert any("gives the target" in e.value for e in app.error)


def test_a_note_already_alive_is_priced(app):
    prefix = open_family(app, "Autocallables", "Phoenix autocall")
    app.number_input(key=f"widget_{prefix}_elapsed").set_value(1.0).run()
    app.number_input(key=f"widget_{prefix}_memory_accrued").set_value(4.0).run()
    app.checkbox(key=f"widget_{prefix}_breached").check().run()
    app.button(key=f"{prefix}_price").click().run()
    assert not app.exception
    assert float(metric(app, "Probability of loss").value.split()[0]) > 5.0


def test_capital_protection_solves_the_participation(app):
    app.switch_page("views/capital_protection.py").run()
    app.button(key="cp_price_button").click().run()
    app.selectbox(key="widget_cp_solve_term").set_value("Participation").run()
    app.button(key="cp_solve_button").click().run()
    assert not app.exception
    assert any("closed form, exact" in s.value for s in app.success)
    app.button(key="cp_solve_apply").click().run()
    assert app.session_state["cp_Participation note_participation"] != 100.0
    app.button(key="cp_price_button").click().run()  # new terms: price again
    app.number_input(key="widget_cp_solve_target").set_value(500.0).run()
    app.button(key="cp_solve_button").click().run()
    assert any("gives the target" in e.value for e in app.error)
