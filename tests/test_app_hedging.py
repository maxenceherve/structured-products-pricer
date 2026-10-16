"""End-to-end tests of the Hedging page and of its explanations."""

from pathlib import Path

import numpy as np
import pytest

import hedging_explain as hx
from spricer import strategies as strat
from spricer.bricks import EuropeanOption
from spricer.hedging import static_hedge
from spricer.market import FlatVol, Market
from spricer.risk import Attribution

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ENTRY = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
FLAT = Market(100.0, 0.03, 0.01, FlatVol(0.2))


@pytest.fixture
def app():
    at = AppTest.from_file(str(ENTRY), default_timeout=120)
    at.run()
    at.switch_page("views/hedging.py").run()
    assert not at.exception
    return at


def bullets(at) -> str:
    return " ".join(m.value for m in at.markdown if m.value.startswith("- "))


@pytest.mark.parametrize("target", ["Delta", "Delta + gamma", "Delta + gamma + vega"])
def test_every_static_hedge_is_shown_and_explained(app, target):
    app.radio(key="widget_hedge_target").set_value(target).run()
    assert not app.exception
    assert "shares" in bullets(app)
    assert len(app.tabs) == 3


def test_a_short_position_reads_as_a_market_maker(app):
    app.radio(key="widget_hedge_side").set_value("Short").run()
    app.radio(key="widget_hedge_target").set_value("Delta + gamma").run()
    text = bullets(app)
    assert "short gamma" in text
    assert "market maker" in text


def test_identical_hedging_options_are_refused(app):
    app.radio(key="widget_hedge_target").set_value("Delta + gamma + vega").run()
    app.radio(key="widget_hedge_h2_kind").set_value("call").run()
    app.number_input(key="widget_hedge_h2_maturity").set_value(0.25).run()
    assert any("proportional" in e.value for e in app.error)


def test_the_simulator_runs_and_compares_with_theory(app):
    app.selectbox(key="widget_hedge_paths").set_value("2,000").run()
    app.selectbox(key="widget_hedge_frequency").set_value("Weekly").run()
    app.slider(key="widget_hedge_realised").set_value(30.0).run()
    app.number_input(key="widget_hedge_cost").set_value(5.0).run()
    app.button(key="hedge_simulate").click().run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert {"Mean P&L", "Theory", "Standard deviation"} <= set(labels)
    text = bullets(app)
    assert "earns" in text
    assert "Transaction costs" in text


def test_pnl_explain_and_its_limits(app):
    app.number_input(key="widget_hedge_move").set_value(-40.0).run()
    assert not app.exception
    assert "residual is large" in bullets(app)
    app.number_input(key="widget_hedge_days").set_value(400.0).run()
    assert any("before maturity" in e.value for e in app.error)


# --- Explanations --------------------------------------------------------------------------------


def test_static_reading_names_every_leg():
    position = strat.call(100.0, 1.0).greeks(FLAT)
    hedge = static_hedge(
        position,
        (
            ("calls", EuropeanOption("call", 100.0, 0.25).greeks(FLAT)),
            ("puts", EuropeanOption("put", 90.0, 2.0).greeks(FLAT)),
        ),
    )
    text = " ".join(hx.static_reading(hedge, 100, short=False))
    assert "long gamma" in text
    assert "Two options are needed" in text
    assert "buyer" in text
    flat_position = static_hedge(strat.synthetic_forward(100.0, 1.0).greeks(FLAT))
    assert "gamma" not in hx.static_reading(flat_position, 1, short=False)[0]


def test_simulation_readings():
    loses = hx.simulation_reading(-1.0, -1.1, 0.5, 0.5, 15.0, 20.0, True, 0.0)
    assert "loses" in loses[0]
    assert not any("Transaction costs" in line for line in loses)
    short_wins = hx.simulation_reading(1.0, 1.1, 0.5, 0.5, 15.0, 20.0, False, 0.0)
    assert "short gamma position earns" in short_wins[0]


def test_attribution_readings():
    terms = dict.fromkeys(("Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Volga"), 0.0)
    small = Attribution(10.0, 9.0, terms | {"Delta": -1.0}, terms, {})
    assert "reliable" in " ".join(hx.attribution_reading(small))
    nothing = Attribution(10.0, 10.0, terms, terms, {})
    assert "(100 %)" in hx.attribution_reading(nothing)[0]
    assert np.isfinite(small.residual)
