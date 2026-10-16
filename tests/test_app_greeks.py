"""End-to-end tests of the Greeks lab and of the Monte Carlo Greeks on the note pages."""

from pathlib import Path

import pytest

from spricer import strategies as strat
from spricer.greeks_lab import greeks_at
from spricer.market import FlatVol, Market
from spricer.market.defaults import default_market

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ENTRY = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
LAB = "views/greeks_lab.py"
INSTRUMENTS = [
    "Call",
    "Put",
    "Digital call",
    "Digital put",
    "Bull call spread",
    "Straddle",
    "Call butterfly",
    "Barrier option",
]


@pytest.fixture
def app():
    at = AppTest.from_file(str(ENTRY), default_timeout=120)
    at.run()
    assert not at.exception
    at.switch_page(LAB).run()
    assert not at.exception
    return at


def metric(at, label):
    return next(m for m in at.metric if m.label == label)


def bullets(at) -> str:
    return " ".join(m.value for m in at.markdown if m.value.startswith("- "))


@pytest.mark.parametrize("instrument", INSTRUMENTS)
def test_every_instrument_is_explained(app, instrument):
    app.pills(key="widget_lab_instrument").set_value(instrument).run()
    assert not app.exception
    assert "**Delta" in bullets(app)
    assert len(app.tabs) == 3


def test_the_lab_prices_like_the_library(app):
    base = default_market()
    flat = Market(base.spot, base.rate, base.dividend_yield, FlatVol(0.2))
    expected = greeks_at(strat.call(100.0, 1.0), flat)
    assert metric(app, "Price (%)").value == f"{float(expected['Price (%)']):.2f}"
    assert metric(app, "Delta (%)").value == f"{float(expected['Delta (%)']):+.2f}"
    assert "At the money" in bullets(app)


def test_moving_the_strike_and_the_position_changes_the_reading(app):
    app.slider(key="widget_lab_strike").set_value(70.0).run()
    assert "In the money" in bullets(app)
    app.radio(key="widget_lab_side").set_value("Short").run()
    assert "Short position" in bullets(app)
    app.radio(key="widget_lab_vol_source").set_value("Smile of the Volatility surface page").run()
    assert not app.exception


def test_a_crossed_barrier_is_flagged(app):
    app.pills(key="widget_lab_instrument").set_value("Barrier option").run()
    app.selectbox(key="widget_lab_barrier_type").set_value("up-and-out").run()
    assert any("already crossed" in w.value for w in app.warning)
    assert "worth nothing" in bullets(app)
    app.selectbox(key="widget_lab_barrier_type").set_value("down-and-in").run()
    assert not app.warning


def test_the_time_tab_follows_the_chosen_greek(app):
    app.selectbox(key="widget_lab_time_greek").set_value("Theta (%)").run()
    assert not app.exception
    assert "**Vanna" in bullets(app)


def test_note_pages_compute_monte_carlo_greeks():
    at = AppTest.from_file(str(ENTRY), default_timeout=180)
    at.run()
    at.switch_page("views/autocallables.py").run()
    at.radio(key="widget_cn_autocallables_paths").set_value("20,000").run()
    at.button(key="cn_autocallables_price").click().run()
    at.button(key="cn_autocallables_greeks_button").click().run()
    assert not at.exception
    assert metric(at, "Rho (%)") is not None
    assert metric(at, "Theta (%)") is not None
    text = " ".join(m.value for m in at.markdown)
    assert "the investor gains" in text
    assert any("common random numbers" in c.value for c in at.caption)
