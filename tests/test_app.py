"""End-to-end tests of the Streamlit app, run headless with streamlit.testing.AppTest."""

from pathlib import Path

import pytest

from spricer.market.defaults import default_market
from spricer.strategies import CATALOG, Family

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ENTRY = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def app():
    at = AppTest.from_file(str(ENTRY), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def choose(at, spec) -> None:
    """Pick a product the way a user does: its family, then the product."""
    at.pills(key="widget_vanilla_family").set_value(spec.family.value).run()
    at.pills(key=f"widget_vanilla_product_{spec.family.name}").set_value(spec.name).run()


def price(at) -> str:
    at.button(key="price_button").click().run()
    assert not at.exception
    return str(at.metric[0].value)


def expected_price(spec, side: str = "Long") -> str:
    market = default_market()
    strategy = spec.build([market.spot * k / 100 for k in spec.default_strikes_pct], 1.0, 10.0)
    if side == "Short":
        strategy = strategy.short()
    return f"{100 * float(strategy.value(market)) / market.spot:.2f} %"


def test_app_opens_on_the_vanilla_pricer(app):
    assert app.title[0].value == "Vanilla options"
    assert any("click **Price**" in info.value for info in app.info)
    assert not app.metric  # nothing priced yet


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: s.name)
def test_every_product_prices_like_the_library(app, spec):
    choose(app, spec)
    assert price(app) == expected_price(spec)
    assert not app.error


def test_short_position_flips_the_price(app):
    spec = next(s for s in CATALOG if s.name == "Strangle")
    choose(app, spec)
    app.radio(key="widget_vanilla_side").set_value("Short").run()
    assert price(app) == expected_price(spec, "Short")


def test_custom_product_prices_its_legs(app):
    import presentation as pres

    app.pills(key="widget_vanilla_family").set_value(Family.CUSTOM.value).run()
    market = default_market()
    strategy = pres.custom_strategy(pres.DEFAULT_CUSTOM_LEGS, 1.0, market.spot)
    assert price(app) == f"{100 * float(strategy.value(market)) / market.spot:.2f} %"


def test_results_are_hidden_when_an_input_changes(app):
    price(app)
    app.number_input(key="widget_vanilla_maturity").set_value(2.0).run()
    assert not app.metric
    assert any("click **Price**" in info.value for info in app.info)


def test_net_of_premium_view_renders(app):
    price(app)
    app.checkbox(key="vanilla_net_of_premium").check().run()
    assert not app.exception


def test_invalid_strikes_show_an_error(app):
    choose(app, next(s for s in CATALOG if s.name == "Bull call spread"))
    app.number_input(key="widget_vanilla_strike_Bull call spread_0").set_value(120.0).run()
    assert not app.exception
    assert any("increasing" in e.value for e in app.error)


def test_volatility_surface_page_is_arbitrage_free_by_default(app):
    app.switch_page("views/volatility_surface.py").run()
    assert not app.exception
    assert app.title[0].value == "Volatility surface"
    assert any("No arbitrage" in s.value for s in app.success)


def test_market_inputs_are_shared_between_pages(app):
    app.number_input(key="widget_market_spot").set_value(120.0).run()
    app.switch_page("views/volatility_surface.py").run()
    assert not app.exception
    # the widget is redrawn from the stored value (AppTest reports widget defaults, not that)
    assert app.session_state["market_spot"] == 120.0
    assert app.session_state["widget_market_spot"] == 120.0
    app.switch_page("views/vanilla_options.py").run()
    assert app.session_state["widget_market_spot"] == 120.0


def test_search_finds_every_butterfly(app):
    app.text_input(key="widget_vanilla_search").set_value("butterfly").run()
    assert not app.exception
    options = list(app.pills(key="widget_vanilla_product_search").options)
    # the butterflies first, then the condor whose description mentions it
    assert options[:3] == ["Call butterfly", "Put butterfly", "Iron butterfly"]
    assert "Call condor" in options


def test_search_without_match_warns(app):
    app.text_input(key="widget_vanilla_search").set_value("zzzz").run()
    assert any("No product matches" in w.value for w in app.warning)


def test_every_family_is_offered_with_its_products(app):
    families = list(app.pills(key="widget_vanilla_family").options)
    assert families == [f.value for f in Family]


# --- Capital protection page ------------------------------------------------------------------

from spricer.products.capital_protection import NOTE_CATALOG  # noqa: E402


def open_capital_protection(at):
    at.switch_page("views/capital_protection.py").run()
    assert not at.exception
    return at


@pytest.mark.parametrize("spec", NOTE_CATALOG, ids=lambda s: s.name)
def test_every_note_prices_like_the_library(app, spec):
    open_capital_protection(app)
    app.pills(key="widget_cp_product").set_value(spec.name).run()
    app.button(key="cp_price_button").click().run()
    assert not app.exception
    expected = float(spec.build({}, 3.0, 100.0).value(default_market()))
    assert app.metric[0].value == f"{expected:.2f} %"


def test_issuer_spread_discounts_the_note(app):
    import math

    open_capital_protection(app)
    app.number_input(key="widget_cp_spread_bp").set_value(100.0).run()
    app.button(key="cp_price_button").click().run()
    spec = NOTE_CATALOG[0]
    expected = math.exp(-0.01 * 3.0) * float(spec.build({}, 3.0, 100.0).value(default_market()))
    assert app.metric[0].value == f"{expected:.2f} %"


def test_invalid_terms_show_an_error(app):
    open_capital_protection(app)
    app.number_input(key="widget_cp_Participation note_cap").set_value(90.0).run()
    assert not app.exception
    assert any("cap" in e.value for e in app.error)


@pytest.mark.parametrize("choice", ["Continuously", "Daily (252 dates a year)"])
def test_shark_barrier_observation_can_be_chosen(app, choice):
    from spricer.monitoring import Monitoring

    open_capital_protection(app)
    app.pills(key="widget_cp_product").set_value("Shark note (up)").run()
    app.radio(key="widget_cp_monitoring").set_value(choice).run()
    app.button(key="cp_price_button").click().run()
    assert not app.exception
    mode = Monitoring.CONTINUOUS if choice == "Continuously" else Monitoring.DISCRETE
    spec = next(s for s in NOTE_CATALOG if s.name == "Shark note (up)")
    note = spec.build({}, 3.0, 100.0, mode, 756 if mode is Monitoring.DISCRETE else None)
    assert app.metric[0].value == f"{float(note.value(default_market())):.2f} %"


def run_monte_carlo_check(at):
    at.button(key="cp_price_button").click().run()
    at.radio(key="widget_cp_mc_paths").set_value("50,000").run()
    at.button(key="cp_mc_button").click().run()
    assert not at.exception
    return at


def test_monte_carlo_check_agrees_with_a_european_note(app):
    open_capital_protection(app)
    run_monte_carlo_check(app)
    labels = [m.label for m in app.metric]
    assert "Monte Carlo" in labels
    assert "95 % interval" in labels
    gap = next(m for m in app.metric if m.label == "MC − closed form")
    assert abs(float(str(gap.delta).split()[0])) < 4.0  # in standard errors
    assert any("Monte Carlo noise" in c.value for c in app.caption)


def test_monte_carlo_check_explains_the_smile_effect_on_a_path_barrier(app):
    open_capital_protection(app)
    app.pills(key="widget_cp_product").set_value("Shark note (up)").run()
    app.radio(key="widget_cp_monitoring").set_value("Continuously").run()
    run_monte_carlo_check(app)
    assert any("local volatility of the whole smile" in i.value for i in app.info)


def test_typed_volatilities_are_flagged_in_the_monte_carlo_check(app):
    open_capital_protection(app)
    app.button(key="cp_price_button").click().run()
    editor = app.session_state["cp_vols"].copy()
    editor.loc[0, "Volatility (%)"] = 30.0
    app.session_state["cp_vols"] = editor
    app.session_state["cp_vols_base"] = editor
    run_monte_carlo_check(app)
    assert any("apply to the price at the top only" in c.value for c in app.caption)
