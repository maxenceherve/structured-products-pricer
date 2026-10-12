"""Unit tests of the app's pure helpers (presentation and market inputs)."""

import numpy as np
import pandas as pd
import pytest

import presentation as pres
import state
from spricer import strategies as st
from spricer.bricks import DigitalOption, EuropeanOption
from spricer.strategies import Leg

T = 1.0


def test_leg_names_say_long_or_short():
    assert pres.leg_name(Leg(EuropeanOption("call", 100.0, T), 1.0)) == "Long call"
    assert pres.leg_name(Leg(EuropeanOption("put", 90.0, T), -1.0)) == "Short put"
    assert pres.leg_name(Leg(EuropeanOption("call", 100.0, T), -2.0)) == "Short 2 calls"
    assert pres.leg_name(Leg(DigitalOption("call", 100.0, T), 10.0)) == "Long digital call"
    names = [pres.leg_name(leg) for leg in st.covered_call(110.0, T).legs]
    assert names == ["Long underlying", "Short call"]


def test_strike_levels_name_the_legs_struck_there(market):
    assert pres.strike_levels(st.straddle(100.0, T), market.spot) == [
        (100.0, "Long call + Long put strike 100%")
    ]
    assert pres.strike_levels(st.risk_reversal(90.0, 110.0, T), market.spot) == [
        (90.0, "Short put strike 90%"),
        (110.0, "Long call strike 110%"),
    ]
    assert pres.strike_levels(st.iron_butterfly(90.0, 100.0, 110.0, T), market.spot)[1] == (
        100.0,
        "Short put + Short call strike 100%",
    )
    assert pres.strike_levels(st.digital_call(105.0, T), market.spot) == [
        (105.0, "Long digital call strike 105%")
    ]


def test_smile_table_is_prefilled_and_typed_vols_override_only_their_strike(market):
    strategy = st.bull_call_spread(100.0, 110.0, T)
    smile = pres.smile_vols(strategy, market)
    assert list(smile["Leg"]) == ["Long call", "Short call"]
    assert list(smile["Strike (% of spot)"]) == [100.0, 110.0]
    assert smile["Volatility (%)"].iloc[0] == pytest.approx(100 * float(market.vol.vol(100.0, T)))
    # unchanged table: the smile applies exactly and nothing is reported as edited
    assert pres.changed_vols(smile, smile).empty
    same = pres.with_vols(market, smile, smile)
    assert float(strategy.value(same)) == pytest.approx(float(strategy.value(market)), rel=1e-14)
    # typing a vol replaces the smile at that strike only
    edited = smile.copy()
    edited.loc[0, "Volatility (%)"] = 30.0
    changed = pres.changed_vols(edited, smile)
    assert list(changed["Leg"]) == ["Long call"]
    assert changed["Smile (%)"].iloc[0] == pytest.approx(smile["Volatility (%)"].iloc[0])
    higher = pres.with_vols(market, edited, smile)
    assert float(higher.vol.vol(100.0, T)) == pytest.approx(0.30)
    assert float(higher.vol.vol(110.0, T)) == float(market.vol.vol(110.0, T))
    assert float(strategy.value(higher)) != float(strategy.value(market))
    for bad in (-5.0, float("nan")):
        edited.loc[0, "Volatility (%)"] = bad
        with pytest.raises(ValueError, match="positive"):
            pres.with_vols(market, edited, smile)


def test_decomposition_styler_colours_signs_and_total(market):
    table = pres.decomposition_table(st.bull_call_spread(100.0, 110.0, T), market)
    formats = {c: "{:+.4f}" for c in table.columns if c != "Leg"}
    html = pres.style_decomposition(table, formats).to_html()
    assert pres.POSITIVE_COLOR in html
    assert pres.NEGATIVE_COLOR in html
    assert "border-top: 2px solid" in html  # the total row is highlighted


def test_custom_strategy_reads_positions_and_cash_quantities(market):
    legs = pd.DataFrame(
        {
            "Position": ["Long", "Short", "Long"],
            "Instrument": ["Call", "Put", "Digital call"],
            "Strike (% of spot)": [100.0, 90.0, 110.0],
            "Quantity": [1.0, 2.0, 5.0],
        }
    )
    strategy = pres.custom_strategy(legs, T, market.spot)
    assert [leg.quantity for leg in strategy.legs] == [1.0, -2.0, 5.0]  # digital: 5 % of 100


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"Quantity": [0.0]}, "positive"),
        ({"Strike (% of spot)": [np.nan]}, "strike"),
        ({"Instrument": ["Swap"]}, "instrument"),
    ],
)
def test_custom_strategy_rejects_incomplete_legs(market, change, message):
    legs = pd.DataFrame(
        {
            "Position": ["Long"],
            "Instrument": ["Put"],
            "Strike (% of spot)": [95.0],
            "Quantity": [1.0],
        }
    )
    for column, values in change.items():
        legs[column] = values
    with pytest.raises(ValueError, match=message):
        pres.custom_strategy(legs, T, market.spot)


def test_decomposition_table(market):
    straddle = pres.decomposition_table(st.straddle(100.0, T), market)
    assert list(straddle["Leg"]) == ["Long call", "Long put", "Total"]
    assert straddle["Price (%)"].iloc[-1] == pytest.approx(straddle["Price (%)"].iloc[:-1].sum())
    assert "of which skew (%)" not in straddle.columns
    assert "Pays (%)" not in straddle.columns
    single = pres.decomposition_table(st.call(100.0, T), market)
    assert list(single["Leg"]) == ["Long call"]  # no total row for a single leg
    digital = pres.decomposition_table(st.digital_call(100.0, T, 10.0), market)
    assert digital["Pays (%)"].iloc[0] == pytest.approx(10.0)
    assert digital["of which skew (%)"].iloc[0] > 0.0


def test_breakevens_of_a_straddle(market):
    strategy = st.straddle(100.0, T)
    premium = 100.0 * float(strategy.value(market)) * np.exp(market.rate * T) / market.spot
    x = np.linspace(50.0, 150.0, 1001)
    pnl = 100.0 * strategy.payoff(x) / market.spot - premium
    assert pres.breakevens(x, pnl) == pytest.approx([100.0 - premium, 100.0 + premium])


def test_build_market_from_the_default_inputs():
    market = state.build_market(100.0, 3.0, 2.0, state.default_grid())
    assert market.vol.check_arbitrage().is_arbitrage_free  # type: ignore[attr-defined]
    shuffled = state.default_grid().iloc[::-1]
    assert float(state.build_market(100.0, 3.0, 2.0, shuffled).vol.vol(90.0, 1.0)) == float(
        market.vol.vol(90.0, 1.0)
    )


@pytest.mark.parametrize(
    ("row", "column", "message"),
    [(0, "100%", "empty cells"), (0, state.MATURITY_COLUMN, "maturity")],
)
def test_build_market_reports_incomplete_grids(row, column, message):
    grid = state.default_grid()
    grid.loc[row, column] = np.nan
    with pytest.raises(ValueError, match=message):
        state.build_market(100.0, 3.0, 2.0, grid)


def test_regrid_keeps_known_strikes_and_reads_new_ones_on_the_smile():
    grid = state.default_grid()
    market = state.build_market(100.0, 3.0, 2.0, grid)
    new = state.regrid(grid, [85.0, 100.0], market)
    assert state.grid_strikes(new) == [85.0, 100.0]
    np.testing.assert_array_equal(new["100%"], grid["100%"])
    expected = np.round(100 * market.vol.vol(85.0, grid[state.MATURITY_COLUMN].to_numpy()), 1)
    np.testing.assert_array_equal(new["85%"], expected)
    assert new["85%"].isna().sum() == 0
    assert state.regrid(grid, [85.0], None)["85%"].isna().all()


def test_signature_changes_with_any_input():
    grid = state.default_grid()
    assert pres.signature("Call", 1.0, grid) == pres.signature("Call", 1.0, grid.copy())
    assert pres.signature("Call", 1.0, grid) != pres.signature("Call", 2.0, grid)
