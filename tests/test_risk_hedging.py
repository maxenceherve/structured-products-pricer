"""P&L explain and hedging: Taylor order, exact steps, static hedges, delta-hedging simulation."""

import itertools
import math

import numpy as np
import pytest

from spricer import strategies as strat
from spricer.bricks import EuropeanOption
from spricer.conventions import Greeks
from spricer.hedging import derman_kamal_std, simulate_delta_hedge, static_hedge
from spricer.market import FlatVol, Market
from spricer.products.capital_protection import find_note
from spricer.risk import Shock, explain, greeks, residual_order, scenario_matrix
from web.options import display_greeks

S, R, Q, VOL = 100.0, 0.03, 0.01, 0.20
FLAT = Market(S, R, Q, FlatVol(VOL))
N_SE = 4.0


def call(t: float):
    return strat.call(100.0, t)


def call_value(m: Market, t: float) -> float:
    return 100.0 * float(call(t).value(m)) / S


# --- P&L explain --------------------------------------------------------------------------------


def test_risk_greeks_are_the_desk_greeks():
    g = greeks(call_value, FLAT, 1.0)
    shown = display_greeks(call(1.0), FLAT)
    assert g["Delta"] == pytest.approx(shown["Delta (%)"] / 100.0, rel=1e-3)
    assert g["Gamma"] == pytest.approx(shown["Gamma (%)"] / 100.0, rel=1e-3)
    assert g["Vega"] == pytest.approx(shown["Vega (%)"], rel=1e-3)
    assert g["Theta"] == pytest.approx(shown["Theta (%)"], rel=1e-2)
    assert g["Rho"] == pytest.approx(shown["Rho (%)"], rel=1e-3)


def test_taylor_residual_is_third_order():
    # halving the move divides the residual by about 8 (third order; a second-order error
    # would give 4); the finite-difference Greeks add a small fourth-order part
    residuals = residual_order(call_value, FLAT, 1.0, [8.0, 4.0, 2.0])
    for big, small in itertools.pairwise(residuals):
        assert 6.0 < abs(big / small) < 14.0


def test_steps_add_up_and_the_residual_is_small():
    shock = Shock(spot_pct=-10.0, vol_pts=3.0, days=30.0, rate_bp=25.0)
    a = explain(call_value, FLAT, 1.0, shock)
    assert sum(a.steps.values()) == pytest.approx(a.full, abs=1e-12)
    assert a.end == pytest.approx(call_value(shock.apply(FLAT), 1.0 - 30.0 / 365.0))
    assert abs(a.residual) < 0.05 * abs(a.full)
    assert a.first_and_second_order == pytest.approx(a.taylor, abs=1e-5)  # ATM: no vanna
    assert set(a.terms) == {"Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Volga"}


def test_carry_is_counted_once():
    """A capital protected note on a still market: the P&L is the theta, not twice it."""
    spec = find_note("Participation note")

    def note_value(m: Market, t: float) -> float:
        return float(spec.build({"participation": 60.0}, t, 100.0).value(m))

    a = explain(note_value, FLAT, 2.0, Shock(days=91.0))
    assert a.full == pytest.approx(a.steps["Time"])
    assert a.terms["Theta"] == pytest.approx(a.full, rel=0.05)
    assert a.taylor == pytest.approx(a.full, rel=0.05)


def test_scenario_matrix_and_inputs():
    grid = scenario_matrix(call_value, FLAT, 1.0, [-10.0, 0.0, 10.0], [-5.0, 0.0, 5.0])
    assert grid.shape == (3, 3)
    assert grid[1, 1] == pytest.approx(0.0, abs=1e-12)
    assert grid[1, 2] > 0.0 > grid[1, 0]
    assert grid[2, 1] > 0.0 > grid[0, 1]
    with pytest.raises(ValueError, match="horizon"):
        scenario_matrix(call_value, FLAT, 0.1, [0.0], [0.0], days=40.0)
    with pytest.raises(ValueError, match="before maturity"):
        explain(call_value, FLAT, 0.1, Shock(days=40.0))


# --- Static hedge ------------------------------------------------------------------------------


def test_static_hedges_neutralise_their_targets():
    position = strat.straddle(100.0, 1.0).short().greeks(FLAT)
    short_dated = EuropeanOption("call", 100.0, 0.25).greeks(FLAT)
    long_dated = EuropeanOption("put", 90.0, 2.0).greeks(FLAT)

    delta = static_hedge(position)
    assert float(delta.hedged.delta) == pytest.approx(0.0, abs=1e-12)
    assert len(delta.legs) == 1

    gamma = static_hedge(position, (("calls", short_dated),))
    assert float(gamma.hedged.delta) == pytest.approx(0.0, abs=1e-12)
    assert float(gamma.hedged.gamma) == pytest.approx(0.0, abs=1e-12)
    assert gamma.legs[0].quantity > 0.0  # short gamma: buy options

    both = static_hedge(position, (("calls", short_dated), ("puts", long_dated)))
    hedged = both.hedged
    for x in (hedged.delta, hedged.gamma, hedged.vega):
        assert float(x) == pytest.approx(0.0, abs=1e-10)


def test_static_hedge_refuses_impossible_hedges():
    position = strat.call(100.0, 1.0).greeks(FLAT)
    no_gamma = Greeks(*(np.asarray(x) for x in (1.0, 0.0, 0.0, 0.0, 0.0)))
    option = EuropeanOption("call", 100.0, 0.5).greeks(FLAT)
    with pytest.raises(ValueError, match="no gamma"):
        static_hedge(position, (("forward", no_gamma),))
    with pytest.raises(ValueError, match="proportional"):
        static_hedge(position, (("a", option), ("b", option * 2.0)))
    with pytest.raises(ValueError, match="at most two"):
        static_hedge(position, (("a", option), ("b", option), ("c", option)))


# --- Delta-hedging simulation ------------------------------------------------------------------


def bs_gap(realised: float, t: float = 1.0) -> float:
    other = Market(S, R, Q, FlatVol(realised))
    return 100.0 * (float(call(t).value(other)) - float(call(t).value(FLAT))) / S


@pytest.mark.parametrize("realised", [0.15, 0.20, 0.30])
def test_mean_hedged_pnl_is_the_price_gap_between_volatilities(realised):
    sim = simulate_delta_hedge(
        call, FLAT, 1.0, realised_vol=realised, rebalancings=52, n_paths=20_000
    )
    assert abs(sim.mean.value - bs_gap(realised)) < N_SE * sim.mean.std_error
    short = simulate_delta_hedge(
        call, FLAT, 1.0, realised_vol=realised, rebalancings=52, quantity=-1.0, n_paths=20_000
    )
    assert short.mean.value == pytest.approx(-sim.mean.value, abs=1e-10)


def test_hedging_noise_follows_derman_kamal():
    vega_pct = 100.0 * float(call(1.0).greeks(FLAT).vega) / S
    stds = {}
    for n in (252, 52):
        sim = simulate_delta_hedge(
            call, FLAT, 1.0, realised_vol=VOL, rebalancings=n, n_paths=20_000
        )
        assert sim.std == pytest.approx(derman_kamal_std(vega_pct, VOL, n), rel=0.15)
        stds[n] = sim.std
    assert stds[52] / stds[252] == pytest.approx(math.sqrt(252 / 52), rel=0.15)


def test_costs_lower_the_pnl_and_the_sample_path_adds_up():
    free = simulate_delta_hedge(call, FLAT, 1.0, realised_vol=VOL, rebalancings=52, n_paths=4_000)
    costly = simulate_delta_hedge(
        call, FLAT, 1.0, realised_vol=VOL, rebalancings=52, cost_bp=10.0, n_paths=4_000
    )
    assert costly.mean.value < free.mean.value - N_SE * free.mean.std_error
    total = float(np.sum(costly.total_pnl))
    assert total == pytest.approx(costly.pnl[0] * math.exp(R * 1.0), rel=1e-10)
    np.testing.assert_allclose(costly.gamma_pnl + costly.theta_pnl, costly.total_pnl)
    assert costly.spot.shape == (53,)
    assert costly.delta.shape == (52,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rebalancings": 0}, "rebalancing"),
        ({"realised_vol": 0.0}, "realised volatility"),
        ({"n_paths": 3}, "even"),
    ],
)
def test_invalid_simulations_raise(kwargs, message):
    args = {"realised_vol": VOL, "rebalancings": 4, "n_paths": 100} | kwargs
    with pytest.raises(ValueError, match=message):
        simulate_delta_hedge(call, FLAT, 1.0, **args)
