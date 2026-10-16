"""Monte Carlo Greeks: against closed forms, the gain of common random numbers, worst-of."""

import math

import numpy as np
import pytest

from spricer.conventions import DAYS_PER_YEAR
from spricer.market import Basket, FlatVol, Market
from spricer.mc.greeks import DELTA_BUMP, GAMMA_BUMP, coupon_note_greeks
from spricer.monitoring import Monitoring
from spricer.products.coupon_notes import (
    CouponNote,
    Observation,
    barrier_reverse_convertible,
    phoenix,
)

N_SE = 4.0
FLAT = Market(100.0, 0.03, 0.02, FlatVol(0.22))


def within(est, exact, n_se=N_SE):
    return abs(est.value - exact) <= n_se * est.std_error + 1e-9


def closed_form_greeks(note: CouponNote):
    """Same bumps as the Monte Carlo, applied to the exact bricks of the note."""
    rep = note.replication(100.0)
    assert rep is not None

    def v(market: Market, n: CouponNote = note) -> float:
        r = n.replication(100.0)
        assert r is not None
        return float(r.value(market))

    def spot(h: float) -> float:
        return v(FLAT.with_spot(100.0 * (1.0 + h)))

    hd, hg = DELTA_BUMP, GAMMA_BUMP
    return {
        "delta": (spot(hd) - spot(-hd)) / (2 * hd),
        "gamma": 0.01 * (spot(hg) - 2 * v(FLAT) + spot(-hg)) / hg**2,
        "exact_delta": (spot(1e-4) - spot(-1e-4)) / 2e-4,
        "exact_gamma": 0.01 * (spot(1e-3) - 2 * v(FLAT) + spot(-1e-3)) / 1e-6,
        "vega": (v(FLAT.with_vol_shift(0.01)) - v(FLAT.with_vol_shift(-0.01))) / 2,
        "rho": (v(FLAT.with_rate(0.0301)) - v(FLAT.with_rate(0.0299))) / 2,
        "theta": v(FLAT, note.aged(1.0 / DAYS_PER_YEAR)) - v(FLAT),
    }


@pytest.mark.parametrize("monitoring", [Monitoring.EUROPEAN, Monitoring.CONTINUOUS])
def test_monte_carlo_greeks_match_the_closed_form(monitoring):
    note = barrier_reverse_convertible(
        maturity=1.0, frequency=4, coupon_pa=6.0, barrier=70.0, monitoring=monitoring
    )
    mc = coupon_note_greeks(note, Basket.single(FLAT), n_paths=100_000, seed=1)
    exact = closed_form_greeks(note)
    asset = mc.assets[0]
    assert within(asset.delta, exact["delta"])
    assert within(asset.gamma, exact["gamma"])
    assert within(asset.vega, exact["vega"])
    assert within(mc.rho, exact["rho"])
    assert mc.theta is not None
    assert within(mc.theta, exact["theta"])
    # the bump sizes keep the bias small against the true derivatives
    assert exact["delta"] == pytest.approx(exact["exact_delta"], rel=0.015)
    assert exact["gamma"] == pytest.approx(exact["exact_gamma"], rel=0.02, abs=0.02)


def test_common_random_numbers_remove_most_of_the_noise(market):
    note = phoenix(maturity=2.0, frequency=4, coupon_pa=8.0)
    basket = Basket.single(market)
    crn = coupon_note_greeks(note, basket, n_paths=20_000, seed=3)
    independent = coupon_note_greeks(
        note, basket, n_paths=20_000, seed=3, common_random_numbers=False
    )
    a, b = crn.assets[0], independent.assets[0]
    assert a.delta.std_error < b.delta.std_error
    assert a.gamma.std_error < b.gamma.std_error
    assert a.vega.std_error < 0.5 * b.vega.std_error
    assert crn.rho.std_error < 0.1 * independent.rho.std_error
    assert crn.theta is not None
    assert independent.theta is not None
    assert crn.theta.std_error < independent.theta.std_error


def test_identical_underlyings_share_the_worst_of_risk():
    basket = Basket.of([FLAT, FLAT], [100.0, 100.0], ["A", "B"], [[1.0, 0.5], [0.5, 1.0]])
    note = phoenix(maturity=1.0, frequency=4, coupon_pa=8.0)
    mc = coupon_note_greeks(note, basket, n_paths=40_000, seed=4)
    a, b = mc.assets
    for x, y in ((a.delta, b.delta), (a.vega, b.vega)):
        assert abs(x.value - y.value) < N_SE * math.hypot(x.std_error, y.std_error)
    assert a.delta.value > 0.0
    assert a.vega.value < 0.0
    assert (a.name, b.name) == ("A", "B")


def test_theta_is_skipped_when_an_observation_falls_within_a_day():
    note = CouponNote(
        "x",
        (Observation(0.001, 1.0, 0.0), Observation(1.0, 1.0, 0.0)),
        protection_barrier=60.0,
    )
    mc = coupon_note_greeks(note, Basket.single(FLAT), n_paths=2_000)
    assert mc.theta is None
    with pytest.raises(ValueError, match="falls within"):
        note.aged(0.01)
    later = note.aged(0.0005)
    np.testing.assert_allclose(later.times, [0.0005, 0.9995])
