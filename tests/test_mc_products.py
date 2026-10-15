"""Notes and bricks priced on Monte Carlo paths, against their closed forms (in standard errors)."""

import math

import numpy as np
import pytest

from spricer.bricks import BarrierOption, DigitalOption, OneTouch, ZeroCoupon
from spricer.market import FlatVol, Market
from spricer.mc import brick_payoff, price_note, simulate
from spricer.monitoring import Monitoring, observation_dates
from spricer.products.capital_protection import NOTE_CATALOG, find_note
from spricer.products.notes import Note, NoteLeg

N_SE = 4.0
SPREAD = 0.01
FLAT = Market(100.0, 0.03, 0.02, FlatVol(0.22))


def default_note(name, maturity=2.0, monitoring=Monitoring.EUROPEAN, observations=None):
    spec = find_note(name)
    values = {p.key: p.default for p in spec.params}
    return spec.build(values, maturity, 100.0, monitoring, observations)


@pytest.mark.parametrize("name", [spec.name for spec in NOTE_CATALOG])
def test_european_notes_reprice_their_closed_form_under_the_smile(market, name):
    note = default_note(name)
    mc = price_note(note, market, SPREAD, n_paths=60_000, seed=1)
    assert abs(mc.total.value - float(note.value(market, SPREAD))) < N_SE * mc.total.std_error
    # the legs are priced on the same paths: they add up exactly, and each one is right
    assert sum(leg.value for leg in mc.legs) == pytest.approx(mc.total.value, abs=1e-9)
    for est, exact in zip(mc.legs, note.leg_values(market, SPREAD), strict=True):
        assert abs(est.value - exact) <= N_SE * est.std_error + 1e-12


@pytest.mark.parametrize(
    ("monitoring", "observations"), [(Monitoring.CONTINUOUS, None), (Monitoring.DISCRETE, 252)]
)
@pytest.mark.parametrize("name", ["Shark note (up)", "Shark note (down)"])
def test_path_dependent_sharks_match_their_closed_form_under_a_flat_volatility(
    name, monitoring, observations
):
    note = default_note(name, 1.0, monitoring, observations)
    mc = price_note(note, FLAT, SPREAD, n_paths=100_000, seed=2)
    assert abs(mc.total.value - float(note.value(FLAT, SPREAD))) < N_SE * mc.total.std_error


@pytest.mark.slow
def test_one_volatility_cannot_price_coupons_and_protection_together(market):
    """Annual 5 % digital coupons above 100 % for 5 years, then 100 minus a European PDI(100, 60).

    Diffusing at the volatility of the barrier (one volatility for the whole product) misprices
    it by about 5 points; the local volatility reprices the smile within its standard error.
    """
    years = [1.0, 2.0, 3.0, 4.0, 5.0]
    legs = [NoteLeg(f"Coupon {t:g}y", DigitalOption("call", 100.0, t), 5.0) for t in years]
    legs += [
        NoteLeg("Redemption", ZeroCoupon(5.0), 100.0),
        NoteLeg("Protection", BarrierOption("put", "down-and-in", 100.0, 60.0, 5.0), -1.0),
    ]
    strip = Note("Coupon strip", 5.0, 100.0, tuple(legs), formula=lambda p: p)
    exact = float(strip.value(market))
    local = price_note(strip, market, n_paths=200_000, seed=6)
    assert abs(local.total.value - exact) < N_SE * local.total.std_error
    barrier_vol = float(market.vol.vol(60.0, 5.0))
    one_vol = Market(100.0, market.rate, market.dividend_yield, FlatVol(barrier_vol))
    flat = price_note(strip, one_vol, n_paths=200_000, seed=6)
    assert exact - flat.total.value > 4.0  # about 5 points of notional
    flat_exact = float(strip.value(one_vol))
    assert abs(flat.total.value - flat_exact) < N_SE * flat.total.std_error


def test_bricks_read_the_barrier_the_way_they_are_monitored():
    paths = simulate(FLAT, [1.0], n_paths=1_000, monitoring=Monitoring.CONTINUOUS)
    european = BarrierOption("put", "down-and-in", 100.0, 80.0, 1.0)
    s = paths.spot_at(1.0)
    np.testing.assert_array_equal(brick_payoff(european, paths), european.payoff(s))
    touch = OneTouch("down", 80.0, 1.0, Monitoring.CONTINUOUS)
    low, _ = paths.extremes(1.0)
    np.testing.assert_array_equal(brick_payoff(touch, paths), (low < 80.0).astype(float))
    knock_in = BarrierOption("put", "down-and-in", 100.0, 80.0, 1.0, Monitoring.CONTINUOUS)
    expected = np.where(low < 80.0, np.maximum(100.0 - s, 0.0), 0.0)
    np.testing.assert_array_equal(brick_payoff(knock_in, paths), expected)


def test_bricks_refuse_paths_that_observe_their_barrier_differently():
    european_paths = simulate(FLAT, [1.0], n_paths=1_000)
    continuous = BarrierOption("call", "up-and-out", 100.0, 120.0, 1.0, Monitoring.CONTINUOUS)
    with pytest.raises(ValueError, match="same monitoring"):
        brick_payoff(continuous, european_paths)
    weekly = simulate(
        FLAT,
        [1.0],
        n_paths=1_000,
        monitoring=Monitoring.DISCRETE,
        monitoring_times=observation_dates(1.0, 52),
    )
    daily = BarrierOption("call", "up-and-out", 100.0, 120.0, 1.0, Monitoring.DISCRETE, 252)
    with pytest.raises(ValueError, match="brick's dates"):
        brick_payoff(daily, weekly)


def test_a_note_needs_one_set_of_observation_dates():
    legs = (
        NoteLeg("a", OneTouch("up", 120.0, 1.0, Monitoring.DISCRETE, 12), 1.0),
        NoteLeg("b", OneTouch("up", 130.0, 1.0, Monitoring.DISCRETE, 52), 1.0),
    )
    note = Note("Mixed", 1.0, 100.0, legs, formula=lambda p: p, monitoring=Monitoring.DISCRETE)
    with pytest.raises(ValueError, match="share their observation dates"):
        price_note(note, FLAT, n_paths=1_000)


def test_the_issuer_spread_discounts_every_leg():
    note = default_note("Participation note")
    plain = price_note(note, FLAT, n_paths=2_000, seed=3)
    spread = price_note(note, FLAT, 0.02, n_paths=2_000, seed=3)
    assert spread.total.value == pytest.approx(plain.total.value * math.exp(-0.02 * 2.0))
