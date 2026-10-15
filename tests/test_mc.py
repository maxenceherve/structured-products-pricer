"""Monte Carlo engine: correlation, local volatility, paths, barriers, standard errors.

Every comparison with a closed form is made in standard errors of the Monte Carlo estimate.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from spricer.bricks import BarrierOption, DigitalOption, EuropeanOption, OneTouch
from spricer.market import FlatVol, InterpolatedVolSurface, Market, ShiftedVol
from spricer.mc import (
    MCEstimate,
    check_correlation,
    local_vol,
    nearest_correlation,
    price_brick,
    simulate,
)
from spricer.mc.local_vol import MAX_LOCAL_VOL, flat_sigma, local_variance_at_moneyness
from spricer.mc.paths import estimate
from spricer.monitoring import Monitoring, observation_dates

S, R, Q, VOL = 100.0, 0.03, 0.02, 0.22
FLAT = Market(S, R, Q, FlatVol(VOL))
N_SE = 4.0  # tolerance in standard errors


def within(est: MCEstimate, exact: float, n_se: float = N_SE) -> bool:
    return abs(est.value - exact) <= n_se * est.std_error


def surface(maturities, strikes_pct, vols) -> InterpolatedVolSurface:
    return InterpolatedVolSurface(
        spot_ref=S,
        rate=R,
        dividend_yield=Q,
        maturities=maturities,
        strikes_pct=strikes_pct,
        vols=vols,
    )


# --- Correlation ---------------------------------------------------------------------------------


def test_a_valid_correlation_matrix_passes():
    c = [[1.0, 0.5], [0.5, 1.0]]
    np.testing.assert_array_equal(check_correlation(c, 2), c)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[1.0, 0.5], [0.5, 1.0]], "shape"),  # checked against 3 assets below
        ([[1.0, 1.5], [1.5, 1.0]], "between -1 and 1"),
        ([[1.0, np.nan], [np.nan, 1.0]], "finite"),
        ([[1.0, 0.5], [0.4, 1.0]], "symmetric"),
        ([[0.9, 0.5], [0.5, 1.0]], "diagonal"),
        ([[1.0, 0.9, 0.1], [0.9, 1.0, 0.8], [0.1, 0.8, 1.0]], "smallest eigenvalue -0.1556"),
    ],
)
def test_invalid_correlation_matrices_are_refused(matrix, message):
    n = 3 if message == "shape" else len(matrix)
    with pytest.raises(ValueError, match=message):
        check_correlation(matrix, n)


def test_nearest_correlation_matches_higham_example():
    # Higham (2002), section 4: the nearest correlation matrix to this one
    a = [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    expected = [[1.0, 0.7607, 0.1573], [0.7607, 1.0, 0.7607], [0.1573, 0.7607, 1.0]]
    c = nearest_correlation(a)
    np.testing.assert_allclose(c, expected, atol=1e-4)
    np.testing.assert_array_equal(np.diag(c), 1.0)
    check_correlation(c, 3)  # strictly positive definite


def test_nearest_correlation_keeps_a_valid_matrix_and_fixes_a_typed_one():
    valid = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.4], [-0.2, 0.4, 1.0]])
    np.testing.assert_allclose(nearest_correlation(valid), valid, atol=1e-9)
    typed = [[1.0, 0.9, 0.1], [0.9, 1.0, 0.8], [0.1, 0.8, 1.0]]
    fixed = nearest_correlation(typed)
    check_correlation(fixed, 3)
    assert np.max(np.abs(fixed - typed)) < 0.15
    with pytest.raises(ValueError, match="square"):
        nearest_correlation([[1.0, 0.5]])
    # stopped early, the result is still a valid correlation matrix
    check_correlation(nearest_correlation(typed, max_iterations=1), 3)


# --- Local volatility ----------------------------------------------------------------------------


def test_a_flat_surface_has_its_own_local_volatility():
    np.testing.assert_allclose(local_vol(FLAT, [50.0, 100.0, 180.0], 1.3), VOL, rtol=1e-6)
    assert flat_sigma(FLAT.vol) == VOL
    assert flat_sigma(ShiftedVol(FLAT.vol, 0.01)) == pytest.approx(VOL + 0.01)
    assert flat_sigma(ShiftedVol(surface([1.0], [90.0, 110.0], [[0.2, 0.2]]), 0.01)) is None


def test_without_smile_the_local_variance_is_the_forward_variance():
    s = surface([1.0, 2.0], [80.0, 100.0, 120.0], [[0.20] * 3, [0.25] * 3])
    m = Market(S, R, Q, s)
    forward_variance = 0.25**2 * 2.0 - 0.20**2 * 1.0  # d(sigma^2 T)/dT between the maturities
    np.testing.assert_allclose(
        local_vol(m, [90.0, 100.0, 115.0], 1.5), math.sqrt(forward_variance), rtol=1e-5
    )
    np.testing.assert_allclose(local_vol(m, 100.0, 0.5), 0.20, rtol=1e-5)  # before the grid
    np.testing.assert_allclose(local_vol(m, 100.0, 3.0), 0.25, rtol=1e-5)  # after the grid


def test_skew_raises_the_local_volatility_on_the_downside(market):
    lv = local_vol(market, [70.0, 100.0, 130.0], 1.0)
    assert lv[0] > lv[1]
    assert np.all((lv > 0.0) & (lv <= MAX_LOCAL_VOL))
    table = local_variance_at_moneyness(market, np.linspace(-0.5, 0.3, 5), np.array([[0.5], [2.0]]))
    assert table.shape == (2, 5)


def test_a_surface_with_arbitrage_has_no_local_volatility():
    calendar = Market(S, R, Q, surface([1.0, 2.0], [90.0, 110.0], [[0.30, 0.30], [0.20, 0.20]]))
    with pytest.raises(ValueError, match="calendar arbitrage"):
        local_vol(calendar, 100.0, 1.5)
    kink = Market(S, R, Q, surface([1.0], [90.0, 100.0, 110.0], [[0.20, 0.60, 0.20]]))
    with pytest.raises(ValueError, match="butterfly arbitrage"):
        local_vol(kink, np.linspace(80.0, 120.0, 41), 1.0)
    with pytest.raises(ValueError, match="positive time"):
        local_vol(FLAT, 100.0, 0.0)


# --- Estimates -----------------------------------------------------------------------------------


def test_estimate_pairs_antithetic_samples_and_combines_richardson_rows():
    e = estimate([1.0, 3.0, 3.0, 1.0], antithetic=True)
    assert (e.value, e.std_error, e.n_paths) == (2.0, 0.0, 4)
    r = estimate([2.0, 2.0, 1.0, 1.0], antithetic=False, richardson=True)
    assert (r.value, r.n_paths) == (3.0, 2)
    plain = MCEstimate(1.0, 0.1, 10)
    assert plain.ci95 == pytest.approx((1.0 - 0.196, 1.0 + 0.196), abs=1e-4)
    assert str(plain) == "1.0000 ± 0.1000"


# --- Paths under a flat volatility: exact steps ------------------------------------------------


def test_flat_volatility_paths_reprice_black_scholes_and_stay_martingales():
    paths = simulate(FLAT, [0.5, 2.0], n_paths=100_000, seed=1)
    assert paths.spots.shape == (100_000, 2, 1)
    assert not paths.richardson
    for t in (0.5, 2.0):
        s = paths.spot_at(t)
        assert within(paths.estimate(s * math.exp(-(R - Q) * t)), S)
        for brick in (
            EuropeanOption("call", 110.0, t),
            EuropeanOption("put", 85.0, t),
            DigitalOption("call", 100.0, t),
        ):
            est = paths.estimate(brick.payoff(s) * paths.discount(t))
            assert within(est, float(brick.value(FLAT))), brick.label


def test_same_seed_same_paths_and_antithetic_variates_reduce_the_error():
    a = simulate(FLAT, [1.0], n_paths=20_000, seed=5)
    b = simulate(FLAT, [1.0], n_paths=20_000, seed=5)
    c = simulate(FLAT, [1.0], n_paths=20_000, seed=6)
    np.testing.assert_array_equal(a.spots, b.spots)
    assert not np.array_equal(a.spots, c.spots)
    call = EuropeanOption("call", 100.0, 1.0)
    plain = simulate(FLAT, [1.0], n_paths=20_000, seed=5, antithetic=False)
    anti_se = a.estimate(call.payoff(a.spot_at(1.0))).std_error
    plain_se = plain.estimate(call.payoff(plain.spot_at(1.0))).std_error
    assert anti_se < 0.9 * plain_se  # same number of paths, about 22 % less error here


@pytest.mark.parametrize(
    "brick",
    [
        BarrierOption("put", "down-and-in", 100.0, 75.0, 2.0, Monitoring.CONTINUOUS),
        BarrierOption("call", "up-and-out", 100.0, 130.0, 1.0, Monitoring.CONTINUOUS),
        OneTouch("up", 120.0, 1.0, Monitoring.CONTINUOUS),
        OneTouch("down", 80.0, 1.5, Monitoring.CONTINUOUS),
    ],
    ids=lambda b: b.label,
)
def test_brownian_bridge_reprices_continuous_barriers(brick):
    # one exact step to maturity: every crossing comes from the bridge
    est = price_brick(brick, FLAT, n_paths=200_000, seed=2)
    assert within(est, float(brick.value(FLAT)))


def test_discrete_barrier_observed_daily_matches_bgk():
    brick = BarrierOption("call", "up-and-out", 100.0, 130.0, 1.0, Monitoring.DISCRETE, 252)
    est = price_brick(brick, FLAT, n_paths=100_000, seed=3)
    continuous = BarrierOption("call", "up-and-out", 100.0, 130.0, 1.0, Monitoring.CONTINUOUS)
    assert within(est, float(brick.value(FLAT)))
    assert est.value - float(continuous.value(FLAT)) > N_SE * est.std_error


def test_extremes_restart_at_each_recorded_date():
    times = [0.5, 1.0]
    paths = simulate(
        FLAT, times, n_paths=2_000, monitoring=Monitoring.DISCRETE, monitoring_times=times
    )
    assert paths.minimum is not None
    np.testing.assert_array_equal(paths.minimum[:, :, 0], paths.spots[:, :, 0])
    low, high = paths.extremes(1.0)
    np.testing.assert_array_equal(low, paths.spots[:, :, 0].min(axis=1))
    np.testing.assert_array_equal(high, paths.spots[:, :, 0].max(axis=1))
    sparse = simulate(
        FLAT, [1.0, 2.0], n_paths=2_000, monitoring=Monitoring.DISCRETE, monitoring_times=[2.0]
    )
    assert np.all(sparse.extremes(1.0)[0] == np.inf)  # nothing observed before t = 1


# --- Local volatility paths ----------------------------------------------------------------------


@pytest.mark.slow
def test_local_volatility_reprices_the_smile(market):
    """Vanillas and digitals of the implied surface, across strikes and maturities."""
    times = [0.5, 2.0, 5.0]
    paths = simulate(market, times, n_paths=200_000, seed=11)
    assert paths.richardson
    assert paths.n_paths == 200_000
    for t in times:
        s = paths.spot_at(t)
        assert within(paths.estimate(s * math.exp(-(R - Q) * t)), S)
        for k in (70.0, 85.0, 100.0, 115.0, 130.0):
            for brick in (
                EuropeanOption("put" if k < 100.0 else "call", k, t),
                DigitalOption("call", k, t),
            ):
                est = paths.estimate(brick.payoff(s) * paths.discount(t))
                assert within(est, float(brick.value(market))), (t, brick.label)


@pytest.mark.slow
def test_richardson_extrapolation_removes_the_euler_bias(market):
    digital = DigitalOption("call", 100.0, 0.5)
    exact = float(digital.value(market))
    estimates = {}
    for rich in (False, True):
        paths = simulate(market, [0.5], n_paths=400_000, seed=4, richardson=rich)
        estimates[rich] = paths.estimate(digital.payoff(paths.spot_at(0.5)) * paths.discount(0.5))
    assert estimates[False].value - exact < -N_SE * estimates[False].std_error  # Euler bias
    assert within(estimates[True], exact)


def test_a_shifted_smile_is_simulated_on_its_own_local_volatility(market):
    """A vega bump shifts the whole surface: the paths follow the shifted smile."""
    bumped = market.with_vol_shift(0.01)
    call = EuropeanOption("call", 100.0, 1.0)
    paths = simulate(bumped, [1.0], n_paths=100_000, seed=9)
    est = paths.estimate(call.payoff(paths.spot_at(1.0)) * paths.discount(1.0))
    assert within(est, float(call.value(bumped)))
    assert est.value > float(call.value(market))


def test_daily_dates_are_stepped_directly(market):
    dates = observation_dates(0.25, 63)
    paths = simulate(
        market, [0.25], n_paths=2_000, monitoring=Monitoring.DISCRETE, monitoring_times=dates
    )
    assert not paths.richardson
    weekly = simulate(market, [0.25], n_paths=2_000)
    assert weekly.richardson
    assert weekly.spots.shape[0] == 4_000  # fine paths, then their coarse twins


# --- Several underlyings -------------------------------------------------------------------------


def margrabe(s1, s2, q1, q2, v1, v2, rho, t):
    v = math.sqrt(v1**2 + v2**2 - 2.0 * rho * v1 * v2)
    d1 = (math.log(s1 / s2) + (q2 - q1 + 0.5 * v**2) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    return s1 * math.exp(-q1 * t) * norm.cdf(d1) - s2 * math.exp(-q2 * t) * norm.cdf(d2)


def test_two_correlated_assets_reprice_margrabe_and_their_correlation():
    a = Market(100.0, R, 0.01, FlatVol(0.25))
    b = Market(95.0, R, 0.03, FlatVol(0.30))
    rho, t = 0.6, 2.0
    paths = simulate([a, b], [t], n_paths=200_000, seed=8, correlation=[[1, rho], [rho, 1]])
    s1, s2 = paths.spot_at(t, 0), paths.spot_at(t, 1)
    est = paths.estimate(np.maximum(s1 - s2, 0.0) * paths.discount(t))
    assert within(est, margrabe(100.0, 95.0, 0.01, 0.03, 0.25, 0.30, rho, t))
    log_returns = np.log(paths.spots[:, 0, :] / [100.0, 95.0])
    assert np.corrcoef(log_returns.T)[0, 1] == pytest.approx(rho, abs=0.01)


def test_a_flat_and_a_smiled_asset_run_together(market):
    other = Market(50.0, R, 0.0, FlatVol(0.3))
    paths = simulate([market, other], [1.0], n_paths=20_000, correlation=[[1, 0.3], [0.3, 1]])
    assert paths.spots.shape == (40_000, 1, 2)
    est = paths.estimate(paths.spot_at(1.0, 1) * paths.discount(1.0))
    assert within(est, 50.0)


# --- Input validation ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"markets": []}, "at least one market"),
        ({"markets": [FLAT, Market(S, 0.05, Q, FlatVol(VOL))]}, "same rate"),
        ({"markets": [FLAT, FLAT]}, "correlation matrix"),
        ({"n_paths": 1_001}, "even"),
        ({"max_step": 0.0}, "max_step"),
        ({"times": [0.0, 1.0]}, "times must be positive"),
        ({"monitoring": Monitoring.DISCRETE}, "needs monitoring_times"),
        ({"monitoring_times": [0.5]}, "only apply to discrete"),
    ],
)
def test_invalid_simulation_inputs_raise(kwargs, message):
    args = {"markets": FLAT, "times": [1.0], "n_paths": 1_000} | kwargs
    markets, times = args.pop("markets"), args.pop("times")
    with pytest.raises(ValueError, match=message):
        simulate(markets, times, **args)


def test_paths_refuse_dates_they_did_not_record():
    paths = simulate(FLAT, [1.0], n_paths=100)
    with pytest.raises(ValueError, match="not a recorded date"):
        paths.spot_at(0.5)
    with pytest.raises(ValueError, match="extremes"):
        paths.extremes(1.0)
    with pytest.raises(ValueError, match="at least one observation"):
        observation_dates(1.0, 0)
