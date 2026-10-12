import numpy as np
import pytest
from scipy.integrate import simpson

from spricer.analytic import black_scholes as bs
from spricer.market import FlatVol, InterpolatedVolSurface, ShiftedVol
from spricer.market.defaults import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_MATURITIES,
    DEFAULT_RATE,
    DEFAULT_SPOT,
    DEFAULT_STRIKES_PCT,
    DEFAULT_VOLS,
    default_vol_surface,
)

R, Q = DEFAULT_RATE, DEFAULT_DIVIDEND_YIELD


def make_surface(maturities, strikes_pct, vols, spot_ref=100.0):
    return InterpolatedVolSurface(
        spot_ref=spot_ref,
        rate=R,
        dividend_yield=Q,
        maturities=maturities,
        strikes_pct=strikes_pct,
        vols=vols,
    )


def forward(t, spot=DEFAULT_SPOT):
    return spot * np.exp((R - Q) * np.asarray(t))


def test_grid_nodes_are_reproduced(surface):
    for t, row in zip(DEFAULT_MATURITIES, DEFAULT_VOLS, strict=True):
        np.testing.assert_allclose(surface.vol(surface.strikes, t), row, atol=1e-14)


def test_default_grid_is_arbitrage_free(surface):
    report = surface.check_arbitrage()
    assert report.is_arbitrage_free, report


def test_flat_grid_gives_a_flat_surface():
    s = make_surface((0.5, 1.0, 2.0), (80, 100, 120), np.full((3, 3), 0.25))
    strikes, maturities = np.linspace(1, 400, 50), np.array([[0.1], [0.75], [3.0]])
    np.testing.assert_allclose(s.vol(strikes, maturities), 0.25, atol=1e-14)
    np.testing.assert_allclose(s.vol_slope(strikes, maturities), 0.0, atol=1e-14)


@pytest.mark.parametrize("t", [0.1, 0.25, 0.7, 1.0, 4.0, 8.0])
def test_vol_slope_matches_finite_differences(surface, t):
    k = np.linspace(20.0, 300.0, 400)  # inside the grid and in both wings
    h = 1e-4
    fd = (surface.vol(k + h, t) - surface.vol(k - h, t)) / (2 * h)
    np.testing.assert_allclose(surface.vol_slope(k, t), fd, atol=1e-8)


@pytest.mark.parametrize("t", [0.25, 1.0, 5.0])
@pytest.mark.parametrize("quantity", ["vol", "vol_slope", "density"])
def test_smile_has_no_jump(surface, t, quantity):
    """Halving the strike step halves the largest move between neighbours: no discontinuity."""
    f = getattr(surface, quantity)

    def largest_step(n):
        return np.max(np.abs(np.diff(f(np.linspace(20.0, 300.0, n), t))))

    assert largest_step(28001) / largest_step(56001) == pytest.approx(2.0, rel=0.05)


def test_strikes_are_absolute_whatever_the_reference_spot():
    """The same absolute strikes quoted against two reference spots give the same surface."""
    vols = np.array(DEFAULT_VOLS)
    a = make_surface(DEFAULT_MATURITIES, DEFAULT_STRIKES_PCT, vols, spot_ref=100.0)
    b = make_surface(
        DEFAULT_MATURITIES, np.array(DEFAULT_STRIKES_PCT) * 100.0 / 80.0, vols, spot_ref=80.0
    )
    k, t = np.linspace(30.0, 200.0, 50), np.array([[0.1], [0.4], [1.5], [7.0]])
    np.testing.assert_allclose(a.vol(k, t), b.vol(k, t), rtol=1e-12)


def test_maturity_interpolation_is_linear_in_total_variance_at_fixed_moneyness(surface):
    t1, t2, t = 1.0, 2.0, 1.4
    for k in (-0.3, 0.0, 0.2):
        w1 = float(surface.vol(forward(t1) * np.exp(k), t1)) ** 2 * t1
        w2 = float(surface.vol(forward(t2) * np.exp(k), t2)) ** 2 * t2
        w = float(surface.vol(forward(t) * np.exp(k), t)) ** 2 * t
        assert w == pytest.approx(w1 + (w2 - w1) * (t - t1) / (t2 - t1), rel=1e-12)


def test_volatility_is_flat_in_time_outside_the_grid_maturities(surface):
    for k in (-0.4, 0.0, 0.3):
        first = float(surface.vol(forward(0.25) * np.exp(k), 0.25))
        last = float(surface.vol(forward(5.0) * np.exp(k), 5.0))
        assert float(surface.vol(forward(0.05) * np.exp(k), 0.05)) == pytest.approx(
            first, rel=1e-12
        )
        assert float(surface.vol(forward(0.0) * np.exp(k), 0.0)) == pytest.approx(first, rel=1e-12)
        assert float(surface.vol(forward(9.0) * np.exp(k), 9.0)) == pytest.approx(last, rel=1e-12)


@pytest.mark.parametrize("t", DEFAULT_MATURITIES)
def test_wings_respect_lee_bounds(surface, t):
    for k in (-30.0, 30.0):
        w = float(surface.vol(forward(t) * np.exp(k), t)) ** 2 * t
        assert 0.0 < w / abs(k) <= 2.0


@pytest.mark.parametrize("t", [0.25, 0.6, 1.0, 3.5, 5.0, 8.0])
def test_density_integrates_to_one_and_recovers_the_forward(surface, t):
    x = np.linspace(np.log(forward(t)) - 40.0, np.log(forward(t)) + 15.0, 2_200_001)
    strikes = np.exp(x)
    mass_density = surface.density(strikes, t) * strikes  # density of log-strike
    mass = simpson(mass_density, x=x)
    mean = simpson(mass_density * strikes, x=x)
    assert mass == pytest.approx(1.0, abs=1e-8)
    assert mean == pytest.approx(float(forward(t)), rel=1e-8)


@pytest.mark.parametrize("t", [0.25, 1.0, 3.0])
def test_density_matches_breeden_litzenberger(surface, t):
    h = 1e-3

    def call(k):
        return bs.vanilla_price(
            "call",
            spot=DEFAULT_SPOT,
            strike=k,
            maturity=t,
            rate=R,
            dividend_yield=Q,
            vol=surface.vol(k, t),
        )

    for k in (60.0, 90.0, 100.0, 115.0, 180.0):
        bl = np.exp(R * t) * (call(k + h) - 2 * call(k) + call(k - h)) / h**2
        assert float(surface.density(k, t)) == pytest.approx(float(bl), rel=1e-4)


def test_butterfly_arbitrage_is_detected():
    vols = [[0.20, 0.20, 0.20, 0.32, 0.20, 0.20, 0.20]]  # a spike at 100 %
    s = make_surface((1.0,), (70, 80, 90, 100, 110, 120, 130), vols)
    report = s.check_arbitrage()
    assert report.butterfly
    assert not report.calendar
    assert any(
        v.strike_pct_from < 100.0 < v.strike_pct_to or 85 < v.strike_pct_from < 115
        for v in report.butterfly
    )


def test_calendar_arbitrage_is_detected():
    s = make_surface((0.5, 1.0), (80, 100, 120), [[0.30, 0.30, 0.30], [0.15, 0.15, 0.15]])
    report = s.check_arbitrage()
    assert report.calendar
    assert report.calendar[0].maturity_from == 0.5
    assert report.calendar[0].max_decrease == pytest.approx(0.30**2 * 0.5 - 0.15**2 * 1.0)


@pytest.mark.parametrize(
    ("maturities", "strikes", "vols", "message"),
    [
        ((1.0, 0.5), (90, 110), [[0.2, 0.2], [0.2, 0.2]], "maturities"),
        ((0.0, 1.0), (90, 110), [[0.2, 0.2], [0.2, 0.2]], "maturities"),
        ((1.0,), (110, 90), [[0.2, 0.2]], "strikes_pct"),
        ((1.0,), (100,), [[0.2]], "strikes_pct"),
        ((1.0,), (90, 110), [[0.2, 0.2, 0.2]], "shape"),
        ((1.0,), (90, 110), [[0.2, -0.1]], "vols"),
        ((1.0,), (90, 110), [[0.2, np.nan]], "vols"),
    ],
)
def test_invalid_grids_raise(maturities, strikes, vols, message):
    with pytest.raises(ValueError, match=message):
        make_surface(maturities, strikes, vols)


def test_negative_strike_or_maturity_raise(surface):
    with pytest.raises(ValueError, match="non-negative"):
        surface.vol(-1.0, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        surface.vol(100.0, -1.0)


def test_flat_and_shifted_vols():
    flat = FlatVol(0.2)
    assert flat.vol(np.ones((2, 3)), 1.0).shape == (2, 3)
    shifted = ShiftedVol(default_vol_surface(), 0.01)
    base = default_vol_surface()
    np.testing.assert_allclose(shifted.vol([90.0, 110.0], 1.0), base.vol([90.0, 110.0], 1.0) + 0.01)
    np.testing.assert_allclose(
        shifted.vol_slope([90.0, 110.0], 1.0), base.vol_slope([90.0, 110.0], 1.0)
    )
    with pytest.raises(ValueError, match="sigma"):
        FlatVol(-0.1)


def test_single_maturity_grid_is_flat_in_time_at_fixed_moneyness():
    s = make_surface((1.0,), (70, 85, 100, 115, 130), [[0.30, 0.24, 0.20, 0.18, 0.17]])
    assert s.check_arbitrage().is_arbitrage_free
    for k in (-0.3, 0.0, 0.25):
        ref = float(s.vol(forward(1.0) * np.exp(k), 1.0))
        np.testing.assert_allclose(
            s.vol(forward(np.array([0.1, 3.0])) * np.exp(k), [0.1, 3.0]), ref
        )


def test_density_needs_a_positive_maturity(surface):
    with pytest.raises(ValueError, match="positive maturity"):
        surface.density(100.0, 0.0)


def test_spot_ref_must_be_positive():
    with pytest.raises(ValueError, match="spot_ref"):
        make_surface((1.0,), (90, 110), [[0.2, 0.2]], spot_ref=0.0)


def test_flat_vol_has_no_slope():
    np.testing.assert_array_equal(FlatVol(0.2).vol_slope([90.0, 110.0], 1.0), [0.0, 0.0])


def test_overridden_vol_replaces_only_the_quoted_strikes(surface):
    from spricer.market import OverriddenVol

    quoted = OverriddenVol(surface, ((110.0, 0.30), (90.0, 0.10)))
    k = np.array([90.0, 100.0, 110.0])
    np.testing.assert_allclose(quoted.vol(k, 1.0), [0.10, float(surface.vol(100.0, 1.0)), 0.30])
    np.testing.assert_array_equal(quoted.vol_slope(k, 1.0), surface.vol_slope(k, 1.0))
    with pytest.raises(ValueError, match="overrides"):
        OverriddenVol(surface, ((100.0, -0.1),))
