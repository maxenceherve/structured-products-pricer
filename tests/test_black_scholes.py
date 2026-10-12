import numpy as np
import pytest

from spricer.analytic import black_scholes as bs

GREEKS = ("delta", "gamma", "vega", "theta", "rho")


def vanilla(kind, s, k, t, r, q, v):
    return bs.vanilla_price(kind, spot=s, strike=k, maturity=t, rate=r, dividend_yield=q, vol=v)


def digital(kind, s, k, t, r, q, v, slope=0.0):
    return bs.digital_price(
        kind, spot=s, strike=k, maturity=t, rate=r, dividend_yield=q, vol=v, vol_slope=slope
    )


@pytest.mark.parametrize(
    ("kind", "s", "k", "t", "r", "q", "v", "expected", "tol"),
    [
        # Hull, Options, Futures, and Other Derivatives: worked example S0 = 42, K = 40.
        ("call", 42.0, 40.0, 0.5, 0.10, 0.0, 0.20, 4.76, 5e-3),
        ("put", 42.0, 40.0, 0.5, 0.10, 0.0, 0.20, 0.81, 5e-3),
        # Haug, The Complete Guide to Option Pricing Formulas, ch. 1 examples.
        ("call", 60.0, 65.0, 0.25, 0.08, 0.0, 0.30, 2.1334, 5e-5),
        ("put", 100.0, 95.0, 0.5, 0.10, 0.05, 0.20, 2.4648, 5e-5),
    ],
)
def test_published_values(kind, s, k, t, r, q, v, expected, tol):
    assert float(vanilla(kind, s, k, t, r, q, v)) == pytest.approx(expected, abs=tol)


def test_put_call_parity_on_random_inputs():
    rng = np.random.default_rng(0)
    n = 5000
    s, k = rng.uniform(1, 200, n), rng.uniform(1, 200, n)
    t, v = rng.uniform(0.0, 10.0, n), rng.uniform(0.0, 1.5, n)
    r, q = rng.uniform(-0.02, 0.08, n), rng.uniform(0.0, 0.06, n)
    lhs = vanilla("call", s, k, t, r, q, v) - vanilla("put", s, k, t, r, q, v)
    rhs = s * np.exp(-q * t) - k * np.exp(-r * t)
    np.testing.assert_allclose(lhs, rhs, atol=1e-10 * float(np.max(s)))


def test_inputs_broadcast():
    out = vanilla(
        "call", np.array([[90.0], [100.0], [110.0]]), np.array([95.0, 105.0]), 1.0, 0.03, 0.0, 0.2
    )
    assert out.shape == (3, 2)
    assert np.all(np.diff(out, axis=0) > 0)  # increasing in spot
    assert np.all(np.diff(out, axis=1) < 0)  # decreasing in strike


# --- Greeks ----------------------------------------------------------------------------------


def finite_difference_greeks(price, s, k, t, r, q, v):
    hs, hv, hr, ht = 1e-4 * s, 1e-5, 1e-6, 1e-6
    return {
        "delta": (price(s + hs, k, t, r, q, v) - price(s - hs, k, t, r, q, v)) / (2 * hs),
        "gamma": (
            price(s + hs, k, t, r, q, v)
            - 2 * price(s, k, t, r, q, v)
            + price(s - hs, k, t, r, q, v)
        )
        / hs**2,
        "vega": (price(s, k, t, r, q, v + hv) - price(s, k, t, r, q, v - hv)) / (2 * hv),
        "theta": -(price(s, k, t + ht, r, q, v) - price(s, k, t - ht, r, q, v)) / (2 * ht),
        "rho": (price(s, k, t, r + hr, q, v) - price(s, k, t, r - hr, q, v)) / (2 * hr),
    }


CASES = [
    (s, k, t, 0.03, 0.01, v)
    for s in (70.0, 100.0, 130.0)
    for k in (80.0, 100.0, 120.0)
    for t in (0.25, 1.0, 3.0)
    for v in (0.15, 0.35)
]


@pytest.mark.parametrize("kind", ["call", "put"])
def test_vanilla_greeks_match_finite_differences(kind):
    for case in CASES:
        g = bs.vanilla_greeks(
            kind,
            **dict(
                zip(
                    ("spot", "strike", "maturity", "rate", "dividend_yield", "vol"),
                    case,
                    strict=True,
                )
            ),
        )
        fd = finite_difference_greeks(lambda *a: float(vanilla(kind, *a)), *case)
        for name in GREEKS:
            assert float(getattr(g, name)) == pytest.approx(fd[name], rel=1e-5, abs=1e-6), (
                case,
                name,
            )


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("slope", [0.0, -0.004, 0.002])
def test_digital_greeks_match_finite_differences(kind, slope):
    for case in CASES:
        args = dict(
            zip(("spot", "strike", "maturity", "rate", "dividend_yield", "vol"), case, strict=True)
        )
        g = bs.digital_greeks(kind, **args, vol_slope=slope)
        fd = finite_difference_greeks(
            lambda s_, k_, t_, r_, q_, v_: float(digital(kind, s_, k_, t_, r_, q_, v_, slope)),
            *case,
        )
        for name in GREEKS:
            assert float(getattr(g, name)) == pytest.approx(fd[name], rel=1e-5, abs=1e-6), (
                case,
                name,
            )


# --- Digitals and the smile ------------------------------------------------------------------


def test_digital_call_plus_put_is_the_discount_factor():
    s = np.array([80.0, 100.0, 100.0, 100.0, 120.0])
    t = np.array([1.0, 1.0, 0.0, 2.0, 0.5])
    v = np.array([0.2, 0.0, 0.2, 0.3, 0.25])
    total = digital("call", s, 100.0, t, 0.04, 0.01, v, -0.003) + digital(
        "put", s, 100.0, t, 0.04, 0.01, v, -0.003
    )
    np.testing.assert_allclose(total, np.exp(-0.04 * t), rtol=1e-14)


def test_digital_is_minus_strike_derivative_of_the_call_along_the_smile():
    def smile(k):
        return 0.20 - 0.004 * (k - 100.0) + 0.00002 * (k - 100.0) ** 2

    def smile_slope(k):
        return -0.004 + 0.00004 * (k - 100.0)

    s, t, r, q, h = 100.0, 1.0, 0.03, 0.01, 1e-4
    for k in (80.0, 95.0, 100.0, 110.0):
        call_spread = (
            vanilla("call", s, k - h, t, r, q, smile(k - h))
            - vanilla("call", s, k + h, t, r, q, smile(k + h))
        ) / (2 * h)
        expected = float(call_spread)
        assert float(digital("call", s, k, t, r, q, smile(k), smile_slope(k))) == pytest.approx(
            expected, abs=1e-7
        )
        flat = float(digital("call", s, k, t, r, q, smile(k)))
        assert abs(flat - expected) > 0.05  # the skew term is a first-order effect


# --- Degenerate inputs ------------------------------------------------------------------------


def test_zero_spot_and_zero_strike_are_exact_limits():
    r, q, t, v = 0.03, 0.01, 1.0, 0.2
    assert float(vanilla("put", 0.0, 100.0, t, r, q, v)) == pytest.approx(100.0 * np.exp(-r))
    assert float(vanilla("call", 0.0, 100.0, t, r, q, v)) == 0.0
    assert float(vanilla("call", 100.0, 0.0, t, r, q, v)) == pytest.approx(100.0 * np.exp(-q))
    assert float(vanilla("put", 100.0, 0.0, t, r, q, v)) == 0.0
    assert float(digital("call", 100.0, 0.0, t, r, q, v)) == pytest.approx(np.exp(-r))
    # continuity: a tiny spot gives the same put value as a zero spot
    assert float(vanilla("put", 1e-9, 100.0, t, r, q, v)) == pytest.approx(100.0 * np.exp(-r))


def test_zero_vol_and_zero_time_are_exact_limits():
    r, q = 0.03, 0.01
    fwd_value = 110.0 * np.exp(-q) - 100.0 * np.exp(-r)
    assert float(vanilla("call", 110.0, 100.0, 1.0, r, q, 0.0)) == pytest.approx(fwd_value)
    assert float(vanilla("call", 110.0, 100.0, 1.0, r, q, 1e-9)) == pytest.approx(fwd_value)
    assert float(vanilla("put", 110.0, 100.0, 0.0, r, q, 0.2)) == 0.0
    assert float(vanilla("put", 90.0, 100.0, 0.0, r, q, 0.2)) == pytest.approx(10.0)
    g = bs.vanilla_greeks(
        "call", spot=110.0, strike=100.0, maturity=1.0, rate=r, dividend_yield=q, vol=0.0
    )
    assert float(g.delta) == pytest.approx(np.exp(-q))
    assert float(g.gamma) == 0.0
    assert float(g.vega) == 0.0
    assert float(g.rho) == pytest.approx(100.0 * np.exp(-r))


@pytest.mark.parametrize("field", ["spot", "strike", "maturity", "vol"])
def test_negative_inputs_raise(field):
    args = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.03,
        "dividend_yield": 0.0,
        "vol": 0.2,
    }
    args[field] = -1.0
    with pytest.raises(ValueError, match=field):
        bs.vanilla_price("call", **args)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        bs.vanilla_price(
            "straddle",  # type: ignore[arg-type]
            spot=100,
            strike=100,
            maturity=1,
            rate=0,
            dividend_yield=0,
            vol=0.2,
        )
