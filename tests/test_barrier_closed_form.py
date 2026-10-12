"""Continuous and discrete barriers: closed forms, BGK correction, one-touch."""

import math

import numpy as np
import pytest

from spricer.analytic import barrier as cf
from spricer.bricks import BARRIER_TYPES, BarrierOption, EuropeanOption, OneTouch
from spricer.market import FlatVol, Market
from spricer.monitoring import Monitoring

S, R, Q, VOL, T = 100.0, 0.03, 0.01, 0.25, 1.0
FLAT = Market(S, R, Q, FlatVol(VOL))


def barrier_for(barrier_type: str) -> float:
    return 120.0 if barrier_type.startswith("up") else 80.0


# --- QuantLib cross-check ---------------------------------------------------------------------


ql = pytest.importorskip("QuantLib")


def quantlib_barrier(kind, barrier_type, strike, barrier, days, vol=VOL):
    today = ql.Date(2, ql.January, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, Q, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, R, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, dc)),
    )
    ql_type = {
        "up-and-out": ql.Barrier.UpOut,
        "up-and-in": ql.Barrier.UpIn,
        "down-and-out": ql.Barrier.DownOut,
        "down-and-in": ql.Barrier.DownIn,
    }[barrier_type]
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, strike)
    option = ql.BarrierOption(ql_type, barrier, 0.0, payoff, ql.EuropeanExercise(today + days))
    option.setPricingEngine(ql.AnalyticBarrierEngine(process))
    return option.NPV()


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
@pytest.mark.parametrize("strike", [70.0, 90.0, 100.0, 110.0, 130.0])
@pytest.mark.parametrize("days", [91, 365, 1095])
def test_continuous_closed_form_matches_quantlib(kind, barrier_type, strike, days):
    barrier = barrier_for(barrier_type)
    option = BarrierOption(kind, barrier_type, strike, barrier, days / 365, Monitoring.CONTINUOUS)
    expected = quantlib_barrier(kind, barrier_type, strike, barrier, days)
    assert float(option.value(FLAT)) == pytest.approx(expected, rel=1e-10, abs=1e-12)


# --- Parities, limits, one-touch --------------------------------------------------------------


@pytest.mark.parametrize("monitoring", [Monitoring.CONTINUOUS, Monitoring.DISCRETE])
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("direction", ["up", "down"])
def test_in_plus_out_is_the_vanilla(market, monitoring, kind, direction):
    n = 52 if monitoring is Monitoring.DISCRETE else None
    barrier = 125.0 if direction == "up" else 75.0
    for strike in (80.0, 100.0, 130.0):
        knock_in = BarrierOption(kind, f"{direction}-and-in", strike, barrier, T, monitoring, n)  # type: ignore[arg-type]
        knock_out = BarrierOption(kind, f"{direction}-and-out", strike, barrier, T, monitoring, n)  # type: ignore[arg-type]
        vanilla = EuropeanOption(kind, strike, T)
        # the closed forms use the strike's volatility: compare with a vanilla at that volatility
        flat = Market(
            market.spot, market.rate, market.dividend_yield, FlatVol(vanilla.implied_vol(market))
        )
        assert float(knock_in.value(market) + knock_out.value(market)) == pytest.approx(
            float(vanilla.value(flat)), rel=1e-12
        )
        g_in, g_out, g_v = knock_in.greeks(market), knock_out.greeks(market), vanilla.greeks(flat)
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            assert float(getattr(g_in, name) + getattr(g_out, name)) == pytest.approx(
                float(getattr(g_v, name)), rel=1e-4, abs=1e-6
            ), name


def test_limits_of_the_barrier():
    far = BarrierOption("call", "up-and-out", 100.0, 1e6, T, Monitoring.CONTINUOUS)
    assert float(far.value(FLAT)) == pytest.approx(
        float(EuropeanOption("call", 100.0, T).value(FLAT))
    )
    already = BarrierOption("put", "down-and-out", 100.0, 120.0, T, Monitoring.CONTINUOUS)
    assert float(already.value(FLAT)) == 0.0  # the spot is already below the down barrier
    knocked_in = BarrierOption("put", "down-and-in", 100.0, 120.0, T, Monitoring.CONTINUOUS)
    assert float(knocked_in.value(FLAT)) == pytest.approx(
        float(EuropeanOption("put", 100.0, T).value(FLAT))
    )
    expired = cf.barrier_price(
        "call",
        "up-and-out",
        spot=110.0,
        strike=100.0,
        barrier=120.0,
        maturity=0.0,
        rate=R,
        dividend_yield=Q,
        vol=VOL,
    )
    assert float(expired) == pytest.approx(10.0)


def test_one_touch_matches_the_reflection_principle():
    nu, h = R - Q - 0.5 * VOL**2, math.log(130.0 / S)
    n = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))  # noqa: E731
    v = VOL * math.sqrt(T)
    probability = n((-h + nu * T) / v) + math.exp(2 * nu * h / VOL**2) * n((-h - nu * T) / v)
    touch = OneTouch("up", 130.0, T, Monitoring.CONTINUOUS)
    assert float(touch.value(FLAT)) == pytest.approx(math.exp(-R * T) * probability, rel=1e-12)
    down = OneTouch("down", 70.0, T, Monitoring.CONTINUOUS)
    assert 0.0 < float(down.value(FLAT)) < math.exp(-R * T)
    european = OneTouch("up", 130.0, T)
    assert float(european.value(FLAT)) < float(
        touch.value(FLAT)
    )  # touched along the path is likelier


def test_one_touch_greeks_and_validation(market):
    touch = OneTouch("up", 130.0, T, Monitoring.DISCRETE, 52)
    g, h = touch.greeks(market), 1e-2
    fd = (float(touch.value(market, S + h)) - float(touch.value(market, S - h))) / (2 * h)
    assert float(g.delta) == pytest.approx(fd, rel=1e-4)
    assert touch.strike is None
    assert touch.label == "One-touch up H=130 1y"
    assert float(OneTouch("up", 130.0, T).skew_adjustment(market)) != 0.0
    assert float(touch.skew_adjustment(market)) == 0.0
    np.testing.assert_array_equal(touch.payoff([120.0, 140.0]), [0.0, 1.0])
    with pytest.raises(ValueError, match="direction"):
        OneTouch("sideways", 130.0, T)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="barrier"):
        OneTouch("up", -1.0, T)


@pytest.mark.parametrize(
    ("monitoring", "observations"),
    [(Monitoring.DISCRETE, None), (Monitoring.DISCRETE, 0), (Monitoring.CONTINUOUS, 12)],
)
def test_observations_go_with_discrete_monitoring_only(monitoring, observations):
    with pytest.raises(ValueError, match="observation"):
        BarrierOption("call", "up-and-out", 100.0, 120.0, T, monitoring, observations)


# --- Discrete monitoring: Monte Carlo vs BGK ---------------------------------------------------


def monte_carlo_up_and_out_call(
    n_obs: int, paths: int = 200_000, seed: int = 7
) -> tuple[float, float]:
    """Discounted payoff mean and standard error, barrier checked at ``n_obs`` dates."""
    rng = np.random.default_rng(seed)
    dt = T / n_obs
    z = rng.standard_normal((paths // 2, n_obs))
    z = np.vstack([z, -z])  # antithetic
    log_s = math.log(S) + np.cumsum((R - Q - 0.5 * VOL**2) * dt + VOL * math.sqrt(dt) * z, axis=1)
    alive = log_s.max(axis=1) < math.log(130.0)
    payoff = np.where(alive, np.maximum(np.exp(log_s[:, -1]) - 100.0, 0.0), 0.0) * math.exp(-R * T)
    pairs = 0.5 * (payoff[: paths // 2] + payoff[paths // 2 :])  # independent antithetic pairs
    return float(pairs.mean()), float(pairs.std(ddof=1) / math.sqrt(paths // 2))


@pytest.mark.slow
@pytest.mark.parametrize(("n_obs", "paths"), [(52, 200_000), (252, 40_000)])
def test_bgk_closed_form_matches_a_discretely_monitored_monte_carlo(n_obs, paths):
    mc, se = monte_carlo_up_and_out_call(n_obs, paths)
    discrete = BarrierOption("call", "up-and-out", 100.0, 130.0, T, Monitoring.DISCRETE, n_obs)
    continuous = BarrierOption("call", "up-and-out", 100.0, 130.0, T, Monitoring.CONTINUOUS)
    assert abs(mc - float(discrete.value(FLAT))) < 4.0 * se  # BGK within Monte Carlo noise
    assert mc - float(continuous.value(FLAT)) > 5.0 * se  # the continuous formula is biased
    assert cf.bgk_warning(n_obs) is None


@pytest.mark.slow
def test_bgk_is_flagged_as_approximate_with_few_dates():
    mc, se = monte_carlo_up_and_out_call(12)
    bgk = float(
        BarrierOption("call", "up-and-out", 100.0, 130.0, T, Monitoring.DISCRETE, 12).value(FLAT)
    )
    assert bgk - mc > 4.0 * se  # monthly dates: BGK overprices by a few % of the price
    assert (bgk - mc) / mc < 0.1
    message = cf.bgk_warning(12)
    assert message is not None
    assert "approximate" in message


def test_closed_forms_validate_their_inputs_and_expire():
    with pytest.raises(ValueError, match="observations"):
        cf.bgk_barrier(130.0, "up", VOL, T, 0)
    with pytest.raises(ValueError, match="positive"):
        cf.barrier_price(
            "call",
            "up-and-out",
            spot=100.0,
            strike=100.0,
            barrier=0.0,
            maturity=T,
            rate=R,
            dividend_yield=Q,
            vol=VOL,
        )
    with pytest.raises(ValueError, match="positive"):
        cf.one_touch_price(
            "up", spot=-1.0, barrier=130.0, maturity=T, rate=R, dividend_yield=Q, vol=VOL
        )
    expired = cf.one_touch_price(
        "up",
        spot=np.array([120.0, 140.0]),
        barrier=130.0,
        maturity=0.0,
        rate=R,
        dividend_yield=Q,
        vol=VOL,
    )
    np.testing.assert_array_equal(expired, [0.0, 1.0])
