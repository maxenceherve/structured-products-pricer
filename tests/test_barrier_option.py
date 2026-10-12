"""European barrier options: replication by vanilla options and digitals."""

import numpy as np
import pytest

from spricer.bricks import BARRIER_TYPES, BarrierOption, EuropeanOption
from spricer.monitoring import Monitoring

T = 1.0
CASES = [
    (kind, barrier_type, strike, barrier)
    for kind in ("call", "put")
    for barrier_type in BARRIER_TYPES
    for strike, barrier in ((100.0, 120.0), (100.0, 80.0), (100.0, 100.0))
]
IDS = [f"{k}-{b}-K{s:g}-H{h:g}" for k, b, s, h in CASES]


@pytest.mark.parametrize(("kind", "barrier_type", "strike", "barrier"), CASES, ids=IDS)
def test_components_replicate_the_payoff(kind, barrier_type, strike, barrier):
    option = BarrierOption(kind, barrier_type, strike, barrier, T)
    s = np.linspace(0.5, 300.0, 59901)  # hits the strike and the barrier exactly
    replicated = sum(q * brick.payoff(s) for q, brick in option.components())
    np.testing.assert_allclose(replicated + 0.0 * s, option.payoff(s), atol=1e-9)


@pytest.mark.parametrize(("kind", "strike", "barrier"), [(k, s, h) for k, _, s, h in CASES[::4]])
def test_in_plus_out_is_the_vanilla(market, kind, strike, barrier):
    vanilla = float(EuropeanOption(kind, strike, T).value(market))
    for direction in ("up", "down"):
        knock_in = BarrierOption(kind, f"{direction}-and-in", strike, barrier, T)  # type: ignore[arg-type]
        knock_out = BarrierOption(kind, f"{direction}-and-out", strike, barrier, T)  # type: ignore[arg-type]
        assert float(knock_in.value(market) + knock_out.value(market)) == pytest.approx(vanilla)


@pytest.mark.parametrize(("kind", "barrier_type", "strike", "barrier"), CASES, ids=IDS)
def test_value_is_the_discounted_expected_payoff(
    market, discounted_expectation, kind, barrier_type, strike, barrier
):
    option = BarrierOption(kind, barrier_type, strike, barrier, T)
    expected = discounted_expectation(market, option.payoff, T, (strike, barrier))
    assert float(option.value(market)) == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_greeks_match_finite_differences(market):
    option = BarrierOption("call", "up-and-out", 100.0, 125.0, T)
    g, h, hv = option.greeks(market), 1e-3, 1e-5
    v = lambda m=market, spot=None: float(option.value(m, spot))  # noqa: E731
    assert float(g.delta) == pytest.approx((v(spot=100 + h) - v(spot=100 - h)) / (2 * h), rel=1e-6)
    assert float(g.vega) == pytest.approx(
        (v(market.with_vol_shift(hv)) - v(market.with_vol_shift(-hv))) / (2 * hv), rel=1e-6
    )
    assert float(option.skew_adjustment(market)) != 0.0  # the digital at the barrier
    assert option.barrier_vol(market) == float(market.vol.vol(125.0, T))


def test_a_down_and_in_put_struck_at_its_barrier_is_a_put(market):
    pdi = BarrierOption("put", "down-and-in", 90.0, 90.0, T)
    assert float(pdi.value(market)) == pytest.approx(
        float(EuropeanOption("put", 90.0, T).value(market))
    )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("swap", "up-and-out", 100.0, 120.0, T), "kind"),
        (("call", "sideways", 100.0, 120.0, T), "barrier_type"),
        (("call", "up-and-out", 100.0, 0.0, T), "barrier"),
    ],
)
def test_invalid_barrier_options_raise(args, message):
    with pytest.raises(ValueError, match=message):
        BarrierOption(*args)


def test_only_a_european_barrier_option_replicates_with_vanillas():
    with pytest.raises(ValueError, match="European"):
        BarrierOption("call", "up-and-out", 100.0, 120.0, T, Monitoring.CONTINUOUS).components()
    assert (
        BarrierOption("put", "down-and-in", 100.0, 70.0, T).label == "Put down-and-in K=100 H=70 1y"
    )
