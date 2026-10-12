import numpy as np
import pytest

from spricer import strategies as st
from spricer.bricks import DigitalOption, EuropeanOption, ZeroCoupon
from spricer.strategies import CATALOG, Family, Leg, Strategy

T = 1.0
PAYOUT = 10.0


def build(spec, spot=100.0):
    strikes = [spot * pct / 100.0 for pct in spec.default_strikes_pct]
    return spec.build(strikes, T, PAYOUT)


def test_catalog_covers_every_family_with_unique_names():
    names = [spec.name for spec in CATALOG]
    assert len(names) == len(set(names))
    by_family = st.specs_by_family()
    assert set(by_family) == set(Family) - {Family.CUSTOM}
    assert st.find_spec("Straddle").family is Family.VOLATILITY
    with pytest.raises(KeyError):
        st.find_spec("Unknown")


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: s.name)
def test_value_is_the_discounted_expected_payoff_under_the_smile_density(
    market, discounted_expectation, spec
):
    strategy = build(spec)
    expected = discounted_expectation(market, strategy.payoff, T, strategy.strikes)
    assert float(strategy.value(market)) == pytest.approx(expected, rel=1e-7, abs=1e-8)


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: s.name)
def test_decomposition_adds_up(market, spec):
    strategy = build(spec)
    rows = strategy.decompose(market)
    assert sum(r.value for r in rows) == pytest.approx(float(strategy.value(market)), abs=1e-12)
    total = strategy.greeks(market)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert sum(float(getattr(r.greeks, name)) for r in rows) == pytest.approx(
            float(getattr(total, name)), abs=1e-12
        )
    for row in rows:
        assert row.value == pytest.approx(row.quantity * row.unit_value)


@pytest.mark.parametrize("spec", CATALOG, ids=lambda s: s.name)
def test_short_is_the_exact_opposite(market, spec):
    long, short = build(spec), build(spec).short()
    s_t = np.linspace(40.0, 160.0, 25)
    assert float(short.value(market)) == -float(long.value(market))
    np.testing.assert_array_equal(short.payoff(s_t), -long.payoff(s_t))
    assert float(short.greeks(market).vega) == -float(long.greeks(market).vega)
    assert short.name.startswith("Short ")


def test_box_spread_is_a_zero_coupon(market):
    box = st.box_spread(90.0, 115.0, T)
    np.testing.assert_allclose(box.payoff(np.linspace(1.0, 300.0, 100)), 25.0)
    assert float(box.value(market)) == pytest.approx(25.0 * np.exp(-market.rate * T), rel=1e-12)


def test_synthetic_forward_is_a_forward(market):
    fwd = st.synthetic_forward(104.0, T)
    expected = market.spot * np.exp(-market.dividend_yield * T) - 104.0 * np.exp(-market.rate * T)
    assert float(fwd.value(market)) == pytest.approx(expected, rel=1e-12)
    assert float(fwd.greeks(market).vega) == pytest.approx(
        0.0, abs=1e-12
    )  # same volatility on both legs


def test_covered_call_is_a_zero_coupon_minus_a_put(market):
    k = 110.0
    covered = st.covered_call(k, T)
    put = EuropeanOption("put", k, T)
    assert float(covered.value(market)) == pytest.approx(
        k * np.exp(-market.rate * T) - float(put.value(market)), rel=1e-12
    )


@pytest.mark.parametrize("strikes", [(90.0, 100.0, 110.0), (80.0, 95.0, 125.0)])
def test_call_and_put_butterflies_coincide(market, strikes):
    low, mid, high = strikes
    call_fly, put_fly = st.call_butterfly(low, mid, high, T), st.put_butterfly(low, mid, high, T)
    s_t = np.linspace(1.0, 300.0, 1000)
    np.testing.assert_allclose(call_fly.payoff(s_t), put_fly.payoff(s_t), atol=1e-12)
    assert float(call_fly.value(market)) == pytest.approx(float(put_fly.value(market)), rel=1e-10)
    outside = (s_t <= strikes[0]) | (s_t >= strikes[2])
    np.testing.assert_allclose(call_fly.payoff(s_t)[outside], 0.0, atol=1e-12)
    assert float(call_fly.payoff(strikes[1])) == pytest.approx(strikes[1] - strikes[0], rel=1e-12)


def test_condor_payoff_vanishes_outside_the_wings():
    condor = st.call_condor(80.0, 90.0, 110.0, 130.0, T)
    s_t = np.linspace(1.0, 300.0, 3000)
    outside = (s_t <= 80.0) | (s_t >= 130.0)
    np.testing.assert_allclose(condor.payoff(s_t)[outside], 0.0, atol=1e-12)


def test_iron_butterfly_is_a_butterfly_minus_a_zero_coupon(market):
    iron, fly = st.iron_butterfly(90.0, 100.0, 110.0, T), st.call_butterfly(90.0, 100.0, 110.0, T)
    zc = 10.0 * float(ZeroCoupon(T).value(market))
    assert float(iron.value(market)) == pytest.approx(float(fly.value(market)) - zc, rel=1e-10)


def test_strip_and_strap_add_one_leg_to_a_straddle(market):
    straddle = float(st.straddle(100.0, T).value(market))
    assert float(st.strip(100.0, T).value(market)) == pytest.approx(
        straddle + float(EuropeanOption("put", 100.0, T).value(market))
    )
    assert float(st.strap(100.0, T).value(market)) == pytest.approx(
        straddle + float(EuropeanOption("call", 100.0, T).value(market))
    )


def test_butterflies_are_never_negative_on_the_default_surface(market):
    for mid in np.arange(55.0, 196.0, 5.0):
        assert float(st.call_butterfly(mid - 2.0, mid, mid + 2.0, T).value(market)) > 0.0


def test_call_spread_replication_converges_to_the_skewed_digital(market):
    digital = PAYOUT * float(DigitalOption("call", 100.0, T).value(market))
    errors = [
        abs(float(st.call_spread_replication(100 - h, 100 + h, T, PAYOUT).value(market)) - digital)
        for h in (4.0, 2.0, 1.0)
    ]
    assert errors[2] < errors[1] < errors[0]
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.1)  # second-order convergence
    assert errors[2] < 2e-3


def test_digital_legs_report_their_skew_adjustment(market):
    rows = st.digital_call(100.0, T, PAYOUT).decompose(market)
    assert rows[0].skew_adjustment == pytest.approx(
        PAYOUT * float(DigitalOption("call", 100.0, T).skew_adjustment(market))
    )
    assert all(r.skew_adjustment == 0.0 for r in st.straddle(100.0, T).decompose(market))


def test_values_are_vectorised_in_spot(market):
    spots = np.linspace(60.0, 140.0, 9)
    collar = st.collar(90.0, 110.0, T)
    np.testing.assert_allclose(
        collar.value(market, spots),
        [float(collar.value(market.with_spot(s))) for s in spots],
        rtol=1e-13,
    )


@pytest.mark.parametrize(
    "build_bad",
    [
        lambda: st.bull_call_spread(110.0, 100.0, T),
        lambda: st.call_butterfly(90.0, 90.0, 110.0, T),
        lambda: st.straddle(-5.0, T),
        lambda: st.find_spec("Straddle").build([100.0, 110.0], T),
        lambda: Strategy("Empty", Family.CUSTOM, ()),
        lambda: Strategy(
            "Calendar",
            Family.CUSTOM,
            (
                Leg(EuropeanOption("call", 100.0, 1.0), 1.0),
                Leg(EuropeanOption("call", 100.0, 2.0), -1.0),
            ),
        ),
        lambda: Strategy("Nan", Family.CUSTOM, (Leg(EuropeanOption("call", 100.0, 1.0), np.nan),)),
    ],
)
def test_invalid_strategies_raise(build_bad):
    with pytest.raises(ValueError, match=r"strike|leg|maturity|finite"):
        build_bad()


def test_strategy_exposes_its_maturity_and_strikes():
    condor = st.iron_condor(80.0, 90.0, 110.0, 120.0, 0.75)
    assert condor.maturity == 0.75
    assert condor.strikes == (80.0, 90.0, 110.0, 120.0)


def test_single_options_are_plain_calls_and_puts(market):
    assert float(st.call(105.0, T).value(market)) == float(
        EuropeanOption("call", 105.0, T).value(market)
    )
    assert float(st.put(95.0, T).value(market)) == float(
        EuropeanOption("put", 95.0, T).value(market)
    )
