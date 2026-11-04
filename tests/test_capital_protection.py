from dataclasses import replace

import numpy as np
import pytest

from spricer.bricks import ZeroCoupon
from spricer.market import FlatVol, InterpolatedVolSurface, Market
from spricer.market.defaults import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_MATURITIES,
    DEFAULT_RATE,
    DEFAULT_STRIKES_PCT,
    DEFAULT_VOLS,
)
from spricer.monitoring import Monitoring
from spricer.products import capital_protection as cp
from spricer.products.notes import Note, NoteLeg

T, INITIAL = 3.0, 100.0
VARIANTS: list[tuple[cp.NoteSpec, dict[str, float]]] = [(spec, {}) for spec in cp.NOTE_CATALOG] + [
    (cp.find_note("Participation note"), {"capped": 0.0}),
    (cp.find_note("Bearish participation note"), {"limited": 0.0}),
    (cp.find_note("Bonus note"), {"protection": 100.0, "barrier": 80.0, "capped": 0.0}),
    # protection below the barrier: the note tracks the underlying between the two
    (cp.find_note("Bonus note"), {"protection": 80.0, "barrier": 90.0, "bonus": 110.0}),
    # a strangle-like twin-win: the call and the put struck apart
    (cp.find_note("Twin-win note"), {"call_strike": 105.0, "put_strike": 95.0}),
]
IDS = [f"{spec.name}{'-' + '-'.join(v) if v else ''}" for spec, v in VARIANTS]


def market_at(perf: float) -> Market:
    """Spot at ``perf`` x initial, with the smile quoted relative to the current spot."""
    spot = perf * INITIAL
    surface = InterpolatedVolSurface(
        spot_ref=spot,
        rate=DEFAULT_RATE,
        dividend_yield=DEFAULT_DIVIDEND_YIELD,
        maturities=DEFAULT_MATURITIES,
        strikes_pct=DEFAULT_STRIKES_PCT,
        vols=DEFAULT_VOLS,
    )
    return Market(spot, DEFAULT_RATE, DEFAULT_DIVIDEND_YIELD, surface)


@pytest.mark.parametrize(("spec", "values"), VARIANTS, ids=IDS)
def test_bricks_replicate_the_term_sheet_payoff(spec, values):
    note = spec.build(values, T, INITIAL)
    perf = np.linspace(0.5, 300.0, 59901)  # step 0.005 %, hits every round level exactly
    np.testing.assert_allclose(note.bricks_payoff(perf), note.payoff(perf), atol=1e-9)


@pytest.mark.parametrize("perf", [1.0, 1.22])
@pytest.mark.parametrize(("spec", "values"), VARIANTS, ids=IDS)
def test_value_is_the_discounted_expected_payoff_under_the_smile(
    discounted_expectation, spec, values, perf
):
    market = market_at(perf)
    note = spec.build(values, T, INITIAL)
    expected = discounted_expectation(
        market, lambda s: note.payoff(100.0 * s / INITIAL), T, note.strikes
    )
    # notes are worth about 100: 1e-6 relative is 0.01 bp of notional
    assert float(note.value(market)) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(("spec", "values"), VARIANTS, ids=IDS)
def test_legs_add_up_and_the_issuer_spread_discounts_everything(spec, values):
    market, note, s = market_at(1.0), spec.build(values, T, INITIAL), 0.012
    assert sum(note.leg_values(market, s)) == pytest.approx(float(note.value(market, s)))
    assert float(note.value(market, s)) == pytest.approx(
        np.exp(-s * T) * float(note.value(market)), rel=1e-14
    )


def test_bond_part_is_the_protection_discounted_at_rate_plus_spread():
    note = cp.find_note("Participation note").build({"protection": 90.0}, T, INITIAL)
    bond = note.leg_values(market_at(1.0), 0.01)[0]
    assert note.legs[0].role == "Zero-coupon bond"
    assert bond == pytest.approx(90.0 * np.exp(-(DEFAULT_RATE + 0.01) * T))


def test_greeks_match_finite_differences_with_a_spread():
    s = 0.01
    spec = cp.find_note("Bonus note")
    flat = Market(100.0, DEFAULT_RATE, DEFAULT_DIVIDEND_YIELD, FlatVol(0.2))
    note = spec.build({}, T, INITIAL)
    g = note.greeks(flat, s)
    h, hv, hr, ht = 1e-3, 1e-5, 1e-6, 1e-6

    def v(m=flat, spot=None, t=T):
        return float(spec.build({}, t, INITIAL).value(m, s, spot))

    assert float(g.delta) == pytest.approx((v(spot=100 + h) - v(spot=100 - h)) / (2 * h), rel=1e-6)
    assert float(g.gamma) == pytest.approx(
        (v(spot=100 + h) - 2 * v() + v(spot=100 - h)) / h**2, rel=1e-3
    )
    assert float(g.vega) == pytest.approx(
        (v(flat.with_vol_shift(hv)) - v(flat.with_vol_shift(-hv))) / (2 * hv), rel=1e-6
    )
    assert float(g.rho) == pytest.approx(
        (v(flat.with_rate(DEFAULT_RATE + hr)) - v(flat.with_rate(DEFAULT_RATE - hr))) / (2 * hr),
        rel=1e-6,
    )
    assert float(g.theta) == pytest.approx(-(v(t=T + ht) - v(t=T - ht)) / (2 * ht), rel=1e-6)


def test_a_shark_with_an_unreachable_barrier_is_a_participation_note():
    market = market_at(1.0)
    shark = cp.shark_note(
        protection=100.0,
        strike=100.0,
        participation=80.0,
        barrier=1e5,
        rebate=0.0,
        direction="up",
        maturity=T,
        initial=INITIAL,
    )
    plain = cp.participation_note(
        protection=100.0, strike=100.0, participation=80.0, cap=None, maturity=T, initial=INITIAL
    )
    assert float(shark.value(market)) == pytest.approx(float(plain.value(market)), abs=1e-9)


def test_degenerate_notes_reduce_to_the_bond():
    market = market_at(1.0)
    bond = 95.0 * np.exp(-DEFAULT_RATE * T)
    no_coupon = cp.digital_coupon_note(
        protection=95.0, coupon=0.0, barrier=100.0, maturity=T, initial=INITIAL
    )
    no_participation = cp.participation_note(
        protection=95.0, strike=100.0, participation=0.0, cap=None, maturity=T, initial=INITIAL
    )
    for note in (no_coupon, no_participation):
        assert len(note.legs) == 1
        assert float(note.value(market)) == pytest.approx(bond)


def test_bonus_equal_to_the_protection_above_the_barrier_has_no_bonus_digital():
    note = cp.bonus_note(
        protection=100.0,
        barrier=90.0,
        bonus=100.0,
        strike=100.0,
        participation=100.0,
        cap=None,
        maturity=T,
        initial=INITIAL,
    )
    assert [leg.role for leg in note.legs] == [
        "Zero-coupon bond",
        "Long call (participation)",
    ]


def test_strikes_are_read_at_the_true_moneyness():
    """With the underlying at 122 % of its initial level, a 108 % strike is at 88.5 % of spot."""
    market = market_at(1.22)
    note = cp.find_note("Bonus note").build({}, T, INITIAL)
    long_call = next(leg for leg in note.legs if leg.role == "Long call (participation)")
    assert long_call.brick.strike == pytest.approx(108.0)
    assert long_call.brick.implied_vol(market) == pytest.approx(float(market.vol.vol(108.0, T)))
    vol_108 = long_call.brick.implied_vol(market)
    assert vol_108 is not None
    assert vol_108 > float(market.vol.vol(market.spot, T))  # skew


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: cp.find_note("Participation note").build({"cap": 90.0}, T, INITIAL), "cap"),
        (lambda: cp.find_note("Shark note (up)").build({"barrier": 95.0}, T, INITIAL), "above"),
        (lambda: cp.find_note("Shark note (down)").build({"barrier": 105.0}, T, INITIAL), "below"),
        (lambda: cp.find_note("Bonus note").build({"bonus": 80.0}, T, INITIAL), "bonus"),
        (lambda: cp.find_note("Twin-win note").build({"floor": 120.0}, T, INITIAL), "floor"),
        (lambda: cp.find_note("Twin-win note").build({"cap": 90.0}, T, INITIAL), "cap"),
        (
            lambda: cp.find_note("Participation note").build({"participation": -5.0}, T, INITIAL),
            "participation",
        ),
        (lambda: cp.find_note("Participation note").build({}, T, 0.0), "initial"),
        (
            lambda: cp.find_note("Bearish participation note").build({"floor": 110.0}, T, INITIAL),
            "floor",
        ),
        (lambda: cp.find_note("Bonus note").build({"cap": 100.0}, T, INITIAL), "cap"),
        (lambda: Note("Empty", T, INITIAL, (), lambda p: p), "leg"),
        (
            lambda: Note("Bad", T, -1.0, (NoteLeg("Bond", ZeroCoupon(T), 100.0),), lambda p: p),
            "initial",
        ),
        (
            lambda: cp.shark_note(
                protection=100.0,
                strike=100.0,
                participation=100.0,
                barrier=130.0,
                rebate=0.0,
                direction="sideways",
                maturity=T,
                initial=INITIAL,
            ),
            "direction",
        ),
    ],
)
def test_invalid_terms_raise(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_notes_replicated_at_maturity_refuse_a_path_dependent_barrier():
    for name in ("Bonus note", "Digital coupon note"):
        with pytest.raises(NotImplementedError, match="European"):
            cp.find_note(name).build({}, T, INITIAL, Monitoring.CONTINUOUS)


def test_sharks_accept_every_monitoring():
    market = market_at(1.0)
    spec = cp.find_note("Shark note (up)")
    values = {
        mode: float(
            spec.build(
                {}, T, INITIAL, mode, 252 * 3 if mode is Monitoring.DISCRETE else None
            ).value(market)
        )
        for mode in Monitoring
    }
    # observed more often, the up-and-out barrier is touched more often: the note is worth less
    assert values[Monitoring.EUROPEAN] > values[Monitoring.DISCRETE] > values[Monitoring.CONTINUOUS]
    roles = [leg.role for leg in spec.build({}, T, INITIAL, Monitoring.CONTINUOUS).legs]
    assert roles[-1] == "Long one-touch (rebate)"


def test_catalog_names_are_unique_and_found():
    names = [spec.name for spec in cp.NOTE_CATALOG]
    assert len(names) == len(set(names))
    assert cp.find_note("Bonus note").has_barrier
    with pytest.raises(KeyError):
        cp.find_note("Unknown")


def test_levels_name_the_term_sheet():
    note = cp.find_note("Bonus note").build({}, T, INITIAL)
    names = [level.name for level in note.levels]
    assert names == [
        "Capital protection",
        "Bonus level",
        "Bonus barrier",
        "Long call strike",
        "Short call strike (cap)",
    ]
    assert replace(note.levels[0]).value_pct == 90.0


def test_bonus_with_protection_below_the_barrier_tracks_the_underlying():
    note = cp.find_note("Bonus note").build({"protection": 80.0, "barrier": 90.0}, T, INITIAL)
    roles = [leg.role for leg in note.legs]
    assert "Short put down-and-in" in roles
    np.testing.assert_allclose(note.payoff([70.0, 85.0, 89.0, 95.0]), [80.0, 85.0, 89.0, 108.0])


def test_bonus_protection_put_cancels_the_down_and_in_put_when_they_coincide():
    market = market_at(1.0)
    note = cp.find_note("Bonus note").build({"protection": 90.0, "barrier": 90.0}, T, INITIAL)
    values = dict(zip((leg.role for leg in note.legs), note.leg_values(market), strict=True))
    assert values["Long put (capital protection)"] == pytest.approx(
        -values["Short put down-and-in"]
    )
    assert note.legs[0].quantity == 90.0  # bond paying the barrier level


def test_a_shark_without_participation_is_a_bond_plus_a_rebate_digital():
    note = cp.find_note("Shark note (up)").build({"participation": 0.0}, T, INITIAL)
    assert [leg.role for leg in note.legs] == ["Zero-coupon bond", "Long digital call (rebate)"]
