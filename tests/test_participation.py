import numpy as np
import pytest

from spricer.market import InterpolatedVolSurface, Market
from spricer.market.defaults import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_MATURITIES,
    DEFAULT_RATE,
    DEFAULT_STRIKES_PCT,
    DEFAULT_VOLS,
)
from spricer.monitoring import Monitoring
from spricer.products import participation as pc
from spricer.products.capital_protection import NoteSpec

T, INITIAL = 3.0, 100.0
SPECS = pc.PARTICIPATION_CATALOG + pc.YIELD_CERTIFICATES + pc.SHARK_WARRANTS
VARIANTS: list[tuple[NoteSpec, dict[str, float]]] = [(spec, {}) for spec in SPECS] + [
    (pc.find_certificate("Bonus certificate"), {"capped": 1.0}),
    (pc.find_certificate("Airbag certificate"), {"capped": 1.0, "participation": 80.0}),
    (pc.find_certificate("Buffer certificate"), {"capped": 1.0, "participation": 150.0}),
    (pc.find_certificate("Twin-win certificate"), {"participation_up": 120.0, "barrier": 60.0}),
    (pc.find_certificate("Outperformance certificate"), {"participation": 100.0}),
    (pc.find_certificate("Reverse bonus certificate"), {"capped": 1.0}),
    (pc.find_certificate("Sprint certificate"), {"participation": 100.0}),
    # every degree of freedom: strikes, participations, caps
    (pc.find_certificate("Outperformance certificate"), {"strike": 105.0, "capped": 1.0}),
    (
        pc.find_certificate("Barrier outperformance certificate"),
        {"strike": 90.0, "capped": 1.0, "cap": 125.0},
    ),
    (pc.find_certificate("Bonus certificate"), {"participation": 150.0}),
    (
        pc.find_certificate("Bonus certificate"),
        {"participation": 80.0, "strike": 105.0, "capped": 1.0, "cap": 160.0},
    ),
    (pc.find_certificate("Bonus certificate"), {"bonus": 105.0, "strike": 110.0}),
    (pc.find_certificate("Twin-win certificate"), {"strike": 95.0, "capped": 1.0}),
    (pc.find_certificate("Airbag certificate"), {"strike": 105.0, "capped": 1.0}),
    (pc.find_certificate("Buffer certificate"), {"strike": 95.0}),
    (pc.find_certificate("Sprint certificate"), {"strike": 105.0}),
    (pc.find_certificate("Reverse bonus certificate"), {"participation": 150.0, "capped": 1.0}),
]
IDS = [f"{spec.name}{'-' + '-'.join(v) if v else ''}" for spec, v in VARIANTS]


def market_at(perf: float) -> Market:
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


@pytest.mark.parametrize("perf", [1.0, 0.85])
@pytest.mark.parametrize(("spec", "values"), VARIANTS, ids=IDS)
def test_value_is_the_discounted_expected_payoff_under_the_smile(
    discounted_expectation, spec, values, perf
):
    market = market_at(perf)
    note = spec.build(values, T, INITIAL)
    expected = discounted_expectation(
        market, lambda s: note.payoff(100.0 * s / INITIAL), T, note.strikes
    )
    assert float(note.value(market)) == pytest.approx(expected, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(("spec", "values"), VARIANTS, ids=IDS)
def test_legs_add_up(spec, values):
    market, note = market_at(1.0), spec.build(values, T, INITIAL)
    assert sum(note.leg_values(market, 0.01)) == pytest.approx(float(note.value(market, 0.01)))


def test_shapes_at_key_levels():
    def at(name, values, *perfs):
        note = pc.find_certificate(name).build(values, T, INITIAL)
        return note.payoff(np.array(perfs)).tolist()

    assert at("Outperformance certificate", {}, 60.0, 100.0, 120.0) == [60.0, 100.0, 130.0]
    assert at("Barrier outperformance certificate", {}, 69.0, 80.0, 110.0) == pytest.approx(
        [69.0, 100.0, 113.0]
    )
    assert at("Bonus certificate", {}, 69.0, 90.0, 130.0) == [69.0, 115.0, 130.0]
    assert at("Bonus certificate", {"capped": 1.0}, 150.0) == [140.0]
    assert at("Twin-win certificate", {}, 69.0, 80.0, 120.0) == [69.0, 120.0, 120.0]
    # airbag at 80 %: 100 % down to 80, then 100 / 80 of the underlying (50 -> 62.5)
    assert at("Airbag certificate", {}, 50.0, 85.0, 130.0) == [62.5, 100.0, 130.0]
    assert at("Buffer certificate", {}, 50.0, 95.0, 130.0) == [60.0, 100.0, 130.0]
    # sprint: twice the rise up to 120 %, i.e. at most 140 %
    assert at("Sprint certificate", {}, 70.0, 110.0, 150.0) == [70.0, 120.0, 140.0]
    # reverse bonus: 200 % - performance, at least 115 % below the 130 % barrier
    assert at("Reverse bonus certificate", {}, 50.0, 100.0, 140.0, 250.0) == [
        150.0,
        115.0,
        60.0,
        0.0,
    ]
    assert at("Reverse bonus certificate", {"capped": 1.0}, 20.0) == [150.0]
    assert at("Discount certificate", {}, 80.0, 120.0) == [80.0, 105.0]
    assert at("Shark warrant (call)", {}, 90.0, 120.0, 130.0) == [0.0, 20.0, 5.0]
    assert at("Shark warrant (put)", {}, 69.0, 80.0, 110.0) == [5.0, 20.0, 0.0]
    # the extra terms: strike, participation, cap
    capped = {"strike": 110.0, "capped": 1.0, "cap": 130.0}
    assert at("Outperformance certificate", capped, 100.0, 120.0, 150.0) == pytest.approx(
        [100.0, 125.0, 140.0]
    )
    # bonus 115 % with 150 % above 100 %: the upside reaches the bonus at 110 %
    geared = {"participation": 150.0}
    assert at("Bonus certificate", geared, 69.0, 105.0, 120.0) == pytest.approx(
        [69.0, 115.0, 130.0]
    )
    assert at("Twin-win certificate", {"strike": 95.0}, 80.0, 105.0) == pytest.approx(
        [110.0, 105.0]
    )
    assert at("Airbag certificate", {"strike": 105.0}, 110.0) == pytest.approx([105.0])
    assert at("Reverse bonus certificate", {"participation": 150.0}, 100.0, 125.0) == pytest.approx(
        [150.0, 115.0]
    )


def test_the_barrier_can_be_observed_along_the_path():
    market = market_at(1.0)
    for name in (
        "Bonus certificate",
        "Twin-win certificate",
        "Barrier outperformance certificate",
        "Reverse bonus certificate",
    ):
        spec = pc.find_certificate(name)
        european = float(spec.build({}, T, INITIAL).value(market))
        continuous = float(spec.build({}, T, INITIAL, Monitoring.CONTINUOUS).value(market))
        assert continuous < european, name  # the bonus is lost more often
    shark = pc.find_certificate("Shark warrant (call)")
    roles = [leg.role for leg in shark.build({}, T, INITIAL, Monitoring.DISCRETE, 36).legs]
    assert roles == ["Long call up-and-out", "Rebate if the barrier is touched"]


def test_a_shark_warrant_without_rebate_is_a_knock_out_option():
    note = pc.find_certificate("Shark warrant (put)").build({"rebate": 0.0}, T, INITIAL)
    assert [leg.role for leg in note.legs] == ["Long put down-and-out"]


def test_airbag_and_buffer_are_european():
    for name in ("Airbag certificate", "Buffer certificate"):
        with pytest.raises(NotImplementedError, match="European"):
            pc.find_certificate(name).build({}, T, INITIAL, Monitoring.CONTINUOUS)


@pytest.mark.parametrize(
    ("name", "values", "message"),
    [
        ("Outperformance certificate", {"participation": 0.0}, "participation"),
        ("Barrier outperformance certificate", {"barrier": 100.0}, "below the initial"),
        ("Bonus certificate", {"bonus": 60.0}, "below the bonus"),
        ("Bonus certificate", {"capped": 1.0, "cap": 110.0}, "cap"),
        ("Twin-win certificate", {"barrier": 105.0}, "below the strike"),
        ("Outperformance certificate", {"capped": 1.0, "cap": 100.0}, "above the strike"),
        ("Bonus certificate", {"participation": 50.0, "capped": 1.0, "cap": 125.0}, "reaches"),
        ("Bonus certificate", {"strike": 0.0}, "strike"),
        ("Reverse bonus certificate", {"participation": 50.0}, "participation x reverse"),
        ("Twin-win certificate", {"participation_down": -1.0}, "participation_down"),
        ("Airbag certificate", {"airbag": 100.0}, "airbag level"),
        ("Airbag certificate", {"capped": 1.0, "cap": 100.0}, "cap"),
        ("Buffer certificate", {"buffer": 120.0}, "buffer level"),
        ("Buffer certificate", {"capped": 1.0, "cap": 90.0}, "cap"),
        ("Sprint certificate", {"cap": 100.0}, "cap"),
        ("Discount certificate", {"cap": 0.0}, "cap"),
        ("Reverse bonus certificate", {"bonus": 200.0}, "bonus"),
        ("Reverse bonus certificate", {"barrier": 80.0}, "barrier"),
        ("Reverse bonus certificate", {"barrier": 210.0}, "barrier"),
        ("Reverse bonus certificate", {"capped": 1.0, "cap": 110.0}, "cap"),
        ("Shark warrant (call)", {"barrier": 95.0}, "beyond the strike"),
        ("Shark warrant (put)", {"barrier": 105.0}, "beyond the strike"),
        ("Shark warrant (put)", {"rebate": -1.0}, "rebate"),
    ],
)
def test_invalid_terms_raise(name, values, message):
    with pytest.raises(ValueError, match=message):
        pc.find_certificate(name).build(values, T, INITIAL)


def test_a_shark_warrant_needs_a_direction():
    with pytest.raises(ValueError, match="direction"):
        pc.shark_warrant(
            direction="sideways",
            strike=100.0,
            barrier=130.0,
            participation=100.0,
            rebate=0.0,
            maturity=T,
            initial=INITIAL,
        )


def test_catalog_names_are_unique_and_found():
    names = [spec.name for spec in SPECS]
    assert len(names) == len(set(names))
    with pytest.raises(KeyError):
        pc.find_certificate("Unknown")


def _legs(name, values=None):
    note = pc.find_certificate(name).build(values or {}, T, INITIAL)
    return [(leg.role.split(" (")[0], round(leg.quantity * INITIAL, 6)) for leg in note.legs]


def test_the_usual_decompositions():
    """As issuers describe them: a zero-strike call plus down-and-out puts."""
    zero_strike = ("Zero-strike call", 100.0)
    # twin-win: the underlying and two down-and-out puts at the strike
    assert _legs("Twin-win certificate") == [zero_strike, ("Long puts down-and-out", 200.0)]
    # bonus: the underlying and a down-and-out put at the bonus level
    assert _legs("Bonus certificate") == [zero_strike, ("Long put down-and-out", 100.0)]
    # barrier outperformance: extra calls, and a down-and-out put giving the capital back
    assert _legs("Barrier outperformance certificate") == [
        zero_strike,
        ("Long call", 30.0),
        ("Long put down-and-out", 100.0),
    ]
    # discount: the underlying less a call at the cap
    assert _legs("Discount certificate") == [zero_strike, ("Short call", -100.0)]
    # sprint at 200 %: one more call at the strike, two sold at the cap
    assert _legs("Sprint certificate") == [
        zero_strike,
        ("Long call", 100.0),
        ("Short call", -200.0),
    ]
