"""Coupon notes: schedules, builders, replication, Monte Carlo legs, probabilities, solver."""

import math
from typing import Any

import numpy as np
import pytest

from spricer.bricks import BarrierOption, DigitalOption, ZeroCoupon
from spricer.market import Basket, FlatVol, Market
from spricer.mc.coupon_notes import LEG_NAMES, evaluate_note, price_coupon_note, simulate_note
from spricer.monitoring import Monitoring
from spricer.products import capital_protection as cp
from spricer.products.coupon_notes import (
    COUPON_NOTE_CATALOG,
    CouponNote,
    LifeState,
    Observation,
    athena,
    autocallable_reverse_convertible,
    barrier_reverse_convertible,
    find_coupon_note,
    phoenix,
    reverse_convertible,
    schedule,
)
from spricer.solve import solve_coupon_note, solve_note

N_SE = 4.0
FLAT = Market(100.0, 0.03, 0.02, FlatVol(0.22))


def within(est, exact, n_se=N_SE):
    return abs(est.value - exact) <= n_se * est.std_error


# --- Schedules and builders ------------------------------------------------------------------


def test_schedule_counts_from_issue_and_drops_past_dates():
    assert schedule(1.0, 4) == [(1, 0.25), (2, 0.5), (3, 0.75), (4, 1.0)]
    seasoned = schedule(2.0, 2, elapsed=0.75)
    assert [k for k, _ in seasoned] == [2, 3, 4]
    np.testing.assert_allclose([t for _, t in seasoned], [0.25, 0.75, 1.25])


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0.0, 4), "maturity"),
        ((1.0, 3), "frequency"),
        ((1.1, 4), "whole number"),
        ((1.0, 4, 1.0), "elapsed"),
    ],
)
def test_invalid_schedules_raise(args, message):
    with pytest.raises(ValueError, match=message):
        schedule(*args)


def test_phoenix_counts_non_call_and_step_down_from_issue():
    new = phoenix(maturity=2.0, frequency=4, coupon_pa=8.0, non_call=2, step_down=5.0)
    calls = [o.autocall_level for o in new.observations]
    assert calls == [None, None, 100.0, 95.0, 90.0, 85.0, 80.0, None]
    assert all(o.coupon == 2.0 and o.coupon_level == 70.0 for o in new.observations)
    alive = phoenix(
        maturity=2.0,
        frequency=4,
        coupon_pa=8.0,
        non_call=2,
        step_down=5.0,
        state=LifeState(elapsed=1.0, memory_accrued=4.0),
    )
    assert [o.autocall_level for o in alive.observations] == [90.0, 85.0, 80.0, None]
    assert alive.memory_accrued == 4.0
    no_memory = phoenix(
        maturity=1.0, frequency=4, coupon_pa=8.0, memory=False, state=LifeState(memory_accrued=4.0)
    )
    assert no_memory.memory_accrued == 0.0


def test_athena_coupons_follow_the_autocall_level():
    note = athena(maturity=1.0, frequency=4, coupon_pa=10.0, step_down=5.0, non_call=1)
    assert [o.autocall_level for o in note.observations] == [None, 100.0, 95.0, None]
    assert [o.coupon_level for o in note.observations] == [100.0, 100.0, 95.0, 95.0]
    assert note.memory
    assert (
        athena(maturity=1.0, frequency=4, coupon_pa=10.0, final_coupon_level=60.0)
        .observations[-1]
        .coupon_level
        == 60.0
    )


def test_reverse_convertibles_have_certain_coupons_and_no_call():
    rc = reverse_convertible(maturity=1.0, frequency=2, coupon_pa=10.0, strike=95.0)
    assert rc.protection_barrier == rc.put_strike == 95.0
    assert not rc.callable
    assert all(o.coupon_level == 0.0 for o in rc.observations)
    brc = barrier_reverse_convertible(
        maturity=1.0, frequency=1, coupon_pa=6.0, barrier=60.0, monitoring=Monitoring.DISCRETE
    )
    assert brc.daily_observations == 252
    assert brc.barrier_dates() is not None
    assert rc.barrier_dates() is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"observations": ()}, "at least one observation"),
        ({"observations": (Observation(1.0, 1.0, 0.0), Observation(0.5, 1.0, 0.0))}, "increasing"),
        ({"protection_barrier": 0.0}, "protection barrier"),
        ({"put_strike": -1.0}, "put strike"),
        ({"memory_accrued": -1.0}, "non-negative"),
        ({"memory_accrued": 2.0}, "with memory"),
    ],
)
def test_invalid_coupon_notes_raise(kwargs, message):
    args = {"name": "x", "observations": (Observation(1.0, 1.0, 0.0),), "protection_barrier": 60.0}
    with pytest.raises(ValueError, match=message):
        CouponNote(**(args | kwargs))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"time": 0.0}, "times"),
        ({"coupon": -1.0}, "coupons"),
        ({"coupon_level": -1.0}, "coupon levels"),
        ({"autocall_level": 0.0}, "autocall levels"),
    ],
)
def test_invalid_observations_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Observation(**({"time": 1.0, "coupon": 1.0, "coupon_level": 0.0} | kwargs))


def test_catalog_builds_every_note_and_is_searchable():
    for spec in COUPON_NOTE_CATALOG:
        note = spec.build({}, 2.0, 4)
        assert note.maturity == 2.0
        assert note.levels[0].name == "Protection barrier"
        assert find_coupon_note(spec.name) is spec
        assert spec.param("coupon_pa").solvable
    with pytest.raises(KeyError):
        find_coupon_note("Unknown")


# --- Payoff at maturity and replication --------------------------------------------------------


def test_maturity_payoff_loses_below_the_strike_only_after_a_breach():
    brc = barrier_reverse_convertible(maturity=1.0, frequency=1, coupon_pa=6.0, barrier=60.0)
    np.testing.assert_allclose(
        brc.maturity_payoff([50.0, 70.0, 120.0], False), [56.0, 106.0, 106.0]
    )
    path = barrier_reverse_convertible(
        maturity=1.0, frequency=1, coupon_pa=6.0, barrier=60.0, monitoring=Monitoring.CONTINUOUS
    )
    np.testing.assert_allclose(path.maturity_payoff([70.0, 120.0], True), [76.0, 106.0])
    np.testing.assert_allclose(path.maturity_payoff([70.0, 120.0], False), [106.0, 106.0])
    phx = phoenix(maturity=1.0, frequency=4, coupon_pa=8.0, coupon_level=70.0)
    np.testing.assert_allclose(phx.maturity_payoff([65.0, 75.0], False), [100.0, 102.0])


@pytest.mark.parametrize(
    "note",
    [
        reverse_convertible(maturity=2.0, frequency=4, coupon_pa=9.0, strike=95.0),
        barrier_reverse_convertible(maturity=2.0, frequency=2, coupon_pa=6.0, barrier=65.0),
        barrier_reverse_convertible(
            maturity=2.0, frequency=2, coupon_pa=6.0, barrier=65.0, monitoring=Monitoring.CONTINUOUS
        ),
        barrier_reverse_convertible(
            maturity=2.0,
            frequency=2,
            coupon_pa=6.0,
            barrier=65.0,
            state=LifeState(barrier_breached=True),
        ),
    ],
    ids=["RC", "BRC European", "BRC continuous", "BRC breached"],
)
def test_bricks_replicate_reverse_convertibles(note):
    replication = note.replication(100.0)
    assert replication is not None
    grid = np.linspace(1.0, 200.0, 400)
    np.testing.assert_allclose(replication.bricks_payoff(grid), replication.payoff(grid), atol=1e-9)


def test_callable_or_conditional_notes_have_no_closed_form():
    assert phoenix(maturity=1.0, frequency=4, coupon_pa=8.0).replication(100.0) is None
    conditional = CouponNote("x", (Observation(1.0, 5.0, 80.0),), protection_barrier=60.0)
    assert conditional.replication(100.0) is None


# --- Monte Carlo against closed forms ----------------------------------------------------------


@pytest.mark.parametrize(
    "note",
    [
        reverse_convertible(maturity=2.0, frequency=4, coupon_pa=9.0, strike=95.0),
        barrier_reverse_convertible(maturity=3.0, frequency=1, coupon_pa=6.0, barrier=60.0),
    ],
    ids=["RC", "BRC"],
)
def test_reverse_convertibles_reprice_their_bricks_under_the_smile(market, note):
    mc = price_coupon_note(note, Basket.single(market), 0.01, n_paths=60_000, seed=2)
    replication = note.replication(100.0)
    assert replication is not None
    exact = float(replication.value(market, 0.01))
    assert within(mc.total, exact)
    assert sum(leg.value for leg in mc.legs) == pytest.approx(mc.total.value, abs=1e-9)
    assert mc.legs[1].value == 0.0  # never called
    assert mc.expected_life.value == pytest.approx(note.maturity)


@pytest.mark.parametrize("monitoring", [Monitoring.CONTINUOUS, Monitoring.DISCRETE])
def test_path_barriers_match_their_closed_form_under_a_flat_volatility(monitoring):
    note = barrier_reverse_convertible(
        maturity=1.0, frequency=4, coupon_pa=6.0, barrier=70.0, monitoring=monitoring
    )
    mc = price_coupon_note(note, Basket.single(FLAT), n_paths=100_000, seed=3)
    replication = note.replication(100.0)
    assert replication is not None
    assert within(mc.total, float(replication.value(FLAT)))


def test_a_single_date_phoenix_is_a_digital_a_bond_and_a_put(market):
    note = phoenix(maturity=1.0, frequency=1, coupon_pa=7.0, coupon_level=80.0, barrier=65.0)
    mc = price_coupon_note(note, Basket.single(market), n_paths=100_000, seed=4)
    exact = (
        7.0 * float(DigitalOption("call", 80.0, 1.0).value(market))
        + 100.0 * float(ZeroCoupon(1.0).value(market))
        - float(BarrierOption("put", "down-and-in", 100.0, 65.0, 1.0).value(market))
    )
    assert within(mc.total, exact)


def test_without_call_and_coupon_barrier_a_phoenix_is_a_barrier_reverse_convertible():
    basket = Basket.single(FLAT)
    brc = barrier_reverse_convertible(maturity=2.0, frequency=4, coupon_pa=6.0, barrier=60.0)
    phx = phoenix(
        maturity=2.0, frequency=4, coupon_pa=6.0, coupon_level=0.0, autocall_level=1e9, barrier=60.0
    )
    paths = simulate_note(brc, basket, n_paths=10_000, seed=5)
    a, b = evaluate_note(brc, basket, paths), evaluate_note(phx, basket, paths)
    assert b.total.value == pytest.approx(a.total.value, abs=1e-10)
    assert all(c.value == 0.0 for c in b.call_probability)


def test_autocallable_reverse_convertibles_are_phoenixes_with_certain_coupons_and_no_memory():
    basket = Basket.single(FLAT)
    terms: dict[str, Any] = {
        "maturity": 2.0,
        "frequency": 4,
        "coupon_pa": 7.0,
        "step_down": 2.0,
        "non_call": 1,
    }
    abrc = autocallable_reverse_convertible(**terms, barrier=60.0)
    arc = autocallable_reverse_convertible(**terms, strike=95.0)
    assert abrc.name == "Autocallable barrier reverse convertible"
    assert arc.name == "Autocallable reverse convertible"
    assert arc.protection_barrier == arc.put_strike == 95.0
    assert all(o.coupon_level == 0.0 for o in abrc.observations)
    assert [o.autocall_level for o in abrc.observations][:3] == [None, 100.0, 98.0]
    assert not abrc.memory
    assert abrc.callable
    phx = phoenix(**terms, coupon_level=0.0, barrier=60.0, memory=False)
    paths = simulate_note(abrc, basket, n_paths=10_000, seed=5)
    called = evaluate_note(abrc, basket, paths)
    assert called.total.value == pytest.approx(evaluate_note(phx, basket, paths).total.value)
    assert sum(c.value for c in called.call_probability) > 0.3
    assert called.expected_life.value < 2.0
    # an autocall that is never reached leaves the barrier reverse convertible
    never = autocallable_reverse_convertible(**terms, barrier=60.0, autocall_level=1e9)
    brc = barrier_reverse_convertible(maturity=2.0, frequency=4, coupon_pa=7.0, barrier=60.0)
    assert evaluate_note(never, basket, paths).total.value == pytest.approx(
        evaluate_note(brc, basket, paths).total.value, abs=1e-10
    )
    # and the coupons stop at the call: fewer coupons are paid than on the plain note
    assert called.legs[0].value < evaluate_note(brc, basket, paths).legs[0].value


def test_memory_pays_at_least_as_much_and_a_breach_costs(market):
    basket = Basket.single(market)
    with_memory = phoenix(maturity=3.0, frequency=4, coupon_pa=8.0, memory=True)
    without = phoenix(maturity=3.0, frequency=4, coupon_pa=8.0, memory=False)
    paths = simulate_note(with_memory, basket, n_paths=20_000, seed=6)
    v_mem = evaluate_note(with_memory, basket, paths).total.value
    v_none = evaluate_note(without, basket, paths).total.value
    assert v_mem > v_none
    accrued = phoenix(maturity=3.0, frequency=4, coupon_pa=8.0, state=LifeState(memory_accrued=6.0))
    assert evaluate_note(accrued, basket, paths).total.value > v_mem
    breached = phoenix(
        maturity=3.0, frequency=4, coupon_pa=8.0, state=LifeState(barrier_breached=True)
    )
    result = evaluate_note(breached, basket, paths)
    assert result.total.value < v_mem
    assert result.breach_probability.value == pytest.approx(
        1.0 - sum(c.value for c in result.call_probability)
    )


def test_call_probabilities_and_expected_life_are_consistent(market):
    note = athena(maturity=3.0, frequency=4, coupon_pa=8.0, non_call=3)
    mc = price_coupon_note(note, Basket.single(market), n_paths=20_000, seed=7)
    probabilities = [c.value for c in mc.call_probability]
    assert probabilities[:3] == [0.0, 0.0, 0.0]
    assert probabilities[-1] == 0.0
    assert 0.0 < sum(probabilities) < 1.0
    assert 1.0 <= mc.expected_life.value < 3.0
    assert 0.0 < mc.loss_probability.value <= mc.breach_probability.value
    assert LEG_NAMES[0] == "Coupons"


def test_worst_of_is_cheaper_than_each_underlying(market):
    other = Market(50.0, market.rate, 0.01, FlatVol(0.30))
    worst = Basket.of([market, other], [100.0, 50.0], ["A", "B"], [[1.0, 0.5], [0.5, 1.0]])
    note = phoenix(maturity=2.0, frequency=4, coupon_pa=8.0)
    wo = price_coupon_note(note, worst, n_paths=40_000, seed=8).total
    for single in (Basket.single(market), Basket.single(other)):
        mono = price_coupon_note(note, single, n_paths=40_000, seed=8).total
        assert wo.value < mono.value - N_SE * (wo.std_error + mono.std_error)


def test_paths_must_observe_the_barrier_like_the_note():
    basket = Basket.single(FLAT)
    european = barrier_reverse_convertible(maturity=1.0, frequency=1, coupon_pa=5.0, barrier=60.0)
    continuous = barrier_reverse_convertible(
        maturity=1.0, frequency=1, coupon_pa=5.0, barrier=60.0, monitoring=Monitoring.CONTINUOUS
    )
    paths = simulate_note(european, basket, n_paths=1_000)
    with pytest.raises(ValueError, match="observe the barrier"):
        evaluate_note(continuous, basket, paths)


# --- Basket ------------------------------------------------------------------------------------


def test_basket_reads_the_worst_performance():
    other = Market(50.0, FLAT.rate, 0.0, FlatVol(0.3))
    basket = Basket.of([FLAT, other], [100.0, 40.0], ["A", "B"], np.eye(2))
    assert basket.size == 2
    assert basket.rate == FLAT.rate
    assert basket.today_worst_pct() == 100.0
    np.testing.assert_allclose(basket.worst_pct([[90.0, 50.0], [120.0, 30.0]]), [90.0, 75.0])
    assert Basket.single(FLAT, 80.0).today_worst_pct() == 125.0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (((), (), ()), "at least one"),
        (((FLAT,), (100.0, 1.0), ("A",)), "one initial level"),
        (((FLAT,), (0.0,), ("A",)), "positive"),
        (((FLAT, FLAT), (1.0, 1.0), ("A", "B")), "correlation"),
    ],
)
def test_invalid_baskets_raise(args, message):
    with pytest.raises(ValueError, match=message):
        Basket(*args)
    with pytest.raises(ValueError, match="same rate"):
        Basket.of([FLAT, Market(1.0, 0.05, 0.0, FlatVol(0.2))], [1.0, 1.0], ["A", "B"], np.eye(2))


# --- Solve for ---------------------------------------------------------------------------------


def test_solving_the_coupon_prices_the_note_at_the_target(market):
    spec = find_coupon_note("Phoenix autocall")
    basket = Basket.single(market)
    solution, result = solve_coupon_note(
        lambda c: spec.build({"coupon_pa": c}, 3.0, 4),
        basket,
        98.0,
        (0.0, 30.0),
        n_paths=20_000,
        what="coupon",
    )
    assert solution.price == pytest.approx(98.0, abs=1e-4)
    assert result.total.value == pytest.approx(98.0, abs=1e-4)
    assert 0.0 < solution.value < 30.0
    assert solution.std_error is not None
    assert 0.0 < solution.std_error < 0.5


def test_solving_a_barrier_and_an_unreachable_target(market):
    spec = find_coupon_note("Barrier reverse convertible")
    basket = Basket.single(market)
    solution, _ = solve_coupon_note(
        lambda b: spec.build({"barrier": b}, 2.0, 4), basket, 100.0, (30.0, 95.0), n_paths=20_000
    )
    assert 30.0 < solution.value < 95.0
    assert solution.price == pytest.approx(100.0, abs=0.05)
    with pytest.raises(ValueError, match="no coupon between"):
        solve_coupon_note(
            lambda c: spec.build({"coupon_pa": c}, 2.0, 4),
            basket,
            150.0,
            (0.0, 10.0),
            n_paths=2_000,
            what="coupon",
        )
    with pytest.raises(ValueError, match="must not change the dates"):
        solve_coupon_note(
            lambda t: spec.build({}, round(t), 4), basket, 100.0, (1.0, 3.0), n_paths=2_000
        )


def test_solving_the_participation_of_a_capital_protected_note(market):
    spec = cp.find_note("Participation note")
    solution = solve_note(
        lambda p: spec.build({"participation": p}, 3.0, 100.0), market, 98.0, (0.0, 300.0)
    )
    assert solution.price == pytest.approx(98.0, abs=1e-6)
    assert solution.std_error is None
    edge = solve_note(
        lambda p: spec.build({"participation": p}, 3.0, 100.0),
        market,
        float(spec.build({"participation": 0.0}, 3.0, 100.0).value(market)),
        (0.0, 50.0),
    )
    assert edge.value == 0.0
    assert math.isfinite(edge.price)
    at_high = float(spec.build({"participation": 50.0}, 3.0, 100.0).value(market))
    top = solve_note(
        lambda p: spec.build({"participation": p}, 3.0, 100.0), market, at_high, (0.0, 50.0)
    )
    assert top.value == 50.0
