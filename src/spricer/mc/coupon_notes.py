"""Coupon notes (reverse convertibles, autocalls) priced on Monte Carlo paths.

The note is evaluated path by path, date by date: coupon first (with memory), then the call.
Its value splits into four legs evaluated on the same paths, which therefore add up to the
total exactly:

* coupons paid;
* par paid back at a call;
* par paid back at maturity;
* the capital at risk: minus ``100 x (K - P) / K`` on the paths where, at maturity, the
  protection barrier was breached and the worst performance ``P`` ends below the strike ``K``.
  It is the down-and-in put on the worst performance the investor has sold.

Every cash flow is discounted at the risk-free rate plus the issuer's credit spread; the
underlyings keep their risk-free drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spricer._typing import FloatArray
from spricer.market.basket import Basket
from spricer.mc.paths import DEFAULT_MAX_STEP, DEFAULT_PATHS, MCEstimate, PathSet, simulate
from spricer.monitoring import Monitoring
from spricer.products.coupon_notes import CouponNote
from spricer.products.notes import NOTIONAL

LEG_NAMES = (
    "Coupons",
    "Redemption at call",
    "Redemption at maturity",
    "Capital at risk (short down-and-in put)",
)


@dataclass(frozen=True)
class CouponNoteMonteCarlo:
    """Monte Carlo value of a coupon note, its legs and its probabilities.

    Attributes
    ----------
    total
        Value in % of notional.
    legs
        Values of :data:`LEG_NAMES`, in the same order; they add up to ``total``.
    call_probability
        Probability of a call at each observation date (zero where no call is possible).
    expected_life
        Expected life of the note in years (call date, or maturity).
    loss_probability
        Probability of losing capital at maturity.
    breach_probability
        Probability of reaching maturity with the protection barrier breached (the loss
        then depends on where the worst performance ends).
    """

    total: MCEstimate
    legs: tuple[MCEstimate, ...]
    call_probability: tuple[MCEstimate, ...]
    expected_life: MCEstimate
    loss_probability: MCEstimate
    breach_probability: MCEstimate


def simulate_note(
    note: CouponNote,
    basket: Basket,
    *,
    n_paths: int = DEFAULT_PATHS,
    seed: int | None = 0,
    max_step: float = DEFAULT_MAX_STEP,
) -> PathSet:
    """Paths of the basket recorded at the note's dates, its barrier observed as defined."""
    return simulate(
        basket.markets,
        note.times,
        n_paths=n_paths,
        seed=seed,
        correlation=basket.correlation,
        monitoring=note.monitoring,
        monitoring_times=note.barrier_dates(),
        max_step=max_step,
    )


def _breached(note: CouponNote, basket: Basket, paths: PathSet, final: FloatArray) -> FloatArray:
    rows = paths.spots.shape[0]
    if note.barrier_breached:
        return np.ones(rows, dtype=bool)
    if note.monitoring is Monitoring.EUROPEAN:
        return np.asarray(final < note.protection_barrier)
    lows = np.column_stack(
        [paths.extremes(note.maturity, asset)[0] for asset in range(basket.size)]
    )
    return np.asarray(basket.worst_pct(lows) < note.protection_barrier)


@dataclass(frozen=True)
class _CashFlows:
    """Discounted cash flows of a note on every row of a path set."""

    legs: tuple[FloatArray, FloatArray, FloatArray, FloatArray]
    called: tuple[FloatArray, ...]
    life: FloatArray
    losing: FloatArray
    breached_alive: FloatArray


def _cash_flows(
    note: CouponNote, basket: Basket, paths: PathSet, credit_spread: float
) -> _CashFlows:
    if paths.monitoring is not note.monitoring:
        raise ValueError("the paths do not observe the barrier the way the note does")
    rows = paths.spots.shape[0]
    rate = basket.rate + credit_spread
    alive = np.ones(rows, dtype=bool)
    memory = np.full(rows, note.memory_accrued)
    coupons = np.zeros(rows)
    called_cash = np.zeros(rows)
    life = np.full(rows, note.maturity)
    calls: list[FloatArray] = []
    last = len(note.observations) - 1
    for j, obs in enumerate(note.observations):
        df = np.exp(-rate * obs.time)
        worst = basket.worst_pct(paths.spots[:, paths.index(obs.time), :])
        paid = alive & (worst >= obs.coupon_level)
        amount = obs.coupon + (memory if note.memory else 0.0)
        coupons += np.where(paid, amount, 0.0) * df
        if note.memory:
            memory = np.where(paid, 0.0, memory + obs.coupon)
        called = np.zeros(rows, dtype=bool)
        if j < last and obs.autocall_level is not None:
            called = alive & (worst >= obs.autocall_level)
            called_cash += np.where(called, NOTIONAL * df, 0.0)
            life = np.where(called, obs.time, life)
            alive &= ~called
        calls.append(called.astype(np.float64))

    final = basket.worst_pct(paths.spots[:, paths.index(note.maturity), :])
    breached = _breached(note, basket, paths, final)
    df_t = np.exp(-rate * note.maturity)
    losing = alive & breached & (final < note.put_strike)
    loss = np.where(losing, NOTIONAL * (note.put_strike - final) / note.put_strike, 0.0) * df_t
    redemption = np.where(alive, NOTIONAL * df_t, 0.0)
    return _CashFlows(
        legs=(coupons, called_cash, redemption, -loss),
        called=tuple(calls),
        life=life,
        losing=losing.astype(np.float64),
        breached_alive=(alive & breached).astype(np.float64),
    )


def note_samples(
    note: CouponNote, basket: Basket, paths: PathSet, credit_spread: float = 0.0
) -> FloatArray:
    """Discounted value of ``note`` on every row of ``paths`` (feed to ``paths.estimate``)."""
    return np.asarray(np.sum(_cash_flows(note, basket, paths, credit_spread).legs, axis=0))


def evaluate_note(
    note: CouponNote, basket: Basket, paths: PathSet, credit_spread: float = 0.0
) -> CouponNoteMonteCarlo:
    """Value ``note`` on paths simulated by :func:`simulate_note`."""
    flows = _cash_flows(note, basket, paths, credit_spread)
    return CouponNoteMonteCarlo(
        total=paths.estimate(np.sum(flows.legs, axis=0)),
        legs=tuple(paths.estimate(leg) for leg in flows.legs),
        call_probability=tuple(paths.estimate(c) for c in flows.called),
        expected_life=paths.estimate(flows.life),
        loss_probability=paths.estimate(flows.losing),
        breach_probability=paths.estimate(flows.breached_alive),
    )


def price_coupon_note(
    note: CouponNote,
    basket: Basket,
    credit_spread: float = 0.0,
    *,
    n_paths: int = DEFAULT_PATHS,
    seed: int | None = 0,
    max_step: float = DEFAULT_MAX_STEP,
) -> CouponNoteMonteCarlo:
    """Monte Carlo value of ``note`` on ``basket`` under local volatility."""
    paths = simulate_note(note, basket, n_paths=n_paths, seed=seed, max_step=max_step)
    return evaluate_note(note, basket, paths, credit_spread)
