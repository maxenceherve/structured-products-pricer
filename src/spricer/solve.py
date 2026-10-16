"""Solve for: the term that gives a product a target price.

A structurer fixes every term but one (the coupon, a barrier, the participation) and solves
for the value that makes the note worth its target, typically the issue price minus the
margin. The root is found by Brent's method on the price as a function of that term.

For a Monte Carlo price the paths are simulated once and reused for every trial value (common
random numbers): the price is then a deterministic function of the term, and the root is the
term at which these paths give the target. Its uncertainty is the standard error of the price
divided by the slope of the price in the term.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from spricer.market.basket import Basket
from spricer.market.core import Market
from spricer.mc.coupon_notes import CouponNoteMonteCarlo, evaluate_note, simulate_note
from spricer.mc.paths import DEFAULT_MAX_STEP, DEFAULT_PATHS
from spricer.products.coupon_notes import CouponNote
from spricer.products.notes import Note

_XTOL = 1e-6


@dataclass(frozen=True)
class Solution:
    """Solved term and the price it gives.

    Attributes
    ----------
    value
        Term that prices the product at the target.
    price
        Price at the solved term (% of notional).
    std_error
        Uncertainty of ``value`` from the Monte Carlo noise (None for a closed form).
    """

    value: float
    price: float
    std_error: float | None = None


def _root(f: Callable[[float], float], low: float, high: float, what: str) -> float:
    f_low, f_high = f(low), f(high)
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if np.sign(f_low) == np.sign(f_high):
        raise ValueError(
            f"no {what} between {low:g} and {high:g} gives the target: the price goes from "
            f"{f_low:+.2f} to {f_high:+.2f} away from it over that range"
        )
    return float(brentq(f, low, high, xtol=_XTOL))


def solve_note(
    build: Callable[[float], Note],
    market: Market,
    target: float,
    bounds: tuple[float, float],
    credit_spread: float = 0.0,
    what: str = "value",
) -> Solution:
    """Term of a closed-form note (e.g. the participation) that prices it at ``target``."""
    value = _root(
        lambda x: float(build(x).value(market, credit_spread)) - target, *bounds, what=what
    )
    return Solution(value, float(build(value).value(market, credit_spread)))


def solve_coupon_note(
    build: Callable[[float], CouponNote],
    basket: Basket,
    target: float,
    bounds: tuple[float, float],
    credit_spread: float = 0.0,
    *,
    what: str = "value",
    n_paths: int = DEFAULT_PATHS,
    seed: int | None = 0,
    max_step: float = DEFAULT_MAX_STEP,
) -> tuple[Solution, CouponNoteMonteCarlo]:
    """Term of a coupon note that prices it at ``target`` on one set of paths.

    Every trial note must share the dates and the barrier observation of ``build`` at the
    middle of ``bounds`` (the solved term must not move the schedule).
    """
    low, high = bounds
    reference = build(0.5 * (low + high))
    paths = simulate_note(reference, basket, n_paths=n_paths, seed=seed, max_step=max_step)

    def result(x: float) -> CouponNoteMonteCarlo:
        note = build(x)
        if note.times != reference.times or note.monitoring is not reference.monitoring:
            raise ValueError("the solved term must not change the dates or the observation")
        return evaluate_note(note, basket, paths, credit_spread)

    value = _root(lambda x: result(x).total.value - target, low, high, what)
    solved = result(value)
    h = 1e-2 * (high - low)  # wide enough to see the slope of a barrier term
    slope = (result(min(value + h, high)).total.value - result(max(value - h, low)).total.value) / (
        min(value + h, high) - max(value - h, low)
    )
    std_error = solved.total.std_error / abs(slope) if slope else None
    return Solution(value, solved.total.value, std_error), solved
