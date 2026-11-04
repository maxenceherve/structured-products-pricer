"""Solve for: the term that gives a product a target price.

A structurer fixes every term but one (the coupon, a barrier, the participation) and solves
for the value that makes the note worth its target, typically the issue price minus the
margin. The root is found by Brent's method on the price as a function of that term;
:func:`solve_on_grid` also finds the interval to search, from any product's price function.

For a Monte Carlo price the paths are simulated once and reused for every trial value (common
random numbers): the price is then a deterministic function of the term, and the root is the
term at which these paths give the target. Its uncertainty is the standard error of the price
divided by the slope of the price in the term.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
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


def solve_on_grid(
    price: Callable[[float], float],
    target: float,
    grid: ArrayLike,
    current: float,
    what: str = "value",
    tolerance: float = 0.01,
) -> Solution:
    """Value nearest to ``current`` at which ``price`` equals ``target``.

    A price need not be monotonic in a term (a strike of a strangle), and a term is only valid
    over part of the grid (a barrier below a strike): the search walks away from ``current``,
    on each side up to the first invalid value (``price`` raises ValueError), and solves in the
    first interval where the price crosses the target.

    Raises
    ------
    ValueError
        If ``current`` is invalid, if no valid value reaches the target, or if the price jumps
        over the target (by more than ``tolerance``) instead of crossing it.
    """
    start = float(price(current))
    if start == target:
        return Solution(current, start)
    nodes = np.unique(np.asarray(grid, dtype=np.float64))
    seen = [start]
    sides = [
        (current, start, iter(nodes[nodes > current])),
        (current, start, iter(nodes[nodes < current][::-1])),
    ]
    while sides:
        remaining = []
        for x0, y0, nodes_left in sides:
            x1 = next(nodes_left, None)
            if x1 is None:
                continue
            try:
                y1 = float(price(float(x1)))
            except ValueError:
                continue
            seen.append(y1)
            if (y0 - target) * (y1 - target) <= 0.0:
                low, high = sorted((x0, float(x1)))
                value = float(brentq(lambda x: price(x) - target, low, high, xtol=_XTOL))
                reached = float(price(value))
                if abs(reached - target) > tolerance:
                    raise ValueError(
                        f"the price jumps over the target when the {what} crosses {value:g}: "
                        f"it reaches {reached:.2f}, not {target:.2f}"
                    )
                return Solution(value, reached)
            remaining.append((float(x1), y1, nodes_left))
        sides = remaining
    raise ValueError(
        f"no {what} gives a price of {target:g}: from the current terms the price only goes "
        f"from {min(seen):.2f} to {max(seen):.2f}"
    )


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
