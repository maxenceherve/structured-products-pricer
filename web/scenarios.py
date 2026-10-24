"""What a product pays at maturity for a few final levels of the underlying: the payoff tables.

Each function returns the levels (in % of today's spot for an option, of the initial level for
a note) and named rows of amounts in % of the notional, ready for a table under the payoff
chart or in the product sheet.
"""

from __future__ import annotations

import numpy as np

from spricer.bricks import BarrierOption
from spricer.monitoring import Monitoring
from spricer.products.coupon_notes import CouponNote
from web.notes import NotePosition
from web.options import Position
from web.options import breakevens as crossings

Rows = list[tuple[str, list[float]]]


def levels_grid(key: list[float], low: float, high: float, step: float = 10.0) -> list[float]:
    """A regular grid of levels with the product's own levels added, sorted and distinct."""
    grid = set(np.round(np.arange(low, high + 1e-9, step), 6).tolist())
    grid |= {round(k, 2) for k in key if low <= k <= high}
    return sorted(grid)


def option_scenarios(position: Position) -> tuple[list[float], Rows]:
    """Paid at maturity and P&L (paid minus the premium compounded) of an option position."""
    market, strategy = position.market, position.strategy
    key = list(position.strikes_pct)
    if position.barrier_pct is not None:
        key.append(position.barrier_pct)
    levels = levels_grid(key, 60.0, 140.0)
    spots = np.array(levels) / 100.0 * market.spot
    scale = 100.0 / market.spot
    premium = float(strategy.value(market)) * float(np.exp(market.rate * strategy.maturity))
    brick = strategy.legs[0].brick
    if isinstance(brick, BarrierOption) and brick.monitoring is not Monitoring.EUROPEAN:
        quantity = strategy.legs[0].quantity
        sign = 1.0 if brick.kind == "call" else -1.0
        vanilla = quantity * np.maximum(sign * (spots - brick.strike), 0.0)
        beyond = spots >= brick.barrier if brick.is_up else spots < brick.barrier
        never = np.where(beyond, np.nan, 0.0 if brick.is_in else vanilla)
        touched = vanilla if brick.is_in else np.zeros_like(spots)
        return levels, [
            ("Paid if the barrier was never touched", list(scale * never)),
            ("Paid if it was touched", list(scale * touched)),
            ("P&L if never touched", list(scale * (never - premium))),
            ("P&L if touched", list(scale * (touched - premium))),
        ]
    paid = np.asarray(strategy.payoff(spots), dtype=np.float64)
    return levels, [
        ("Paid at maturity", list(scale * paid)),
        ("P&L at maturity", list(scale * (paid - premium))),
    ]


def note_scenarios(position: NotePosition) -> tuple[list[float], Rows]:
    """Paid at maturity by a note, and the gain or loss on its price today."""
    note = position.note
    levels = levels_grid([lv.value_pct for lv in note.levels], 40.0, 160.0)
    paid = np.asarray(note.payoff(np.array(levels)), dtype=np.float64)
    price = position.value()
    return levels, [
        ("Paid at maturity", list(paid)),
        ("Gain on the price paid", list(paid - price)),
    ]


def coupon_scenarios(note: CouponNote) -> tuple[list[float], Rows]:
    """Paid at maturity by a coupon note never called (both cases for a path barrier)."""
    levels = levels_grid([lv.value_pct for lv in note.levels], 40.0, 160.0)
    perf = np.array(levels)
    never = np.asarray(note.maturity_payoff(perf, False), dtype=np.float64)
    if note.monitoring is Monitoring.EUROPEAN or note.barrier_breached:
        return levels, [("Paid at maturity", list(never))]
    breached = np.asarray(note.maturity_payoff(perf, True), dtype=np.float64)
    return levels, [
        ("Paid, barrier never breached", list(never)),
        ("Paid, barrier breached during the life", list(breached)),
    ]


def breakevens(levels: list[float], values: list[float]) -> list[float]:
    """Levels where a row of the table crosses zero (linear interpolation)."""
    x, y = np.array(levels), np.array(values)
    keep = ~np.isnan(y)
    return crossings(x[keep], y[keep])


def option_breakevens(position: Position) -> list[float]:
    """Spots (% of today) where the P&L at maturity crosses zero.

    On a fine grid, so that a payoff that jumps (a digital) crosses zero at its jump; for a
    barrier observed along the path, the P&L if the barrier was never touched.
    """
    levels, rows = option_scenarios(position)
    if len(rows) > 2:
        return breakevens(levels, rows[2][1])
    market, strategy = position.market, position.strategy
    growth = float(np.exp(market.rate * strategy.maturity))
    fine = np.linspace(50.0, 150.0, 20001)
    paid = 100.0 * np.asarray(strategy.payoff(fine / 100.0 * market.spot), dtype=float)
    return breakevens(list(fine), list(paid / market.spot - position.price_pct() * growth))


def note_breakevens(position: NotePosition) -> list[float]:
    """Final levels (% of initial) where what the note pays equals its price today."""
    note = position.note
    top = max(160.0, 1.2 * max(lv.value_pct for lv in note.levels))
    fine = np.linspace(0.0, top, 40001)
    gain = np.asarray(note.payoff(fine), dtype=float) - position.value()
    return breakevens(list(fine), list(gain))
