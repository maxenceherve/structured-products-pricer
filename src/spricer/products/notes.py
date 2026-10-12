"""Structured notes: a payoff in percent of notional, replicated by bricks.

A :class:`Note` is written in percent of its notional and in percent of the underlying's
initial level, the way a term sheet is. Each leg is a brick held in a quantity per note of
notional 100, with a role that says what it does ("Capital protection (bond)", "Bonus
digital"...). Strikes are converted once to absolute levels (``percent x initial``), so the
volatility smile is always read at the true moneyness, whatever the current spot.

The note is a liability of its issuer: every cash flow, the bond part and the options alike,
is discounted at the risk-free rate plus the issuer's credit spread, while the underlying keeps
its risk-free drift. With a spread ``s`` the value is ``e^{-sT}`` times the risk-free value of
the replicating bricks.

Every note also carries its payoff as a direct formula of the final performance. The tests
check that the bricks replicate it exactly, independently of any pricing model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray
from spricer.bricks import Brick
from spricer.conventions import Greeks
from spricer.market.core import Market
from spricer.monitoring import Monitoring, require_european

NOTIONAL = 100.0  # quantities are per note of notional 100, so values read in percent


class LevelKind(StrEnum):
    """How a named level is drawn on the payoff chart."""

    PERFORMANCE = "performance"  # a vertical line: a level of the underlying (strike, barrier)
    PAYOFF = "payoff"  # a horizontal line: a level of the payoff (floor, bonus)


@dataclass(frozen=True)
class Level:
    """A named level of the term sheet, in percent (of initial or of notional)."""

    name: str
    value_pct: float
    kind: LevelKind


@dataclass(frozen=True)
class NoteLeg:
    """A brick held in ``quantity`` units per note of notional 100, with its role."""

    role: str
    brick: Brick
    quantity: float


@dataclass(frozen=True)
class Note:
    """A structured note replicated by bricks.

    Attributes
    ----------
    name
        Product name.
    maturity
        Maturity in years.
    initial
        Initial level of the underlying (the 100 % of the term sheet).
    legs
        Replicating bricks; the bond part is the leg holding a zero-coupon.
    formula
        Payoff at maturity in % of notional, as a function of the final performance in %
        of the initial level. Used only to check the replication and to draw the payoff.
    levels
        Named levels for the chart.
    monitoring
        Barrier observation, or None for a note without barrier.
    """

    name: str
    maturity: float
    initial: float
    legs: tuple[NoteLeg, ...]
    formula: Callable[[FloatArray], FloatArray]
    levels: tuple[Level, ...] = ()
    monitoring: Monitoring | None = None

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("a note needs at least one leg")
        if not (np.isfinite(self.initial) and self.initial > 0.0):
            raise ValueError("the initial level must be positive")
        if self.monitoring is not None:
            require_european(self.monitoring)

    def discount_spread(self, credit_spread: float) -> float:
        """Extra discount factor ``e^{-sT}`` for the issuer's credit spread."""
        return float(np.exp(-credit_spread * self.maturity))

    def value(
        self, market: Market, credit_spread: float = 0.0, spot: ArrayLike | None = None
    ) -> FloatArray:
        """Value in % of notional."""
        risk_free = sum(
            (leg.quantity * leg.brick.value(market, spot) for leg in self.legs),
            start=np.zeros(()),
        )
        return np.asarray(risk_free * self.discount_spread(credit_spread), dtype=np.float64)

    def leg_values(self, market: Market, credit_spread: float = 0.0) -> list[float]:
        """Value of each leg in % of notional (they add up to :meth:`value`)."""
        df = self.discount_spread(credit_spread)
        return [leg.quantity * float(leg.brick.value(market)) * df for leg in self.legs]

    def leg_greeks(self, market: Market, credit_spread: float = 0.0) -> list[Greeks]:
        """Raw Greeks of each leg, in % of notional per unit of the underlying."""
        df = self.discount_spread(credit_spread)
        out = []
        for leg in self.legs:
            g = leg.brick.greeks(market) * leg.quantity
            v = leg.quantity * leg.brick.value(market)
            # theta = -d/dT [e^{-sT} V] = e^{-sT} (theta_V + s V)
            out.append(
                Greeks(
                    delta=g.delta * df,
                    gamma=g.gamma * df,
                    vega=g.vega * df,
                    theta=(g.theta + credit_spread * v) * df,
                    rho=g.rho * df,
                )
            )
        return out

    def greeks(self, market: Market, credit_spread: float = 0.0) -> Greeks:
        """Raw Greeks of the note (sum of the legs')."""
        total = Greeks.zero()
        for g in self.leg_greeks(market, credit_spread):
            total = total + g
        return total

    def bricks_payoff(self, performance_pct: ArrayLike) -> FloatArray:
        """Payoff of the replicating bricks, in % of notional, at a final performance in %."""
        s_t = np.asarray(performance_pct, dtype=np.float64) / 100.0 * self.initial
        return sum((leg.quantity * leg.brick.payoff(s_t) for leg in self.legs), start=np.zeros(()))

    def payoff(self, performance_pct: ArrayLike) -> FloatArray:
        """Payoff at maturity in % of notional (term-sheet formula)."""
        return self.formula(np.asarray(performance_pct, dtype=np.float64))

    @property
    def strikes(self) -> tuple[float, ...]:
        """Distinct absolute strikes and barriers of the legs (where the smile is read)."""
        levels = {leg.brick.strike for leg in self.legs} | {
            getattr(leg.brick, "barrier", None) for leg in self.legs
        }
        return tuple(sorted(k for k in levels if k))
