"""Same payoff, different packaging: equivalent constructions and what separates them.

Put-call parity, ``C(K) - P(K) = S e^{-qT} - K e^{-rT}``, makes several pairs of positions pay
the same at maturity up to a fixed amount of cash. The two sides of a pair have the same
delta, gamma and vega; what differs is the cash: one side pays a premium, the other receives
one and must post collateral against its worst case, and the side that sells options carries
the obligation to pay when they end in the money.

Each construction is a :class:`~spricer.strategies.Strategy` whose legs may include a
zero-coupon (cash put aside today, ``K e^{-rT}`` for ``K`` at maturity) and the underlying
(``S e^{-qT}``, dividends left out), so that both sides of a pair have exactly the same payoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spricer.bricks import EuropeanOption, Underlying, ZeroCoupon
from spricer.finder import View
from spricer.market.core import Market
from spricer.strategies import Family, Leg, Strategy

GRID_TOP = 3.0  # payoffs are scanned from 0 to 3 times today's spot


@dataclass(frozen=True)
class Construction:
    """One way to build a payoff, and what it implies for the cash.

    Attributes
    ----------
    name
        Name of the construction.
    strategy
        Its legs (options, zero-coupons, underlying).
    obligation
        What the holder commits to, in words.
    """

    name: str
    strategy: Strategy
    obligation: str

    def cash_today(self, market: Market) -> float:
        """Value today in % of the spot: paid when positive, received when negative."""
        return 100.0 * float(self.strategy.value(market)) / market.spot

    def option_premium(self, market: Market) -> float:
        """Value of the option legs only (% of spot): paid when positive, received if not."""
        return (
            100.0
            * sum(
                leg.quantity * float(leg.brick.value(market))
                for leg in self.strategy.legs
                if isinstance(leg.brick, EuropeanOption)
            )
            / market.spot
        )

    def collateral(self, market: Market) -> float:
        """Worst amount the option and stock legs can owe at maturity, in % of spot.

        This is what a counterparty asks to be secured when options are sold; a bond leg of the
        construction can cover it (a cash-secured put, a protected note).
        """
        grid = np.linspace(0.0, GRID_TOP * market.spot, 3001)
        payoff = sum(
            leg.quantity * leg.brick.payoff(grid)
            for leg in self.strategy.legs
            if not isinstance(leg.brick, ZeroCoupon)
        )
        return 100.0 * max(0.0, -float(np.min(np.asarray(payoff)))) / market.spot

    def bond(self) -> float:
        """Cash put aside, paid at maturity (% of spot is up to the caller's spot)."""
        return float(
            sum(leg.quantity for leg in self.strategy.legs if isinstance(leg.brick, ZeroCoupon))
        )


@dataclass(frozen=True)
class Equivalence:
    """Two constructions with the same payoff up to cash, and the lesson of the pair."""

    title: str
    first: Construction
    second: Construction
    lesson: str

    def cash_gap(self, market: Market) -> float:
        """Difference of value today (first minus second), % of spot: the parity cash."""
        return self.first.cash_today(market) - self.second.cash_today(market)


def _strategy(name: str, legs: list[Leg]) -> Strategy:
    return Strategy(name, Family.CUSTOM, tuple(legs))


def _c(k: float, t: float, q: float = 1.0) -> Leg:
    return Leg(EuropeanOption("call", k, t), q)


def _p(k: float, t: float, q: float = 1.0) -> Leg:
    return Leg(EuropeanOption("put", k, t), q)


def _zc(amount: float, t: float) -> Leg:
    return Leg(ZeroCoupon(t), amount)


def equivalences(view: View, spot: float, maturity: float) -> list[Equivalence]:
    """Pairs of constructions relevant to ``view``, struck around ``spot``."""
    t = maturity
    k = spot
    low, high = 0.9 * spot, 1.1 * spot
    width = high - low
    stock = Leg(Underlying(t), 1.0)
    out: list[Equivalence] = []
    if view in (View.BULLISH, View.MILD_BULL):
        out.append(
            Equivalence(
                "Capital protected and bullish: buy a call spread, or sell a put spread",
                Construction(
                    "Bond + long call spread",
                    _strategy(
                        "Bond + long call spread", [_zc(spot, t), _c(low, t), _c(high, t, -1.0)]
                    ),
                    "Pays the call spread premium upfront; can lose nothing beyond it.",
                ),
                Construction(
                    "Bigger bond + short put spread",
                    _strategy(
                        "Bigger bond + short put spread",
                        [_zc(spot + width, t), _p(high, t, -1.0), _p(low, t)],
                    ),
                    "Receives the put spread premium, but is short puts: must post collateral "
                    "for the worst case and pay if the spot ends below the upper strike.",
                ),
                "Both pay the capital plus the rise between the strikes, and have the same "
                "Greeks. The buyer of the call spread pays and is done. The seller of the put "
                "spread is paid, so a larger bond is needed to protect the capital, and the "
                "counterparty asks for collateral: the risk is the same, the funding is not.",
            )
        )
    if view in (View.BEARISH, View.MILD_BEAR):
        out.append(
            Equivalence(
                "Bearish: buy a put spread, or sell a call spread",
                Construction(
                    "Long put spread",
                    _strategy("Long put spread", [_p(high, t), _p(low, t, -1.0)]),
                    "Pays the premium upfront; can lose nothing beyond it.",
                ),
                Construction(
                    "Short call spread + cash",
                    _strategy(
                        "Short call spread + cash",
                        [_c(low, t, -1.0), _c(high, t), _zc(width, t)],
                    ),
                    "Receives the premium, but is short a call: posts collateral and pays if "
                    "the spot ends above the lower strike.",
                ),
                "Same payoff and Greeks once the cash for the spread width is put aside. The "
                "seller collects the premium today but carries the obligation to pay.",
            )
        )
    if view in (View.BULLISH, View.MILD_BULL, View.BIG_MOVE):
        out.append(
            Equivalence(
                "Protected upside: stock plus a put, or a bond plus a call",
                Construction(
                    "Stock + put",
                    _strategy("Stock + put", [stock, _p(k, t)]),
                    "Holds the stock (and its voting rights, without the dividends here) and "
                    "buys insurance below the strike.",
                ),
                Construction(
                    "Bond + call",
                    _strategy("Bond + call", [_zc(k, t), _c(k, t)]),
                    "Holds cash in a bond and buys the upside: the structure of a capital "
                    "protected note.",
                ),
                "Put-call parity: S + P = C + K e^(-rT). A capital protected note is a "
                "protective put in disguise; the bond leg makes the issuer's credit risk part "
                "of the product.",
            )
        )
    if view in (View.RANGE, View.MILD_BULL):
        out.append(
            Equivalence(
                "Income: a covered call, or a put sold with the cash aside",
                Construction(
                    "Covered call",
                    _strategy("Covered call", [stock, _c(k, t, -1.0)]),
                    "Owns the stock and sells the upside above the strike.",
                ),
                Construction(
                    "Cash-secured short put",
                    _strategy("Cash-secured short put", [_zc(k, t), _p(k, t, -1.0)]),
                    "Keeps the strike in cash and sells a put: buys the stock at the strike if "
                    "it falls below. A reverse convertible is this, wrapped in a note.",
                ),
                "Same payoff: the premium is the income, and below the strike the investor ends "
                "up with the stock. Both are short volatility: they suit high implied "
                "volatility.",
            )
        )
    if view is View.BIG_MOVE:
        out.append(
            Equivalence(
                "Big move: a straddle, or two calls and a short stock",
                Construction(
                    "Straddle",
                    _strategy("Straddle", [_c(k, t), _p(k, t)]),
                    "Pays both premiums upfront; can lose nothing beyond them.",
                ),
                Construction(
                    "Two calls + short stock + cash",
                    _strategy(
                        "Two calls + short stock + cash",
                        [_c(k, t, 2.0), Leg(Underlying(t), -1.0), _zc(k, t)],
                    ),
                    "Borrows and sells the stock: owes the stock at maturity, and the lender "
                    "asks for collateral.",
                ),
                "Replacing the put by a call and a short forward gives the same convexity: what "
                "changes is borrowing the stock.",
            )
        )
    return out
