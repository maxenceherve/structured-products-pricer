"""Vanilla option strategies, grouped by family.

The families follow Hull, *Options, Futures, and Other Derivatives*, ch. 12 ("Trading strategies
involving options"). A strategy is a list of legs (a brick and a quantity) sharing one maturity.
Its value, payoff and Greeks are the sums of those of its legs, and :meth:`Strategy.decompose`
returns the leg-by-leg breakdown: the price of each brick and its contribution.

Strikes are absolute levels. :data:`CATALOG` lists every predefined strategy with default strikes
in percent of spot, which is what the app reads to build its menus.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray
from spricer.bricks import Brick, DigitalOption, EuropeanOption, Underlying
from spricer.conventions import Greeks
from spricer.market.core import Market


class Family(StrEnum):
    """Strategy families."""

    SINGLE_OPTIONS = "Single option"
    VERTICAL_SPREADS = "Spreads"
    VOLATILITY = "Volatility plays"
    BUTTERFLIES_CONDORS = "Butterflies & condors"
    OPTION_AND_UNDERLYING = "With the underlying"
    DIGITALS = "Digitals"
    CUSTOM = "Custom"


@dataclass(frozen=True)
class Leg:
    """A brick held in a given quantity (negative for a short position)."""

    brick: Brick
    quantity: float


@dataclass(frozen=True)
class LegValuation:
    """Valuation of one leg, as returned by :meth:`Strategy.decompose`.

    Attributes
    ----------
    label
        Description of the brick.
    quantity
        Quantity held (negative if short).
    strike
        Strike of the brick, if any.
    implied_vol
        Volatility used to price the brick, if any.
    unit_value
        Value of one unit of the brick.
    value
        Contribution of the leg: ``quantity * unit_value``.
    skew_adjustment
        Part of ``value`` coming from the smile slope (non-zero for digitals only).
    greeks
        Raw Greeks of the leg (already multiplied by the quantity).
    """

    label: str
    quantity: float
    strike: float | None
    implied_vol: float | None
    unit_value: float
    value: float
    skew_adjustment: float
    greeks: Greeks


@dataclass(frozen=True)
class Strategy:
    """A combination of bricks with a single maturity."""

    name: str
    family: Family
    legs: tuple[Leg, ...]

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("a strategy needs at least one leg")
        maturities = {leg.brick.maturity for leg in self.legs}
        if max(maturities) - min(maturities) > 1e-12:
            raise ValueError("all legs of a strategy must share the same maturity")
        if not all(np.isfinite(leg.quantity) for leg in self.legs):
            raise ValueError("quantities must be finite")

    @property
    def maturity(self) -> float:
        """Common maturity of the legs, in years."""
        return self.legs[0].brick.maturity

    @property
    def strikes(self) -> tuple[float, ...]:
        """Sorted distinct strikes of the legs."""
        return tuple(
            sorted({leg.brick.strike for leg in self.legs if leg.brick.strike is not None})
        )

    def value(self, market: Market, spot: ArrayLike | None = None) -> FloatArray:
        """Present value; ``spot`` overrides the market spot (vectorised)."""
        return sum(
            (leg.quantity * leg.brick.value(market, spot) for leg in self.legs),
            start=np.zeros(()),
        )

    def payoff(self, spot_at_maturity: ArrayLike) -> FloatArray:
        """Cash amount paid at maturity."""
        return sum(
            (leg.quantity * leg.brick.payoff(spot_at_maturity) for leg in self.legs),
            start=np.zeros(()),
        )

    def greeks(self, market: Market, spot: ArrayLike | None = None) -> Greeks:
        """Raw Greeks, sum of the legs'."""
        total = Greeks.zero()
        for leg in self.legs:
            total = total + leg.brick.greeks(market, spot) * leg.quantity
        return total

    def decompose(self, market: Market) -> tuple[LegValuation, ...]:
        """Leg-by-leg valuation at the market spot."""
        rows = []
        for leg in self.legs:
            unit = float(leg.brick.value(market))
            skew = 0.0
            if isinstance(leg.brick, DigitalOption):
                skew = leg.quantity * float(leg.brick.skew_adjustment(market))
            rows.append(
                LegValuation(
                    label=leg.brick.label,
                    quantity=leg.quantity,
                    strike=leg.brick.strike,
                    implied_vol=leg.brick.implied_vol(market),
                    unit_value=unit,
                    value=leg.quantity * unit,
                    skew_adjustment=skew,
                    greeks=leg.brick.greeks(market) * leg.quantity,
                )
            )
        return tuple(rows)

    def short(self) -> Strategy:
        """The opposite position."""
        return Strategy(
            name=f"Short {self.name}",
            family=self.family,
            legs=tuple(Leg(leg.brick, -leg.quantity) for leg in self.legs),
        )


# --- Builders -----------------------------------------------------------------------------


def _increasing(*strikes: float) -> None:
    if any(not np.isfinite(k) or k <= 0.0 for k in strikes):
        raise ValueError("strikes must be finite and positive")
    if any(b <= a for a, b in itertools.pairwise(strikes)):
        raise ValueError("strikes must be strictly increasing")


def _call(strike: float, maturity: float) -> EuropeanOption:
    return EuropeanOption("call", strike, maturity)


def _put(strike: float, maturity: float) -> EuropeanOption:
    return EuropeanOption("put", strike, maturity)


def call(strike: float, maturity: float) -> Strategy:
    """A single European call."""
    _increasing(strike)
    return Strategy("Call", Family.SINGLE_OPTIONS, (Leg(_call(strike, maturity), 1.0),))


def put(strike: float, maturity: float) -> Strategy:
    """A single European put."""
    _increasing(strike)
    return Strategy("Put", Family.SINGLE_OPTIONS, (Leg(_put(strike, maturity), 1.0),))


def bull_call_spread(low: float, high: float, maturity: float) -> Strategy:
    """Long call ``low``, short call ``high``."""
    _increasing(low, high)
    legs = (Leg(_call(low, maturity), 1.0), Leg(_call(high, maturity), -1.0))
    return Strategy("Bull call spread", Family.VERTICAL_SPREADS, legs)


def bear_call_spread(low: float, high: float, maturity: float) -> Strategy:
    """Short call ``low``, long call ``high``."""
    _increasing(low, high)
    legs = (Leg(_call(low, maturity), -1.0), Leg(_call(high, maturity), 1.0))
    return Strategy("Bear call spread", Family.VERTICAL_SPREADS, legs)


def bull_put_spread(low: float, high: float, maturity: float) -> Strategy:
    """Long put ``low``, short put ``high``."""
    _increasing(low, high)
    legs = (Leg(_put(low, maturity), 1.0), Leg(_put(high, maturity), -1.0))
    return Strategy("Bull put spread", Family.VERTICAL_SPREADS, legs)


def bear_put_spread(low: float, high: float, maturity: float) -> Strategy:
    """Short put ``low``, long put ``high``."""
    _increasing(low, high)
    legs = (Leg(_put(low, maturity), -1.0), Leg(_put(high, maturity), 1.0))
    return Strategy("Bear put spread", Family.VERTICAL_SPREADS, legs)


def box_spread(low: float, high: float, maturity: float) -> Strategy:
    """Bull call spread plus bear put spread on the same strikes: pays ``high - low``."""
    _increasing(low, high)
    legs = (
        Leg(_call(low, maturity), 1.0),
        Leg(_call(high, maturity), -1.0),
        Leg(_put(low, maturity), -1.0),
        Leg(_put(high, maturity), 1.0),
    )
    return Strategy("Box spread", Family.VERTICAL_SPREADS, legs)


def straddle(strike: float, maturity: float) -> Strategy:
    """Long call and long put at the same strike."""
    _increasing(strike)
    legs = (Leg(_call(strike, maturity), 1.0), Leg(_put(strike, maturity), 1.0))
    return Strategy("Straddle", Family.VOLATILITY, legs)


def strangle(put_strike: float, call_strike: float, maturity: float) -> Strategy:
    """Long put ``put_strike``, long call ``call_strike``."""
    _increasing(put_strike, call_strike)
    legs = (Leg(_put(put_strike, maturity), 1.0), Leg(_call(call_strike, maturity), 1.0))
    return Strategy("Strangle", Family.VOLATILITY, legs)


def strip(strike: float, maturity: float) -> Strategy:
    """One call and two puts at the same strike."""
    _increasing(strike)
    legs = (Leg(_call(strike, maturity), 1.0), Leg(_put(strike, maturity), 2.0))
    return Strategy("Strip", Family.VOLATILITY, legs)


def strap(strike: float, maturity: float) -> Strategy:
    """Two calls and one put at the same strike."""
    _increasing(strike)
    legs = (Leg(_call(strike, maturity), 2.0), Leg(_put(strike, maturity), 1.0))
    return Strategy("Strap", Family.VOLATILITY, legs)


def _butterfly_weight(low: float, mid: float, high: float) -> float:
    """Weight of the high-strike leg that makes the payoff vanish beyond ``high``."""
    return (mid - low) / (high - mid)


def call_butterfly(low: float, mid: float, high: float, maturity: float) -> Strategy:
    """Long call ``low``, short ``1 + w`` calls ``mid``, long ``w`` calls ``high``.

    With ``w = (mid - low) / (high - mid)`` the payoff is zero outside ``[low, high]`` and peaks
    at ``mid - low``; symmetric strikes give the usual 1 / -2 / 1 weights.
    """
    _increasing(low, mid, high)
    w = _butterfly_weight(low, mid, high)
    legs = (
        Leg(_call(low, maturity), 1.0),
        Leg(_call(mid, maturity), -(1.0 + w)),
        Leg(_call(high, maturity), w),
    )
    return Strategy("Call butterfly", Family.BUTTERFLIES_CONDORS, legs)


def put_butterfly(low: float, mid: float, high: float, maturity: float) -> Strategy:
    """Same payoff as :func:`call_butterfly`, built with puts."""
    _increasing(low, mid, high)
    w = _butterfly_weight(low, mid, high)
    legs = (
        Leg(_put(low, maturity), 1.0),
        Leg(_put(mid, maturity), -(1.0 + w)),
        Leg(_put(high, maturity), w),
    )
    return Strategy("Put butterfly", Family.BUTTERFLIES_CONDORS, legs)


def iron_butterfly(low: float, mid: float, high: float, maturity: float) -> Strategy:
    """Short put and short call ``mid``, long put ``low``, long call ``high``."""
    _increasing(low, mid, high)
    legs = (
        Leg(_put(low, maturity), 1.0),
        Leg(_put(mid, maturity), -1.0),
        Leg(_call(mid, maturity), -1.0),
        Leg(_call(high, maturity), 1.0),
    )
    return Strategy("Iron butterfly", Family.BUTTERFLIES_CONDORS, legs)


def call_condor(k1: float, k2: float, k3: float, k4: float, maturity: float) -> Strategy:
    """Long call ``k1``, short call ``k2``, short ``w`` calls ``k3``, long ``w`` calls ``k4``.

    With ``w = (k2 - k1) / (k4 - k3)`` the payoff is zero above ``k4``.
    """
    _increasing(k1, k2, k3, k4)
    w = (k2 - k1) / (k4 - k3)
    legs = (
        Leg(_call(k1, maturity), 1.0),
        Leg(_call(k2, maturity), -1.0),
        Leg(_call(k3, maturity), -w),
        Leg(_call(k4, maturity), w),
    )
    return Strategy("Call condor", Family.BUTTERFLIES_CONDORS, legs)


def iron_condor(k1: float, k2: float, k3: float, k4: float, maturity: float) -> Strategy:
    """Long put ``k1``, short put ``k2``, short call ``k3``, long call ``k4``."""
    _increasing(k1, k2, k3, k4)
    legs = (
        Leg(_put(k1, maturity), 1.0),
        Leg(_put(k2, maturity), -1.0),
        Leg(_call(k3, maturity), -1.0),
        Leg(_call(k4, maturity), 1.0),
    )
    return Strategy("Iron condor", Family.BUTTERFLIES_CONDORS, legs)


def covered_call(strike: float, maturity: float) -> Strategy:
    """Long underlying (prepaid forward), short call."""
    _increasing(strike)
    legs = (Leg(Underlying(maturity), 1.0), Leg(_call(strike, maturity), -1.0))
    return Strategy("Covered call", Family.OPTION_AND_UNDERLYING, legs)


def protective_put(strike: float, maturity: float) -> Strategy:
    """Long underlying (prepaid forward), long put."""
    _increasing(strike)
    legs = (Leg(Underlying(maturity), 1.0), Leg(_put(strike, maturity), 1.0))
    return Strategy("Protective put", Family.OPTION_AND_UNDERLYING, legs)


def collar(put_strike: float, call_strike: float, maturity: float) -> Strategy:
    """Long underlying, long put ``put_strike``, short call ``call_strike``."""
    _increasing(put_strike, call_strike)
    legs = (
        Leg(Underlying(maturity), 1.0),
        Leg(_put(put_strike, maturity), 1.0),
        Leg(_call(call_strike, maturity), -1.0),
    )
    return Strategy("Collar", Family.OPTION_AND_UNDERLYING, legs)


def risk_reversal(put_strike: float, call_strike: float, maturity: float) -> Strategy:
    """Short put ``put_strike``, long call ``call_strike``."""
    _increasing(put_strike, call_strike)
    legs = (Leg(_put(put_strike, maturity), -1.0), Leg(_call(call_strike, maturity), 1.0))
    return Strategy("Risk reversal", Family.OPTION_AND_UNDERLYING, legs)


def synthetic_forward(strike: float, maturity: float) -> Strategy:
    """Long call and short put at the same strike."""
    _increasing(strike)
    legs = (Leg(_call(strike, maturity), 1.0), Leg(_put(strike, maturity), -1.0))
    return Strategy("Synthetic forward", Family.OPTION_AND_UNDERLYING, legs)


def digital_call(strike: float, maturity: float, payout: float = 1.0) -> Strategy:
    """Digital call paying ``payout`` if the spot ends at or above ``strike``."""
    _increasing(strike)
    legs = (Leg(DigitalOption("call", strike, maturity), payout),)
    return Strategy("Digital call", Family.DIGITALS, legs)


def digital_put(strike: float, maturity: float, payout: float = 1.0) -> Strategy:
    """Digital put paying ``payout`` if the spot ends below ``strike``."""
    _increasing(strike)
    legs = (Leg(DigitalOption("put", strike, maturity), payout),)
    return Strategy("Digital put", Family.DIGITALS, legs)


def call_spread_replication(
    low: float, high: float, maturity: float, payout: float = 1.0
) -> Strategy:
    """``payout / (high - low)`` call spreads: tends to a digital at the mid-strike."""
    _increasing(low, high)
    n = payout / (high - low)
    legs = (Leg(_call(low, maturity), n), Leg(_call(high, maturity), -n))
    return Strategy("Digital call by call spread", Family.DIGITALS, legs)


# --- Catalog ------------------------------------------------------------------------------

Builder = Callable[[Sequence[float], float, float], Strategy]


@dataclass(frozen=True)
class StrategySpec:
    """A predefined strategy, as offered by the app.

    Attributes
    ----------
    name, family, description
        What the strategy is and when to use it.
    strike_labels
        Names of the strikes, in increasing order.
    default_strikes_pct
        Default strikes in percent of spot.
    builder
        Function ``(strikes, maturity, payout) -> Strategy``.
    uses_payout
        True for digitals, whose cash payout is a parameter.
    """

    name: str
    family: Family
    description: str
    strike_labels: tuple[str, ...]
    default_strikes_pct: tuple[float, ...]
    builder: Builder
    uses_payout: bool = False

    def build(self, strikes: Sequence[float], maturity: float, payout: float = 1.0) -> Strategy:
        """Build the strategy for absolute ``strikes`` and ``maturity`` in years."""
        if len(strikes) != len(self.strike_labels):
            raise ValueError(f"{self.name} needs {len(self.strike_labels)} strike(s)")
        return self.builder(strikes, maturity, payout)


CATALOG: tuple[StrategySpec, ...] = (
    StrategySpec(
        "Call",
        Family.SINGLE_OPTIONS,
        "Right to buy at the strike: gains if the spot rises, loss limited to the premium.",
        ("Call strike",),
        (100.0,),
        lambda k, t, _: call(k[0], t),
    ),
    StrategySpec(
        "Put",
        Family.SINGLE_OPTIONS,
        "Right to sell at the strike: gains if the spot falls, loss limited to the premium.",
        ("Put strike",),
        (100.0,),
        lambda k, t, _: put(k[0], t),
    ),
    StrategySpec(
        "Bull call spread",
        Family.VERTICAL_SPREADS,
        "Buy a call, sell a higher call: bullish, gain capped, cheaper than the call alone.",
        ("Long call strike", "Short call strike"),
        (100.0, 110.0),
        lambda k, t, _: bull_call_spread(k[0], k[1], t),
    ),
    StrategySpec(
        "Bear call spread",
        Family.VERTICAL_SPREADS,
        "Sell a call, buy a higher call: collects a premium, loss capped by the higher call.",
        ("Short call strike", "Long call strike"),
        (100.0, 110.0),
        lambda k, t, _: bear_call_spread(k[0], k[1], t),
    ),
    StrategySpec(
        "Bull put spread",
        Family.VERTICAL_SPREADS,
        "Buy a put, sell a higher put: collects a premium, kept if the spot ends high.",
        ("Long put strike", "Short put strike"),
        (90.0, 100.0),
        lambda k, t, _: bull_put_spread(k[0], k[1], t),
    ),
    StrategySpec(
        "Bear put spread",
        Family.VERTICAL_SPREADS,
        "Sell a put, buy a higher put: bearish view, gain capped, cheaper than the put alone.",
        ("Short put strike", "Long put strike"),
        (90.0, 100.0),
        lambda k, t, _: bear_put_spread(k[0], k[1], t),
    ),
    StrategySpec(
        "Box spread",
        Family.VERTICAL_SPREADS,
        "Bull call spread plus bear put spread: always pays the strike gap, a zero-coupon.",
        ("Lower strike", "Upper strike"),
        (90.0, 110.0),
        lambda k, t, _: box_spread(k[0], k[1], t),
    ),
    StrategySpec(
        "Straddle",
        Family.VOLATILITY,
        "Buy a call and a put at the same strike: gains from a large move in either direction.",
        ("Strike (call and put)",),
        (100.0,),
        lambda k, t, _: straddle(k[0], t),
    ),
    StrategySpec(
        "Strangle",
        Family.VOLATILITY,
        "Buy a put below and a call above: cheaper than a straddle, needs a larger move.",
        ("Put strike", "Call strike"),
        (90.0, 110.0),
        lambda k, t, _: strangle(k[0], k[1], t),
    ),
    StrategySpec(
        "Strip",
        Family.VOLATILITY,
        "One call and two puts at the same strike: a volatility bet tilted to the downside.",
        ("Strike (1 call, 2 puts)",),
        (100.0,),
        lambda k, t, _: strip(k[0], t),
    ),
    StrategySpec(
        "Strap",
        Family.VOLATILITY,
        "Two calls and one put at the same strike: a volatility bet tilted to the upside.",
        ("Strike (2 calls, 1 put)",),
        (100.0,),
        lambda k, t, _: strap(k[0], t),
    ),
    StrategySpec(
        "Call butterfly",
        Family.BUTTERFLIES_CONDORS,
        "Buy the wings, sell the body, with calls: bets that the spot ends near the body.",
        ("Lower wing strike", "Body strike", "Upper wing strike"),
        (90.0, 100.0, 110.0),
        lambda k, t, _: call_butterfly(k[0], k[1], k[2], t),
    ),
    StrategySpec(
        "Put butterfly",
        Family.BUTTERFLIES_CONDORS,
        "Same payoff as the call butterfly, built with puts: same price by put-call parity.",
        ("Lower wing strike", "Body strike", "Upper wing strike"),
        (90.0, 100.0, 110.0),
        lambda k, t, _: put_butterfly(k[0], k[1], k[2], t),
    ),
    StrategySpec(
        "Iron butterfly",
        Family.BUTTERFLIES_CONDORS,
        "Sell a straddle at the body, protected by a put and a call: bets that the spot stays put.",
        ("Long put strike", "Short put & call strike", "Long call strike"),
        (90.0, 100.0, 110.0),
        lambda k, t, _: iron_butterfly(k[0], k[1], k[2], t),
    ),
    StrategySpec(
        "Call condor",
        Family.BUTTERFLIES_CONDORS,
        "A butterfly with a flat top: bets that the spot ends between the two short strikes.",
        (
            "Long call strike (low)",
            "Short call strike (low)",
            "Short call strike (high)",
            "Long call strike (high)",
        ),
        (80.0, 90.0, 110.0, 120.0),
        lambda k, t, _: call_condor(k[0], k[1], k[2], k[3], t),
    ),
    StrategySpec(
        "Iron condor",
        Family.BUTTERFLIES_CONDORS,
        "Sell a strangle protected by a further put and call: bets that the spot stays in a range.",
        ("Long put strike", "Short put strike", "Short call strike", "Long call strike"),
        (80.0, 90.0, 110.0, 120.0),
        lambda k, t, _: iron_condor(k[0], k[1], k[2], k[3], t),
    ),
    StrategySpec(
        "Covered call",
        Family.OPTION_AND_UNDERLYING,
        "Hold the underlying and sell a call: gives up the upside above the strike for a premium.",
        ("Short call strike",),
        (110.0,),
        lambda k, t, _: covered_call(k[0], t),
    ),
    StrategySpec(
        "Protective put",
        Family.OPTION_AND_UNDERLYING,
        "Hold the underlying and buy a put: floors the payoff at the strike and keeps the upside.",
        ("Long put strike",),
        (90.0,),
        lambda k, t, _: protective_put(k[0], t),
    ),
    StrategySpec(
        "Collar",
        Family.OPTION_AND_UNDERLYING,
        "Hold the underlying, buy a put, sell a call: floor and cap, the call pays for the put.",
        ("Long put strike", "Short call strike"),
        (90.0, 110.0),
        lambda k, t, _: collar(k[0], k[1], t),
    ),
    StrategySpec(
        "Risk reversal",
        Family.OPTION_AND_UNDERLYING,
        "Sell a put, buy a call: leveraged bullish view whose price reflects the skew.",
        ("Short put strike", "Long call strike"),
        (90.0, 110.0),
        lambda k, t, _: risk_reversal(k[0], k[1], t),
    ),
    StrategySpec(
        "Synthetic forward",
        Family.OPTION_AND_UNDERLYING,
        "Buy a call and sell a put at the same strike: replicates a forward.",
        ("Strike (long call, short put)",),
        (100.0,),
        lambda k, t, _: synthetic_forward(k[0], t),
    ),
    StrategySpec(
        "Digital call",
        Family.DIGITALS,
        "Pays a fixed amount if the spot ends at or above the strike; priced with the smile slope.",
        ("Strike",),
        (100.0,),
        lambda k, t, p: digital_call(k[0], t, p),
        uses_payout=True,
    ),
    StrategySpec(
        "Digital put",
        Family.DIGITALS,
        "Pays a fixed amount if the spot ends below the strike; priced with the smile slope.",
        ("Strike",),
        (100.0,),
        lambda k, t, p: digital_put(k[0], t, p),
        uses_payout=True,
    ),
    StrategySpec(
        "Digital call by call spread",
        Family.DIGITALS,
        "The hedge of a digital: a tight call spread, converging to it as strikes get closer.",
        ("Lower strike", "Upper strike"),
        (99.0, 101.0),
        lambda k, t, p: call_spread_replication(k[0], k[1], t, p),
        uses_payout=True,
    ),
)


def specs_by_family() -> dict[Family, tuple[StrategySpec, ...]]:
    """Catalog grouped by family, in catalog order."""
    grouped: dict[Family, list[StrategySpec]] = {}
    for spec in CATALOG:
        grouped.setdefault(spec.family, []).append(spec)
    return {family: tuple(specs) for family, specs in grouped.items()}


def find_spec(name: str) -> StrategySpec:
    """Catalog entry with the given name."""
    for spec in CATALOG:
        if spec.name == name:
            return spec
    raise KeyError(name)
