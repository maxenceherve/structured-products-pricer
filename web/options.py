"""Options and option strategies of the site: their terms, and the position they build.

A product page keeps its terms as plain JSON (``terms`` and per-strike volatility overrides);
:func:`build_position` turns them into a priced position on the session market. Amounts are in
% of the notional, one unit of underlying worth today's spot, as everywhere on the site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from spricer.bricks import (
    BarrierOption,
    BarrierType,
    DigitalOption,
    EuropeanOption,
    Underlying,
    ZeroCoupon,
)
from spricer.conventions import to_desk
from spricer.greeks_lab import ATM_BAND, Instrument, Moneyness, moneyness
from spricer.market import Market, OverriddenVol
from spricer.monitoring import Monitoring
from spricer.strategies import Family, Leg, Strategy, StrategySpec, find_spec
from web import registry
from web.explain import Situation
from web.registry import Product

Terms = dict[str, Any]
DAILY = 252  # observation dates a year for a daily barrier
MONITORING = {
    "european": Monitoring.EUROPEAN,
    "daily": Monitoring.DISCRETE,
    "continuous": Monitoring.CONTINUOUS,
}
MONITORING_LABELS = {
    "european": "European",
    "continuous": "American (continuous)",
    "daily": "Discrete (daily closes)",
}
MONITORING_HELP = (
    "European: the barrier is looked at on the final date only. American: at every moment of "
    "the life. Discrete: at each daily close."
)
GREEKS = ("Delta (%)", "Gamma (%)", "Vega (%)", "Theta (%)", "Rho (%)")

_EXPLAIN_FAMILY = {
    Family.VERTICAL_SPREADS: "spread",
    Family.VOLATILITY: "volatility",
    Family.BUTTERFLIES_CONDORS: "spread",
    Family.OPTION_AND_UNDERLYING: "spread",
    Family.DIGITALS: "digital",
}


@dataclass(frozen=True)
class OptionSetup:
    """What a product page of an option or a strategy offers to set.

    Attributes
    ----------
    product
        The gallery product.
    description
        One sentence on what the position is for.
    strike_labels, default_strikes_pct
        Strikes to set (% of spot), in increasing order.
    uses_payout
        True for digitals: the cash paid is a term.
    barrier_type
        Barrier of a barrier option, None otherwise.
    kind
        "call" or "put": side of the main option (zones in the money, time profiles).
    family
        Family used by the explanations: "vanilla", "digital", "spread", "volatility" or
        "barrier".
    spec
        The library strategy, None for a barrier option.
    """

    product: Product
    description: str
    strike_labels: tuple[str, ...]
    default_strikes_pct: tuple[float, ...]
    uses_payout: bool
    barrier_type: BarrierType | None
    kind: str
    family: str
    spec: StrategySpec | None

    def default_terms(self) -> Terms:
        """Terms of a new page."""
        terms: Terms = {"side": "Long", "maturity": 1.0, "notional": 1_000_000.0}
        for i, k in enumerate(self.default_strikes_pct):
            terms[f"strike_{i}"] = k
        if self.uses_payout:
            terms["payout"] = 10.0
        if self.barrier_type is not None:
            terms["barrier"] = self.product.marker
            terms["monitoring"] = "continuous"
        return terms


def setup_of(product: Product) -> OptionSetup | None:
    """Setup of an option or strategy page, None for a structured product."""
    if product.family == registry.OPTIONS:
        spec = find_spec(product.name)
        kind = "put" if "put" in spec.name.lower() else "call"
        family = _EXPLAIN_FAMILY.get(spec.family, "vanilla")
        return OptionSetup(
            product,
            spec.description,
            spec.strike_labels,
            spec.default_strikes_pct,
            spec.uses_payout,
            None,
            kind,
            family,
            spec,
        )
    if product.family == registry.BARRIERS and product.group != "Sharks":
        barrier_type, kind = product.name.lower().rsplit(" ", 1)
        knock = "comes alive" if barrier_type.endswith("in") else "dies"
        move = "rises to" if barrier_type.startswith("up") else "falls below"
        return OptionSetup(
            product,
            f"A {kind} that {knock} if the underlying {move} the barrier before maturity: "
            f"cheaper than the vanilla {kind}, with a payoff that depends on the path.",
            ("Strike",),
            (100.0,),
            False,
            barrier_type,  # type: ignore[arg-type]  # one of BARRIER_TYPES by construction
            kind,
            "barrier",
            None,
        )
    return None


# --- Building the position ---------------------------------------------------------------------


def _number(terms: Terms, key: str, label: str, *, positive: bool = True) -> float:
    value = terms.get(key)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number") from None
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"{label} must be positive")
    return number


@dataclass(frozen=True)
class Position:
    """A priced option position on the session market (volatility overrides applied)."""

    setup: OptionSetup
    short: bool
    maturity: float
    notional: float
    strikes_pct: tuple[float, ...]
    market: Market
    instrument: Instrument
    barrier_pct: float | None = None

    @property
    def strategy(self) -> Strategy:
        """The position at its maturity."""
        return self.instrument(self.maturity)

    @property
    def name(self) -> str:
        """'Long bull call spread', 'Short call'..."""
        return f"{'Short' if self.short else 'Long'} {self.setup.product.name.lower()}"

    @property
    def single(self) -> bool:
        """One strike: the position is in, at or out of the money."""
        return len(self.strikes_pct) == 1 and self.setup.family != "volatility"

    @property
    def moneyness(self) -> Moneyness | None:
        """ATM, ITM or OTM for a single strike, None otherwise."""
        if not self.single:
            return None
        strike = self.strikes_pct[0] / 100.0 * self.market.spot
        return moneyness(self.setup.kind, strike, self.market, self.maturity)

    @property
    def barrier_crossed(self) -> bool:
        """The barrier of a barrier option is already on the wrong side of today's spot."""
        if self.barrier_pct is None or self.setup.barrier_type is None:
            return False
        if self.setup.barrier_type.startswith("up"):
            return self.barrier_pct <= 100.0
        return self.barrier_pct > 100.0

    @property
    def situation(self) -> Situation:
        """What the explanations need to know."""
        barrier = self.setup.barrier_type or ""
        return Situation(
            self.setup.product.name,
            self.setup.family,
            self.short,
            self.moneyness,
            knock_out=barrier.endswith("out"),
        )

    def atm_bands(self) -> tuple[tuple[float, float], ...]:
        """Spots (% of today) where each strike is at the money (a quarter of a deviation)."""
        out = []
        carry = math.exp((self.market.rate - self.market.dividend_yield) * self.maturity)
        for k in self.strikes_pct:
            strike = k / 100.0 * self.market.spot
            sd = float(self.market.vol.vol(strike, self.maturity)) * math.sqrt(self.maturity)
            centre = k / carry
            out.append((centre * math.exp(-ATM_BAND * sd), centre * math.exp(ATM_BAND * sd)))
        return tuple(out)

    def price_pct(self) -> float:
        """Value in % of the notional."""
        return 100.0 * float(self.strategy.value(self.market)) / self.market.spot


def smile_vols(setup: OptionSetup, terms: Terms, market: Market) -> list[float | None]:
    """Volatility (%) the smile gives each strike of the terms, None where a term is invalid."""
    out: list[float | None] = []
    try:
        maturity = _number(terms, "maturity", "The maturity")
    except ValueError:
        return [None] * len(setup.strike_labels)
    for i in range(len(setup.strike_labels)):
        try:
            k = _number(terms, f"strike_{i}", "strike")
        except ValueError:
            out.append(None)
            continue
        out.append(round(100.0 * float(market.vol.vol(k / 100.0 * market.spot, maturity)), 2))
    return out


def _with_overrides(
    market: Market, strikes_pct: tuple[float, ...], vols: list[Any] | None
) -> Market:
    overrides = []
    for k, typed in zip(strikes_pct, vols or [], strict=False):
        if typed is None or typed == "":
            continue
        try:
            vol = float(typed)
        except (TypeError, ValueError):
            raise ValueError(f"the volatility at {k:g}% must be a number") from None
        if not (math.isfinite(vol) and vol > 0.0):
            raise ValueError(f"the volatility at {k:g}% must be positive")
        overrides.append((k / 100.0 * market.spot, vol / 100.0))
    if not overrides:
        return market
    return Market(
        market.spot, market.rate, market.dividend_yield, OverriddenVol(market.vol, tuple(overrides))
    )


def build_position(
    setup: OptionSetup, terms: Terms, vols: list[Any] | None, market: Market
) -> Position:
    """The position the terms describe, on ``market``; ValueError with a readable message."""
    maturity = _number(terms, "maturity", "The maturity")
    notional = _number(terms, "notional", "The notional")
    strikes_pct = tuple(
        _number(terms, f"strike_{i}", label) for i, label in enumerate(setup.strike_labels)
    )
    short = terms.get("side") == "Short"
    spot = market.spot
    strikes = [k / 100.0 * spot for k in strikes_pct]
    barrier_pct = None
    if setup.spec is not None:
        spec = setup.spec
        payout = _number(terms, "payout", "The amount paid") if setup.uses_payout else 10.0

        def build(t: float) -> Strategy:
            return spec.build(strikes, t, payout / 100.0 * spot)

    else:
        barrier_pct = _number(terms, "barrier", "The barrier")
        monitoring = MONITORING.get(str(terms.get("monitoring")), Monitoring.CONTINUOUS)
        barrier_type = setup.barrier_type
        kind = setup.kind
        level = barrier_pct / 100.0 * spot

        def build(t: float) -> Strategy:
            observations = max(1, round(DAILY * t)) if monitoring is Monitoring.DISCRETE else None
            option = BarrierOption(
                kind,  # type: ignore[arg-type]
                barrier_type,  # type: ignore[arg-type]
                strikes[0],
                level,
                t,
                monitoring,
                observations,
            )
            return Strategy(setup.product.name, Family.CUSTOM, (Leg(option, 1.0),))

    def instrument(t: float) -> Strategy:
        strategy = build(t)
        return strategy.short() if short else strategy

    instrument(maturity)  # validates the terms now, not inside a chart
    return Position(
        setup,
        short,
        maturity,
        notional,
        strikes_pct,
        _with_overrides(market, strikes_pct, vols),
        instrument,
        barrier_pct,
    )


# --- Legs --------------------------------------------------------------------------------------


def leg_name(leg: Leg) -> str:
    """'Long call', 'Short 2 calls', 'Long digital call'... (digital quantities are cash)."""
    brick = leg.brick
    side = "Long" if leg.quantity >= 0 else "Short"
    noun: str
    if isinstance(brick, EuropeanOption):
        noun = brick.kind
    elif isinstance(brick, DigitalOption):
        noun = f"digital {brick.kind}"
    elif isinstance(brick, Underlying):
        noun = "underlying"
    elif isinstance(brick, ZeroCoupon):
        noun = "zero-coupon"
    else:
        noun = brick.label
    size = abs(leg.quantity)
    if isinstance(brick, DigitalOption | ZeroCoupon | BarrierOption) or math.isclose(size, 1.0):
        return f"{side} {noun}"
    plural = noun if noun == "underlying" else f"{noun}s"
    return f"{side} {size:g} {plural}"


def display_greeks(item: Strategy | Leg, market: Market) -> dict[str, float]:
    """Greeks of a strategy or one leg in the desk units of the site (% of notional)."""
    raw = (
        item.brick.greeks(market) * item.quantity if isinstance(item, Leg) else item.greeks(market)
    )
    desk = to_desk(raw, spot=market.spot, notional=market.spot)
    return {
        "Delta (%)": 100.0 * float(desk.delta),
        "Gamma (%)": 100.0 * float(desk.gamma),
        "Vega (%)": float(desk.vega),
        "Theta (%)": float(desk.theta),
        "Rho (%)": float(desk.rho),
    }


def decomposition(position: Position) -> list[dict[str, Any]]:
    """One row per leg and a total row: price, skew and Greeks in % of notional."""
    market, strategy = position.market, position.strategy
    rows: list[dict[str, Any]] = []
    for leg, valuation in zip(strategy.legs, strategy.decompose(market), strict=True):
        brick = leg.brick
        terms = []
        if brick.strike is not None:
            terms.append(f"K {100.0 * brick.strike / market.spot:g}%")
        if isinstance(brick, DigitalOption):
            terms.append(f"pays {100.0 * abs(leg.quantity) / market.spot:g}%")
        vol = brick.implied_vol(market)
        rows.append(
            {
                "Leg": leg_name(leg),
                "Terms": " · ".join(terms),
                "Volatility (%)": None if vol is None else 100.0 * vol,
                "Price (%)": 100.0 * valuation.value / market.spot,
                "Skew (%)": 100.0 * valuation.skew_adjustment / market.spot,
                **display_greeks(leg, market),
            }
        )
    if len(rows) > 1:
        total: dict[str, Any] = {"Leg": "Total", "Terms": "", "Volatility (%)": None}
        for key in ("Price (%)", "Skew (%)", *GREEKS):
            total[key] = sum(r[key] for r in rows)
        rows.append(total)
    return rows


def breakevens(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Points where a piecewise-linear curve crosses zero (linear interpolation).

    Values within rounding of zero count as zero, so a flat zero P&L has no breakeven.
    """
    y = np.where(np.abs(y) < 1e-9 * max(1.0, float(np.max(np.abs(y)))), 0.0, y)
    if not np.any(y):
        return []
    out = []
    for i in range(len(x) - 1):
        a, b = float(y[i]), float(y[i + 1])
        if a == 0.0 and (i == 0 or y[i - 1] != 0.0):
            out.append(float(x[i]))
        elif a * b < 0.0:
            out.append(float(x[i] - a * (x[i + 1] - x[i]) / (b - a)))
    return out
