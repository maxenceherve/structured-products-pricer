"""Elementary bricks: the building blocks of every strategy and structured product.

A brick is a European claim on a single underlying, with one maturity, a payoff function and a
closed-form value in a :class:`~spricer.market.Market`. Every brick is priced consistently with
the volatility surface: its value is the discounted risk-neutral expectation of its payoff under
the density implied by the smile. Vanilla options are therefore priced at the volatility of their
own strike, and digitals carry the skew correction ``-vega * dsigma/dK``.

The smile is sticky-strike: Greeks hold the implied volatility of each strike fixed when the spot
moves; ``vega`` is the sensitivity to a parallel shift of the surface; ``theta`` holds implied
volatilities constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spricer._typing import FloatArray
from spricer.analytic import black_scholes as bs
from spricer.analytic.black_scholes import OptionKind
from spricer.conventions import Greeks
from spricer.market.core import Market
from spricer.monitoring import Monitoring, require_european


def _spot(market: Market, spot: ArrayLike | None) -> FloatArray:
    return np.asarray(market.spot if spot is None else spot, dtype=np.float64)


def _check_maturity(maturity: float) -> None:
    if not (np.isfinite(maturity) and maturity >= 0.0):
        raise ValueError("maturity must be finite and non-negative")


def _check_strike(strike: float) -> None:
    if not (np.isfinite(strike) and strike >= 0.0):
        raise ValueError("strike must be finite and non-negative")


class Brick(Protocol):
    """Interface shared by every brick."""

    @property
    def maturity(self) -> float:
        """Maturity in years."""
        ...

    @property
    def strike(self) -> float | None:
        """Strike, or None for a brick without one."""
        ...

    @property
    def label(self) -> str:
        """Short human-readable description."""
        ...

    def payoff(self, spot_at_maturity: ArrayLike) -> FloatArray:
        """Cash amount paid at maturity per unit of brick."""
        ...

    def value(self, market: Market, spot: ArrayLike | None = None) -> FloatArray:
        """Present value per unit; ``spot`` overrides the market spot (vectorised)."""
        ...

    def greeks(self, market: Market, spot: ArrayLike | None = None) -> Greeks:
        """Raw Greeks per unit; ``spot`` overrides the market spot (vectorised)."""
        ...

    def implied_vol(self, market: Market) -> float | None:
        """Implied volatility used to price the brick, or None if it does not use one."""
        ...


@dataclass(frozen=True)
class ZeroCoupon:
    """Pays one unit of currency at maturity."""

    maturity: float

    def __post_init__(self) -> None:
        _check_maturity(self.maturity)

    @property
    def strike(self) -> None:
        """A zero-coupon has no strike."""
        return None

    @property
    def label(self) -> str:
        """Short description."""
        return f"Zero-coupon {self.maturity:g}y"

    def payoff(self, spot_at_maturity: ArrayLike) -> FloatArray:
        """One, whatever the spot."""
        return np.ones_like(np.asarray(spot_at_maturity, dtype=np.float64))

    def value(self, market: Market, spot: ArrayLike | None = None) -> FloatArray:
        """Discount factor ``e^{-rT}``."""
        s = _spot(market, spot)
        return np.broadcast_to(market.discount(self.maturity), s.shape).copy()

    def greeks(self, market: Market, spot: ArrayLike | None = None) -> Greeks:
        """Only theta (``r e^{-rT}``) and rho (``-T e^{-rT}``) are non-zero."""
        s = _spot(market, spot)
        df = market.discount(self.maturity)
        zero = np.zeros_like(s)
        return Greeks(zero, zero, zero, zero + market.rate * df, zero - self.maturity * df)

    def implied_vol(self, market: Market) -> None:
        """A zero-coupon uses no volatility."""
        return None


@dataclass(frozen=True)
class Underlying:
    """Delivers one unit of the underlying at maturity, without the dividends paid meanwhile.

    This is a prepaid forward, worth ``S e^{-qT}``: the performance leg of a structured product,
    or the stock leg of a covered call held to maturity with dividends left out.
    """

    maturity: float

    def __post_init__(self) -> None:
        _check_maturity(self.maturity)

    @property
    def strike(self) -> None:
        """The underlying has no strike."""
        return None

    @property
    def label(self) -> str:
        """Short description."""
        return f"Underlying {self.maturity:g}y"

    def payoff(self, spot_at_maturity: ArrayLike) -> FloatArray:
        """The spot at maturity."""
        return np.asarray(spot_at_maturity, dtype=np.float64).copy()

    def value(self, market: Market, spot: ArrayLike | None = None) -> FloatArray:
        """``S e^{-qT}``."""
        return np.asarray(
            _spot(market, spot) * np.exp(-market.dividend_yield * self.maturity), dtype=np.float64
        )

    def greeks(self, market: Market, spot: ArrayLike | None = None) -> Greeks:
        """Delta ``e^{-qT}`` and theta ``q S e^{-qT}``."""
        s = _spot(market, spot)
        dq = np.exp(-market.dividend_yield * self.maturity)
        zero = np.zeros_like(s)
        return Greeks(zero + dq, zero, zero, market.dividend_yield * s * dq, zero)

    def implied_vol(self, market: Market) -> None:
        """The prepaid forward uses no volatility."""
        return None


@dataclass(frozen=True)
class EuropeanOption:
    """European call or put, priced at the implied volatility of its strike."""

    kind: OptionKind
    strike: float
    maturity: float

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")
        _check_strike(self.strike)
        _check_maturity(self.maturity)

    @property
    def label(self) -> str:
        """Short description."""
        return f"{self.kind.capitalize()} K={self.strike:g} {self.maturity:g}y"

    def payoff(self, spot_at_maturity: ArrayLike) -> FloatArray:
        """``max(S_T - K, 0)`` for a call, ``max(K - S_T, 0)`` for a put."""
        s = np.asarray(spot_at_maturity, dtype=np.float64)
        sign = 1.0 if self.kind == "call" else -1.0
        return np.maximum(sign * (s - self.strike), 0.0)

    def implied_vol(self, market: Market) -> float:
        """Volatility of the surface at the strike and maturity."""
        return float(market.vol.vol(self.strike, self.maturity))

    def value(self, market: Market, spot: ArrayLike | None = None) -> FloatArray:
        """Black-Scholes-Merton value at the volatility of the strike."""
        return bs.vanilla_price(
            self.kind,
            spot=_spot(market, spot),
            strike=self.strike,
            maturity=self.maturity,
            rate=market.rate,
            dividend_yield=market.dividend_yield,
            vol=self.implied_vol(market),
        )

    def greeks(self, market: Market, spot: ArrayLike | None = None) -> Greeks:
        """Sticky-strike Black-Scholes-Merton Greeks."""
        return bs.vanilla_greeks(
            self.kind,
            spot=_spot(market, spot),
            strike=self.strike,
            maturity=self.maturity,
            rate=market.rate,
            dividend_yield=market.dividend_yield,
            vol=self.implied_vol(market),
        )


@dataclass(frozen=True)
class DigitalOption:
    """Cash-or-nothing digital paying one unit of currency.

    The call pays if ``S_T >= K``, the put if ``S_T < K``. With ``skew_adjusted`` (the default)
    the price includes the smile term ``-vega * dsigma/dK``, which makes it equal to the limit of
    a tight call spread on the same surface. Without it, the digital is the flat Black-Scholes
    formula at the volatility of its strike, kept only to measure the size of the skew effect.
    """

    kind: OptionKind
    strike: float
    maturity: float
    skew_adjusted: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")
        _check_strike(self.strike)
        _check_maturity(self.maturity)

    @property
    def label(self) -> str:
        """Short description."""
        return f"Digital {self.kind} K={self.strike:g} {self.maturity:g}y"

    def payoff(self, spot_at_maturity: ArrayLike) -> FloatArray:
        """One if the digital pays, zero otherwise."""
        s = np.asarray(spot_at_maturity, dtype=np.float64)
        paid = s >= self.strike if self.kind == "call" else s < self.strike
        return paid.astype(np.float64)

    def implied_vol(self, market: Market) -> float:
        """Volatility of the surface at the strike and maturity."""
        return float(market.vol.vol(self.strike, self.maturity))

    def vol_slope(self, market: Market) -> float:
        """Smile slope used by the skew correction (zero if not skew-adjusted)."""
        if not self.skew_adjusted:
            return 0.0
        return float(market.vol.vol_slope(self.strike, self.maturity))

    def _args(self, market: Market, spot: ArrayLike | None) -> dict[str, ArrayLike]:
        return {
            "spot": _spot(market, spot),
            "strike": self.strike,
            "maturity": self.maturity,
            "rate": market.rate,
            "dividend_yield": market.dividend_yield,
            "vol": self.implied_vol(market),
        }

    def value(self, market: Market, spot: ArrayLike | None = None) -> FloatArray:
        """Digital value, skew-adjusted unless ``skew_adjusted`` is False."""
        return bs.digital_price(
            self.kind, **self._args(market, spot), vol_slope=self.vol_slope(market)
        )

    def skew_adjustment(self, market: Market, spot: ArrayLike | None = None) -> FloatArray:
        """Part of the value that comes from the smile slope (zero if not skew-adjusted)."""
        flat = bs.digital_price(self.kind, **self._args(market, spot))
        return self.value(market, spot) - flat

    def greeks(self, market: Market, spot: ArrayLike | None = None) -> Greeks:
        """Sticky-strike Greeks, including the derivatives of the skew term."""
        return bs.digital_greeks(
            self.kind, **self._args(market, spot), vol_slope=self.vol_slope(market)
        )


BarrierType = Literal["up-and-out", "up-and-in", "down-and-out", "down-and-in"]
BARRIER_TYPES: tuple[BarrierType, ...] = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")


@dataclass(frozen=True)
class BarrierOption:
    """European call or put activated (``-in``) or cancelled (``-out``) by a barrier ``H``.

    The barrier is touched at or above ``H`` for an up barrier, strictly below ``H`` for a down
    barrier, the same convention as the digitals. Observed at maturity (European monitoring),
    a barrier option is exactly a vanilla option restricted to a region of ``S_T``, which
    replicates with vanilla options and a digital (Haug, ch. 4) and therefore inherits the
    smile, read at the strike and at the barrier. For instance

    * up-and-in call, ``H >= K``: ``C(H) + (H - K) D_call(H)``;
    * down-and-in put, ``H <= K``: ``P(H) + (K - H) D_put(H)``;

    and ``in + out = vanilla`` in every case. Continuous and discrete monitoring use closed
    forms that come with the barrier module and are refused until then.
    """

    kind: OptionKind
    barrier_type: BarrierType
    strike: float
    barrier: float
    maturity: float
    monitoring: Monitoring = Monitoring.EUROPEAN

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")
        if self.barrier_type not in BARRIER_TYPES:
            raise ValueError(f"barrier_type must be one of {BARRIER_TYPES}")
        _check_strike(self.strike)
        if not (np.isfinite(self.barrier) and self.barrier > 0.0):
            raise ValueError("barrier must be finite and positive")
        _check_maturity(self.maturity)
        require_european(self.monitoring)

    @property
    def is_up(self) -> bool:
        """True for an up barrier."""
        return self.barrier_type.startswith("up")

    @property
    def is_in(self) -> bool:
        """True for a knock-in."""
        return self.barrier_type.endswith("in")

    @property
    def label(self) -> str:
        """Short description."""
        return (
            f"{self.kind.capitalize()} {self.barrier_type} K={self.strike:g} "
            f"H={self.barrier:g} {self.maturity:g}y"
        )

    def _touched(self, s: FloatArray) -> NDArray[np.bool_]:
        return np.asarray((s >= self.barrier) if self.is_up else (s < self.barrier))

    def payoff(self, spot_at_maturity: ArrayLike) -> FloatArray:
        """Vanilla payoff where the option is alive at maturity, zero elsewhere."""
        s = np.asarray(spot_at_maturity, dtype=np.float64)
        sign = 1.0 if self.kind == "call" else -1.0
        vanilla = np.maximum(sign * (s - self.strike), 0.0)
        touched = self._touched(s)
        alive = touched if self.is_in else ~touched
        return np.where(alive, vanilla, 0.0)

    def components(self) -> tuple[tuple[float, Brick], ...]:
        """Replicating vanilla options and digitals, as ``(quantity, brick)`` pairs."""
        k, h, t = self.strike, self.barrier, self.maturity
        vanilla = EuropeanOption(self.kind, k, t)
        knocked: tuple[tuple[float, Brick], ...]
        if self.kind == "call":
            if self.is_up:  # call x 1{S >= H}
                knocked = (
                    ((1.0, EuropeanOption("call", h, t)), (h - k, DigitalOption("call", h, t)))
                    if h >= k
                    else ((1.0, vanilla),)
                )
            else:  # call x 1{S < H}
                knocked = (
                    (
                        (1.0, vanilla),
                        (-1.0, EuropeanOption("call", h, t)),
                        (-(h - k), DigitalOption("call", h, t)),
                    )
                    if h > k
                    else ()
                )
        elif self.is_up:  # put x 1{S >= H}
            knocked = (
                (
                    (1.0, vanilla),
                    (-1.0, EuropeanOption("put", h, t)),
                    (-(k - h), DigitalOption("put", h, t)),
                )
                if h < k
                else ()
            )
        else:  # put x 1{S < H}
            knocked = (
                ((1.0, EuropeanOption("put", h, t)), (k - h, DigitalOption("put", h, t)))
                if h <= k
                else ((1.0, vanilla),)
            )
        if self.is_in:
            return tuple((q, b) for q, b in knocked if q)
        return ((1.0, vanilla), *((-q, b) for q, b in knocked if q))

    def implied_vol(self, market: Market) -> float:
        """Volatility of the surface at the strike."""
        return float(market.vol.vol(self.strike, self.maturity))

    def barrier_vol(self, market: Market) -> float:
        """Volatility of the surface at the barrier."""
        return float(market.vol.vol(self.barrier, self.maturity))

    def value(self, market: Market, spot: ArrayLike | None = None) -> FloatArray:
        """Sum of the replicating bricks' values."""
        total = np.zeros_like(_spot(market, spot))
        for q, brick in self.components():
            total = total + q * brick.value(market, spot)
        return total

    def skew_adjustment(self, market: Market, spot: ArrayLike | None = None) -> FloatArray:
        """Part of the value coming from the smile slope (through the digital)."""
        total = np.zeros_like(_spot(market, spot))
        for q, brick in self.components():
            if isinstance(brick, DigitalOption):
                total = total + q * brick.skew_adjustment(market, spot)
        return total

    def greeks(self, market: Market, spot: ArrayLike | None = None) -> Greeks:
        """Sum of the replicating bricks' Greeks."""
        total = Greeks.zero()
        for q, brick in self.components():
            total = total + brick.greeks(market, spot) * q
        return total
