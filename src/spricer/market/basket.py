"""Underlyings of a structured product: one market each, initial levels, correlation."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray
from spricer.market.core import Market


@dataclass(frozen=True)
class Basket:
    """The underlyings a product is written on.

    A product on several underlyings looks at the worst performance: the lowest of
    ``spot_i / initial_i``. With one underlying the basket is that underlying.

    Attributes
    ----------
    markets
        One market per underlying (spot, dividend yield, smile); one shared rate.
    initials
        Initial level of each underlying (the 100 % of the term sheet).
    names
        Display names.
    correlation
        Correlation matrix of the underlyings, None for a single one.
    """

    markets: tuple[Market, ...]
    initials: tuple[float, ...]
    names: tuple[str, ...]
    correlation: FloatArray | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        n = len(self.markets)
        if n == 0:
            raise ValueError("a basket needs at least one underlying")
        if len(self.initials) != n or len(self.names) != n:
            raise ValueError("one initial level and one name per underlying")
        if any(not (np.isfinite(s) and s > 0.0) for s in self.initials):
            raise ValueError("initial levels must be positive")
        if n > 1 and self.correlation is None:
            raise ValueError("several underlyings need a correlation matrix")
        if len({m.rate for m in self.markets}) != 1:
            raise ValueError("all underlyings must share the same rate (one currency)")

    @classmethod
    def single(
        cls, market: Market, initial: float | None = None, name: str = "Underlying"
    ) -> Basket:
        """A basket of one underlying; its initial level defaults to today's spot."""
        return cls((market,), (market.spot if initial is None else initial,), (name,))

    @classmethod
    def of(
        cls,
        markets: Sequence[Market],
        initials: Sequence[float],
        names: Sequence[str],
        correlation: ArrayLike | None = None,
    ) -> Basket:
        """Basket from sequences."""
        corr = None if correlation is None else np.asarray(correlation, dtype=np.float64)
        return cls(tuple(markets), tuple(initials), tuple(names), corr)

    def with_market(self, index: int, market: Market) -> Basket:
        """Same basket with the market of underlying ``index`` replaced (for a bump)."""
        markets = list(self.markets)
        markets[index] = market
        return dataclasses.replace(self, markets=tuple(markets))

    def map_markets(self, change: Callable[[Market], Market]) -> Basket:
        """Same basket with every market changed (e.g. a rate bump)."""
        return dataclasses.replace(self, markets=tuple(change(m) for m in self.markets))

    @property
    def size(self) -> int:
        """Number of underlyings."""
        return len(self.markets)

    @property
    def rate(self) -> float:
        """Shared risk-free rate."""
        return self.markets[0].rate

    def performances_pct(self, spots: ArrayLike) -> FloatArray:
        """Performance of each underlying in % of its initial level (last axis: underlyings)."""
        return np.asarray(100.0 * np.asarray(spots, dtype=np.float64) / np.asarray(self.initials))

    def worst_pct(self, spots: ArrayLike) -> FloatArray:
        """Worst performance in % of the initial levels (reduces the last axis)."""
        return np.asarray(self.performances_pct(spots).min(axis=-1))

    def today_worst_pct(self) -> float:
        """Worst performance today."""
        return float(self.worst_pct([m.spot for m in self.markets]))
