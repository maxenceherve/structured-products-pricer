"""Single-underlying market environment."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray
from spricer.market.vol_surface import ShiftedVol, VolSurface


@dataclass(frozen=True)
class Market:
    """Spot, flat rate, flat dividend yield and an implied volatility surface.

    Attributes
    ----------
    spot
        Current spot level (currency).
    rate
        Continuously compounded risk-free rate (decimal), used both to discount and in the
        risk-neutral drift.
    dividend_yield
        Continuous dividend yield (decimal).
    vol
        Implied volatility surface, read by absolute strike (sticky strike).
    """

    spot: float
    rate: float
    dividend_yield: float
    vol: VolSurface

    def __post_init__(self) -> None:
        if not (np.isfinite(self.spot) and self.spot > 0.0):
            raise ValueError("spot must be finite and positive")
        if not (np.isfinite(self.rate) and np.isfinite(self.dividend_yield)):
            raise ValueError("rate and dividend_yield must be finite")

    def forward(self, maturity: ArrayLike) -> FloatArray:
        """Forward price for delivery at ``maturity`` (years)."""
        t = np.asarray(maturity, dtype=np.float64)
        return self.spot * np.exp((self.rate - self.dividend_yield) * t)

    def discount(self, maturity: ArrayLike) -> FloatArray:
        """Discount factor to ``maturity`` (years)."""
        return np.exp(-self.rate * np.asarray(maturity, dtype=np.float64))

    def with_spot(self, spot: float) -> Market:
        """Same market with another spot; the smile does not move (sticky strike)."""
        return dataclasses.replace(self, spot=spot)

    def with_rate(self, rate: float) -> Market:
        """Same market with another rate."""
        return dataclasses.replace(self, rate=rate)

    def with_vol_shift(self, shift: float) -> Market:
        """Same market with the whole surface shifted by ``shift`` (decimal)."""
        return dataclasses.replace(self, vol=ShiftedVol(self.vol, shift))
