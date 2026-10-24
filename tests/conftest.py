import itertools
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from scipy.integrate import simpson

from spricer._typing import FloatArray
from spricer.market import InterpolatedVolSurface, Market
from spricer.market.defaults import default_market

Payoff = Callable[[FloatArray], FloatArray]


@pytest.fixture
def market() -> Market:
    return default_market()


@pytest.fixture
def surface(market: Market) -> InterpolatedVolSurface:
    assert isinstance(market.vol, InterpolatedVolSurface)
    return market.vol


def _discounted_expectation(
    market: Market, payoff: Payoff, maturity: float, breakpoints: Sequence[float] = ()
) -> float:
    """``e^{-rT} E[payoff(S_T)]`` under the density implied by the market's surface.

    Composite Simpson rule in log-strike, split at the payoff's kinks and jumps.
    """
    assert isinstance(market.vol, InterpolatedVolSurface)
    log_forward = float(np.log(market.forward(maturity)))
    # Wide range: the Lee-type left wing has a fat tail in log-strike.
    ends = [log_forward - 40.0, log_forward + 15.0]
    cuts = sorted({*ends, *(float(np.log(b)) for b in breakpoints if b > 0.0)})
    total = 0.0
    for lo, hi in itertools.pairwise(cuts):
        x = np.linspace(lo, hi, 1 + 2 * max(1000, int(2000 * (hi - lo))))
        strikes = np.exp(x)
        # evaluate the payoff just inside the segment: digitals jump at its ends
        inside = np.exp(np.clip(x, lo + 1e-12, hi - 1e-12))
        integrand = payoff(inside) * market.vol.density(strikes, maturity) * strikes
        total += float(simpson(integrand, x=x))
    return float(market.discount(maturity)) * total


@pytest.fixture
def discounted_expectation() -> Callable[..., float]:
    return _discounted_expectation
