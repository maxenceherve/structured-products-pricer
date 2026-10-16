"""Greeks of Monte Carlo prices, underlying by underlying, with their standard errors.

Bump and revalue with common random numbers (Glasserman 2003, section 7.1): every bumped
scenario is simulated with the same seed as the base price, so the same normal draws (and the
same Brownian-bridge uniforms) drive every path. A finite difference is then the mean of
per-path differences, whose noise largely cancels, and its standard error follows directly.

Bumps, in the desk units of :mod:`spricer.conventions`:

* Delta (%): central difference with a relative spot bump of 3 % of one underlying, others
  frozen; Gamma (%): second difference with a 5 % bump. Structured payoffs jump (barrier,
  autocall and coupon levels), and with common random numbers only the paths crossing a jump
  between the two bumps contribute: a small bump is unbiased but very noisy. Measured on a
  barrier reverse convertible against its closed form with 100,000 paths, a 1 % bump leaves
  the gamma with 57 % noise; 5 % brings it to 5 % with a 0.7 % bias, and 3 % keeps the delta
  within 1 % of its exact value (tests in ``tests/test_mc_greeks.py``);
* Vega (%): parallel shift of one underlying's implied surface by ±1 volatility point, the
  local volatility being rebuilt from the shifted surface;
* Rho (%): every rate ±1 basis point (drift and discounting);
* Theta (%): every date of the note one calendar day closer, the market unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spricer._typing import FloatArray
from spricer.conventions import DAYS_PER_YEAR
from spricer.market.basket import Basket
from spricer.market.core import Market
from spricer.mc.coupon_notes import note_samples, simulate_note
from spricer.mc.paths import DEFAULT_MAX_STEP, DEFAULT_PATHS, MCEstimate, PathSet
from spricer.products.coupon_notes import CouponNote

DELTA_BUMP = 0.03
GAMMA_BUMP = 0.05
VOL_BUMP = 0.01  # one volatility point
RATE_BUMP = 1e-4  # one basis point


@dataclass(frozen=True)
class AssetGreeks:
    """Sensitivities to one underlying, in % of notional.

    Attributes
    ----------
    name
        Underlying.
    delta
        Exposure: change in value for a +1 % spot move, times 100.
    gamma
        Change of Delta (%) for a +1 % spot move.
    vega
        Change in value for +1 volatility point on this underlying's smile.
    """

    name: str
    delta: MCEstimate
    gamma: MCEstimate
    vega: MCEstimate


@dataclass(frozen=True)
class NoteGreeks:
    """Monte Carlo Greeks of a note, in % of notional.

    Attributes
    ----------
    price
        Base value.
    assets
        Delta, gamma and vega of each underlying.
    theta
        Change in value after one calendar day, or None when an observation falls within it.
    rho
        Change in value for +1 basis point on the rate.
    """

    price: MCEstimate
    assets: tuple[AssetGreeks, ...]
    theta: MCEstimate | None
    rho: MCEstimate


def coupon_note_greeks(
    note: CouponNote,
    basket: Basket,
    credit_spread: float = 0.0,
    *,
    n_paths: int = DEFAULT_PATHS,
    seed: int = 0,
    max_step: float = DEFAULT_MAX_STEP,
    common_random_numbers: bool = True,
) -> NoteGreeks:
    """Greeks of ``note`` by bump and revalue.

    With ``common_random_numbers`` False each scenario gets its own seed, which shows how
    much noise the common random numbers remove (the error then adds up across scenarios).
    """
    counter = iter(range(1, 10_000))

    def run(b: Basket, n: CouponNote = note) -> tuple[FloatArray, PathSet]:
        s = seed if common_random_numbers else seed + next(counter)
        paths = simulate_note(n, b, n_paths=n_paths, seed=s, max_step=max_step)
        return note_samples(n, b, paths, credit_spread), paths

    base, paths = run(basket)

    def difference(samples: FloatArray, scale: float) -> MCEstimate:
        return paths.estimate(samples * scale)

    def combine(parts: list[tuple[FloatArray, float]]) -> MCEstimate:
        """Linear combination of scenarios, with the error of independent ones if no CRN."""
        if common_random_numbers:
            return difference(np.sum([w * x for x, w in parts], axis=0), 1.0)
        estimates = [paths.estimate(x) for x, _ in parts]
        value = sum(w * e.value for (_, w), e in zip(parts, estimates, strict=True))
        error = float(
            np.sqrt(sum((w * e.std_error) ** 2 for (_, w), e in zip(parts, estimates, strict=True)))
        )
        return MCEstimate(value, error, estimates[0].n_paths)

    assets = []
    for i, market in enumerate(basket.markets):

        def bumped(h: float, m: Market = market, i: int = i) -> FloatArray:
            return run(basket.with_market(i, m.with_spot(m.spot * (1.0 + h))))[0]

        hd, hg = DELTA_BUMP, GAMMA_BUMP
        vol_up, _ = run(basket.with_market(i, market.with_vol_shift(VOL_BUMP)))
        vol_down, _ = run(basket.with_market(i, market.with_vol_shift(-VOL_BUMP)))
        assets.append(
            AssetGreeks(
                name=basket.names[i],
                delta=combine([(bumped(hd), 0.5 / hd), (bumped(-hd), -0.5 / hd)]),
                gamma=combine(
                    [(bumped(hg), 0.01 / hg**2), (base, -0.02 / hg**2), (bumped(-hg), 0.01 / hg**2)]
                ),
                vega=combine([(vol_up, 0.5), (vol_down, -0.5)]),
            )
        )

    def rate_shift(d: float) -> Callable[[Basket], Basket]:
        return lambda b: b.map_markets(lambda m: m.with_rate(m.rate + d))

    rho_up, _ = run(rate_shift(RATE_BUMP)(basket))
    rho_down, _ = run(rate_shift(-RATE_BUMP)(basket))
    theta: MCEstimate | None = None
    try:
        aged = note.aged(1.0 / DAYS_PER_YEAR)
    except ValueError:
        aged = None
    if aged is not None:
        later, _ = run(basket, aged)
        theta = combine([(later, 1.0), (base, -1.0)])
    return NoteGreeks(
        price=paths.estimate(base),
        assets=tuple(assets),
        theta=theta,
        rho=combine([(rho_up, 0.5), (rho_down, -0.5)]),
    )
