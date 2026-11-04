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

:func:`coupon_note_profile` draws the Greeks against the level of the underlyings without a
simulation per level. A local volatility tabulated in forward log-moneyness scales with the
spot, so paths from another spot are today's paths rescaled, which amounts to dividing the
initial levels: the smile moves with the underlying (sticky moneyness), where the Greeks above
keep the implied volatility of each strike (sticky strike).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray
from spricer.conventions import DAYS_PER_YEAR
from spricer.market.basket import Basket
from spricer.market.core import Market
from spricer.mc.coupon_notes import LEG_NAMES, leg_samples, note_samples, simulate_note
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
class LegGreeks:
    """Sensitivities of one leg of the note, every underlying moved, in % of notional.

    Attributes
    ----------
    name
        Leg, from :data:`~spricer.mc.coupon_notes.LEG_NAMES`.
    delta, gamma, vega
        Sum over the underlyings.
    theta
        After one calendar day, None when an observation falls within it.
    rho
        For +1 basis point on the rate.
    """

    name: str
    delta: MCEstimate
    gamma: MCEstimate
    vega: MCEstimate
    theta: MCEstimate | None
    rho: MCEstimate


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
    legs
        The same Greeks leg by leg; on common random numbers they add up to the note's.
    """

    price: MCEstimate
    assets: tuple[AssetGreeks, ...]
    theta: MCEstimate | None
    rho: MCEstimate
    legs: tuple[LegGreeks, ...] = ()


def coupon_note_greeks(
    note: CouponNote,
    basket: Basket,
    credit_spread: float = 0.0,
    *,
    n_paths: int = DEFAULT_PATHS,
    seed: int = 0,
    max_step: float = DEFAULT_MAX_STEP,
    common_random_numbers: bool = True,
    delta_bump: float = DELTA_BUMP,
    gamma_bump: float = GAMMA_BUMP,
) -> NoteGreeks:
    """Greeks of ``note`` by bump and revalue.

    With ``common_random_numbers`` False each scenario gets its own seed, which shows how
    much noise the common random numbers remove (the error then adds up across scenarios).
    """
    counter = iter(range(1, 10_000))

    def run(b: Basket, n: CouponNote = note) -> tuple[FloatArray, PathSet]:
        """Discounted value of each leg on every row, and the paths."""
        s = seed if common_random_numbers else seed + next(counter)
        paths = simulate_note(n, b, n_paths=n_paths, seed=s, max_step=max_step)
        return leg_samples(n, b, paths, credit_spread), paths

    base_legs, paths = run(basket)
    base = np.sum(base_legs, axis=0)

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

    def summed(parts: list[tuple[FloatArray, float]]) -> list[tuple[FloatArray, float]]:
        return [(np.sum(x, axis=0), w) for x, w in parts]

    def of_leg(parts: list[tuple[FloatArray, float]], leg: int) -> MCEstimate:
        return paths.estimate(np.sum([w * x[leg] for x, w in parts], axis=0))

    assets = []
    deltas: list[tuple[FloatArray, float]] = []
    gammas: list[tuple[FloatArray, float]] = []
    vegas: list[tuple[FloatArray, float]] = []
    for i, market in enumerate(basket.markets):

        def bumped(h: float, m: Market = market, i: int = i) -> FloatArray:
            return run(basket.with_market(i, m.with_spot(m.spot * (1.0 + h))))[0]

        hd, hg = delta_bump, gamma_bump
        delta = [(bumped(hd), 0.5 / hd), (bumped(-hd), -0.5 / hd)]
        gamma = [
            (bumped(hg), 0.01 / hg**2),
            (base_legs, -0.02 / hg**2),
            (bumped(-hg), 0.01 / hg**2),
        ]
        vega = [
            (run(basket.with_market(i, market.with_vol_shift(VOL_BUMP)))[0], 0.5),
            (run(basket.with_market(i, market.with_vol_shift(-VOL_BUMP)))[0], -0.5),
        ]
        assets.append(
            AssetGreeks(
                name=basket.names[i],
                delta=combine(summed(delta)),
                gamma=combine(summed(gamma)),
                vega=combine(summed(vega)),
            )
        )
        deltas += delta
        gammas += gamma
        vegas += vega

    def rate_shift(d: float) -> Callable[[Basket], Basket]:
        return lambda b: b.map_markets(lambda m: m.with_rate(m.rate + d))

    rho = [
        (run(rate_shift(RATE_BUMP)(basket))[0], 0.5),
        (run(rate_shift(-RATE_BUMP)(basket))[0], -0.5),
    ]
    aging: list[tuple[FloatArray, float]] | None = None
    try:
        aged = note.aged(1.0 / DAYS_PER_YEAR)
    except ValueError:
        aged = None
    if aged is not None:
        aging = [(run(basket, aged)[0], 1.0), (base_legs, -1.0)]
    return NoteGreeks(
        price=paths.estimate(base),
        assets=tuple(assets),
        theta=None if aging is None else combine(summed(aging)),
        rho=combine(summed(rho)),
        legs=tuple(
            LegGreeks(
                name,
                of_leg(deltas, j),
                of_leg(gammas, j),
                of_leg(vegas, j),
                None if aging is None else of_leg(aging, j),
                of_leg(rho, j),
            )
            for j, name in enumerate(LEG_NAMES)
        ),
    )


def coupon_note_profile(
    note: CouponNote,
    basket: Basket,
    levels_pct: ArrayLike,
    credit_spread: float = 0.0,
    *,
    n_paths: int = DEFAULT_PATHS,
    seed: int = 0,
    max_step: float = DEFAULT_MAX_STEP,
) -> dict[str, FloatArray]:
    """Price and Greeks of ``note`` if every underlying stood at ``levels_pct`` % of today.

    All the underlyings move together, so each Greek is the sum over them. A few simulations
    in all (the base, the volatility, the rate and the time bumps), their paths rescaled for
    every level; theta is NaN when an observation falls within the next day.
    """
    scales = np.asarray(levels_pct, dtype=np.float64) / 100.0

    def curve(b: Basket, n: CouponNote = note, *factors: float) -> list[FloatArray]:
        """The price against the level, for each factor applied to the level (1 by default)."""
        paths = simulate_note(n, b, n_paths=n_paths, seed=seed, max_step=max_step)
        return [
            np.array(
                [
                    paths.estimate(
                        note_samples(
                            n,
                            replace(b, initials=tuple(i / (s * f) for i in b.initials)),
                            paths,
                            credit_spread,
                        )
                    ).value
                    for s in scales
                ]
            )
            for f in factors or (1.0,)
        ]

    hd, hg = DELTA_BUMP, GAMMA_BUMP
    price, up_d, down_d, up_g, down_g = curve(
        basket, note, 1.0, 1.0 + hd, 1.0 - hd, 1.0 + hg, 1.0 - hg
    )
    vol_up = curve(basket.map_markets(lambda m: m.with_vol_shift(VOL_BUMP)))[0]
    vol_down = curve(basket.map_markets(lambda m: m.with_vol_shift(-VOL_BUMP)))[0]
    rho_up = curve(basket.map_markets(lambda m: m.with_rate(m.rate + RATE_BUMP)))[0]
    rho_down = curve(basket.map_markets(lambda m: m.with_rate(m.rate - RATE_BUMP)))[0]
    try:
        theta = curve(basket, note.aged(1.0 / DAYS_PER_YEAR))[0] - price
    except ValueError:
        theta = np.full_like(price, np.nan)
    return {
        "Price (%)": price,
        "Delta (%)": 0.5 / hd * (up_d - down_d),
        "Gamma (%)": 0.01 / hg**2 * (up_g - 2.0 * price + down_g),
        "Vega (%)": 0.5 * (vol_up - vol_down),
        "Theta (%)": theta,
        "Rho (%)": 0.5 * (rho_up - rho_down),
    }
