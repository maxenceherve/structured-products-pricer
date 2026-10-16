"""Hedging an option position: static Greek neutralisation and dynamic delta hedging.

Static hedge
    A position is made delta-neutral with the underlying; gamma-neutral with one vanilla and
    then delta-neutral again; gamma- and vega-neutral with two vanillas (a 2 x 2 linear system)
    and then delta-neutral. The underlying has no gamma and no vega, so it always comes last.

Dynamic delta hedge (Hull, ch. 19; Derman and Kamal 1999, *When you cannot hedge
continuously*)
    The underlying follows a geometric Brownian motion with the *realised* volatility, with
    the risk-neutral drift ``r - q``. The hedger revalues the option and its delta in the market
    it trades in (the implied volatility) and rebalances the hedge at regular dates, paying a
    proportional transaction cost on every share traded. Cash earns the rate and the shares
    pay their dividend yield. At maturity the option pays off and the hedge is unwound.

    Two exact facts make the simulation testable. Discounted hedging gains have zero mean under
    the risk-neutral drift, whatever the hedge ratio, so the mean discounted P&L of a long
    option bought at the implied volatility is ``BS(realised) - BS(implied)`` (minus costs).
    And for equal volatilities the standard deviation of the P&L falls like ``1 / sqrt(N)``
    with the number of rebalancings: about ``sqrt(pi / 4) * sigma * vega / sqrt(N)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spricer._typing import FloatArray
from spricer.conventions import Greeks
from spricer.market.core import Market
from spricer.mc.paths import MCEstimate, estimate
from spricer.strategies import Strategy

Instrument = Callable[[float], Strategy]  # maturity in years -> strategy


@dataclass(frozen=True)
class HedgeLeg:
    """An instrument of the hedge and the quantity traded (positive: bought)."""

    name: str
    quantity: float
    greeks: Greeks  # per unit


@dataclass(frozen=True)
class StaticHedge:
    """A position, its hedge, and the Greeks before and after (raw, in currency)."""

    position: Greeks
    legs: tuple[HedgeLeg, ...]

    @property
    def hedged(self) -> Greeks:
        """Greeks of the position plus its hedge."""
        total = self.position
        for leg in self.legs:
            total = total + leg.greeks * leg.quantity
        return total


def _underlying_greeks() -> Greeks:
    one, zero = np.ones(()), np.zeros(())
    return Greeks(one, zero, zero, zero, zero)


def static_hedge(
    position: Greeks,
    hedges: tuple[tuple[str, Greeks], ...] = (),
) -> StaticHedge:
    """Neutralise the delta, and the gamma (one hedge) or gamma and vega (two hedges).

    Parameters
    ----------
    position
        Raw Greeks of the position to hedge.
    hedges
        Zero, one or two hedging options with their raw Greeks per unit: none for a delta hedge,
        one for a gamma hedge, two for a gamma-vega hedge.

    Raises
    ------
    ValueError
        With more than two options, an option without gamma, or two options whose gamma and
        vega are proportional (they cannot be told apart).
    """
    legs: list[HedgeLeg] = []
    exposure = position
    if len(hedges) == 1:
        name, g = hedges[0]
        if abs(float(g.gamma)) < 1e-14:
            raise ValueError(f"{name} has no gamma: it cannot hedge the gamma")
        q = -float(position.gamma) / float(g.gamma)
        legs.append(HedgeLeg(name, q, g))
        exposure = exposure + g * q
    elif len(hedges) == 2:
        (n1, g1), (n2, g2) = hedges
        system = np.array([[float(g1.gamma), float(g2.gamma)], [float(g1.vega), float(g2.vega)]])
        if abs(np.linalg.det(system)) < 1e-14 * max(1.0, float(np.abs(system).max()) ** 2):
            raise ValueError(
                "the two options have proportional gamma and vega: choose another strike or "
                "maturity"
            )
        q1, q2 = np.linalg.solve(system, -np.array([float(position.gamma), float(position.vega)]))
        legs += [HedgeLeg(n1, float(q1), g1), HedgeLeg(n2, float(q2), g2)]
        exposure = exposure + g1 * float(q1) + g2 * float(q2)
    elif hedges:
        raise ValueError("a static hedge uses at most two options")
    legs.append(HedgeLeg("Underlying", -float(exposure.delta), _underlying_greeks()))
    return StaticHedge(position, tuple(legs))


@dataclass(frozen=True)
class HedgeSimulation:
    """Result of a dynamic delta-hedging simulation, in % of the notional (today's spot).

    Attributes
    ----------
    pnl
        Discounted P&L of each path.
    mean, std
        Mean P&L with its standard error, and the standard deviation of the P&L.
    times, spot, delta, gamma_pnl, theta_pnl, total_pnl
        One sample path: rebalancing dates, spot, delta of the option position, and the P&L of
        each step of the hedged position (undiscounted) split into its gamma part
        ``1/2 Gamma dS^2`` and the rest (theta, carry and costs), and their sum.
    """

    pnl: FloatArray
    mean: MCEstimate
    std: float
    times: FloatArray
    spot: FloatArray
    delta: FloatArray
    gamma_pnl: FloatArray
    theta_pnl: FloatArray
    total_pnl: FloatArray


def simulate_delta_hedge(
    instrument: Instrument,
    market: Market,
    maturity: float,
    *,
    realised_vol: float,
    rebalancings: int,
    quantity: float = 1.0,
    cost_bp: float = 0.0,
    n_paths: int = 5_000,
    seed: int | None = 0,
) -> HedgeSimulation:
    """Delta-hedge ``quantity`` units of the instrument (negative: short) until maturity.

    The hedger values the option in ``market`` (its implied volatilities); the spot moves with
    ``realised_vol``. ``cost_bp`` is paid on the value of every share traded, at inception, at
    each rebalancing and at the final unwind.
    """
    if rebalancings < 1:
        raise ValueError("at least one rebalancing")
    if not (np.isfinite(realised_vol) and realised_vol > 0.0):
        raise ValueError("the realised volatility must be positive")
    if n_paths < 2 or n_paths % 2:
        raise ValueError("n_paths must be even (antithetic paths)")
    r, q, s0 = market.rate, market.dividend_yield, market.spot
    dt = maturity / rebalancings
    times = np.linspace(0.0, maturity, rebalancings + 1)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths // 2, rebalancings))
    z = np.concatenate([z, -z])
    log_steps = (r - q - 0.5 * realised_vol**2) * dt + realised_vol * np.sqrt(dt) * z
    spots = s0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)], 1))
    cost = cost_bp * 1e-4

    value0 = quantity * float(instrument(maturity).value(market))
    shares = -quantity * np.asarray(instrument(maturity).greeks(market, spots[:, 0]).delta)
    cash = -value0 - shares * spots[:, 0] - cost * np.abs(shares) * spots[:, 0]
    # sample path: portfolio value (option + shares + cash) after each date's trades, starting
    # from zero before the first trade (the first step carries the initial costs)
    sample_delta = [float(-shares[0])]
    portfolio = [0.0]
    for k in range(1, rebalancings + 1):
        s = spots[:, k]
        cash = cash * np.exp(r * dt) + shares * spots[:, k - 1] * (np.exp(q * dt) - 1.0)
        if k < rebalancings:
            strategy = instrument(maturity - times[k])
            target = -quantity * np.asarray(strategy.greeks(market, s).delta)
            cash -= (target - shares) * s + cost * np.abs(target - shares) * s
            shares = target
            option = quantity * float(strategy.value(market, s[0]))
            portfolio.append(option + float(shares[0] * s[0] + cash[0]))
            sample_delta.append(float(-shares[0]))
    final = spots[:, -1]
    payoff = quantity * np.asarray(instrument(maturity).payoff(final))
    cash += shares * final - cost * np.abs(shares) * final
    pnl = 100.0 * (payoff + cash) * np.exp(-r * maturity) / s0
    portfolio.append(float(payoff[0] + cash[0]))

    # attribution of each step of the first path: 1/2 Gamma dS^2, and the rest (theta, carry,
    # costs); the steps add up to the undiscounted P&L of that path
    path = spots[0]
    step_pnl = np.diff(np.array(portfolio))
    gammas = np.array(
        [
            quantity * float(instrument(maturity - t).greeks(market, path[i]).gamma)
            for i, t in enumerate(times[:-1])
        ]
    )
    gamma_pnl = 0.5 * gammas * np.diff(path) ** 2
    scale = 100.0 / s0
    return HedgeSimulation(
        pnl=pnl,
        mean=estimate(pnl, antithetic=True),
        std=float(np.std(pnl, ddof=1)),
        times=times,
        spot=path,
        delta=np.array(sample_delta),
        gamma_pnl=gamma_pnl * scale,
        theta_pnl=(step_pnl - gamma_pnl) * scale,
        total_pnl=step_pnl * scale,
    )


def derman_kamal_std(vega: float, vol: float, rebalancings: int) -> float:
    """Approximate standard deviation of the hedging error, ``sqrt(pi/4) vol vega / sqrt(N)``.

    ``vega`` is ``dV/dsigma`` (per unit of volatility) in the unit of the P&L.
    """
    return float(np.sqrt(np.pi / 4.0) * vol * abs(vega) / np.sqrt(rebalancings))
