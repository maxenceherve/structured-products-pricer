"""Bricks and notes priced on Monte Carlo paths.

Every brick of a note is evaluated on the same paths, so the decomposition is exact path by
path: the legs add up to the total, and each leg comes with its own standard error. Barriers
are read from the paths the way the brick defines them (at maturity, continuously, or at its
discrete dates), so a brick can only be priced on paths simulated with its monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spricer._typing import FloatArray
from spricer.bricks import BarrierOption, Brick, EuropeanOption, OneTouch
from spricer.market.core import Market
from spricer.mc.paths import DEFAULT_MAX_STEP, DEFAULT_PATHS, MCEstimate, PathSet, simulate
from spricer.monitoring import Monitoring, observation_dates
from spricer.products.notes import Note

_TIME_TOL = 1e-10


def _check_paths(
    monitoring: Monitoring, observations: int | None, maturity: float, paths: PathSet
) -> None:
    """The paths must observe the barrier exactly as the brick does."""
    if paths.monitoring is not monitoring:
        raise ValueError(
            f"a brick monitored {monitoring.value.lower()} needs paths simulated with the same "
            f"monitoring (got {paths.monitoring.value.lower()})"
        )
    if monitoring is Monitoring.DISCRETE:
        assert observations is not None
        assert paths.monitoring_times is not None
        simulated = paths.monitoring_times[paths.monitoring_times <= maturity + _TIME_TOL]
        dates = observation_dates(maturity, observations)
        if simulated.size != len(dates) or not np.allclose(simulated, dates, atol=_TIME_TOL):
            raise ValueError("the paths do not observe the barrier on the brick's dates")


def _touched(
    direction_up: bool, barrier: float, paths: PathSet, maturity: float, asset: int
) -> NDArray[np.bool_]:
    low, high = paths.extremes(maturity, asset)
    return np.asarray(high >= barrier if direction_up else low < barrier)


def brick_payoff(brick: Brick, paths: PathSet, asset: int = 0) -> FloatArray:
    """Cash flow of one unit of ``brick`` at its maturity, on every row of ``paths``.

    The brick's maturity must be a recorded date. Barrier bricks observed continuously or on
    dates read the extremes of the paths; every other brick reads the spot at maturity.
    """
    spot = paths.spot_at(brick.maturity, asset)
    if isinstance(brick, BarrierOption) and brick.monitoring is not Monitoring.EUROPEAN:
        _check_paths(brick.monitoring, brick.observations, brick.maturity, paths)
        touched = _touched(brick.is_up, brick.barrier, paths, brick.maturity, asset)
        alive = touched if brick.is_in else ~touched
        vanilla = EuropeanOption(brick.kind, brick.strike, brick.maturity).payoff(spot)
        return np.asarray(np.where(alive, vanilla, 0.0))
    if isinstance(brick, OneTouch) and brick.monitoring is not Monitoring.EUROPEAN:
        _check_paths(brick.monitoring, brick.observations, brick.maturity, paths)
        touched = _touched(brick.direction == "up", brick.barrier, paths, brick.maturity, asset)
        return np.asarray(touched, dtype=np.float64)
    return brick.payoff(spot)


def price_brick(
    brick: Brick,
    market: Market,
    *,
    n_paths: int = DEFAULT_PATHS,
    seed: int | None = 0,
    max_step: float = DEFAULT_MAX_STEP,
) -> MCEstimate:
    """Monte Carlo value of one unit of ``brick`` under the local volatility of ``market``."""
    monitoring = getattr(brick, "monitoring", Monitoring.EUROPEAN)
    observations = getattr(brick, "observations", None)
    paths = simulate(
        market,
        [brick.maturity],
        n_paths=n_paths,
        seed=seed,
        monitoring=monitoring,
        monitoring_times=(
            observation_dates(brick.maturity, observations) if observations is not None else None
        ),
        max_step=max_step,
    )
    return paths.estimate(brick_payoff(brick, paths) * paths.discount(brick.maturity))


@dataclass(frozen=True)
class NoteMonteCarlo:
    """Monte Carlo value of a note and of each of its legs, in % of notional.

    The legs are evaluated on the same paths: their values add up to the total exactly.
    """

    total: MCEstimate
    legs: tuple[MCEstimate, ...]


def price_note(
    note: Note,
    market: Market,
    credit_spread: float = 0.0,
    *,
    n_paths: int = DEFAULT_PATHS,
    seed: int | None = 0,
    max_step: float = DEFAULT_MAX_STEP,
) -> NoteMonteCarlo:
    """Monte Carlo value of ``note`` under the local volatility of ``market``.

    Cash flows are discounted at the risk-free rate plus ``credit_spread``, as in
    :meth:`Note.value`; the underlying keeps its risk-free drift.
    """
    monitoring = note.monitoring or Monitoring.EUROPEAN
    monitoring_times = None
    if monitoring is Monitoring.DISCRETE:
        counts: set[int] = {
            n for leg in note.legs if (n := getattr(leg.brick, "observations", None)) is not None
        }
        if len(counts) != 1:
            raise ValueError("the barrier legs of a note must share their observation dates")
        monitoring_times = observation_dates(note.maturity, counts.pop())
    paths = simulate(
        market,
        sorted({leg.brick.maturity for leg in note.legs}),
        n_paths=n_paths,
        seed=seed,
        monitoring=monitoring,
        monitoring_times=monitoring_times,
        max_step=max_step,
    )
    samples = [
        leg.quantity
        * brick_payoff(leg.brick, paths)
        * paths.discount(leg.brick.maturity)
        * np.exp(-credit_spread * leg.brick.maturity)
        for leg in note.legs
    ]
    return NoteMonteCarlo(
        total=paths.estimate(np.sum(samples, axis=0)),
        legs=tuple(paths.estimate(s) for s in samples),
    )
