"""Monte Carlo paths of one or several correlated underlyings.

Each underlying follows a local volatility diffusion under the risk-neutral measure,

    dS_i / S_i = (r - q_i) dt + sigma_i(t, S_i) dW_i,   d<W_i, W_j> = rho_ij dt,

with ``sigma_i`` the Dupire local volatility of its own market (:mod:`spricer.mc.local_vol`).

Time stepping
    When every surface is flat the log-spot step is exact, so the grid holds only the dates the
    product needs. Under a local volatility the log-spot follows an Euler scheme, the local
    volatility read at the middle of each step, whose bias is proportional to the step. Steps
    grow from daily near today (the short-dated smile makes the local volatility steep) to
    ``max_step``, and the surface's own maturities are grid nodes. The first-order bias is then
    removed by Richardson extrapolation (Talay and Tubaro 1990): every path is also simulated
    on a grid twice as coarse, driven by the same Brownian increments, and the estimator of a
    payoff ``f`` is ``2 f(fine path) - f(coarse path)``, path by path.

Barriers (Glasserman, *Monte Carlo Methods in Financial Engineering*, 2003, section 6.4)
    * discrete monitoring: the spot is checked exactly at the observation dates, which are
      grid nodes;
    * continuous monitoring: the extremum of each step is drawn from its distribution given the
      two ends of the step (Brownian bridge in log-spot), so no crossing between steps is
      missed. With several assets each asset's bridge is drawn independently, an approximation
      that vanishes as steps shrink.

Standard error
    Antithetic variates are on by default: the second half of the paths uses the opposite
    normal draws. The error is computed on independent samples only: one per antithetic pair,
    after the Richardson combination.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray
from spricer.market.core import Market
from spricer.market.vol_surface import (
    InterpolatedVolSurface,
    OverriddenVol,
    ShiftedVol,
    VolSurface,
)
from spricer.mc.correlation import check_correlation
from spricer.mc.local_vol import flat_sigma, local_variance_at_moneyness
from spricer.monitoring import Monitoring

DEFAULT_PATHS = 100_000
DEFAULT_MAX_STEP = 1.0 / 52.0  # longest fine step under a local volatility (weekly)
_MIN_STEP = 1.0 / 252.0  # fine steps start daily...
_STEP_RATIO = 0.1  # ...and grow as t / 10 up to max_step
_DENSE_STEP = 1.25 * _MIN_STEP
Z_95 = 1.959963984540054
_TIME_TOL = 1e-10
# The local volatility is tabulated at each step on a forward log-moneyness grid spanning
# -10 to +8 ATM standard deviations (within [-4, 3]); beyond, paths keep the edge value.
_K_POINTS = 401
_K_SD_DOWN, _K_SD_UP, _K_MIN, _K_MAX = 10.0, 8.0, -4.0, 3.0


@dataclass(frozen=True)
class MCEstimate:
    """A Monte Carlo estimate with its standard error.

    Attributes
    ----------
    value
        Estimate of the expectation.
    std_error
        Standard error, computed on independent samples.
    n_paths
        Number of simulated paths.
    """

    value: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        """95 % confidence interval."""
        half = Z_95 * self.std_error
        return self.value - half, self.value + half

    def __str__(self) -> str:
        return f"{self.value:.4f} ± {self.std_error:.4f}"


def estimate(samples: ArrayLike, antithetic: bool, richardson: bool = False) -> MCEstimate:
    """Mean and standard error of per-path ``samples``.

    With ``richardson`` the second half of the rows are the coarse paths, combined as
    ``2 fine - coarse``; with ``antithetic`` the second half of the (combined) samples pairs
    with the first half and the error is computed on the pair averages.
    """
    x = np.asarray(samples, dtype=np.float64)
    if richardson:
        n = x.shape[0] // 2
        x = 2.0 * x[:n] - x[n:]
    n_paths = x.shape[0]
    if antithetic:
        half = n_paths // 2
        x = 0.5 * (x[:half] + x[half:])
    return MCEstimate(float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.shape[0])), n_paths)


@dataclass(frozen=True)
class PathSet:
    """Simulated paths, recorded at the dates a product needs.

    A payoff is evaluated row by row on :attr:`spots` (and the extremes) and handed to
    :meth:`estimate`, which knows how the rows combine (antithetic pairs, Richardson).

    Attributes
    ----------
    times
        Recorded dates in years, increasing and positive.
    spots
        Spots at the recorded dates, shape ``(rows, n_times, n_assets)``. With Richardson
        extrapolation the rows are the fine paths followed by their coarse twins.
    minimum, maximum
        Lowest and highest spot observed since the previous recorded date (since today for the
        first one), same shape; None when barriers are observed at maturity only. Observation
        is continuous or at ``monitoring_times`` depending on ``monitoring``.
    rate
        Risk-free rate used for the drift and the discounting.
    monitoring, monitoring_times
        How the extremes were observed.
    antithetic, richardson
        Variance reduction and bias correction used.
    """

    times: FloatArray
    spots: FloatArray
    minimum: FloatArray | None
    maximum: FloatArray | None
    rate: float
    monitoring: Monitoring
    monitoring_times: FloatArray | None
    antithetic: bool
    richardson: bool

    @property
    def n_paths(self) -> int:
        """Number of simulated paths (a fine path and its coarse twin count as one)."""
        rows = int(self.spots.shape[0])
        return rows // 2 if self.richardson else rows

    def index(self, time: float) -> int:
        """Position of a recorded date."""
        hits = np.flatnonzero(np.abs(self.times - time) <= _TIME_TOL)
        if hits.size == 0:
            raise ValueError(f"t = {time:g}y is not a recorded date of these paths")
        return int(hits[0])

    def spot_at(self, time: float, asset: int = 0) -> FloatArray:
        """Spot of ``asset`` at a recorded date, one value per row."""
        return np.asarray(self.spots[:, self.index(time), asset])

    def extremes(self, time: float, asset: int = 0) -> tuple[FloatArray, FloatArray]:
        """Lowest and highest observed spot of ``asset`` between today and ``time``."""
        if self.minimum is None or self.maximum is None:
            raise ValueError("these paths do not track the extremes (European monitoring)")
        i = self.index(time)
        return (
            np.asarray(self.minimum[:, : i + 1, asset].min(axis=1)),
            np.asarray(self.maximum[:, : i + 1, asset].max(axis=1)),
        )

    def discount(self, time: float) -> float:
        """Risk-free discount factor."""
        return float(np.exp(-self.rate * time))

    def estimate(self, samples: ArrayLike) -> MCEstimate:
        """Estimate from per-row samples (Richardson and antithetic pairs respected)."""
        return estimate(samples, self.antithetic, self.richardson)


def _graded_steps(a: float, b: float, max_step: float, scale: float) -> list[float]:
    """Dates after ``a`` up to ``b``: steps ``min(max_step, max(min, ratio t))``, times scale."""
    out, t = [], a
    while t < b:
        h = scale * min(max_step, max(_MIN_STEP, _STEP_RATIO * t))
        t = b if t + 1.25 * h >= b else t + h
        out.append(t)
    return out


def _grid(points: FloatArray, max_step: float | None, pillars: FloatArray) -> FloatArray:
    """Coarse step dates from 0 through every point.

    With ``max_step`` None (flat volatilities, exact steps) the grid is the points themselves.
    Otherwise the local volatility's own maturities (``pillars``, where its time derivative
    jumps) are nodes too, and the graded steps are twice the fine ones: each coarse step is
    later split in two equal fine steps.
    """
    if max_step is None:
        return np.concatenate([[0.0], points])
    nodes = np.unique(np.concatenate([[0.0], points, pillars[pillars < points[-1]]]))
    out = [0.0]
    for a, b in itertools.pairwise(nodes):
        out.extend(_graded_steps(float(a), float(b), max_step, 2.0))
    return np.asarray(out)


def _pillars(surface: VolSurface) -> FloatArray:
    """Maturities of an interpolated surface (possibly shifted or overridden), else none."""
    if isinstance(surface, InterpolatedVolSurface):
        return np.asarray(surface.maturities, dtype=np.float64)
    if isinstance(surface, ShiftedVol | OverriddenVol):
        return _pillars(surface.base)
    return np.empty(0)


class _LocalVolTable:
    """Local volatility of one market tabulated at the middle of every step.

    Row ``j`` holds the volatility on a uniform forward log-moneyness grid spanning -10 to +8
    ATM standard deviations at ``mid_times[j]`` (within [-4, 3]); beyond, paths keep the edge
    value. The whole table is computed in one vectorised call before the simulation.
    """

    def __init__(self, market: Market, mid_times: FloatArray) -> None:
        t = mid_times[:, None]
        sd = market.vol.vol(market.forward(t), t) * np.sqrt(t)
        self.k_low = np.maximum(_K_MIN, -_K_SD_DOWN * sd)
        self.k_step = (np.minimum(_K_MAX, _K_SD_UP * sd) - self.k_low) / (_K_POINTS - 1)
        k = self.k_low + self.k_step * np.arange(_K_POINTS)[None, :]
        self.vols = np.sqrt(local_variance_at_moneyness(market, k, t))
        self.log_forward = np.log(market.forward(mid_times))

    def lookup(self, log_spot: FloatArray, j: int) -> FloatArray:
        """Local volatility at the given log-spots for step ``j`` (linear interpolation)."""
        u = (log_spot - self.log_forward[j] - self.k_low[j, 0]) / self.k_step[j, 0]
        u = np.clip(u, 0.0, _K_POINTS - 1.0)
        i = np.minimum(u.astype(np.intp), _K_POINTS - 2)
        frac = u - i
        row = self.vols[j]
        return np.asarray(row[i] + frac * (row[i + 1] - row[i]))


def _unique_times(values: ArrayLike, name: str) -> FloatArray:
    t = np.asarray(values, dtype=np.float64).ravel()
    if t.size == 0 or not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError(f"{name} must be positive")
    t = np.sort(t)
    keep = np.concatenate([[True], np.diff(t) > _TIME_TOL])
    return np.asarray(t[keep])


class _Diffusion:
    """Euler (or exact, for a flat volatility) step of the log-spots, with bridge extremes."""

    def __init__(
        self,
        markets: list[Market],
        flat: list[float | None],
        continuous: bool,
        mid_times: FloatArray,
    ) -> None:
        self.flat = flat
        self.continuous = continuous
        self.rate = markets[0].rate
        self.q = np.array([m.dividend_yield for m in markets])
        self.tables = [
            _LocalVolTable(m, mid_times) if s is None else None
            for m, s in zip(markets, flat, strict=True)
        ]

    def sigma(self, x: FloatArray, j: int) -> FloatArray:
        """Volatility of each row and asset for step ``j``."""
        out = np.empty_like(x)
        for i, (s_flat, table) in enumerate(zip(self.flat, self.tables, strict=True)):
            out[:, i] = s_flat if table is None else table.lookup(x[:, i], j)
        return out

    def step(
        self,
        x: FloatArray,
        j: int,
        dt: float,
        z: FloatArray,
        u: FloatArray | None,
        low: FloatArray,
        high: FloatArray,
    ) -> FloatArray:
        """Advance ``x`` by step ``j`` of length ``dt``; update ``low``/``high`` with the bridge."""
        sigma = self.sigma(x, j)
        x_next = x + (self.rate - self.q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        if self.continuous and u is not None:
            spread = 2.0 * sigma**2 * dt
            jump = (x_next - x) ** 2
            np.minimum(low, 0.5 * (x + x_next - np.sqrt(jump - spread * np.log(u[0]))), out=low)
            np.maximum(high, 0.5 * (x + x_next + np.sqrt(jump - spread * np.log(u[1]))), out=high)
        return np.asarray(x_next)


def simulate(
    markets: Market | Sequence[Market],
    times: ArrayLike,
    *,
    n_paths: int = DEFAULT_PATHS,
    seed: int | None = 0,
    correlation: ArrayLike | None = None,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    monitoring_times: ArrayLike | None = None,
    max_step: float = DEFAULT_MAX_STEP,
    antithetic: bool = True,
    richardson: bool = True,
) -> PathSet:
    """Simulate correlated paths and record them at ``times``.

    Parameters
    ----------
    markets
        One market per underlying; they must share the same rate (one currency).
    times
        Dates (years) at which spots are recorded: observation and payment dates.
    n_paths
        Number of paths (even with antithetic variates).
    seed
        Seed of the random generator; the same seed gives the same paths.
    correlation
        Correlation matrix of the Brownian motions, required for several underlyings.
    monitoring
        ``EUROPEAN``: no extremes; ``CONTINUOUS``: Brownian-bridge extremes of every step;
        ``DISCRETE``: extremes over ``monitoring_times``.
    monitoring_times
        Barrier observation dates for discrete monitoring (grid nodes).
    max_step
        Longest fine step under a local volatility (ignored when every surface is flat).
    antithetic
        Use antithetic variates.
    richardson
        Remove the first-order time-discretisation bias under a local volatility (ignored, and
        not needed, when every surface is flat).
    """
    mkts = [markets] if isinstance(markets, Market) else list(markets)
    if not mkts:
        raise ValueError("at least one market is needed")
    rate = mkts[0].rate
    if any(m.rate != rate for m in mkts):
        raise ValueError("all underlyings must share the same rate (one currency)")
    n_assets = len(mkts)
    if n_assets == 1 and correlation is None:
        chol = np.ones((1, 1))
    elif correlation is None:
        raise ValueError("several underlyings need a correlation matrix")
    else:
        chol = np.linalg.cholesky(check_correlation(correlation, n_assets))
    if n_paths < 2 or (antithetic and n_paths % 2):
        raise ValueError("n_paths must be at least 2, and even with antithetic variates")
    if not max_step > 0.0:
        raise ValueError("max_step must be positive")

    record = _unique_times(times, "times")
    obs: FloatArray | None = None
    if monitoring is Monitoring.DISCRETE:
        if monitoring_times is None:
            raise ValueError("discrete monitoring needs monitoring_times")
        obs = _unique_times(monitoring_times, "monitoring_times")
    elif monitoring_times is not None:
        raise ValueError("monitoring_times only apply to discrete monitoring")
    points = record if obs is None else _unique_times(np.concatenate([record, obs]), "times")

    flat = [flat_sigma(m.vol) for m in mkts]
    exact = all(s is not None for s in flat)
    # Dates at most a day apart (daily barrier observation) are already daily steps: the grid
    # is the dates themselves, without Richardson extrapolation.
    dense = bool(np.all(np.diff(np.concatenate([[0.0], points])) <= _DENSE_STEP))
    richardson = richardson and not exact and not dense
    pillars = np.concatenate([_pillars(m.vol) for m in mkts])
    grid = _grid(points, None if exact or dense else max_step, pillars)
    # Coarse steps are split in two fine steps (the fine path alone without Richardson).
    split = not (exact or dense)
    on_grid = np.round(grid, 12)
    is_record = np.isin(on_grid, np.round(record, 12))
    is_obs = np.isin(on_grid, np.round(obs, 12)) if obs is not None else np.zeros(grid.size, bool)

    continuous = monitoring is Monitoring.CONTINUOUS
    # mid-times of the steps, three per coarse step (two fine, one coarse) when it is split
    t0s, t2s = grid[:-1], grid[1:]
    h = 0.5 * (t2s - t0s)
    mids = np.column_stack([t0s + 0.5 * h, t0s + 1.5 * h, t0s + h]).ravel() if split else t0s + h
    diffusion = _Diffusion(mkts, flat, continuous, mids)
    rng = np.random.default_rng(seed)
    draws = n_paths // 2 if antithetic else n_paths

    def normals() -> FloatArray:
        z = rng.standard_normal((draws, n_assets)) @ chol.T
        return np.concatenate([z, -z]) if antithetic else z

    def uniforms() -> FloatArray | None:
        if not continuous:
            return None
        u = rng.random((2, draws, n_assets))
        return np.concatenate([u, u], axis=1) if antithetic else u

    states = 2 if richardson else 1
    start = np.tile(np.log([m.spot for m in mkts]), (n_paths, 1))
    x = [start.copy() for _ in range(states)]  # fine, then coarse
    low = [np.full_like(start, np.inf) for _ in range(states)]
    high = [np.full_like(start, -np.inf) for _ in range(states)]
    track = monitoring is not Monitoring.EUROPEAN
    rows = states * n_paths
    spots = np.empty((rows, record.size, n_assets))
    mins = np.empty_like(spots) if track else None
    maxs = np.empty_like(spots) if track else None

    j = 0
    for n, (t0, t2, record_here, observe_here) in enumerate(
        zip(grid[:-1], grid[1:], is_record[1:], is_obs[1:], strict=True)
    ):
        if split:
            half = 0.5 * (t2 - t0)
            z1, z2 = normals(), normals()
            x[0] = diffusion.step(x[0], 3 * n, half, z1, uniforms(), low[0], high[0])
            x[0] = diffusion.step(x[0], 3 * n + 1, half, z2, uniforms(), low[0], high[0])
            if richardson:
                z = (z1 + z2) / np.sqrt(2.0)
                x[1] = diffusion.step(x[1], 3 * n + 2, t2 - t0, z, uniforms(), low[1], high[1])
        else:
            x[0] = diffusion.step(x[0], n, t2 - t0, normals(), uniforms(), low[0], high[0])
        if observe_here:
            for s in range(states):
                np.minimum(low[s], x[s], out=low[s])
                np.maximum(high[s], x[s], out=high[s])
        if record_here:
            spots[:, j, :] = np.exp(np.concatenate(x))
            if mins is not None and maxs is not None:
                mins[:, j, :] = np.exp(np.concatenate(low))
                maxs[:, j, :] = np.exp(np.concatenate(high))
                for s in range(states):
                    low[s].fill(np.inf)
                    high[s].fill(-np.inf)
            j += 1

    return PathSet(
        times=record,
        spots=spots,
        minimum=mins,
        maximum=maxs,
        rate=rate,
        monitoring=monitoring,
        monitoring_times=obs,
        antithetic=antithetic,
        richardson=richardson,
    )
