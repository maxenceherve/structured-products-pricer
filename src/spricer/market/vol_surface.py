"""Implied volatility surfaces: flat, and interpolated from a user-entered grid.

The interpolated surface is built from a grid of implied volatilities quoted at strikes in
percent of a reference spot, for a few maturities. Strikes are converted to absolute levels
once, at construction: every lookup is then made by absolute strike, so a product struck in
percent of its own initial level can never be read at the wrong moneyness. The smile is
sticky-strike: moving the spot of a market does not move the surface.

Interpolation, in total variance ``w = sigma^2 T`` and forward log-moneyness ``k = ln(K/F(T))``:

* in strike, a natural cubic spline of ``w`` in ``k`` for each maturity. Beyond the grid, the
  slope of ``w`` moves smoothly, over one grid spacing, from its value at the last node to that
  value clipped to Lee's moment-formula bounds (between -2 and 0 on the left wing, between 0
  and 2 on the right wing; Lee 2004), then stays constant. ``w`` is twice continuously
  differentiable everywhere, so the implied density is continuous, with no point mass at the
  grid edges; when the end slope is already admissible the wing is simply linear;
* in maturity, linear in ``w`` at fixed ``k``; before the first maturity the volatility of the
  first maturity is used, after the last one the volatility of the last one.

No-arbitrage diagnostics follow Gatheral and Jacquier (2014), *Arbitrage-free SVI volatility
surfaces*: absence of butterfly arbitrage is the positivity of

    g(k) = (1 - k w'/(2w))^2 - (w'^2 / 4) (1/w + 1/4) + w''/2,

and absence of calendar arbitrage is ``w`` non-decreasing in ``T`` at fixed ``k``. The
risk-neutral density of ``S_T`` follows from Breeden and Litzenberger (1978):
``p(K) = g(k) n(d_-(k)) / (K sqrt(w))`` with ``d_- = -k/sqrt(w) - sqrt(w)/2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline

from spricer._typing import FloatArray
from spricer.analytic.black_scholes import norm_pdf

_LEE_BOUND = 2.0  # maximal asymptotic slope of total variance in log-moneyness (Lee 2004)
_K_CLIP = 50.0  # log-moneyness clip: exp(50) is far beyond any meaningful strike


class VolSurface(Protocol):
    """Anything that returns an implied volatility for an absolute strike and a maturity."""

    def vol(self, strike: ArrayLike, maturity: ArrayLike) -> FloatArray:
        """Implied volatility (decimal) at absolute ``strike`` and ``maturity`` in years."""
        ...

    def vol_slope(self, strike: ArrayLike, maturity: ArrayLike) -> FloatArray:
        """Slope of the smile ``dsigma/dK`` per unit of absolute strike."""
        ...


@dataclass(frozen=True)
class FlatVol:
    """The same implied volatility at every strike and maturity."""

    sigma: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError("sigma must be finite and non-negative")

    def vol(self, strike: ArrayLike, maturity: ArrayLike) -> FloatArray:
        """Return ``sigma`` broadcast to the shape of the inputs."""
        shape = np.broadcast_shapes(np.shape(strike), np.shape(maturity))
        return np.full(shape, self.sigma, dtype=np.float64)

    def vol_slope(self, strike: ArrayLike, maturity: ArrayLike) -> FloatArray:
        """Return zeros: a flat smile has no slope."""
        shape = np.broadcast_shapes(np.shape(strike), np.shape(maturity))
        return np.zeros(shape, dtype=np.float64)


@dataclass(frozen=True)
class ShiftedVol:
    """A surface shifted in parallel by ``shift`` (decimal), e.g. for a vega bump."""

    base: VolSurface
    shift: float

    def vol(self, strike: ArrayLike, maturity: ArrayLike) -> FloatArray:
        """Base volatility plus the shift."""
        return self.base.vol(strike, maturity) + self.shift

    def vol_slope(self, strike: ArrayLike, maturity: ArrayLike) -> FloatArray:
        """Base slope: a parallel shift leaves the slope unchanged."""
        return self.base.vol_slope(strike, maturity)


@dataclass(frozen=True)
class OverriddenVol:
    """A surface whose implied volatility is replaced by quoted values at a few strikes.

    Each override applies to one absolute strike, at every maturity; elsewhere the base surface
    applies. The smile slope always comes from the base surface, so a digital keeps its skew
    term when the volatility of its strike is overridden.
    """

    base: VolSurface
    overrides: tuple[tuple[float, float], ...]  # (absolute strike, volatility)

    def __post_init__(self) -> None:
        for strike, vol in self.overrides:
            if not (np.isfinite(strike) and np.isfinite(vol) and vol >= 0.0):
                raise ValueError("overrides need finite strikes and non-negative volatilities")

    def vol(self, strike: ArrayLike, maturity: ArrayLike) -> FloatArray:
        """Quoted volatility at an overridden strike, base volatility elsewhere."""
        out = self.base.vol(strike, maturity)
        k = np.broadcast_to(np.asarray(strike, dtype=np.float64), out.shape)
        for quoted_strike, quoted_vol in self.overrides:
            out = np.where(np.isclose(k, quoted_strike, rtol=1e-12, atol=0.0), quoted_vol, out)
        return np.asarray(out, dtype=np.float64)

    def vol_slope(self, strike: ArrayLike, maturity: ArrayLike) -> FloatArray:
        """Slope of the base surface."""
        return self.base.vol_slope(strike, maturity)


@dataclass(frozen=True)
class ButterflyViolation:
    """A strike range where the implied density is negative, for one grid maturity."""

    maturity: float
    strike_pct_from: float
    strike_pct_to: float
    min_g: float


@dataclass(frozen=True)
class CalendarViolation:
    """A forward-moneyness range where total variance decreases between two maturities."""

    maturity_from: float
    maturity_to: float
    moneyness_pct_from: float
    moneyness_pct_to: float
    max_decrease: float


@dataclass(frozen=True)
class ArbitrageReport:
    """Result of :meth:`InterpolatedVolSurface.check_arbitrage`."""

    butterfly: tuple[ButterflyViolation, ...]
    calendar: tuple[CalendarViolation, ...]

    @property
    def is_arbitrage_free(self) -> bool:
        """True when no violation was found."""
        return not self.butterfly and not self.calendar


@dataclass(frozen=True)
class _Wing:
    """Extrapolation of ``w`` on one side of the grid, as a function of ``x = |k - k_edge|``.

    Over a distance ``delta`` the slope moves from the edge slope to the asymptotic slope along
    the smoothstep ``3u^2 - 2u^3`` (``u = x / delta``), whose derivative vanishes at both ends;
    beyond, ``w`` is linear. The curvature is therefore zero where the wing meets the natural
    spline and where it becomes linear: ``w`` is twice continuously differentiable and the
    implied density is continuous.
    """

    w_edge: float
    slope_edge: float  # dw/dx at the edge
    slope_far: float  # dw/dx beyond delta
    delta: float

    @classmethod
    def build(cls, w_edge: float, slope_edge: float, spacing: float) -> _Wing:
        """Wing whose asymptotic slope is the edge slope clipped to Lee's bounds [0, 2]."""
        slope_far = min(max(slope_edge, 0.0), _LEE_BOUND)
        delta = spacing
        if slope_edge < 0.0:
            # w decreases first: keep it above half its edge value.
            delta = min(delta, w_edge / -slope_edge)
        return cls(w_edge, slope_edge, slope_far, delta)

    def evaluate(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``w``, ``dw/dx`` and ``d2w/dx2`` at distance ``x >= 0`` from the edge."""
        jump = self.slope_far - self.slope_edge
        u = np.minimum(x, self.delta) / self.delta
        w = (
            self.w_edge
            + self.slope_edge * self.delta * u
            + jump * self.delta * (u**3 - 0.5 * u**4)
            + self.slope_far * np.maximum(x - self.delta, 0.0)
        )
        dw = self.slope_edge + jump * (3.0 * u**2 - 2.0 * u**3)
        d2w = jump * 6.0 * u * (1.0 - u) / self.delta
        return w, dw, d2w


class _Slice:
    """Total variance of one maturity as a function of forward log-moneyness."""

    def __init__(self, log_moneyness: FloatArray, total_variance: FloatArray) -> None:
        self.k_min = float(log_moneyness[0])
        self.k_max = float(log_moneyness[-1])
        self.spline = CubicSpline(log_moneyness, total_variance, bc_type="natural")
        # Wings are written in x = |k - k_edge|, so the left slope changes sign.
        self.left = _Wing.build(
            float(total_variance[0]),
            -float(self.spline(self.k_min, 1)),
            float(log_moneyness[1] - log_moneyness[0]),
        )
        self.right = _Wing.build(
            float(total_variance[-1]),
            float(self.spline(self.k_max, 1)),
            float(log_moneyness[-1] - log_moneyness[-2]),
        )

    def evaluate(self, k: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``w``, ``dw/dk`` and ``d2w/dk2`` at ``k``."""
        kc = np.clip(k, self.k_min, self.k_max)
        w_in = np.asarray(self.spline(kc), dtype=np.float64)
        dw_in = np.asarray(self.spline(kc, 1), dtype=np.float64)
        d2w_in = np.asarray(self.spline(kc, 2), dtype=np.float64)
        w_l, dw_l, d2w_l = self.left.evaluate(np.maximum(self.k_min - k, 0.0))
        w_r, dw_r, d2w_r = self.right.evaluate(np.maximum(k - self.k_max, 0.0))
        left, right = k < self.k_min, k > self.k_max
        w = np.where(left, w_l, np.where(right, w_r, w_in))
        dw = np.where(left, -dw_l, np.where(right, dw_r, dw_in))
        d2w = np.where(left, d2w_l, np.where(right, d2w_r, d2w_in))
        return w, dw, d2w


def _durrleman_g(k: FloatArray, w: FloatArray, dw: FloatArray, d2w: FloatArray) -> FloatArray:
    return (1.0 - k * dw / (2.0 * w)) ** 2 - dw**2 / 4.0 * (1.0 / w + 0.25) + d2w / 2.0


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Index ranges ``[start, end]`` of consecutive True values."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


class InterpolatedVolSurface:
    """Implied volatility surface interpolated from a strike x maturity grid.

    Parameters
    ----------
    spot_ref
        Spot level the strike percentages refer to.
    rate, dividend_yield
        Flat continuously compounded rate and dividend yield, used to locate the forward of
        each maturity (the interpolation variable is forward log-moneyness).
    maturities
        Grid maturities in years, strictly increasing and positive.
    strikes_pct
        Grid strikes in percent of ``spot_ref``, strictly increasing and positive, at least two.
    vols
        Implied volatilities (decimal), shape ``(len(maturities), len(strikes_pct))``.
    """

    def __init__(
        self,
        *,
        spot_ref: float,
        rate: float,
        dividend_yield: float,
        maturities: ArrayLike,
        strikes_pct: ArrayLike,
        vols: ArrayLike,
    ) -> None:
        t = np.asarray(maturities, dtype=np.float64).ravel()
        m = np.asarray(strikes_pct, dtype=np.float64).ravel()
        v = np.asarray(vols, dtype=np.float64)
        if not (np.isfinite(spot_ref) and spot_ref > 0.0):
            raise ValueError("spot_ref must be finite and positive")
        if t.size < 1 or np.any(t <= 0.0) or np.any(np.diff(t) <= 0.0):
            raise ValueError("maturities must be positive and strictly increasing")
        if m.size < 2 or np.any(m <= 0.0) or np.any(np.diff(m) <= 0.0):
            raise ValueError("strikes_pct must hold at least two positive, increasing values")
        if v.shape != (t.size, m.size):
            raise ValueError(f"vols must have shape {(t.size, m.size)}, got {v.shape}")
        if not np.all(np.isfinite(v)) or np.any(v <= 0.0):
            raise ValueError("vols must be finite and positive")

        self.spot_ref = float(spot_ref)
        self.rate = float(rate)
        self.dividend_yield = float(dividend_yield)
        self.maturities = t
        self.strikes_pct = m
        self.strikes = m / 100.0 * self.spot_ref
        self.vols = v.copy()
        for arr in (self.maturities, self.strikes_pct, self.strikes, self.vols):
            arr.setflags(write=False)

        forwards = self._forward(t)
        log_moneyness = np.log(self.strikes[None, :] / forwards[:, None])
        total_variance = v**2 * t[:, None]
        self._slices = tuple(_Slice(log_moneyness[i], total_variance[i]) for i in range(t.size))

    def _forward(self, maturity: FloatArray) -> FloatArray:
        return self.spot_ref * np.exp((self.rate - self.dividend_yield) * maturity)

    def _log_moneyness(self, strike: FloatArray, maturity: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore"):
            k = np.log(strike / self._forward(maturity))
        return np.clip(k, -_K_CLIP, _K_CLIP)

    def _variance_rate(
        self, k: FloatArray, maturity: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``w/T`` and its first two derivatives in ``k`` at fixed ``T``."""
        evals = [s.evaluate(k) for s in self._slices]
        w = np.stack([e[0] for e in evals])
        dw = np.stack([e[1] for e in evals])
        d2w = np.stack([e[2] for e in evals])
        tp = self.maturities
        if tp.size == 1:
            return w[0] / tp[0], dw[0] / tp[0], d2w[0] / tp[0]

        hi = np.clip(np.asarray(np.searchsorted(tp, maturity, side="right")), 1, tp.size - 1)
        lo = hi - 1
        alpha = np.clip((maturity - tp[lo]) / (tp[hi] - tp[lo]), 0.0, 1.0)
        t_safe = np.where(maturity > 0.0, maturity, 1.0)

        def blend(a: FloatArray) -> FloatArray:
            a_lo = np.take_along_axis(a, lo[None, ...], axis=0)[0]
            a_hi = np.take_along_axis(a, hi[None, ...], axis=0)[0]
            inside = ((1.0 - alpha) * a_lo + alpha * a_hi) / t_safe
            below = a[0] / tp[0]
            above = a[-1] / tp[-1]
            blended = np.where(
                maturity <= tp[0], below, np.where(maturity >= tp[-1], above, inside)
            )
            return np.asarray(blended, dtype=np.float64)

        return blend(w), blend(dw), blend(d2w)

    def _broadcast(self, strike: ArrayLike, maturity: ArrayLike) -> tuple[FloatArray, FloatArray]:
        k_abs, t = np.broadcast_arrays(
            np.asarray(strike, dtype=np.float64), np.asarray(maturity, dtype=np.float64)
        )
        if np.any(k_abs < 0.0) or np.any(t < 0.0):
            raise ValueError("strike and maturity must be non-negative")
        return k_abs, t

    def vol(self, strike: ArrayLike, maturity: ArrayLike) -> FloatArray:
        """Implied volatility at absolute ``strike`` and ``maturity`` (years)."""
        k_abs, t = self._broadcast(strike, maturity)
        rate, _, _ = self._variance_rate(self._log_moneyness(k_abs, t), t)
        return np.sqrt(rate)

    def vol_slope(self, strike: ArrayLike, maturity: ArrayLike) -> FloatArray:
        """Smile slope ``dsigma/dK`` per unit of absolute strike (zero at a zero strike)."""
        k_abs, t = self._broadcast(strike, maturity)
        rate, drate, _ = self._variance_rate(self._log_moneyness(k_abs, t), t)
        sigma = np.sqrt(rate)
        k_safe = np.where(k_abs > 0.0, k_abs, 1.0)
        return np.where(k_abs > 0.0, drate / (2.0 * sigma * k_safe), 0.0)

    def density(self, strike: ArrayLike, maturity: ArrayLike) -> FloatArray:
        """Risk-neutral density of ``S_T`` at absolute ``strike`` (zero at a zero strike)."""
        k_abs, t = self._broadcast(strike, maturity)
        if np.any(t <= 0.0):
            raise ValueError("density needs a positive maturity")
        k = self._log_moneyness(k_abs, t)
        rate, drate, d2rate = self._variance_rate(k, t)
        w, dw, d2w = rate * t, drate * t, d2rate * t
        g = _durrleman_g(k, w, dw, d2w)
        sqrt_w = np.sqrt(w)
        d_minus = -k / sqrt_w - 0.5 * sqrt_w
        k_safe = np.where(k_abs > 0.0, k_abs, 1.0)
        return np.where(k_abs > 0.0, g * norm_pdf(d_minus) / (k_safe * sqrt_w), 0.0)

    def check_arbitrage(self, n_points: int = 801, tol: float = 1e-10) -> ArbitrageReport:
        """Look for butterfly and calendar arbitrage.

        Parameters
        ----------
        n_points
            Number of log-moneyness points per check; the range covers the grid plus half its
            width on each side.
        tol
            Tolerance below which a negative ``g`` or a decrease in total variance is ignored.

        Returns
        -------
        ArbitrageReport
            Strike ranges with a negative density, per grid maturity, and forward-moneyness
            ranges where total variance decreases between consecutive maturities.
        """
        butterfly: list[ButterflyViolation] = []
        for t_i, sl in zip(self.maturities, self._slices, strict=True):
            margin = 0.5 * (sl.k_max - sl.k_min)
            k = np.linspace(sl.k_min - margin, sl.k_max + margin, n_points)
            g = _durrleman_g(k, *sl.evaluate(k))
            strikes_pct = 100.0 * self._forward(np.asarray(t_i)) * np.exp(k) / self.spot_ref
            for start, end in _runs(g < -tol):
                butterfly.append(
                    ButterflyViolation(
                        float(t_i),
                        float(strikes_pct[start]),
                        float(strikes_pct[end]),
                        float(np.min(g[start : end + 1])),
                    )
                )

        calendar: list[CalendarViolation] = []
        for i in range(len(self._slices) - 1):
            near, far = self._slices[i], self._slices[i + 1]
            lo, hi = min(near.k_min, far.k_min), max(near.k_max, far.k_max)
            margin = 0.5 * (hi - lo)
            k = np.linspace(lo - margin, hi + margin, n_points)
            decrease = near.evaluate(k)[0] - far.evaluate(k)[0]
            for start, end in _runs(decrease > tol):
                calendar.append(
                    CalendarViolation(
                        float(self.maturities[i]),
                        float(self.maturities[i + 1]),
                        float(100.0 * np.exp(k[start])),
                        float(100.0 * np.exp(k[end])),
                        float(np.max(decrease[start : end + 1])),
                    )
                )
        return ArbitrageReport(tuple(butterfly), tuple(calendar))
