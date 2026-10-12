"""Plotly figures used by the pages. Every amount is in percent of the notional (the spot)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

import presentation
from spricer._typing import FloatArray
from spricer.bricks import Underlying
from spricer.market import InterpolatedVolSurface, Market
from spricer.strategies import Strategy

TOTAL_COLOR = "#1f4e79"
DELTA_ONE_COLOR = "#7f7f7f"
LONG_COLORS = ("#1a9850", "#2c7fb8", "#00a087", "#7fbc41", "#41b6c4")
SHORT_COLORS = ("#d73027", "#f46d43", "#c51b7d", "#8c510a", "#e08214")


def leg_colors(strategy: Strategy) -> list[str]:
    """Greens and blues for long legs, reds and oranges for short legs."""
    longs = shorts = 0
    colors = []
    for leg in strategy.legs:
        if leg.quantity >= 0:
            colors.append(LONG_COLORS[longs % len(LONG_COLORS)])
            longs += 1
        else:
            colors.append(SHORT_COLORS[shorts % len(SHORT_COLORS)])
            shorts += 1
    return colors


def spot_axis(strategy: Strategy, spot: float, n: int = 801) -> FloatArray:
    """Spot grid covering the strikes and the current spot with a margin."""
    levels = [spot, *strategy.strikes]
    lo, hi = 0.6 * min(levels), 1.4 * max(levels)
    return np.linspace(max(lo, 1e-6), hi, n)


def payoff_figure(strategy: Strategy, market: Market, *, net_of_premium: bool) -> go.Figure:
    """Payoff at maturity (or P&L net of the premium), with each leg, strikes and delta one.

    Amounts are in % of the notional, one unit of underlying worth the current spot. The P&L
    subtracts each leg's premium compounded at the risk-free rate to maturity.
    """
    notional = market.spot
    spots = spot_axis(strategy, market.spot)
    x = 100.0 * spots / notional
    growth = float(np.exp(market.rate * strategy.maturity))

    curves = []
    for leg in strategy.legs:
        y = leg.quantity * leg.brick.payoff(spots)
        if net_of_premium:
            y = y - leg.quantity * float(leg.brick.value(market)) * growth
        curves.append(100.0 * y / notional)
    total = np.sum(curves, axis=0)
    span = np.concatenate([total, *curves]) if len(curves) > 1 else total
    lo, hi = float(np.min(span)), float(np.max(span))
    pad = 0.1 * max(hi - lo, 1.0)
    y_range = (lo - pad, hi + pad)

    colors = leg_colors(strategy)
    fig = go.Figure()
    name = "P&L at maturity" if net_of_premium else "Payoff at maturity"
    fig.add_trace(
        go.Scatter(x=x, y=total, name=name, mode="lines", line={"color": TOTAL_COLOR, "width": 4})
    )
    if len(curves) > 1:
        for i, (leg, y) in enumerate(zip(strategy.legs, curves, strict=True)):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=presentation.leg_name(leg),
                    mode="lines",
                    line={"color": colors[i], "dash": "dash", "width": 2},
                )
            )
    if not net_of_premium:
        holds_underlying = any(isinstance(leg.brick, Underlying) for leg in strategy.legs)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=x if holds_underlying else x - 100.0,
                name="Delta one",
                mode="lines",
                line={"color": DELTA_ONE_COLOR, "dash": "dot", "width": 1.5},
            )
        )
    for strike, label in presentation.strike_levels(strategy, market.spot):
        level = 100.0 * strike / notional
        fig.add_trace(
            go.Scatter(
                x=[level, level],
                y=list(y_range),
                name=label,
                mode="lines",
                hoverinfo="skip",
                line={
                    "color": colors[presentation.first_leg_index(strategy, strike)],
                    "dash": "dashdot",
                    "width": 1.5,
                },
            )
        )
    if net_of_premium:
        points = presentation.breakevens(x, total)
        if points:
            fig.add_trace(
                go.Scatter(
                    x=points,
                    y=[0.0] * len(points),
                    mode="markers",
                    name="Breakeven " + " / ".join(f"{p:.1f}%" for p in points),
                    marker={"color": TOTAL_COLOR, "size": 10, "symbol": "diamond"},
                )
            )
    fig.add_hline(y=0.0, line={"color": "black", "width": 1})
    fig.add_vline(
        x=100.0,
        line={"dash": "dot", "color": "black", "width": 1},
        annotation_text="spot today",
        annotation_position="top",
    )
    fig.update_layout(
        title=f"{strategy.name} - {name.lower()}",
        xaxis_title="Spot at maturity (% of current spot)",
        yaxis_title=f"{name} (% of notional)",
        yaxis_range=list(y_range),
        legend={"title": {"text": "Legend"}, "x": 1.02, "y": 1.0},
        margin={"t": 60, "r": 10},
        hovermode="x unified",
        height=520,
    )
    return fig


def smile_figure(surface: InterpolatedVolSurface) -> go.Figure:
    """Input volatilities (markers) and the interpolated smile (lines), one per maturity."""
    pct = surface.strikes_pct
    fine = np.linspace(0.8 * pct[0], 1.2 * pct[-1], 400)
    n = len(surface.maturities)
    # short maturities light, long ones dark
    colors = sample_colorscale("Viridis", [0.85 * (1 - i / max(n - 1, 1)) for i in range(n)])
    fig = go.Figure()
    for i, maturity in enumerate(surface.maturities):
        label = f"{maturity:g}y"
        color = colors[i]
        vols = surface.vol(fine / 100.0 * surface.spot_ref, maturity)
        fig.add_trace(
            go.Scatter(
                x=fine,
                y=100.0 * vols,
                name=label,
                mode="lines",
                legendgroup=label,
                line={"color": color},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pct,
                y=100.0 * surface.vols[i],
                name=label,
                mode="markers",
                marker={"color": color, "size": 7},
                legendgroup=label,
                showlegend=False,
            )
        )
    fig.update_layout(
        xaxis_title="Strike (% of spot)",
        yaxis_title="Implied volatility (%)",
        margin={"t": 30},
        hovermode="x unified",
    )
    return fig


def density_figure(surface: InterpolatedVolSurface, maturity: float) -> go.Figure:
    """Risk-neutral density of the spot at ``maturity``, implied by the smile."""
    pct = surface.strikes_pct
    fine = np.linspace(max(0.5 * pct[0], 1.0), 1.5 * pct[-1], 600)
    strikes = fine / 100.0 * surface.spot_ref
    # density per 1 % of spot, so that the area under the curve is the probability
    density = surface.density(strikes, maturity) * surface.spot_ref / 100.0
    fig = go.Figure(
        go.Scatter(
            x=fine,
            y=100.0 * density,
            mode="lines",
            fill="tozeroy",
            line={"color": TOTAL_COLOR},
            name="density",
        )
    )
    fig.update_layout(
        xaxis_title="Spot at maturity (% of spot)",
        yaxis_title="Probability per 1 % of spot (%)",
        margin={"t": 30},
        showlegend=False,
    )
    return fig
