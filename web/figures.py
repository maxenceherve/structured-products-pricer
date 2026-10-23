"""Plotly figures of the product pages. Amounts are in % of the notional (today's spot = 100 %)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spricer._typing import FloatArray
from spricer.bricks import BarrierOption, Underlying
from spricer.monitoring import Monitoring
from spricer.strategies import Strategy
from web.options import Position, breakevens, leg_name

PRIMARY = "#1f4e79"
GUIDE = "#7f7f7f"
LONG_COLORS = ("#1a9850", "#2c7fb8", "#00a087", "#7fbc41", "#41b6c4")
SHORT_COLORS = ("#d73027", "#f46d43", "#c51b7d", "#8c510a", "#e08214")
ZONE_COLORS = {"ITM": "rgba(26, 152, 80, 0.08)", "OTM": "rgba(215, 48, 39, 0.07)"}
ATM_COLOR = "rgba(31, 78, 121, 0.14)"
FONT = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"


def _style(fig: go.Figure, height: int, **layout: object) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font={"family": FONT, "size": 13},
        height=height,
        margin={"t": 50, "r": 10, "l": 10, "b": 10},
        hoverlabel={"font_family": FONT},
        **layout,
    )
    return fig


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


def _path_barrier(strategy: Strategy) -> BarrierOption | None:
    brick = strategy.legs[0].brick
    if isinstance(brick, BarrierOption) and brick.monitoring is not Monitoring.EUROPEAN:
        return brick
    return None


def payoff_figure(position: Position, *, net_of_premium: bool) -> tuple[go.Figure, list[float]]:
    """Payoff at maturity (or P&L net of the premium) with each leg, and the breakevens.

    The P&L subtracts each leg's premium compounded at the risk-free rate. A barrier observed
    along the path is drawn twice: if the barrier was never touched, and if it was.
    """
    market, strategy = position.market, position.strategy
    spot = market.spot
    levels = [spot, *strategy.strikes]
    if position.barrier_pct is not None:
        levels.append(position.barrier_pct / 100.0 * spot)
    spots = np.linspace(max(0.6 * min(levels), 1e-6), 1.4 * max(levels), 801)
    x = 100.0 * spots / spot
    growth = float(np.exp(market.rate * strategy.maturity))
    premiums = [
        leg.quantity * float(leg.brick.value(market)) * growth if net_of_premium else 0.0
        for leg in strategy.legs
    ]
    title = "P&L at maturity" if net_of_premium else "Payoff at maturity"
    fig = go.Figure()
    curves: list[FloatArray] = []
    path_barrier = _path_barrier(strategy)
    points: list[float] = []
    if path_barrier is not None:
        leg = strategy.legs[0]
        sign = 1.0 if path_barrier.kind == "call" else -1.0
        vanilla = leg.quantity * np.maximum(sign * (spots - path_barrier.strike), 0.0)
        beyond = (
            spots >= path_barrier.barrier if path_barrier.is_up else spots < path_barrier.barrier
        )
        never = np.where(beyond, np.nan, 0.0 if path_barrier.is_in else vanilla)
        touched = vanilla if path_barrier.is_in else np.zeros_like(spots)
        for y, name, color in (
            (never, "Barrier never touched", PRIMARY),
            (touched, "Barrier touched", SHORT_COLORS[1]),
        ):
            curve = 100.0 * (y - premiums[0]) / spot
            curves.append(curve)
            fig.add_trace(
                go.Scatter(x=x, y=curve, name=name, mode="lines", line={"color": color, "width": 4})
            )
    else:
        legs = []
        for leg, premium in zip(strategy.legs, premiums, strict=True):
            legs.append(100.0 * (leg.quantity * leg.brick.payoff(spots) - premium) / spot)
        total = np.sum(legs, axis=0)
        curves = [total, *legs] if len(legs) > 1 else [total]
        fig.add_trace(
            go.Scatter(x=x, y=total, name=title, mode="lines", line={"color": PRIMARY, "width": 4})
        )
        if len(legs) > 1:
            for leg, y, color in zip(strategy.legs, legs, leg_colors(strategy), strict=True):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=leg_name(leg),
                        mode="lines",
                        line={"color": color, "dash": "dash", "width": 2},
                    )
                )
        if net_of_premium:
            points = breakevens(x, total)
            if points:
                fig.add_trace(
                    go.Scatter(
                        x=points,
                        y=[0.0] * len(points),
                        mode="markers",
                        name="Breakeven",
                        marker={"color": PRIMARY, "size": 11, "symbol": "diamond"},
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
                line={"color": GUIDE, "dash": "dot", "width": 1.5},
            )
        )
    span = np.concatenate(curves)
    lo, hi = float(np.nanmin(span)), float(np.nanmax(span))
    pad = 0.1 * max(hi - lo, 1.0)
    for k in position.strikes_pct:
        fig.add_vline(x=k, line={"dash": "dashdot", "color": GUIDE, "width": 1})
    if position.barrier_pct is not None:
        fig.add_vline(
            x=position.barrier_pct,
            line={"dash": "dash", "color": SHORT_COLORS[1], "width": 1.5},
            annotation_text="barrier",
            annotation_position="top",
        )
    fig.add_hline(y=0.0, line={"color": "black", "width": 1})
    fig.add_vline(
        x=100.0,
        line={"dash": "dot", "color": "black", "width": 1},
        annotation_text="spot today",
        annotation_position="top left",
    )
    _style(
        fig,
        460,
        xaxis_title="Spot at maturity (% of today's spot)",
        yaxis_title=f"{title} (% of notional)",
        yaxis_range=[lo - pad, hi + pad],
        legend={"orientation": "h", "y": -0.18},
        hovermode="x unified",
    )
    return fig, points


def greeks_grid_figure(
    position: Position,
    spots_pct: FloatArray,
    profile: Mapping[str, FloatArray],
    names: tuple[str, ...],
) -> go.Figure:
    """Small multiples: each Greek against the spot, with the money zones and today's spot."""
    cols = 2
    rows = int(np.ceil(len(names) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names, horizontal_spacing=0.1)
    lo, hi = float(spots_pct[0]), float(spots_pct[-1])
    strikes = position.strikes_pct
    bands = position.atm_bands()
    today = int(np.argmin(np.abs(spots_pct - 100.0)))
    for i, name in enumerate(names):
        r, c = i // cols + 1, i % cols + 1
        if position.single:
            k = strikes[0]
            above, below = ("ITM", "OTM") if position.setup.kind == "call" else ("OTM", "ITM")
            for x0, x1, zone in ((k, hi, above), (lo, k, below)):
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=ZONE_COLORS[zone],
                    line_width=0,
                    row=r,
                    col=c,
                    exclude_empty_subplots=False,
                )
        for x0, x1 in bands:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=ATM_COLOR,
                line_width=0,
                row=r,
                col=c,
                exclude_empty_subplots=False,
            )
        for k in strikes:
            fig.add_vline(
                x=k,
                line={"dash": "dash", "color": GUIDE, "width": 1},
                row=r,
                col=c,
                exclude_empty_subplots=False,
            )
        fig.add_trace(
            go.Scatter(
                x=spots_pct,
                y=profile[name],
                name=name,
                mode="lines",
                line={"color": PRIMARY, "width": 2.5},
                showlegend=False,
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=[100.0],
                y=[float(profile[name][today])],
                mode="markers",
                marker={"color": SHORT_COLORS[0], "size": 9},
                showlegend=False,
                hovertemplate="today: %{y:.4f}<extra></extra>",
            ),
            row=r,
            col=c,
        )
    fig.update_xaxes(title_text="Spot (% of today)", title_font={"size": 11})
    return _style(fig, 270 * rows + 60, hovermode="x unified")


def greek_time_figure(
    maturities: FloatArray, curves: Mapping[str, FloatArray], greek: str
) -> go.Figure:
    """One Greek against the remaining maturity, in, at and out of the money."""
    colors = (LONG_COLORS[0], PRIMARY, SHORT_COLORS[0])
    fig = go.Figure()
    for (label, values), color in zip(curves.items(), colors, strict=False):
        fig.add_trace(
            go.Scatter(
                x=maturities,
                y=values,
                name=label,
                mode="lines",
                line={"color": color, "width": 2.5},
            )
        )
    return _style(
        fig,
        420,
        title=f"{greek} as expiry approaches",
        xaxis_title="Time to maturity (years)",
        yaxis_title=greek,
        xaxis_autorange="reversed",
        legend={"orientation": "h", "y": -0.22},
        hovermode="x unified",
    )


def hedge_histogram_figure(pnl: FloatArray, expected: float) -> go.Figure:
    """Distribution of the hedged P&L with its mean and the theoretical mean."""
    fig = go.Figure(
        go.Histogram(x=pnl, nbinsx=60, name="Hedged P&L", marker_color=LONG_COLORS[1], opacity=0.8)
    )
    fig.add_vline(
        x=float(np.mean(pnl)),
        line={"color": PRIMARY, "width": 2},
        annotation_text="mean",
        annotation_position="top",
    )
    fig.add_vline(
        x=expected,
        line={"color": SHORT_COLORS[0], "width": 2, "dash": "dash"},
        annotation_text="theory",
        annotation_position="bottom",
    )
    return _style(
        fig,
        380,
        title="Hedged P&L at maturity across paths",
        xaxis_title="P&L (% of notional)",
        yaxis_title="Paths",
        showlegend=False,
    )


def hedge_path_figure(
    times: FloatArray,
    spot: FloatArray,
    delta: FloatArray,
    gamma_pnl: FloatArray,
    theta_pnl: FloatArray,
) -> go.Figure:
    """One hedged path: spot and delta, then the cumulative gamma and theta P&L."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{}]],
        subplot_titles=("Spot and delta of the option", "Cumulative P&L of the hedged position"),
        vertical_spacing=0.12,
    )
    fig.add_trace(go.Scatter(x=times, y=spot, name="Spot", line={"color": PRIMARY}), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=times[: len(delta)],
            y=delta,
            name="Delta (shares per option)",
            line={"color": LONG_COLORS[0], "dash": "dot"},
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    steps = times[1 : len(gamma_pnl) + 1]
    for y, name, line in (
        (np.cumsum(gamma_pnl), "Gamma: 1/2 Γ dS²", {"color": LONG_COLORS[1]}),
        (np.cumsum(theta_pnl), "Theta, carry and costs", {"color": SHORT_COLORS[0]}),
        (np.cumsum(gamma_pnl + theta_pnl), "Total", {"color": "black", "width": 2.5}),
    ):
        fig.add_trace(go.Scatter(x=steps, y=y, name=name, line=line), row=2, col=1)
    fig.update_xaxes(title_text="Years", row=2, col=1)
    return _style(fig, 560, legend={"orientation": "h", "y": -0.15})


def scenario_heatmap_figure(
    spot_moves: list[float], vol_moves: list[float], pnl: FloatArray, days: float
) -> go.Figure:
    """Full-revaluation P&L for each spot and volatility move."""
    limit = float(np.max(np.abs(pnl))) or 1.0
    fig = go.Figure(
        go.Heatmap(
            x=[f"{m:+g} %" for m in spot_moves],
            y=[f"{v:+g} pts" for v in vol_moves],
            z=pnl,
            zmin=-limit,
            zmax=limit,
            colorscale=[[0.0, SHORT_COLORS[0]], [0.5, "#ffffff"], [1.0, LONG_COLORS[0]]],
            text=np.round(pnl, 2),
            texttemplate="%{text}",
            colorbar={"title": "P&L %"},
            hovertemplate="spot %{x}, vol %{y}: %{z:.3f} %<extra></extra>",
        )
    )
    return _style(
        fig,
        400,
        title=f"Full revaluation after {days:g} days",
        xaxis_title="Spot move",
        yaxis_title="Volatility move",
    )
