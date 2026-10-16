"""Plotly figures used by the pages. Every amount is in percent of the notional (the spot)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

import presentation
from spricer._typing import FloatArray
from spricer.bricks import Underlying
from spricer.market import InterpolatedVolSurface, Market
from spricer.mc.coupon_notes import CouponNoteMonteCarlo
from spricer.monitoring import Monitoring
from spricer.products.coupon_notes import CouponNote
from spricer.products.notes import LevelKind, Note
from spricer.strategies import Strategy

TOTAL_COLOR = "#1f4e79"
DELTA_ONE_COLOR = "#7f7f7f"
LONG_COLORS = ("#1a9850", "#2c7fb8", "#00a087", "#7fbc41", "#41b6c4")
SHORT_COLORS = ("#d73027", "#f46d43", "#c51b7d", "#8c510a", "#e08214")
POSITIVE_HEAT, NEGATIVE_HEAT = "#1a9850", "#d73027"


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


PERFORMANCE_LEVEL_COLORS = ("#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#17becf")
PAYOFF_LEVEL_COLORS = ("#8b0000", "#6a3d9a", "#b15928", "#1f78b4")


def note_payoff_figure(note: Note, market: Market) -> go.Figure:
    """Payoff of a note at maturity against the final performance, with its named levels.

    Vertical lines are levels of the underlying (strike, barrier, cap), horizontal lines levels
    of the payoff (capital protection, bonus); the grey dotted line is the underlying itself.
    """
    today = 100.0 * market.spot / note.initial
    top = max(200.0, 1.3 * max([today, *(lv.value_pct for lv in note.levels)]))
    perf = np.linspace(0.0, top, 2001)
    payoff = note.payoff(perf)
    y_top = 1.1 * max(float(np.max(payoff)), top)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=perf,
            y=perf,
            name="Delta one",
            mode="lines",
            line={"color": DELTA_ONE_COLOR, "dash": "dot", "width": 1.5},
        )
    )
    vertical = horizontal = 0
    for level in note.levels:
        if level.kind is LevelKind.PERFORMANCE:
            color = PERFORMANCE_LEVEL_COLORS[vertical % len(PERFORMANCE_LEVEL_COLORS)]
            vertical += 1
            x, y, dash = [level.value_pct] * 2, [0.0, y_top], "dash"
        else:
            color = PAYOFF_LEVEL_COLORS[horizontal % len(PAYOFF_LEVEL_COLORS)]
            horizontal += 1
            x, y, dash = [0.0, top], [level.value_pct] * 2, "dashdot"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=f"{level.name} {level.value_pct:g}%",
                mode="lines",
                hoverinfo="skip",
                line={"color": color, "dash": dash, "width": 1.5},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=perf,
            y=payoff,
            name="Payoff",
            mode="lines",
            line={"color": TOTAL_COLOR, "width": 4},
        )
    )
    fig.add_vline(
        x=today,
        line={"dash": "dot", "color": "black", "width": 1},
        annotation_text=f"today {today:.1f}%",
        annotation_position="top",
    )
    fig.update_layout(
        title=f"{note.name} - payoff at maturity",
        xaxis_title="Underlying at maturity (% of initial level)",
        yaxis_title="Payoff (% of notional)",
        xaxis_range=[0.0, top],
        yaxis_range=[0.0, y_top],
        legend={"title": {"text": "Legend"}, "x": 1.02, "y": 1.0},
        margin={"t": 60, "r": 10},
        hovermode="x unified",
        height=540,
    )
    return fig


def coupon_payoff_figure(note: CouponNote, today_pct: float) -> go.Figure:
    """Payoff at maturity of a coupon note never called, against the worst performance.

    With a barrier observed along the path the capital is at risk below the strike only if the
    barrier was breached: the chart then shows both cases.
    """
    levels = [lv.value_pct for lv in note.levels]
    top = max(160.0, 1.25 * max([today_pct, *levels]))
    perf = np.linspace(0.0, top, 1601)
    y_top = 1.1 * max(top, float(np.max(note.maturity_payoff(perf, False))))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=perf,
            y=perf,
            name="Delta one",
            mode="lines",
            line={"color": DELTA_ONE_COLOR, "dash": "dot", "width": 1.5},
        )
    )
    for i, level in enumerate(note.levels):
        fig.add_trace(
            go.Scatter(
                x=[level.value_pct] * 2,
                y=[0.0, y_top],
                name=f"{level.name} {level.value_pct:g}%",
                mode="lines",
                hoverinfo="skip",
                line={
                    "color": PERFORMANCE_LEVEL_COLORS[i % len(PERFORMANCE_LEVEL_COLORS)],
                    "dash": "dash",
                    "width": 1.5,
                },
            )
        )
    path_barrier = note.monitoring is not Monitoring.EUROPEAN and not note.barrier_breached
    if path_barrier:
        fig.add_trace(
            go.Scatter(
                x=perf,
                y=note.maturity_payoff(perf, True),
                name="Barrier breached during the life",
                mode="lines",
                line={"color": SHORT_COLORS[0], "width": 3, "dash": "dash"},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=perf,
            y=note.maturity_payoff(perf, False),
            name="Barrier never breached" if path_barrier else "Payoff",
            mode="lines",
            line={"color": TOTAL_COLOR, "width": 4},
        )
    )
    fig.add_vline(
        x=today_pct,
        line={"dash": "dot", "color": "black", "width": 1},
        annotation_text=f"today {today_pct:.1f}%",
        annotation_position="top",
    )
    fig.update_layout(
        title="If never called: paid at maturity",
        xaxis_title="Worst performance at maturity (% of initial level)",
        yaxis_title="Paid at maturity (% of notional)",
        xaxis_range=[0.0, top],
        yaxis_range=[0.0, y_top],
        legend={"orientation": "h", "y": -0.2},
        margin={"t": 60, "r": 10},
        hovermode="x unified",
        height=520,
    )
    return fig


def call_probability_figure(note: CouponNote, result: CouponNoteMonteCarlo) -> go.Figure:
    """Probability of a call at each date, and of reaching maturity, with the cumulative line."""
    times = [o.time for o in note.observations]
    calls = [100.0 * p.value for p in result.call_probability]
    reach = 100.0 - sum(calls)
    labels = [f"{t:.2f}y" for t in times]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=labels, y=calls, name="Called at this date", marker_color=LONG_COLORS[1])
    )
    fig.add_trace(
        go.Bar(
            x=[f"{note.maturity:.2f}y (maturity)"],
            y=[reach],
            name="Reaches maturity",
            marker_color=SHORT_COLORS[1],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=np.cumsum(calls),
            name="Called by this date",
            mode="lines+markers",
            line={"color": TOTAL_COLOR, "width": 2},
        )
    )
    fig.update_layout(
        title="Probability of a call at each date",
        xaxis_title="Observation date (years from today)",
        yaxis_title="Probability (%)",
        yaxis_range=[0.0, 100.0],
        legend={"orientation": "h", "y": -0.25},
        margin={"t": 60, "r": 10},
        height=480,
    )
    return fig


ZONE_COLORS = {"ITM": "rgba(26, 152, 80, 0.08)", "OTM": "rgba(215, 48, 39, 0.07)"}
ATM_COLOR = "rgba(31, 78, 121, 0.14)"


def greeks_grid_figure(
    spots_pct: FloatArray,
    profile: dict[str, FloatArray],
    names: tuple[str, ...],
    *,
    strikes_pct: tuple[float, ...],
    atm_bands: tuple[tuple[float, float], ...],
    itm_side: str | None,
    title: str,
) -> go.Figure:
    """Small multiples: each Greek against the spot, with the money zones and today's spot.

    ``atm_bands`` are the spot ranges where an option counts as at the money; ``itm_side``
    ("above" or "below" the strike) shades the in- and out-of-the-money sides for a single
    option, None for several strikes.
    """
    from plotly.subplots import make_subplots

    cols = 2
    rows = int(np.ceil(len(names) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names, horizontal_spacing=0.08)
    lo, hi = float(spots_pct[0]), float(spots_pct[-1])
    for i, name in enumerate(names):
        r, c = i // cols + 1, i % cols + 1
        if itm_side is not None and strikes_pct:
            k = strikes_pct[0]
            above, below = ("ITM", "OTM") if itm_side == "above" else ("OTM", "ITM")
            fig.add_vrect(
                x0=k,
                x1=hi,
                fillcolor=ZONE_COLORS[above],
                line_width=0,
                row=r,
                col=c,
                exclude_empty_subplots=False,
            )
            fig.add_vrect(
                x0=lo,
                x1=k,
                fillcolor=ZONE_COLORS[below],
                line_width=0,
                row=r,
                col=c,
                exclude_empty_subplots=False,
            )
        for x0, x1 in atm_bands:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=ATM_COLOR,
                line_width=0,
                row=r,
                col=c,
                exclude_empty_subplots=False,
            )
        for k in strikes_pct:
            fig.add_vline(
                x=k,
                line={"dash": "dash", "color": "#7f7f7f", "width": 1},
                row=r,
                col=c,
                exclude_empty_subplots=False,
            )
        fig.add_vline(
            x=100.0,
            line={"dash": "dot", "color": "black", "width": 1},
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
                line={"color": TOTAL_COLOR, "width": 2.5},
                showlegend=False,
            ),
            row=r,
            col=c,
        )
        idx = int(np.argmin(np.abs(spots_pct - 100.0)))
        fig.add_trace(
            go.Scatter(
                x=[100.0],
                y=[float(profile[name][idx])],
                mode="markers",
                marker={"color": SHORT_COLORS[0], "size": 9},
                showlegend=False,
                hovertemplate="today: %{y:.4f}<extra></extra>",
            ),
            row=r,
            col=c,
        )
    fig.update_xaxes(title_text="Spot (% of today)", title_font={"size": 11})
    fig.update_layout(
        title=title,
        height=300 * rows + 80,
        margin={"t": 80, "r": 10, "l": 10},
        hovermode="x unified",
    )
    return fig


def greek_time_figure(
    maturities: FloatArray, curves: dict[str, FloatArray], greek: str
) -> go.Figure:
    """One Greek against the remaining maturity, for a few spots (ITM, ATM, OTM)."""
    colors = {"In the money": LONG_COLORS[0], "At the money": TOTAL_COLOR}
    fig = go.Figure()
    for label, values in curves.items():
        fig.add_trace(
            go.Scatter(
                x=maturities,
                y=values,
                name=label,
                mode="lines",
                line={"color": colors.get(label.split(" (")[0], SHORT_COLORS[0]), "width": 2.5},
            )
        )
    fig.update_layout(
        title=f"{greek} as expiry approaches",
        xaxis_title="Time to maturity (years)",
        yaxis_title=greek,
        xaxis_autorange="reversed",
        legend={"orientation": "h", "y": -0.25},
        hovermode="x unified",
        height=460,
        margin={"t": 60, "r": 10},
    )
    return fig


def hedge_histogram_figure(pnl: FloatArray, expected: float) -> go.Figure:
    """Distribution of the hedged P&L with its mean and the theoretical mean."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=pnl, nbinsx=60, name="Hedged P&L", marker_color=LONG_COLORS[1], opacity=0.8)
    )
    fig.add_vline(
        x=float(np.mean(pnl)),
        line={"color": TOTAL_COLOR, "width": 2},
        annotation_text="mean",
        annotation_position="top",
    )
    fig.add_vline(
        x=expected,
        line={"color": SHORT_COLORS[0], "width": 2, "dash": "dash"},
        annotation_text="theory",
        annotation_position="bottom",
    )
    fig.update_layout(
        title="Hedged P&L at maturity across paths",
        xaxis_title="P&L (% of notional)",
        yaxis_title="Paths",
        showlegend=False,
        height=420,
        margin={"t": 60, "r": 10},
    )
    return fig


def hedge_path_figure(
    times: FloatArray,
    spot: FloatArray,
    delta: FloatArray,
    gamma_pnl: FloatArray,
    theta_pnl: FloatArray,
) -> go.Figure:
    """One hedged path: spot and delta, then the cumulative gamma and theta P&L."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{}]],
        subplot_titles=("Spot and delta of the option", "Cumulative P&L of the hedged position"),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(x=times, y=spot, name="Spot", line={"color": TOTAL_COLOR}), row=1, col=1
    )
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
    fig.add_trace(
        go.Scatter(
            x=steps, y=np.cumsum(gamma_pnl), name="Gamma: 1/2 Γ dS²", line={"color": LONG_COLORS[1]}
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=np.cumsum(theta_pnl),
            name="Theta, carry and costs",
            line={"color": SHORT_COLORS[0]},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=np.cumsum(gamma_pnl + theta_pnl),
            name="Total",
            line={"color": "black", "width": 2.5},
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Years", row=2, col=1)
    fig.update_layout(
        height=620, legend={"orientation": "h", "y": -0.15}, margin={"t": 60, "r": 10}
    )
    return fig


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
            colorscale=[[0.0, NEGATIVE_HEAT], [0.5, "#ffffff"], [1.0, POSITIVE_HEAT]],
            text=np.round(pnl, 2),
            texttemplate="%{text}",
            colorbar={"title": "P&L %"},
            hovertemplate="spot %{x}, vol %{y}: %{z:.3f} %<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Full revaluation after {days:g} days",
        xaxis_title="Spot move",
        yaxis_title="Volatility move",
        height=460,
        margin={"t": 60, "r": 10},
    )
    return fig
