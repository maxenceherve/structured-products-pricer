"""Plotly figures of the product pages. Amounts are in % of the notional (today's spot = 100 %)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spricer._typing import FloatArray
from spricer.bricks import BarrierOption
from spricer.monitoring import Monitoring
from spricer.packaging import Equivalence
from spricer.products.coupon_notes import CouponNote
from spricer.products.notes import LevelKind, Note
from spricer.strategies import Strategy
from web.options import Position, breakevens

PRIMARY = "#1f4e79"
GUIDE = "#7f7f7f"
LONG_COLORS = ("#1a9850", "#2c7fb8", "#00a087", "#7fbc41", "#41b6c4")
SHORT_COLORS = ("#d73027", "#f46d43", "#c51b7d", "#8c510a", "#e08214")
ZONE_COLORS = {"ITM": "rgba(26, 152, 80, 0.08)", "OTM": "rgba(215, 48, 39, 0.07)"}
ATM_COLOR = "rgba(31, 78, 121, 0.14)"
FONT = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"


def _style(fig: go.Figure, height: int, **layout: object) -> go.Figure:
    base: dict[str, object] = {"margin": {"t": 50, "r": 10, "l": 10, "b": 10}}
    fig.update_layout(
        template="plotly_white",
        font={"family": FONT, "size": 13},
        height=height,
        hoverlabel={"font_family": FONT},
        **(base | layout),
    )
    return fig


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
        total = np.sum(
            [
                100.0 * (leg.quantity * leg.brick.payoff(spots) - premium) / spot
                for leg, premium in zip(strategy.legs, premiums, strict=True)
            ],
            axis=0,
        )
        curves = [total]
        fig.add_trace(
            go.Scatter(x=x, y=total, name=title, mode="lines", line={"color": PRIMARY, "width": 4})
        )
        # the strikes, marked on the payoff itself
        ks = np.array(position.strikes_pct)
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=np.interp(ks, x, total),
                mode="markers+text",
                name="Strikes",
                text=[f"K {k:g}%" for k in ks],
                textposition="top center",
                marker={"color": PRIMARY, "size": 9, "line": {"color": "white", "width": 2}},
                hovertemplate="strike %{x:g} %<extra></extra>",
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
    span = np.concatenate(curves)
    lo, hi = float(np.nanmin(span)), float(np.nanmax(span))
    pad = 0.1 * max(hi - lo, 1.0)
    if position.barrier_pct is not None:
        fig.add_vline(
            x=position.barrier_pct,
            line={"dash": "dash", "color": SHORT_COLORS[1], "width": 1.5},
            annotation_text="barrier",
            annotation_position="top",
        )
    fig.add_hline(y=0.0, line={"color": "black", "width": 1})
    _style(
        fig,
        460,
        xaxis_title="Spot at maturity (% of today's spot = 100)",
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
    keys = [("At the money zone", ATM_COLOR, "square")]
    if position.single:
        keys += [
            ("In the money", ZONE_COLORS["ITM"].replace("0.08", "0.4"), "square"),
            ("Out of the money", ZONE_COLORS["OTM"].replace("0.07", "0.4"), "square"),
        ]
    keys.append(("Today's spot", SHORT_COLORS[0], "circle"))
    for label, color, symbol in keys:  # legend only: what the shading and the dot mean
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=label,
                marker={"color": color, "size": 12, "symbol": symbol},
            )
        )
    fig.update_xaxes(title_text="Spot (% of today)", title_font={"size": 11})
    return _style(
        fig,
        270 * rows + 90,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0.0},
        margin={"t": 80, "r": 10, "l": 10, "b": 10},
    )


def greek_time_figure(
    maturities: FloatArray,
    curves: Mapping[str, FloatArray],
    greek: str,
    highlight: str | None = None,
) -> go.Figure:
    """One Greek against the time left, at a few spot levels; ``highlight`` drawn thickest."""
    colors = (LONG_COLORS[0], PRIMARY, SHORT_COLORS[0], GUIDE)
    fig = go.Figure()
    for (label, values), color in zip(curves.items(), colors, strict=False):
        main = label == highlight
        fig.add_trace(
            go.Scatter(
                x=maturities,
                y=values,
                name=label,
                mode="lines",
                line={
                    "color": "black" if main else color,
                    "width": 4 if main else 2,
                    "dash": "solid" if main else "dash",
                },
            )
        )
        if main:
            fig.add_trace(
                go.Scatter(
                    x=[maturities[0]],
                    y=[values[0]],
                    mode="markers+text",
                    marker={"color": SHORT_COLORS[0], "size": 10},
                    text=["now"],
                    textposition="top center",
                    showlegend=False,
                    hovertemplate="today: %{y:.3f}<extra></extra>",
                )
            )
    return _style(
        fig,
        440,
        xaxis_title="Years to maturity (expiry on the right)",
        yaxis_title=greek,
        xaxis_autorange="reversed",
        legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 0.0},
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


# --- Notes ---------------------------------------------------------------------------------------

LEVEL_COLORS = ("#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#17becf")
PAYOFF_LEVEL_COLORS = ("#8b0000", "#6a3d9a", "#b15928", "#1f78b4")


def note_payoff_figure(note: Note, today_pct: float, *, underlying: bool = True) -> go.Figure:
    """Payoff of a note at maturity against the final performance, with its named levels.

    Vertical lines are levels of the underlying (strike, barrier, cap), horizontal lines levels
    of the payoff (protection, bonus); the grey dotted line is the underlying itself.
    """
    top = max(200.0, 1.3 * max([today_pct, *(lv.value_pct for lv in note.levels)]))
    perf = np.linspace(0.0, top, 2001)
    payoff = np.asarray(note.payoff(perf))
    low = min(0.0, float(np.min(payoff)))
    y_top = 1.1 * max(float(np.max(payoff)), top)
    fig = go.Figure()
    if underlying:
        fig.add_trace(
            go.Scatter(
                x=perf,
                y=perf,
                name="Underlying",
                mode="lines",
                line={"color": GUIDE, "dash": "dot", "width": 1.5},
            )
        )
    else:  # a warrant pays a fraction of the notional: the underlying would flatten the chart
        y_top = 1.15 * max(float(np.max(payoff)), 1.0)
    vertical = horizontal = 0
    for level in note.levels:
        if level.kind is LevelKind.PERFORMANCE:
            color = LEVEL_COLORS[vertical % len(LEVEL_COLORS)]
            vertical += 1
            x, y, dash = [level.value_pct] * 2, [low, y_top], "dash"
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
            x=perf, y=payoff, name="Payoff", mode="lines", line={"color": PRIMARY, "width": 4}
        )
    )
    fig.add_vline(
        x=today_pct,
        line={"dash": "dot", "color": "black", "width": 1},
        annotation_text=f"today {today_pct:.1f}%",
        annotation_position="top",
    )
    return _style(
        fig,
        480,
        xaxis_title="Underlying at maturity (% of initial level)",
        yaxis_title="Paid at maturity (% of notional)",
        xaxis_range=[0.0, top],
        yaxis_range=[low, y_top],
        legend={"orientation": "h", "y": -0.18},
        hovermode="x unified",
    )


def note_greeks_figure(
    note: Note, perf_pct: FloatArray, profile: Mapping[str, FloatArray], today_pct: float
) -> go.Figure:
    """Price, delta, gamma and vega of a note against the underlying's level."""
    names = tuple(profile)
    fig = make_subplots(rows=2, cols=2, subplot_titles=names, horizontal_spacing=0.1)
    today = int(np.argmin(np.abs(perf_pct - today_pct)))
    levels = [lv.value_pct for lv in note.levels if lv.kind is LevelKind.PERFORMANCE]
    for i, name in enumerate(names):
        r, c = i // 2 + 1, i % 2 + 1
        for level in levels:
            fig.add_vline(
                x=level,
                line={"dash": "dash", "color": GUIDE, "width": 1},
                row=r,
                col=c,
                exclude_empty_subplots=False,
            )
        fig.add_trace(
            go.Scatter(
                x=perf_pct,
                y=profile[name],
                mode="lines",
                line={"color": PRIMARY, "width": 2.5},
                showlegend=False,
                name=name,
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=[perf_pct[today]],
                y=[float(profile[name][today])],
                mode="markers",
                marker={"color": SHORT_COLORS[0], "size": 9},
                showlegend=False,
                hovertemplate="today: %{y:.4f}<extra></extra>",
            ),
            row=r,
            col=c,
        )
    fig.update_xaxes(title_text="Underlying (% of initial)", title_font={"size": 11})
    return _style(fig, 600, hovermode="x unified")


def coupon_payoff_figure(note: CouponNote, today_pct: float) -> go.Figure:
    """Paid at maturity by a coupon note never called, against the worst performance.

    With a barrier observed along the path the capital is at risk below the strike only if the
    barrier was breached: the chart then shows both cases.
    """
    top = max(160.0, 1.25 * max([today_pct, *(lv.value_pct for lv in note.levels)]))
    perf = np.linspace(0.0, top, 1601)
    never = np.asarray(note.maturity_payoff(perf, False))
    y_top = 1.1 * max(top, float(np.max(never)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=perf,
            y=perf,
            name="Underlying",
            mode="lines",
            line={"color": GUIDE, "dash": "dot", "width": 1.5},
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
                line={"color": LEVEL_COLORS[i % len(LEVEL_COLORS)], "dash": "dash", "width": 1.5},
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
            y=never,
            name="Barrier never breached" if path_barrier else "Paid at maturity",
            mode="lines",
            line={"color": PRIMARY, "width": 4},
        )
    )
    fig.add_vline(
        x=today_pct,
        line={"dash": "dot", "color": "black", "width": 1},
        annotation_text=f"today {today_pct:.1f}%",
        annotation_position="top",
    )
    return _style(
        fig,
        460,
        title="If never called: paid at maturity",
        xaxis_title="Worst performance at maturity (% of initial level)",
        yaxis_title="Paid at maturity (% of notional)",
        xaxis_range=[0.0, top],
        yaxis_range=[0.0, y_top],
        legend={"orientation": "h", "y": -0.2},
        hovermode="x unified",
    )


def call_probability_figure(note: CouponNote, calls_pct: list[float]) -> go.Figure:
    """Probability of a call at each date, of reaching maturity, and the cumulative line."""
    labels = [f"{o.time:.2f}y" for o in note.observations]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=labels, y=calls_pct, name="Called at this date", marker_color=LONG_COLORS[1])
    )
    fig.add_trace(
        go.Bar(
            x=[f"{note.maturity:.2f}y (maturity)"],
            y=[100.0 - sum(calls_pct)],
            name="Reaches maturity",
            marker_color=SHORT_COLORS[1],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=np.cumsum(calls_pct),
            name="Called by this date",
            mode="lines+markers",
            line={"color": PRIMARY, "width": 2},
        )
    )
    return _style(
        fig,
        420,
        title="When does the note end?",
        xaxis_title="Observation date (years from today)",
        yaxis_title="Probability (%)",
        yaxis_range=[0.0, 100.0],
        legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 1.0, "xanchor": "right"},
    )


def packaging_figure(equivalence: Equivalence, spot: float) -> go.Figure:
    """Payoffs at maturity of both constructions of a pair: the two lines coincide."""
    grid = np.linspace(0.0, 2.0 * spot, 401)
    fig = go.Figure()
    for construction, line in (
        (equivalence.first, {"color": PRIMARY, "width": 5}),
        (equivalence.second, {"color": SHORT_COLORS[1], "width": 2, "dash": "dash"}),
    ):
        fig.add_trace(
            go.Scatter(
                x=100.0 * grid / spot,
                y=100.0 * np.asarray(construction.strategy.payoff(grid)) / spot,
                name=construction.name,
                mode="lines",
                line=line,
            )
        )
    return _style(
        fig,
        340,
        xaxis_title="Underlying at maturity (% of today)",
        yaxis_title="Paid (% of today's spot)",
        legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 0.0},
        hovermode="x unified",
    )
