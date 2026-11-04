"""Plotly figures of the product pages: one payoff chart and one Greeks chart for every product.

Amounts are in % of the notional, levels in % of today's spot or of the initial level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spricer._typing import FloatArray
from spricer.bricks import BarrierOption
from spricer.monitoring import Monitoring
from spricer.products.coupon_notes import CouponNote
from spricer.products.notes import LevelKind, Note
from web.options import Position

# The ledger: ink lines, red ink for what is sold or lost, ledger green for the rest
PRIMARY = "#1b2a41"
GUIDE = "#9aa6b4"
GREEN = "#1e6b57"
RED = "#b3261e"
LEVEL_COLORS = ("#1e6b57", "#b3261e", "#8a6d1d", "#46546b", "#6e7a8c")
PAYOFF_LEVEL_COLORS = ("#8a6d1d", "#1e6b57", "#b3261e", "#46546b")
FONT = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"
GRID = "#e6efe2"
TEXT = "#1b2a41"
PAID = "Paid at maturity (% of notional)"


def _style(fig: go.Figure, height: int, **layout: object) -> go.Figure:
    base: dict[str, object] = {"margin": {"t": 50, "r": 10, "l": 10, "b": 10}}
    fig.update_layout(
        template="plotly_white",
        font={"family": FONT, "size": 12, "color": TEXT},
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        hoverlabel={"font_family": FONT, "bgcolor": "#ffffff", "bordercolor": GRID},
        **(base | layout),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


# --- The payoff --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Curve:
    """A curve of a payoff chart; the first of a chart is the payoff itself, in thick ink."""

    name: str
    y: FloatArray
    color: str = PRIMARY
    dash: str = "solid"
    width: float = 4.0


def payoff_chart(
    x: FloatArray,
    curves: Sequence[Curve],
    verticals: Sequence[tuple[str, float]],
    horizontals: Sequence[tuple[str, float]],
    today: float,
    x_title: str,
    guide: bool,
) -> go.Figure:
    """What a product pays at maturity: the same chart for every product.

    Vertical dashed lines are levels of the underlying (strike, barrier, cap), horizontal ones
    levels of the payoff (protection, bonus); ``guide`` adds the underlying itself, dotted.
    """
    top = float(x[-1])
    values = np.concatenate([c.y for c in curves])
    low = min(0.0, float(np.nanmin(values)))
    high = max(float(np.nanmax(values)), top if guide else 0.0)
    pad = 0.1 * max(high - low, 1.0)
    bottom = low - pad if low < 0.0 else 0.0
    fig = go.Figure()
    if guide:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=x,
                name="Underlying",
                mode="lines",
                line={"color": GUIDE, "dash": "dot", "width": 1.5},
            )
        )
    for i, (name, value) in enumerate(verticals):
        fig.add_trace(
            go.Scatter(
                x=[value, value],
                y=[bottom, high + pad],
                name=f"{name} {value:g}%",
                mode="lines",
                hoverinfo="skip",
                line={"color": LEVEL_COLORS[i % len(LEVEL_COLORS)], "dash": "dash", "width": 1.5},
            )
        )
    for i, (name, value) in enumerate(horizontals):
        fig.add_trace(
            go.Scatter(
                x=[0.0, top],
                y=[value, value],
                name=f"{name} {value:g}%",
                mode="lines",
                hoverinfo="skip",
                line={
                    "color": PAYOFF_LEVEL_COLORS[i % len(PAYOFF_LEVEL_COLORS)],
                    "dash": "dashdot",
                    "width": 1.5,
                },
            )
        )
    for curve in reversed(curves):  # the payoff last, on top of the others
        fig.add_trace(
            go.Scatter(
                x=x,
                y=curve.y,
                name=curve.name,
                mode="lines",
                line={"color": curve.color, "dash": curve.dash, "width": curve.width},
            )
        )
    if low < 0.0:
        fig.add_hline(y=0.0, line={"color": "black", "width": 1})
    fig.add_vline(
        x=today,
        line={"dash": "dot", "color": "black", "width": 1},
        annotation_text=f"today {today:.1f}%",
        annotation_position="top",
    )
    return _style(
        fig,
        470,
        xaxis_title=x_title,
        yaxis_title=PAID,
        xaxis_range=[0.0, top],
        yaxis_range=[bottom, high + pad],
        legend={"orientation": "h", "y": -0.18},
        hovermode="x unified",
    )


def _axis(today: float, levels: Sequence[float]) -> FloatArray:
    """Final levels drawn: from 0 to beyond today and the product's own levels."""
    return np.linspace(0.0, max(160.0, 1.3 * max([today, *levels])), 1601)


def option_payoff_figure(position: Position) -> go.Figure:
    """Payoff of an option position; a barrier observed along the path is drawn in both cases."""
    strategy, spot = position.strategy, position.market.spot
    verticals = [
        (label, k)
        for label, k in zip(position.setup.strike_labels, position.strikes_pct, strict=True)
    ]
    if position.barrier_pct is not None:
        verticals.append(("Barrier", position.barrier_pct))
    x = _axis(100.0, [value for _, value in verticals])
    spots = x / 100.0 * spot
    leg, brick = strategy.legs[0], strategy.legs[0].brick
    if isinstance(brick, BarrierOption) and brick.monitoring is not Monitoring.EUROPEAN:
        sign = 1.0 if brick.kind == "call" else -1.0
        vanilla = leg.quantity * np.maximum(sign * (spots - brick.strike), 0.0)
        beyond = spots >= brick.barrier if brick.is_up else spots < brick.barrier
        never = np.where(beyond, np.nan, 0.0 if brick.is_in else vanilla)
        touched = vanilla if brick.is_in else np.zeros_like(spots)
        curves = [
            Curve("Barrier never touched", 100.0 * never / spot),
            Curve("Barrier touched", 100.0 * touched / spot, RED, "dash", 3.0),
        ]
    else:
        paid = 100.0 * np.asarray(strategy.payoff(spots), dtype=np.float64) / spot
        curves = [Curve("Paid at maturity", paid)]
    return payoff_chart(
        x, curves, verticals, (), 100.0, "Underlying at maturity (% of today's spot)", False
    )


def note_payoff_figure(note: Note, today_pct: float, *, underlying: bool = True) -> go.Figure:
    """Payoff of a note against the final level, with its named levels."""
    verticals = [(lv.name, lv.value_pct) for lv in note.levels if lv.kind is LevelKind.PERFORMANCE]
    horizontals = [
        (lv.name, lv.value_pct) for lv in note.levels if lv.kind is not LevelKind.PERFORMANCE
    ]
    x = _axis(today_pct, [lv.value_pct for lv in note.levels])
    return payoff_chart(
        x,
        [Curve("Paid at maturity", np.asarray(note.payoff(x), dtype=np.float64))],
        verticals,
        horizontals,
        today_pct,
        "Underlying at maturity (% of initial level)",
        underlying,
    )


def coupon_payoff_figure(note: CouponNote, today_pct: float) -> go.Figure:
    """What a coupon note never called pays back at maturity, against the worst performance.

    The capital in thick ink, flat above the barrier for every coupon note; the final coupon
    apart, dashed: a conditional coupon steps up at its barrier, a certain one does not. With a
    barrier observed along the path the capital is at risk only if it was breached: both shown.
    """
    x = _axis(today_pct, [lv.value_pct for lv in note.levels])
    path_barrier = note.monitoring is not Monitoring.EUROPEAN and not note.barrier_breached
    capital = "Capital, barrier never breached" if path_barrier else "Capital paid back"
    curves = [
        Curve(capital, note.redemption(x, False)),
        Curve("With the final coupon", note.maturity_payoff(x, False), GREEN, "dash", 2.0),
    ]
    if path_barrier:
        curves.append(
            Curve(
                "Capital, barrier breached during the life",
                note.redemption(x, True),
                RED,
                "dash",
                3.0,
            )
        )
    return payoff_chart(
        x,
        curves,
        [(lv.name, lv.value_pct) for lv in note.levels],
        (),
        today_pct,
        "Worst performance at maturity (% of initial level)",
        True,
    )


# --- The Greeks --------------------------------------------------------------------------------


def greeks_figure(
    x: FloatArray,
    profile: Mapping[str, FloatArray],
    today: float,
    levels: Sequence[float],
    x_title: str,
) -> go.Figure:
    """Price and each Greek against the underlying: the same small multiples for every product."""
    names = tuple(profile)
    rows = int(np.ceil(len(names) / 2))
    fig = make_subplots(rows=rows, cols=2, subplot_titles=names, horizontal_spacing=0.1)
    index = int(np.argmin(np.abs(x - today)))
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
                x=x,
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
                x=[float(x[index])],
                y=[float(profile[name][index])],
                mode="markers",
                marker={"color": RED, "size": 9},
                showlegend=False,
                hovertemplate="today: %{y:.4f}<extra></extra>",
            ),
            row=r,
            col=c,
        )
    fig.update_xaxes(title_text=x_title, title_font={"size": 11})
    return _style(fig, 270 * rows + 40, hovermode="x unified")


GREEKS_CAPTION = (
    "Each panel: the Greek if the underlying stood elsewhere today. Red dot: today. Dashed lines: "
    "the product's levels."
)
