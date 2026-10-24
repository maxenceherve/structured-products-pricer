"""Risk & hedging tab, shared by the option and note pages.

A static hedge of the Greeks with the underlying and listed options, and a P&L explain with
its stress matrix. Both work on a :class:`Subject`: the option position on an option page; on a
note page, the issuer who sold the note hedges it, and the P&L is the note holder's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, dcc, html

from spricer.bricks import EuropeanOption
from spricer.conventions import Greeks, to_desk
from spricer.hedging import static_hedge
from spricer.market import Market
from spricer.products.notes import NOTIONAL
from spricer.risk import STEPS, TERMS, Shock, explain, scenario_matrix
from web import explain as ex
from web import figures, registry, ui
from web.market import MarketData, funding_bp, market_from_data
from web.notes import NotePosition, build_note, note_spec
from web.options import Position, build_position, setup_of

HEDGES = ("Delta", "Delta + gamma", "Delta + gamma + vega")
STATIC, SIMULATOR, PNL = "Static hedge", "Delta-hedging simulator", "P&L explain"
SPOT_GRID = [-30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0]
VOL_GRID = [-10.0, -5.0, 0.0, 5.0, 10.0]
RISK_VIEWS_FULL = (STATIC, SIMULATOR, PNL)
TAB = "risk-hedging"


def resolve(store: dict[str, Any] | None, data: MarketData) -> Position | NotePosition:
    """The priced product of a page's stored terms; ValueError when they cannot be priced."""
    if not store:
        raise ValueError("Set the terms of the product.")
    product = registry.find(store.get("slug", ""))
    if product is None:
        raise ValueError("This page has no product to price.")
    try:
        market = market_from_data(data)
    except ValueError as exc:
        raise ValueError(f"The market data are invalid: {exc}") from None
    setup = setup_of(product)
    if setup is not None:
        return build_position(setup, store["terms"], store.get("vols"), market)
    spec = note_spec(product)
    if spec is not None:
        return build_note(spec, store["terms"], market, funding_bp(data))
    raise ValueError("This page has no product to price.")


@dataclass(frozen=True)
class Subject:
    """What the risk tab hedges and explains.

    Attributes
    ----------
    hedged
        Name of the position hedged ("Position: long call", "Issuer, short the note").
    short
        The hedged position is short (a market maker or an issuer).
    greeks
        Raw Greeks of one unit of the hedged position.
    unit
        Size of a unit in the count typed by the user (1 option, or 100 of notional).
    desk_notional
        Notional of one unit, to show the Greeks in % (the spot for an option, 100 for a note).
    market, maturity
        Today's market and the years left.
    value
        Value of the position whose P&L is explained, in %, on a market with years left.
    """

    hedged: str
    short: bool
    greeks: Greeks
    unit: float
    desk_notional: float
    market: Market
    maturity: float
    value: Callable[[Market, float], float]


def subject_of(item: Position | NotePosition) -> Subject:
    """The risk subject of an option position or of a note."""
    if isinstance(item, NotePosition):
        return Subject(
            f"Issuer, short the {item.note.name.lower()}",
            True,
            item.greeks() * -1.0,
            NOTIONAL,
            NOTIONAL,
            item.market,
            item.maturity,
            item.value,
        )
    spot, instrument = item.market.spot, item.instrument

    def value(m: Market, t: float) -> float:
        return 100.0 * float(instrument(t).value(m)) / spot

    return Subject(
        f"Position: {item.name}",
        item.short,
        item.strategy.greeks(item.market),
        1.0,
        spot,
        item.market,
        item.maturity,
        value,
    )


# --- Layout ------------------------------------------------------------------------------------


def _hedge_option(n: int, kind: str, maturity: float) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(html.Div(f"Hedging option {n}", className="field-label pt-2"), width=3),
            dbc.Col(ui.select(f"hedge{n}-kind", {"call": "Call", "put": "Put"}, kind), width=3),
            dbc.Col(
                dbc.InputGroup(
                    [ui.number(f"hedge{n}-strike", 100.0, 5.0), dbc.InputGroupText("% K")]
                ),
                width=3,
            ),
            dbc.Col(
                dbc.InputGroup(
                    [ui.number(f"hedge{n}-maturity", maturity, 0.25), dbc.InputGroupText("y")]
                ),
                width=3,
            ),
        ],
        className="g-2 mb-2",
        id=f"hedge{n}-row",
    )


def _block(view: str, content: list[Any], shown: bool) -> html.Div:
    return html.Div(
        content, id={"type": "risk-block", "view": view}, style=ui.SHOWN if shown else ui.HIDDEN
    )


def risk_tab(
    count_label: str, count_default: float, simulator: list[Any] | None = None
) -> html.Div:
    """Controls and outputs of the tab; ``simulator`` adds the delta-hedging simulator."""
    views = [STATIC, *([SIMULATOR] if simulator is not None else []), PNL]
    static = [
        dbc.Row(
            [
                dbc.Col(ui.field("Neutralise", ui.segmented("hedge-target", HEDGES, HEDGES[0]))),
                dbc.Col(
                    ui.field(count_label, ui.number("hedge-options", count_default, 1_000, min=1)),
                    width=3,
                ),
            ]
        ),
        _hedge_option(1, "call", 0.25),
        _hedge_option(2, "put", 2.0),
        dcc.Loading(html.Div(id="hedge-static")),
    ]
    pnl = [
        dbc.Row(
            [
                dbc.Col(ui.field("Spot move (%)", ui.number("pnl-move", -10.0, 1.0))),
                dbc.Col(ui.field("Volatility move (pts)", ui.number("pnl-vol", 2.0, 0.5))),
                dbc.Col(ui.field("Days elapsed", ui.number("pnl-days", 7, 1, min=0))),
                dbc.Col(ui.field("Rate move (bp)", ui.number("pnl-rate", 0.0, 5.0))),
            ],
            className="g-2",
        ),
        dcc.Loading(html.Div(id="pnl-result")),
    ]
    blocks = [_block(STATIC, static, True)]
    if simulator is not None:
        blocks.append(_block(SIMULATOR, simulator, False))
    blocks.append(_block(PNL, pnl, False))
    return html.Div([html.Div(ui.segmented("risk-view", views, STATIC), className="mb-3"), *blocks])


# --- Pure functions behind the callbacks -------------------------------------------------------


def risk_blocks(view: str | None, ids: list[dict[str, str]]) -> list[dict[str, str]]:
    """Show the controls of the chosen risk view."""
    return [ui.SHOWN if i["view"] == view else ui.HIDDEN for i in ids]


def _desk_row(name: Any, g: Greeks, spot: float, notional: float) -> list[Any]:
    d = to_desk(g, spot=spot, notional=notional)
    return [
        name,
        ui.fmt(100.0 * float(d.delta), 2),
        ui.fmt(100.0 * float(d.gamma), 3),
        ui.fmt(float(d.vega), 4),
        ui.fmt(float(d.theta), 4),
    ]


def static_view(
    store: dict[str, Any] | None,
    data: MarketData,
    active: str | None,
    view: str | None,
    count: float | None,
    target: str | None,
    kind1: str | None,
    strike1: float | None,
    maturity1: float | None,
    kind2: str | None,
    strike2: float | None,
    maturity2: float | None,
) -> tuple[Any, dict[str, str], dict[str, str]]:
    """Static hedge table and its reading, and which hedging options are needed."""
    needed = HEDGES.index(target) if target in HEDGES else 0
    rows_shown = (ui.SHOWN if needed >= 1 else ui.HIDDEN, ui.SHOWN if needed >= 2 else ui.HIDDEN)
    if active != TAB or view != STATIC:
        return dash.no_update, *rows_shown
    try:
        subject = subject_of(resolve(store, data))
        units = float(count or 0.0) / subject.unit
        if units < 1e-9:
            raise ValueError("The size of the position must be positive.")
        hedges = []
        options = ((kind1, strike1, maturity1), (kind2, strike2, maturity2))[:needed]
        for kind, strike, maturity in options:
            if not strike or not maturity or strike <= 0 or maturity <= 0:
                raise ValueError("Every hedging option needs a positive strike and maturity.")
            option = EuropeanOption(
                "put" if kind == "put" else "call",
                strike / 100.0 * subject.market.spot,
                float(maturity),
            )
            name = f"{option.kind}s K {strike:g}% {maturity:g}y"
            hedges.append((name, option.greeks(subject.market)))
        hedge = static_hedge(subject.greeks, tuple(hedges))
    except ValueError as exc:
        return ui.error(str(exc)), *rows_shown
    spot, notional = subject.market.spot, subject.desk_notional
    rows = [_desk_row(subject.hedged, hedge.position, spot, notional)]
    for leg in hedge.legs:
        quantity = leg.quantity * units
        verb = "Buy" if quantity > 0 else "Sell"
        label = f"{verb} {abs(quantity):,.0f} {leg.name if leg.name != 'Underlying' else 'shares'}"
        rows.append(
            _desk_row(
                html.Span(label, className=ui.side_class(label)),
                leg.greeks * leg.quantity,
                spot,
                notional,
            )
        )
    rows.append(_desk_row("Total", hedge.hedged, spot, notional))
    return (
        html.Div(
            [
                ui.table(["", "Delta (%)", "Gamma (%)", "Vega (%)", "Theta (%)"], rows, total=True),
                ui.note(
                    "Greeks per unit of the position in % of its notional; the quantities are "
                    "for the whole position. The neutralised Greeks show 0 in the total row."
                ),
                ui.bullets(ex.static_reading(hedge, round(units), subject.short)),
            ]
        ),
        *rows_shown,
    )


def pnl_view(
    store: dict[str, Any] | None,
    data: MarketData,
    active: str | None,
    view: str | None,
    move: float | None,
    dvol: float | None,
    days: float | None,
    dr: float | None,
) -> Any:
    """P&L explain for a market move, and the stress matrix."""
    if active != TAB or view != PNL:
        return dash.no_update
    try:
        subject = subject_of(resolve(store, data))
        shock = Shock(float(move or 0.0), float(dvol or 0.0), float(days or 0.0), float(dr or 0.0))
        attribution = explain(subject.value, subject.market, subject.maturity, shock)
        grid = scenario_matrix(
            subject.value, subject.market, subject.maturity, SPOT_GRID, VOL_GRID, shock.days
        )
    except ValueError as exc:
        return ui.error(str(exc))
    taylor = [[k, ui.fmt(attribution.terms[k], 4), ui.fmt(attribution.greeks[k], 5)] for k in TERMS]
    taylor += [
        [html.B("Taylor total"), ui.fmt(attribution.taylor, 4), ""],
        [html.B("Full revaluation"), ui.fmt(attribution.full, 4), ""],
        [html.B("Unexplained"), ui.fmt(attribution.residual, 4), ""],
    ]
    steps = [[k, ui.fmt(attribution.steps[k], 4)] for k in STEPS]
    steps.append(["Total", ui.fmt(sum(attribution.steps.values()), 4)])
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6("Taylor at today's Greeks"),
                            ui.table(["Term", "P&L (%)", "Greek"], taylor),
                            ui.note(
                                "Greeks per 1 % spot, per (1 %)², per vol point, per day, per bp."
                            ),
                        ],
                        md=7,
                    ),
                    dbc.Col(
                        [
                            html.H6("Step by step (exact)"),
                            ui.table(["Step", "P&L (%)"], steps, total=True),
                            ui.note(
                                "Revalued moving time, then the spot, then the volatility, then "
                                "the rate."
                            ),
                        ],
                        md=5,
                    ),
                ]
            ),
            ui.bullets(ex.attribution_reading(attribution)),
            dcc.Graph(
                figure=figures.scenario_heatmap_figure(SPOT_GRID, VOL_GRID, grid, shock.days)
            ),
            ui.note("Stress test: every cell is a full revaluation, never a Taylor expansion."),
        ]
    )


def register(app: dash.Dash) -> None:
    """Register the callbacks of the risk tab."""
    store, market = Input("product-store", "data"), Input("market-data", "data")
    tab, view = Input("product-tabs", "active_tab"), Input("risk-view", "value")
    app.callback(
        Output({"type": "risk-block", "view": ALL}, "style"),
        Input("risk-view", "value"),
        State({"type": "risk-block", "view": ALL}, "id"),
    )(risk_blocks)
    app.callback(
        Output("hedge-static", "children"),
        Output("hedge1-row", "style"),
        Output("hedge2-row", "style"),
        store,
        market,
        tab,
        view,
        Input("hedge-options", "value"),
        Input("hedge-target", "value"),
        *[
            Input(f"hedge{n}-{field}", "value")
            for n in (1, 2)
            for field in ("kind", "strike", "maturity")
        ],
    )(static_view)
    app.callback(
        Output("pnl-result", "children"),
        store,
        market,
        tab,
        view,
        Input("pnl-move", "value"),
        Input("pnl-vol", "value"),
        Input("pnl-days", "value"),
        Input("pnl-rate", "value"),
    )(pnl_view)
