"""Product page of a note priced by bricks: capital protection, certificates, shark warrants.

Tabs: Payoff, Decomposition, Greeks and Hedging, as on every product page. Each callback
delegates to a pure function of this module.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from spricer.monitoring import Monitoring
from spricer.products.capital_protection import NoteSpec
from spricer.products.notes import LevelKind
from spricer.products.participation import SHARK_WARRANTS
from web import explain as ex
from web import figures, risk_panel, sheet_route, ui
from web.market import MarketData
from web.notes import (
    TOGGLED,
    NotePosition,
    decomposition,
    default_terms,
    desk_greeks,
    note_spec,
    spot_profile,
)
from web.registry import Product
from web.scenarios import note_breakevens, note_scenarios

GREEK_PERF = np.linspace(40.0, 180.0, 141)
AXIS = "the initial level"


# --- Layout ------------------------------------------------------------------------------------


def _terms_card(spec: NoteSpec, terms: dict[str, Any]) -> dbc.Card:
    toggles = {p.key: p for p in spec.params if p.kind == "toggle"}
    levels = []
    for p in spec.params:
        if p.kind == "toggle":
            continue
        control: Any = ui.number(ui.term(p.key), terms[p.key], 1.0)
        toggle = toggles.get(TOGGLED.get(p.key, ""))
        if toggle is not None:
            control = html.Div(
                [
                    dbc.Switch(
                        id=ui.term(toggle.key), label=toggle.label, value=bool(toggle.default)
                    ),
                    control,
                ]
            )
        levels.append(dbc.Col(ui.field(f"{p.label} (%)", control), xs=6))
    fields: list[Any] = [
        dbc.Row(
            [
                dbc.Col(
                    ui.field(
                        "Maturity (years)", ui.number(ui.term("maturity"), terms["maturity"], 0.5)
                    )
                ),
                dbc.Col(
                    ui.field("Notional", ui.number(ui.term("notional"), terms["notional"], 1e5))
                ),
            ],
            className="g-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    ui.field(
                        "Initial level",
                        ui.number(ui.term("initial"), None, 1.0, placeholder="= spot today"),
                    )
                ),
                dbc.Col(
                    ui.field(
                        "Funding spread (bp)",
                        ui.number(ui.term("spread_bp"), None, 5.0, placeholder="= market"),
                    )
                ),
            ],
            className="g-2",
        ),
        html.Hr(),
        dbc.Row(levels, className="g-2"),
    ]
    if spec.monitoring_choice:
        fields.append(ui.monitoring_field("Barrier observed", terms["monitoring"]))
    fields.append(
        ui.note(
            "Levels in % of the initial level, amounts in % of the notional. The issuer's "
            "funding spread discounts every cash flow of the note, not the underlying's drift."
        )
    )
    solvable = {p.key: p.label for p in spec.params if p.kind != "toggle"}
    solvable |= {"maturity": "Maturity", "spread_bp": "Funding spread"}
    fields += [ui.price_block(), ui.solver_block(solvable, 5.0 if spec in SHARK_WARRANTS else 98.0)]
    return dbc.Card(dbc.CardBody([html.H5("Terms"), *fields]), className="terms-card")


def note_page(product: Product, prefilled: dict[str, float] | None = None) -> html.Div:
    """Product page of a note priced by bricks (``product`` must be one)."""
    spec = note_spec(product)
    assert spec is not None  # the router only sends notes here
    terms = default_terms(spec)
    applied = ui.apply_prefill(terms, prefilled or {})
    return ui.product_shell(
        product,
        spec.description,
        _terms_card(spec, terms),
        {
            "Payoff": dcc.Loading(
                [
                    ui.payoff_graph("note-payoff-graph"),
                    html.Div(id="note-payoff-notes"),
                ]
            ),
            "Decomposition": dcc.Loading(html.Div(id="note-decomposition")),
            "Greeks": dcc.Loading(html.Div(id="note-greeks")),
            "Hedging": risk_panel.hedging_tab(),
        },
        prefilled=applied,
    )


# --- Pure functions behind the callbacks -------------------------------------------------------


def _note(store: dict[str, Any] | None, data: MarketData) -> NotePosition:
    item = risk_panel.resolve(store, data)
    if not isinstance(item, NotePosition):
        raise ValueError("This page has no note to price.")
    return item


def kpis(store: dict[str, Any] | None, data: MarketData) -> Any:
    """Price and Greeks of the note."""
    try:
        position = _note(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    price = position.value()
    bond = position.bond_pct()
    facts = [f"Underlying today {position.performance_pct:.1f} % of its initial level"]
    if bond:
        facts.insert(0, f"Bond {bond:.2f} % and options {price - bond:+.2f} %")
    warnings = []
    if position.monitoring is not Monitoring.EUROPEAN:
        warnings.append(
            ui.note(
                "Barrier observed along the path: the closed form reads one volatility per strike."
            )
        )
    warning = position.warning()
    if warning:
        warnings.append(ui.error(warning))
    return ui.headline(
        f"Holder of the {position.note.name.lower()}",
        f"{price:.2f} %",
        f"{price / 100.0 * position.notional:,.0f} for the notional",
        desk_greeks(position.greeks(), position.market.spot),
        facts=facts,
        sheet=ui.sheet_link(sheet_route.href({"store": store, "market": data})),
        below=warnings,
    )


def payoff_view(
    store: dict[str, Any] | None, data: MarketData, active: str | None
) -> tuple[Any, Any]:
    """Payoff at maturity with the named levels."""
    if active != "payoff":
        return dash.no_update, dash.no_update
    try:
        position = _note(store, data)
    except ValueError as exc:
        return go.Figure(), ui.error(str(exc))
    fig = figures.note_payoff_figure(
        position.note, position.performance_pct, underlying=position.spec not in SHARK_WARRANTS
    )
    levels, rows = note_scenarios(position)
    notes: list[Any] = [
        html.H6("Paid at maturity", className="mt-2"),
        ui.scenario_table("Underlying at maturity", levels, rows, note_breakevens(position)),
        ui.note(
            "In % of the notional, against the underlying's final level in % of its initial "
            "level; the gain is on the price of the note today."
        ),
    ]
    if position.monitoring is not Monitoring.EUROPEAN:
        notes.append(
            ui.note(
                "The barrier is observed along the path: the chart shows the payoff when it was "
                "never touched before maturity; once touched, the note pays as written for a "
                "breached barrier whatever the final level."
            )
        )
    return fig, html.Div(notes)


def decomposition_view(store: dict[str, Any] | None, data: MarketData, active: str | None) -> Any:
    """The bricks of the note: price and Greeks."""
    if active != "decomposition":
        return dash.no_update
    try:
        position = _note(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    return html.Div(
        [
            ui.bricks_table(decomposition(position)),
            ui.note(
                "The note is replicated exactly by these bricks, each priced in closed form at "
                "the implied volatility of its own strike. In % of notional: " + ui.GREEK_UNITS_TEXT
            ),
        ]
    )


def greeks_view(store: dict[str, Any] | None, data: MarketData, active: str | None) -> Any:
    """Greeks today, what they mean, and against the underlying's level."""
    if active != "greeks":
        return dash.no_update
    try:
        position = _note(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    today = position.performance_pct
    perf = np.union1d(GREEK_PERF, [round(today, 6)])
    profile = spot_profile(position, perf)
    levels = {
        lv.name: lv.value_pct for lv in position.note.levels if lv.kind is LevelKind.PERFORMANCE
    }
    now = desk_greeks(position.greeks(), position.market.spot)
    cards, notes = ex.greek_cards("note", now, perf, profile, levels, AXIS)
    return html.Div(
        [
            ui.greek_cards(cards, notes),
            dcc.Graph(
                figure=figures.greeks_figure(
                    perf, profile, today, list(levels.values()), "Underlying (% of initial level)"
                )
            ),
            ui.note(figures.GREEKS_CAPTION),
        ]
    )


# --- Callbacks ---------------------------------------------------------------------------------


def register(app: dash.Dash) -> None:
    """Register the callbacks of the note pages."""
    store, market = Input("product-store", "data"), Input("priced-market", "data")
    tab = Input("product-tabs", "active_tab")
    app.callback(
        Output("note-payoff-graph", "figure"),
        Output("note-payoff-notes", "children"),
        store,
        market,
        tab,
    )(payoff_view)
    app.callback(Output("note-decomposition", "children"), store, market, tab)(decomposition_view)
    app.callback(Output("note-greeks", "children"), store, market, tab)(greeks_view)


__all__ = ["note_page", "register"]
