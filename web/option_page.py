"""Product page of an option or an option strategy: terms on the left, results in tabs.

Tabs: Payoff, Decomposition, Greeks and Hedging, as on every product page. Each callback
delegates to a pure function of this module, tested without a browser; a tab is computed only
while it is shown.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from spricer.greeks_lab import greeks_at
from web import explain as ex
from web import figures, risk_panel, sheet_route, ui
from web.market import MarketData
from web.options import (
    OptionSetup,
    Position,
    decomposition,
    display_greeks,
    setup_of,
)
from web.registry import Product
from web.scenarios import option_breakevens, option_scenarios

SPOTS_PCT = np.linspace(50.0, 150.0, 201)
AXIS = "today's spot"


# --- Layout ------------------------------------------------------------------------------------


def _terms_card(setup: OptionSetup, terms: dict[str, Any]) -> dbc.Card:
    strikes = [
        dbc.Col(
            ui.field(f"{label} (%)", ui.number(ui.term(f"strike_{i}"), terms[f"strike_{i}"], 1.0)),
            xs=6,
        )
        for i, label in enumerate(setup.strike_labels)
    ]
    fields: list[Any] = [
        ui.field("Position", ui.segmented(ui.term("side"), ("Long", "Short"), terms["side"])),
        dbc.Row(
            [
                dbc.Col(
                    ui.field(
                        "Maturity (years)", ui.number(ui.term("maturity"), terms["maturity"], 0.25)
                    )
                ),
                dbc.Col(
                    ui.field("Notional", ui.number(ui.term("notional"), terms["notional"], 1e5))
                ),
            ],
            className="g-2",
        ),
        html.Hr(),
        dbc.Row(strikes, className="g-2"),
    ]
    if setup.uses_payout:
        fields.append(
            ui.field(
                "Amount paid (% of notional)", ui.number(ui.term("payout"), terms["payout"], 1.0)
            )
        )
    if setup.barrier_type is not None:
        fields.append(
            dbc.Row(
                [
                    dbc.Col(
                        ui.field(
                            "Barrier (%)", ui.number(ui.term("barrier"), terms["barrier"], 1.0)
                        )
                    ),
                    dbc.Col(ui.monitoring_field("Barrier observed", terms["monitoring"])),
                ],
                className="g-2",
            )
        )
    fields.append(
        ui.note("Strikes and barriers in % of today's spot, amounts in % of the notional.")
    )
    solvable = {f"strike_{i}": label for i, label in enumerate(setup.strike_labels)}
    if setup.uses_payout:
        solvable["payout"] = "Amount paid"
    if setup.barrier_type is not None:
        solvable["barrier"] = "Barrier"
    solvable["maturity"] = "Maturity"
    fields += [ui.price_block(), ui.solver_block(solvable, 5.0)]
    return dbc.Card(dbc.CardBody([html.H5("Terms"), *fields]), className="terms-card")


def option_page(product: Product, prefilled: dict[str, float] | None = None) -> html.Div:
    """Product page of an option or a strategy (``product`` must have a setup)."""
    setup = setup_of(product)
    assert setup is not None  # the router only sends options here
    terms = setup.default_terms()
    applied = ui.apply_prefill(terms, prefilled or {})
    return ui.product_shell(
        product,
        setup.description,
        _terms_card(setup, terms),
        {
            "Payoff": dcc.Loading(
                [
                    dcc.Graph(id="payoff-graph", config={"displaylogo": False}, responsive=True),
                    html.Div(id="payoff-notes"),
                ]
            ),
            "Decomposition": dcc.Loading(html.Div(id="decomposition-result")),
            "Greeks": dcc.Loading(html.Div(id="greeks-result")),
            "Hedging": risk_panel.hedging_tab(),
        },
        prefilled=applied,
    )


# --- Pure functions behind the callbacks -------------------------------------------------------


def _position(store: dict[str, Any] | None, data: MarketData) -> Position:
    """Position of the stored terms on the session market; ValueError when either is invalid."""
    item = risk_panel.resolve(store, data)
    if not isinstance(item, Position):
        raise ValueError("This page has no option to price.")
    return item


def kpis(store: dict[str, Any] | None, data: MarketData) -> Any:
    """Price and Greeks of the position."""
    try:
        position = _position(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    price = position.price_pct()
    premium = price / 100.0 * position.notional
    warnings = []
    if position.barrier_crossed:
        warnings.append(
            ui.error(
                "The barrier is already crossed at today's spot: a knock-out option is dead, a "
                "knock-in option is a vanilla. Move the barrier to the other side of 100 %."
            )
        )
    return ui.headline(
        position.name,
        f"{price:.2f} %",
        f"{abs(premium):,.0f} {'paid' if premium >= 0 else 'received'}",
        display_greeks(position.strategy, position.market),
        sheet=ui.sheet_link(sheet_route.href({"store": store, "market": data})),
        below=warnings,
    )


def payoff_view(
    store: dict[str, Any] | None, data: MarketData, active: str | None
) -> tuple[Any, Any]:
    """Payoff chart and what is paid at a few final levels."""
    if active != "payoff":
        return dash.no_update, dash.no_update
    try:
        position = _position(store, data)
    except ValueError as exc:
        return go.Figure(), ui.error(str(exc))
    levels, rows = option_scenarios(position)
    points = option_breakevens(position)
    notes: list[Any] = [
        html.H6("Paid at maturity", className="mt-2"),
        ui.scenario_table("Underlying at maturity", levels, rows, points),
    ]
    if not points:
        notes.append(
            ui.note(
                "No breakeven: the gain keeps one sign, or stays at zero, wherever the spot ends."
            )
        )
    notes.append(
        ui.note(
            "In % of the notional, against the underlying's final level in % of today's spot; "
            "the gain is on the price of the position today."
        )
    )
    return figures.option_payoff_figure(position), html.Div(notes)


def decomposition_view(store: dict[str, Any] | None, data: MarketData, active: str | None) -> Any:
    """The bricks of the position: price and Greeks."""
    if active != "decomposition":
        return dash.no_update
    try:
        position = _position(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    return html.Div(
        [
            ui.bricks_table(decomposition(position)),
            ui.note(
                "The position is the sum of these bricks, each priced in closed form at the "
                "implied volatility of its own strike. In % of notional: " + ui.GREEK_UNITS_TEXT
            ),
        ]
    )


def greeks_view(store: dict[str, Any] | None, data: MarketData, active: str | None) -> Any:
    """Greeks today, what they mean, and against the spot."""
    if active != "greeks":
        return dash.no_update
    try:
        position = _position(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    market = position.market
    spots = SPOTS_PCT / 100.0 * market.spot
    profile = greeks_at(position.strategy, market, spots)
    levels = {
        label: k
        for label, k in zip(position.setup.strike_labels, position.strikes_pct, strict=True)
    }
    if position.barrier_pct is not None:
        levels["Barrier"] = position.barrier_pct
    now = display_greeks(position.strategy, market)
    cards, notes = ex.greek_cards("position", now, SPOTS_PCT, profile, levels, AXIS)
    return html.Div(
        [
            ui.greek_cards(cards, notes),
            dcc.Graph(
                figure=figures.greeks_figure(
                    SPOTS_PCT,
                    profile,
                    100.0,
                    list(levels.values()),
                    "Underlying (% of today's spot)",
                )
            ),
            ui.note(figures.GREEKS_CAPTION),
        ]
    )


# --- Callbacks ---------------------------------------------------------------------------------


def register(app: dash.Dash) -> None:
    """Register the callbacks of the option pages."""
    store = Input("product-store", "data")
    market = Input("priced-market", "data")
    tab = Input("product-tabs", "active_tab")
    app.callback(
        Output("payoff-graph", "figure"),
        Output("payoff-notes", "children"),
        store,
        market,
        tab,
    )(payoff_view)
    app.callback(Output("decomposition-result", "children"), store, market, tab)(decomposition_view)
    app.callback(Output("greeks-result", "children"), store, market, tab)(greeks_view)


__all__ = ["option_page", "register"]
