"""Hedging tab, the same on every product page: a static hedge of the Greeks.

Shares of the underlying for the delta, listed options for the gamma and the vega, for the
notional of the terms. It works on :class:`Subject`: the option position on an option page; on
a note page the issuer who sold the note hedges it, underlying by underlying for a worst-of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, dcc, html

from spricer.bricks import EuropeanOption
from spricer.conventions import Greeks, to_desk
from spricer.hedging import static_hedge
from spricer.market import Market
from spricer.products.notes import NOTIONAL
from web import explain as ex
from web import registry, ui
from web.coupons import coupon_spec, underlying_market
from web.market import MarketData, funding_bp, market_from_data
from web.notes import NotePosition, build_note, note_spec
from web.options import Position, build_position, setup_of

HEDGES = ("Delta", "Delta + gamma", "Delta + gamma + vega")
TAB = "hedging"


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
        return build_position(setup, store["terms"], market)
    spec = note_spec(product)
    if spec is not None:
        return build_note(spec, store["terms"], market, funding_bp(data))
    raise ValueError("This page has no product to price.")


@dataclass(frozen=True)
class Subject:
    """What the tab hedges.

    Attributes
    ----------
    hedged
        Name of the position hedged ("Position: long call", "Issuer, short the note").
    short
        The hedged position is short (a market maker or an issuer).
    greeks
        Raw Greeks of one unit of the hedged position.
    units
        Units in the notional of the terms (options, or lots of 100 of notional).
    desk_notional
        Notional of one unit, to show the Greeks in % (the spot for an option, 100 for a note).
    market
        Today's market of the underlying.
    """

    hedged: str
    short: bool
    greeks: Greeks
    units: float
    desk_notional: float
    market: Market


def subject_of(item: Position | NotePosition) -> Subject:
    """The subject of an option position or of a note."""
    if isinstance(item, NotePosition):
        return Subject(
            f"Issuer, short the {item.note.name.lower()}",
            True,
            item.greeks() * -1.0,
            item.notional / NOTIONAL,
            NOTIONAL,
            item.market,
        )
    spot = item.market.spot
    return Subject(
        f"Position: {item.name}",
        item.short,
        item.strategy.greeks(item.market),
        item.notional / spot,
        spot,
        item.market,
    )


def coupon_subjects(greeks: dict[str, Any], data: MarketData) -> list[Subject]:
    """The issuer's position in each underlying of a coupon note, from its Monte Carlo Greeks."""
    out = []
    for asset, row in zip(greeks["assets"], greeks["rows"], strict=True):
        market = underlying_market(row, data)
        spot = market.spot
        raw = Greeks(
            -asset["delta"][0] / spot,
            -100.0 * asset["gamma"][0] / spot**2,
            -100.0 * asset["vega"][0],
            np.zeros(()),
            np.zeros(()),
        )
        out.append(
            Subject(
                f"Issuer, short the note on {asset['name']}",
                True,
                raw,
                greeks["notional"] / NOTIONAL,
                NOTIONAL,
                market,
            )
        )
    return out


def subjects_of(
    store: dict[str, Any] | None, data: MarketData, greeks: dict[str, Any] | None
) -> list[Subject]:
    """What a page hedges: one subject, or one per underlying of a coupon note."""
    product = registry.find((store or {}).get("slug", ""))
    if product is not None and coupon_spec(product) is not None:
        if not greeks:
            raise ValueError("The Greeks of the note are being computed.")
        return coupon_subjects(greeks, data)
    return [subject_of(resolve(store, data))]


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


def hedging_tab() -> html.Div:
    """Controls and output of the tab."""
    return html.Div(
        [
            ui.field("Neutralise", ui.segmented("hedge-target", HEDGES, HEDGES[0])),
            _hedge_option(1, "call", 0.25),
            _hedge_option(2, "put", 2.0),
            dcc.Loading(html.Div(id="hedge-static")),
        ]
    )


# --- Pure functions behind the callbacks -------------------------------------------------------


def _desk_row(name: Any, g: Greeks, spot: float, notional: float) -> list[Any]:
    d = to_desk(g, spot=spot, notional=notional)
    return [
        name,
        ui.fmt(100.0 * float(d.delta), 2),
        ui.fmt(100.0 * float(d.gamma), 3),
        ui.fmt(float(d.vega), 4),
    ]


def _hedge_block(subject: Subject, options: list[tuple[Any, Any, Any]]) -> list[Any]:
    hedges = []
    for kind, strike, maturity in options:
        if not strike or not maturity or strike <= 0 or maturity <= 0:
            raise ValueError("Every hedging option needs a positive strike and maturity.")
        option = EuropeanOption(
            "put" if kind == "put" else "call",
            strike / 100.0 * subject.market.spot,
            float(maturity),
        )
        hedges.append(
            (f"{option.kind}s K {strike:g}% {maturity:g}y", option.greeks(subject.market))
        )
    hedge = static_hedge(subject.greeks, tuple(hedges))
    spot, notional = subject.market.spot, subject.desk_notional
    rows = [_desk_row(subject.hedged, hedge.position, spot, notional)]
    for leg in hedge.legs:
        quantity = leg.quantity * subject.units
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
    return [
        ui.table(["", "Delta (%)", "Gamma (%)", "Vega (%)"], rows, total=True),
        ui.bullets(ex.static_reading(hedge, round(subject.units), subject.short)),
    ]


def static_view(
    store: dict[str, Any] | None,
    data: MarketData,
    greeks: dict[str, Any] | None,
    active: str | None,
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
    if active != TAB:
        return dash.no_update, *rows_shown
    options = [(kind1, strike1, maturity1), (kind2, strike2, maturity2)][:needed]
    try:
        subjects = subjects_of(store, data, greeks)
        blocks = [_hedge_block(subject, options) for subject in subjects]
    except ValueError as exc:
        return ui.error(str(exc)), *rows_shown
    children: list[Any] = [piece for block in blocks for piece in block]
    children.append(
        ui.note(
            "Quantities for the notional of the terms; Greeks per unit of the position, in % of "
            "its notional. The neutralised Greeks show 0 in the total row."
        )
    )
    return html.Div(children), *rows_shown


def register(app: dash.Dash) -> None:
    """Register the callback of the tab."""
    app.callback(
        Output("hedge-static", "children"),
        Output("hedge1-row", "style"),
        Output("hedge2-row", "style"),
        Input("product-store", "data"),
        Input("priced-market", "data"),
        Input("product-greeks", "data"),
        Input("product-tabs", "active_tab"),
        Input("hedge-target", "value"),
        *[
            Input(f"hedge{n}-{field}", "value")
            for n in (1, 2)
            for field in ("kind", "strike", "maturity")
        ],
    )(static_view)
