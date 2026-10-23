"""Product page of an option or an option strategy: terms on the left, results in tabs.

Tabs: Payoff, Decomposition (leg by leg), Greeks (against the spot, against time, second
order, with their reading) and Risk & hedging (static hedge, delta-hedging simulator, P&L
explain). Each callback delegates to a pure function of this module, tested without a browser;
a tab is computed only while it is shown.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, dcc, html

from spricer.bricks import BarrierOption, EuropeanOption
from spricer.conventions import Greeks, to_desk
from spricer.greeks_lab import (
    FIRST_ORDER,
    SECOND_ORDER,
    Moneyness,
    greeks_at,
    second_order_at,
    spot_profile,
    time_profile,
)
from spricer.hedging import derman_kamal_std, simulate_delta_hedge, static_hedge
from spricer.market import FlatVol, Market
from spricer.monitoring import Monitoring
from spricer.risk import STEPS, TERMS, Shock, explain, scenario_matrix
from web import explain as ex
from web import figures, registry
from web.market import MarketData, market_from_data
from web.options import (
    GREEKS,
    MONITORING_LABELS,
    OptionSetup,
    Position,
    Terms,
    build_position,
    decomposition,
    display_greeks,
    setup_of,
    smile_vols,
)
from web.registry import Product

TABS = ("Payoff", "Decomposition", "Greeks", "Risk & hedging")
TAB_IDS = tuple(registry.slugify(t) for t in TABS)
GREEK_VIEWS = ("Against the spot", "Against time", "Second order")
RISK_VIEWS = ("Static hedge", "Delta-hedging simulator", "P&L explain")
HEDGES = ("Delta", "Delta + gamma", "Delta + gamma + vega")
REBALANCING = {"Daily": 252, "Weekly": 52, "Monthly": 12}
SIM_PATHS = (2_000, 5_000, 10_000)
SPOT_GRID = [-30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0]
VOL_GRID = [-10.0, -5.0, 0.0, 5.0, 10.0]
SPOTS_PCT = np.linspace(50.0, 150.0, 201)
HIDDEN, SHOWN = {"display": "none"}, {"display": "block"}
MONEYNESS_COLORS = {Moneyness.ITM: "success", Moneyness.ATM: "primary", Moneyness.OTM: "danger"}


# --- Small pieces ------------------------------------------------------------------------------


def _field(label: str, control: Any) -> html.Div:
    return html.Div([dbc.Label(label, className="field-label"), control], className="mb-3")


def _number(id_: Any, value: Any, step: float, **kwargs: Any) -> dbc.Input:
    return dbc.Input(id=id_, type="number", value=value, step=step, debounce=True, **kwargs)


def _segmented(id_: Any, options: Any, value: str) -> dbc.RadioItems:
    """Radio buttons drawn as a segmented control."""
    return dbc.RadioItems(
        id=id_,
        options=[{"label": o, "value": o} for o in options],
        value=value,
        className="btn-group segmented",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary btn-sm",
        labelCheckedClassName="active",
    )


def _term(key: str) -> dict[str, str]:
    return {"type": "term", "key": key}


def _fmt(value: float, digits: int = 2, signed: bool = True) -> html.Span:
    if abs(value) < 0.5 * 10.0**-digits:
        value = 0.0
    text = f"{value:+.{digits}f}" if signed and value else f"{value:.{digits}f}"
    tone = "pos" if value > 5e-5 else "neg" if value < -5e-5 else ""
    return html.Span(text, className=tone if signed else "")


def _bullets(lines: list[str]) -> html.Div:
    return html.Div([dcc.Markdown(f"- {line}") for line in lines], className="reading")


def _error(message: str) -> dbc.Alert:
    return dbc.Alert(message, color="warning", className="py-2")


def _table(
    header: list[str], rows: list[list[Any]], total: bool = False, class_name: str = ""
) -> dbc.Table:
    last = len(rows) - 1
    body = [
        html.Tr([html.Td(c) for c in row], className="total-row" if total and i == last else "")
        for i, row in enumerate(rows)
    ]
    return dbc.Table(
        [html.Thead(html.Tr([html.Th(h) for h in header])), html.Tbody(body)],
        bordered=False,
        hover=True,
        responsive=True,
        size="sm",
        className=f"result-table {class_name}",
    )


# --- Layout ------------------------------------------------------------------------------------


def _terms_card(setup: OptionSetup) -> dbc.Card:
    terms = setup.default_terms()
    fields: list[Any] = [
        _field("Position", _segmented(_term("side"), ("Long", "Short"), terms["side"])),
        dbc.Row(
            [
                dbc.Col(
                    _field("Maturity (years)", _number(_term("maturity"), 1.0, 0.25, min=0.01))
                ),
                dbc.Col(_field("Notional", _number(_term("notional"), terms["notional"], 100_000))),
            ],
            className="g-2",
        ),
    ]
    strikes = [
        dbc.Col(
            _field(f"{label} (%)", _number(_term(f"strike_{i}"), terms[f"strike_{i}"], 1.0)),
            xs=6,
        )
        for i, label in enumerate(setup.strike_labels)
    ]
    fields.append(dbc.Row(strikes, className="g-2"))
    if setup.uses_payout:
        fields.append(
            _field("Amount paid (% of notional)", _number(_term("payout"), terms["payout"], 1.0))
        )
    if setup.barrier_type is not None:
        fields.append(
            dbc.Row(
                [
                    dbc.Col(
                        _field("Barrier (%)", _number(_term("barrier"), terms["barrier"], 1.0))
                    ),
                    dbc.Col(
                        _field(
                            "Observed",
                            dbc.Select(
                                id=_term("monitoring"),
                                options=[
                                    {"label": v, "value": k} for k, v in MONITORING_LABELS.items()
                                ],
                                value=terms["monitoring"],
                            ),
                        )
                    ),
                ],
                className="g-2",
            )
        )
    vols = [
        dbc.Col(
            _field(
                f"σ at {label.lower()}",
                _number({"type": "vol", "index": i}, None, 0.1, min=0.01),
            ),
            xs=6,
        )
        for i, label in enumerate(setup.strike_labels)
    ]
    fields += [
        html.Hr(),
        html.Div("Volatility (%)", className="field-label"),
        html.Div(
            "Read on the smile of the market panel (shown in grey). Type a value to override it.",
            className="field-help mb-2",
        ),
        dbc.Row(vols, className="g-2"),
        html.Div("Strikes and barriers in % of today's spot.", className="field-help"),
    ]
    return dbc.Card(dbc.CardBody([html.H5("Terms"), *fields]), className="terms-card")


def _tab_payoff() -> html.Div:
    return html.Div(
        [
            dbc.Switch(
                id="payoff-net",
                label="Net of premium (P&L at maturity)",
                value=False,
                className="mb-2",
            ),
            dcc.Loading(
                [
                    dcc.Graph(id="payoff-graph", config={"displaylogo": False}),
                    html.Div(id="payoff-notes"),
                ]
            ),
        ]
    )


def _tab_greeks() -> html.Div:
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(_segmented("greeks-view", GREEK_VIEWS, GREEK_VIEWS[0]), width="auto"),
                    dbc.Col(
                        dbc.Select(
                            id="greeks-time-greek",
                            options=[{"label": g, "value": g} for g in FIRST_ORDER[1:]],
                            value="Gamma (%)",
                        ),
                        id="greeks-time-greek-box",
                        width=3,
                        style=HIDDEN,
                    ),
                ],
                className="g-2 mb-2 align-items-center",
            ),
            dcc.Loading(
                [
                    html.Div(id="greeks-reading"),
                    dcc.Graph(id="greeks-graph", config={"displaylogo": False}),
                    html.Div(id="greeks-caption", className="field-help"),
                ]
            ),
        ]
    )


def _hedge_option(n: int, kind: str, maturity: float) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(html.Div(f"Hedging option {n}", className="field-label pt-2"), width=3),
            dbc.Col(
                dbc.Select(
                    id=f"hedge{n}-kind",
                    options=[{"label": k.capitalize(), "value": k} for k in ("call", "put")],
                    value=kind,
                ),
                width=3,
            ),
            dbc.Col(
                dbc.InputGroup(
                    [_number(f"hedge{n}-strike", 100.0, 5.0), dbc.InputGroupText("% K")]
                ),
                width=3,
            ),
            dbc.Col(
                dbc.InputGroup(
                    [_number(f"hedge{n}-maturity", maturity, 0.25), dbc.InputGroupText("y")]
                ),
                width=3,
            ),
        ],
        className="g-2 mb-2",
        id=f"hedge{n}-row",
    )


def _tab_risk() -> html.Div:
    static = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(_field("Neutralise", _segmented("hedge-target", HEDGES, HEDGES[0]))),
                    dbc.Col(
                        _field(
                            "Options in the position",
                            _number("hedge-options", 10_000, 1_000, min=1),
                        ),
                        width=3,
                    ),
                ]
            ),
            _hedge_option(1, "call", 0.25),
            _hedge_option(2, "put", 2.0),
            dcc.Loading(html.Div(id="hedge-static")),
        ],
        id="risk-static",
    )
    simulator = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        _field(
                            "Realised volatility (%)",
                            _number("sim-realised", None, 0.5, min=0.1, placeholder="= implied"),
                        )
                    ),
                    dbc.Col(
                        _field(
                            "Rebalancing",
                            dbc.Select(
                                id="sim-frequency",
                                options=[{"label": k, "value": k} for k in REBALANCING],
                                value="Daily",
                            ),
                        )
                    ),
                    dbc.Col(_field("Costs (bp per trade)", _number("sim-cost", 0.0, 1.0, min=0))),
                    dbc.Col(
                        _field(
                            "Paths",
                            dbc.Select(
                                id="sim-paths",
                                options=[{"label": f"{n:,}", "value": n} for n in SIM_PATHS],
                                value=SIM_PATHS[1],
                            ),
                        )
                    ),
                ],
                className="g-2",
            ),
            dbc.Button(
                [html.I(className="bi bi-play-fill me-1"), "Simulate"],
                id="sim-run",
                color="primary",
                n_clicks=0,
                className="mb-3",
            ),
            dcc.Loading(html.Div(id="sim-result")),
        ],
        id="risk-simulator",
        style=HIDDEN,
    )
    pnl = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(_field("Spot move (%)", _number("pnl-move", -10.0, 1.0))),
                    dbc.Col(_field("Volatility move (pts)", _number("pnl-vol", 2.0, 0.5))),
                    dbc.Col(_field("Days elapsed", _number("pnl-days", 7, 1, min=0))),
                    dbc.Col(_field("Rate move (bp)", _number("pnl-rate", 0.0, 5.0))),
                ],
                className="g-2",
            ),
            dcc.Loading(html.Div(id="pnl-result")),
        ],
        id="risk-pnl",
        style=HIDDEN,
    )
    return html.Div(
        [
            html.Div(_segmented("risk-view", RISK_VIEWS, RISK_VIEWS[0]), className="mb-3"),
            static,
            simulator,
            pnl,
        ]
    )


def option_page(product: Product) -> html.Div:
    """Product page of an option or a strategy (``product`` must have a setup)."""
    setup = setup_of(product)
    assert setup is not None  # the router only sends options here
    tabs = {
        "payoff": _tab_payoff(),
        "decomposition": dcc.Loading(html.Div(id="decomposition-result")),
        "greeks": _tab_greeks(),
        "risk-hedging": _tab_risk(),
    }
    return html.Div(
        [
            dcc.Store(id="option-store"),
            dbc.Breadcrumb(
                items=[
                    {"label": "Home", "href": "/"},
                    {"label": product.family, "href": f"/#{registry.slugify(product.family)}"},
                    {"label": product.name, "active": True},
                ]
            ),
            html.H1(product.name, className="product-title mb-1"),
            html.P(setup.description, className="product-description"),
            dbc.Row(
                [
                    dbc.Col(_terms_card(setup), lg=4, className="mb-4"),
                    dbc.Col(
                        [
                            html.Div(id="option-kpis", className="mb-3"),
                            dbc.Tabs(
                                [
                                    dbc.Tab(label=t, tab_id=i)
                                    for t, i in zip(TABS, TAB_IDS, strict=True)
                                ],
                                id="product-tabs",
                                active_tab=TAB_IDS[0],
                                className="mb-3",
                            ),
                            *[
                                html.Div(
                                    content,
                                    id=f"tab-{key}",
                                    style=SHOWN if key == "payoff" else HIDDEN,
                                )
                                for key, content in tabs.items()
                            ],
                        ],
                        lg=8,
                    ),
                ]
            ),
        ],
        className="product-page",
    )


# --- Pure functions behind the callbacks -------------------------------------------------------


def collect(
    values: list[Any], ids: list[dict[str, str]], vols: list[Any], pathname: str | None
) -> dict[str, Any]:
    """The page's terms as stored JSON: product slug, terms, volatility overrides."""
    slug = (pathname or "").rstrip("/").removeprefix("/product/")
    terms = {i["key"]: v for i, v in zip(ids, values, strict=True)}
    return {"slug": slug, "terms": terms, "vols": list(vols)}


def _position(store: dict[str, Any] | None, data: MarketData) -> Position:
    """Position of the stored terms on the session market; ValueError when either is invalid."""
    if not store:
        raise ValueError("Set the terms of the product.")
    product = registry.find(store["slug"])
    setup = setup_of(product) if product is not None else None
    if setup is None:
        raise ValueError("This page has no option to price.")
    try:
        market = market_from_data(data)
    except ValueError as exc:
        raise ValueError(f"The market data are invalid: {exc}") from None
    return build_position(setup, store["terms"], store.get("vols"), market)


def placeholders(store: dict[str, Any] | None, data: MarketData) -> list[str]:
    """Smile volatility of each strike, shown in the empty override inputs."""
    if not store:
        return []
    product = registry.find(store["slug"])
    setup = setup_of(product) if product is not None else None
    n = len(store.get("vols") or [])
    if setup is None:
        return [""] * n
    try:
        market = market_from_data(data)
    except ValueError:
        return [""] * n
    return ["" if v is None else f"{v:.2f}" for v in smile_vols(setup, store["terms"], market)]


def _kpi(label: str, value: str, sub: str = "", tone: str = "") -> dbc.Col:
    return dbc.Col(
        html.Div(
            [
                html.Div(label, className="kpi-label"),
                html.Div(value, className=f"kpi-value {tone}"),
                html.Div(sub, className="kpi-sub"),
            ],
            className="kpi",
        ),
        xs=6,
        md=True,
    )


def kpis(store: dict[str, Any] | None, data: MarketData) -> Any:
    """Price, premium and Greeks of the position, with its moneyness."""
    try:
        position = _position(store, data)
    except ValueError as exc:
        return _error(str(exc))
    price = position.price_pct()
    greeks = display_greeks(position.strategy, position.market)
    premium = price / 100.0 * position.notional
    paid = "paid" if premium >= 0 else "received"
    children: list[Any] = []
    badge = position.moneyness
    head = [html.Span(position.name, className="position-name")]
    if badge is not None:
        head.append(dbc.Badge(badge.value, color=MONEYNESS_COLORS[badge], className="ms-2"))
    children.append(html.Div(head, className="mb-2"))
    children.append(
        dbc.Row(
            [
                _kpi("Price", f"{price:.2f} %", f"{abs(premium):,.0f} {paid}"),
                _kpi("Delta", f"{greeks['Delta (%)']:+.1f} %", "per +1 % spot: Δ/100 %"),
                _kpi("Gamma", f"{greeks['Gamma (%)']:+.2f}", "Δ points per +1 % spot"),
                _kpi("Vega", f"{greeks['Vega (%)']:+.3f} %", "per vol point"),
                _kpi("Theta", f"{greeks['Theta (%)']:+.4f} %", "per day"),
            ],
            className="g-2",
        )
    )
    if position.barrier_crossed:
        children.append(
            _error(
                "The barrier is already crossed at today's spot: a knock-out option is dead, a "
                "knock-in option is a vanilla. Move the barrier to the other side of 100 %."
            )
        )
    return html.Div(children)


def show_tab(active: str | None) -> list[dict[str, str]]:
    """Show the active tab's content, hide the others."""
    return [SHOWN if active == t else HIDDEN for t in TAB_IDS]


def payoff_view(
    store: dict[str, Any] | None, data: MarketData, net: bool | None, active: str | None
) -> tuple[Any, Any]:
    """Payoff (or P&L) chart and breakevens."""
    if active != "payoff":
        return dash.no_update, dash.no_update
    try:
        position = _position(store, data)
    except ValueError as exc:
        return go.Figure(), _error(str(exc))
    fig, points = figures.payoff_figure(position, net_of_premium=bool(net))
    notes: list[Any] = []
    if net:
        text = (
            "Breakeven at " + " and ".join(f"{p:.1f} %" for p in points) + " of today's spot."
            if points
            else "No breakeven: the P&L keeps one sign, or stays at zero, wherever the spot ends."
        )
        notes.append(html.P(text, className="field-help"))
    notes.append(
        html.P(
            "Amounts in % of the notional; the P&L subtracts the premium compounded at the "
            "risk-free rate to maturity.",
            className="field-help",
        )
    )
    return fig, html.Div(notes)


def decomposition_view(store: dict[str, Any] | None, data: MarketData, active: str | None) -> Any:
    """Price and Greeks leg by leg."""
    if active != "decomposition":
        return dash.no_update
    try:
        position = _position(store, data)
    except ValueError as exc:
        return _error(str(exc))
    rows = decomposition(position)
    skew = any(abs(r["Skew (%)"]) > 1e-12 for r in rows)
    header = ["Leg", "Terms", "σ (%)", "Price (%)", *(["of which skew"] if skew else []), *GREEKS]
    body = []
    for r in rows:
        line: list[Any] = [
            html.Span(r["Leg"], className=_side_class(r["Leg"])),
            r["Terms"],
            "" if r["Volatility (%)"] is None else f"{r['Volatility (%)']:.2f}",
            _fmt(r["Price (%)"], 3),
        ]
        if skew:
            line.append(_fmt(r["Skew (%)"], 3))
        line += [_fmt(r[g], 4 if g in ("Theta (%)", "Rho (%)") else 2) for g in GREEKS]
        body.append(line)
    notes = [
        html.P(
            "Each leg is priced in closed form at the implied volatility of its own strike; the "
            "product is the sum of its legs. Greeks in % of notional: Delta per +1 % spot is "
            "Delta/100 %, Gamma in Delta points per +1 % spot, Vega per volatility point, Theta "
            "per day, Rho per basis point.",
            className="field-help",
        )
    ]
    if skew:
        total = rows[-1]
        notes.append(
            dbc.Alert(
                f"The smile slope adds {total['Skew (%)']:+.2f} % to the digitals: priced with "
                "the flat Black-Scholes formula at the volatility of their strike they would be "
                f"worth {total['Price (%)'] - total['Skew (%)']:.2f} % instead of "
                f"{total['Price (%)']:.2f} %.",
                color="info",
                className="py-2",
            )
        )
    return html.Div([_table(header, body, total=len(rows) > 1), *notes])


def _side_class(name: str) -> str:
    if name.startswith("Long"):
        return "pos fw-semibold"
    if name.startswith("Short"):
        return "neg fw-semibold"
    return "fw-bold"


def greeks_view(
    store: dict[str, Any] | None,
    data: MarketData,
    view: str | None,
    greek: str | None,
    active: str | None,
) -> tuple[Any, Any, Any, Any]:
    """Reading, chart and caption of the Greeks tab, and whether the Greek picker shows."""
    if active != "greeks":
        return (dash.no_update,) * 4
    picker = SHOWN if view == GREEK_VIEWS[1] else HIDDEN
    try:
        position = _position(store, data)
    except ValueError as exc:
        return _error(str(exc)), go.Figure(), "", picker
    market, instrument, maturity = position.market, position.instrument, position.maturity
    spots = SPOTS_PCT / 100.0 * market.spot
    if view == GREEK_VIEWS[1]:
        name = greek if greek in FIRST_ORDER else "Gamma (%)"
        times = np.linspace(maturity, min(0.02, 0.5 * maturity), 60)
        k = position.strikes_pct[0] / 100.0 * market.spot
        put = position.setup.kind == "put"
        levels = {
            "In the money": 0.85 if put else 1.15,
            "At the money": 1.0,
            "Out of the money": 1.15 if put else 0.85,
        }
        curves = {
            f"{label} (spot {100 * f * k / market.spot:.0f} %)": time_profile(
                instrument, market, times, f * k
            )[name]
            for label, f in levels.items()
        }
        caption = (
            "The curves run toward expiry (right). At the money, gamma and theta grow without "
            "bound as expiry approaches; in and out of the money they fade to zero."
        )
        return "", figures.greek_time_figure(times, curves, name), caption, picker
    profile = spot_profile(instrument, market, maturity, spots)
    now = {
        k: float(v)
        for k, v in (
            greeks_at(position.strategy, market) | second_order_at(instrument, market, maturity)
        ).items()
    }
    caption = (
        "Red dot: today. Dashed lines: strikes. Blue band: at the money (within a quarter of a "
        "standard deviation); green: in the money, red: out of the money."
    )
    if view == GREEK_VIEWS[2]:
        reading = _bullets(ex.second_order(now))
        fig = figures.greeks_grid_figure(position, SPOTS_PCT, profile, SECOND_ORDER)
        return reading, fig, caption, picker
    quarter = (
        float(greeks_at(instrument(maturity / 4.0), market)["Gamma (%)"])
        if maturity > 0.08
        else None
    )
    lines = ex.first_order(position.situation, now, SPOTS_PCT, profile, quarter)
    fig = figures.greeks_grid_figure(position, SPOTS_PCT, profile, FIRST_ORDER)
    return _bullets(lines), fig, caption, picker


def risk_blocks(view: str | None) -> list[dict[str, str]]:
    """Show the controls of the chosen risk view."""
    return [SHOWN if view == v else HIDDEN for v in RISK_VIEWS]


def _desk_row(name: Any, g: Greeks, spot: float) -> list[Any]:
    d = to_desk(g, spot=spot, notional=spot)
    return [
        name,
        _fmt(100.0 * float(d.delta), 2),
        _fmt(100.0 * float(d.gamma), 3),
        _fmt(float(d.vega), 4),
        _fmt(float(d.theta), 4),
    ]


def static_view(
    store: dict[str, Any] | None,
    data: MarketData,
    active: str | None,
    view: str | None,
    options: float | None,
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
    rows_shown = (SHOWN if needed >= 1 else HIDDEN, SHOWN if needed >= 2 else HIDDEN)
    if active != "risk-hedging" or view != RISK_VIEWS[0]:
        return dash.no_update, *rows_shown
    try:
        position = _position(store, data)
        count = int(options or 0)
        if count < 1:
            raise ValueError("The number of options must be at least 1.")
        hedges = []
        for kind, strike, maturity in ((kind1, strike1, maturity1), (kind2, strike2, maturity2))[
            :needed
        ]:
            if not strike or not maturity or strike <= 0 or maturity <= 0:
                raise ValueError("Every hedging option needs a positive strike and maturity.")
            option = EuropeanOption(
                "put" if kind == "put" else "call",
                strike / 100.0 * position.market.spot,
                float(maturity),
            )
            name = f"{option.kind}s K {strike:g}% {maturity:g}y"
            hedges.append((name, option.greeks(position.market)))
        hedge = static_hedge(position.strategy.greeks(position.market), tuple(hedges))
    except ValueError as exc:
        return _error(str(exc)), *rows_shown
    spot = position.market.spot
    rows = [_desk_row(f"Position: {position.name}", hedge.position, spot)]
    for leg in hedge.legs:
        units = leg.quantity * count
        verb = "Buy" if units > 0 else "Sell"
        label = html.Span(
            f"{verb} {abs(units):,.0f} {leg.name if leg.name != 'Underlying' else 'shares'}",
            className="pos fw-semibold" if units > 0 else "neg fw-semibold",
        )
        rows.append(_desk_row(label, leg.greeks * leg.quantity, spot))
    rows.append(_desk_row("Total", hedge.hedged, spot))
    table = _table(["", "Delta (%)", "Gamma (%)", "Vega (%)", "Theta (%)"], rows, total=True)
    note = html.P(
        "Greeks per option in % of today's spot; the quantities are for the whole position. "
        "The neutralised Greeks show 0 in the total row.",
        className="field-help",
    )
    reading = _bullets(ex.static_reading(hedge, count, position.short))
    return html.Div([table, note, reading]), *rows_shown


def simulate_view(
    n_clicks: int | None,
    store: dict[str, Any] | None,
    data: MarketData,
    realised_pct: float | None,
    frequency: str | None,
    cost_bp: float | None,
    paths: Any,
) -> Any:
    """Delta-hedge the position along simulated paths."""
    if not n_clicks:
        return html.P(
            "The hedger values the option at its implied volatility and rebalances the delta; "
            "the spot moves with the realised volatility. Click Simulate.",
            className="field-help",
        )
    try:
        position = _position(store, data)
    except ValueError as exc:
        return _error(str(exc))
    brick = position.strategy.legs[0].brick
    if isinstance(brick, BarrierOption) and brick.monitoring is not Monitoring.EUROPEAN:
        return _error(
            "The simulator settles the option on the final spot only: it is available for "
            "barriers observed at maturity. Set Observed to At maturity."
        )
    market, maturity = position.market, position.maturity
    implied = float(market.vol.vol(position.strikes_pct[0] / 100.0 * market.spot, maturity))
    realised = implied if realised_pct is None else float(realised_pct) / 100.0
    if not realised > 0.0:
        return _error("The realised volatility must be positive.")
    rebalancings = max(1, round(REBALANCING.get(frequency or "", 252) * maturity))
    cost = float(cost_bp or 0.0)
    sim = simulate_delta_hedge(
        position.instrument,
        market,
        maturity,
        realised_vol=realised,
        rebalancings=rebalancings,
        cost_bp=cost,
        n_paths=int(paths or SIM_PATHS[1]),
    )
    spot = market.spot

    def flat(vol: float) -> float:
        flat_market = Market(spot, market.rate, market.dividend_yield, FlatVol(vol))
        return float(position.strategy.value(flat_market))

    # the hedge earns, on average, the value at the realised volatility minus the price paid
    theory = 100.0 * (flat(realised) - float(position.strategy.value(market))) / spot
    vols = [v for leg in position.strategy.legs if (v := leg.brick.implied_vol(market)) is not None]
    implied_range = (100.0 * min(vols, default=implied), 100.0 * max(vols, default=implied))
    vega_pct = 100.0 * float(position.strategy.greeks(market).vega) / spot
    dk = derman_kamal_std(vega_pct, implied, rebalancings)
    metrics = dbc.Row(
        [
            _kpi("Mean P&L", f"{sim.mean.value:+.3f} %", f"± {sim.mean.std_error:.3f} (1 s.e.)"),
            _kpi("Theory", f"{theory:+.3f} %", "value at realised vol − price paid"),
            _kpi("Std deviation", f"{sim.std:.3f} %", f"Derman-Kamal {dk:.3f} %"),
        ],
        className="g-2 mb-2",
    )
    reading = ex.simulation_reading(
        sim.mean.value,
        theory,
        sim.std,
        dk,
        100.0 * realised,
        implied_range,
        float(position.strategy.greeks(market).gamma) > 0.0,
        cost,
    )
    return html.Div(
        [
            metrics,
            _bullets(reading),
            dcc.Graph(figure=figures.hedge_histogram_figure(sim.pnl, theory)),
            dcc.Graph(
                figure=figures.hedge_path_figure(
                    sim.times, sim.spot, sim.delta, sim.gamma_pnl, sim.theta_pnl
                )
            ),
            html.P(
                f"Implied volatility {100 * implied:.2f} % at the main strike; {rebalancings} "
                "rebalancing dates; P&L discounted, in % of notional. The path chart shows the "
                "first simulated path, undiscounted.",
                className="field-help",
            ),
        ]
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
    if active != "risk-hedging" or view != RISK_VIEWS[2]:
        return dash.no_update
    try:
        position = _position(store, data)
        shock = Shock(float(move or 0.0), float(dvol or 0.0), float(days or 0.0), float(dr or 0.0))
        spot, instrument = position.market.spot, position.instrument

        def value(m: Market, t: float) -> float:
            return 100.0 * float(instrument(t).value(m)) / spot

        attribution = explain(value, position.market, position.maturity, shock)
        grid = scenario_matrix(
            value, position.market, position.maturity, SPOT_GRID, VOL_GRID, shock.days
        )
    except ValueError as exc:
        return _error(str(exc))
    taylor = [[k, _fmt(attribution.terms[k], 4), _fmt(attribution.greeks[k], 5)] for k in TERMS]
    taylor += [
        [html.B("Taylor total"), _fmt(attribution.taylor, 4), ""],
        [html.B("Full revaluation"), _fmt(attribution.full, 4), ""],
        [html.B("Unexplained"), _fmt(attribution.residual, 4), ""],
    ]
    steps = [[k, _fmt(attribution.steps[k], 4)] for k in STEPS]
    steps.append(["Total", _fmt(sum(attribution.steps.values()), 4)])
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6("Taylor at today's Greeks"),
                            _table(["Term", "P&L (%)", "Greek"], taylor),
                            html.Div(
                                "Greeks per 1 % spot, per (1 %)², per vol point, per day, per bp.",
                                className="field-help",
                            ),
                        ],
                        md=7,
                    ),
                    dbc.Col(
                        [
                            html.H6("Step by step (exact)"),
                            _table(["Step", "P&L (%)"], steps, total=True),
                            html.Div(
                                "Revalued moving time, then the spot, then the volatility, then "
                                "the rate.",
                                className="field-help",
                            ),
                        ],
                        md=5,
                    ),
                ]
            ),
            _bullets(ex.attribution_reading(attribution)),
            dcc.Graph(
                figure=figures.scenario_heatmap_figure(SPOT_GRID, VOL_GRID, grid, shock.days)
            ),
            html.P(
                "Stress test: every cell is a full revaluation, never a Taylor expansion.",
                className="field-help",
            ),
        ]
    )


# --- Callbacks ---------------------------------------------------------------------------------


def register(app: dash.Dash) -> None:
    """Register the callbacks of the option pages."""
    store = Input("option-store", "data")
    market = Input("market-data", "data")
    tab = Input("product-tabs", "active_tab")
    app.callback(
        Output("option-store", "data"),
        Input(_term(ALL), "value"),  # type: ignore[arg-type]
        State(_term(ALL), "id"),  # type: ignore[arg-type]
        Input({"type": "vol", "index": ALL}, "value"),
        State("url", "pathname"),
    )(collect)
    app.callback(
        Output({"type": "vol", "index": ALL}, "placeholder"),
        store,
        market,
    )(placeholders)
    app.callback(Output("option-kpis", "children"), store, market)(kpis)
    app.callback([Output(f"tab-{t}", "style") for t in TAB_IDS], tab)(show_tab)
    app.callback(
        Output("payoff-graph", "figure"),
        Output("payoff-notes", "children"),
        store,
        market,
        Input("payoff-net", "value"),
        tab,
    )(payoff_view)
    app.callback(Output("decomposition-result", "children"), store, market, tab)(decomposition_view)
    app.callback(
        Output("greeks-reading", "children"),
        Output("greeks-graph", "figure"),
        Output("greeks-caption", "children"),
        Output("greeks-time-greek-box", "style"),
        store,
        market,
        Input("greeks-view", "value"),
        Input("greeks-time-greek", "value"),
        tab,
    )(greeks_view)
    app.callback(
        [Output(f"risk-{v}", "style") for v in ("static", "simulator", "pnl")],
        Input("risk-view", "value"),
    )(risk_blocks)
    app.callback(
        Output("hedge-static", "children"),
        Output("hedge1-row", "style"),
        Output("hedge2-row", "style"),
        store,
        market,
        tab,
        Input("risk-view", "value"),
        Input("hedge-options", "value"),
        Input("hedge-target", "value"),
        *[
            Input(f"hedge{n}-{field}", "value")
            for n in (1, 2)
            for field in ("kind", "strike", "maturity")
        ],
    )(static_view)
    app.callback(
        Output("sim-result", "children"),
        Input("sim-run", "n_clicks"),
        State("option-store", "data"),
        State("market-data", "data"),
        State("sim-realised", "value"),
        State("sim-frequency", "value"),
        State("sim-cost", "value"),
        State("sim-paths", "value"),
    )(simulate_view)
    app.callback(
        Output("pnl-result", "children"),
        store,
        market,
        tab,
        Input("risk-view", "value"),
        Input("pnl-move", "value"),
        Input("pnl-vol", "value"),
        Input("pnl-days", "value"),
        Input("pnl-rate", "value"),
    )(pnl_view)


__all__ = ["Terms", "option_page", "register"]
