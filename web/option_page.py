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

from spricer.bricks import BarrierOption
from spricer.greeks_lab import (
    FIRST_ORDER,
    SECOND_ORDER,
    Moneyness,
    greeks_at,
    second_order_at,
    spot_profile,
    time_profile,
)
from spricer.hedging import derman_kamal_std, simulate_delta_hedge
from spricer.market import FlatVol, Market
from spricer.monitoring import Monitoring
from web import explain as ex
from web import figures, registry, risk_panel, sheet, ui
from web.market import MarketData, market_from_data
from web.options import (
    GREEKS,
    OptionSetup,
    Position,
    decomposition,
    display_greeks,
    setup_of,
    smile_vols,
)
from web.registry import Product
from web.scenarios import option_scenarios
from web.sheets import option_sheet

GREEK_VIEWS = ("Against the spot", "Against time", "Second order")
REBALANCING = {"Daily": 252, "Weekly": 52, "Monthly": 12}
SIM_PATHS = (2_000, 5_000, 10_000)
SPOTS_PCT = np.linspace(50.0, 150.0, 201)
TODAY = "Today's spot (100 %)"
MONEYNESS_COLORS = {Moneyness.ITM: "success", Moneyness.ATM: "primary", Moneyness.OTM: "danger"}


# --- Layout ------------------------------------------------------------------------------------


def _terms_card(setup: OptionSetup) -> dbc.Card:
    terms = setup.default_terms()
    term = ui.term
    fields: list[Any] = [
        ui.field("Position", ui.segmented(term("side"), ("Long", "Short"), terms["side"])),
        dbc.Row(
            [
                dbc.Col(
                    ui.field("Maturity (years)", ui.number(term("maturity"), 1.0, 0.25, min=0.01))
                ),
                dbc.Col(ui.field("Notional", ui.number(term("notional"), terms["notional"], 1e5))),
            ],
            className="g-2",
        ),
    ]
    strikes = [
        dbc.Col(
            ui.field(f"{label} (%)", ui.number(term(f"strike_{i}"), terms[f"strike_{i}"], 1.0)),
            xs=6,
        )
        for i, label in enumerate(setup.strike_labels)
    ]
    fields.append(dbc.Row(strikes, className="g-2"))
    if setup.uses_payout:
        fields.append(
            ui.field("Amount paid (% of notional)", ui.number(term("payout"), terms["payout"], 1))
        )
    if setup.barrier_type is not None:
        fields.append(
            dbc.Row(
                [
                    dbc.Col(
                        ui.field("Barrier (%)", ui.number(term("barrier"), terms["barrier"], 1.0))
                    ),
                    dbc.Col(ui.monitoring_field("Barrier observed", terms["monitoring"])),
                ],
                className="g-2",
            )
        )
    vols = [
        dbc.Col(
            ui.field(
                f"σ at {label.lower()}", ui.number({"type": "vol", "index": i}, None, 0.1, min=0.01)
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
        ui.note("Strikes and barriers in % of today's spot."),
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
                    dbc.Col(ui.segmented("greeks-view", GREEK_VIEWS, GREEK_VIEWS[0]), width="auto"),
                    dbc.Col(
                        ui.select("greeks-time-greek", FIRST_ORDER[1:], "Delta (%)"),
                        id="greeks-time-greek-box",
                        width=3,
                        style=ui.HIDDEN,
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


def _simulator() -> list[Any]:
    return [
        dbc.Row(
            [
                dbc.Col(
                    ui.field(
                        "Realised volatility (%)",
                        ui.number("sim-realised", None, 0.5, min=0.1, placeholder="= implied"),
                    )
                ),
                dbc.Col(ui.field("Rebalancing", ui.select("sim-frequency", REBALANCING, "Daily"))),
                dbc.Col(ui.field("Costs (bp per trade)", ui.number("sim-cost", 0.0, 1.0, min=0))),
                dbc.Col(
                    ui.field(
                        "Paths",
                        ui.select("sim-paths", {n: f"{n:,}" for n in SIM_PATHS}, SIM_PATHS[1]),
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
    ]


def option_page(product: Product) -> html.Div:
    """Product page of an option or a strategy (``product`` must have a setup)."""
    setup = setup_of(product)
    assert setup is not None  # the router only sends options here
    return ui.product_shell(
        product,
        setup.description,
        _terms_card(setup),
        {
            "Payoff": _tab_payoff(),
            "Decomposition": dcc.Loading(html.Div(id="decomposition-result")),
            "Greeks": _tab_greeks(),
            "Risk & hedging": risk_panel.risk_tab(
                "Options in the position", 10_000, simulator=_simulator()
            ),
        },
    )


# --- Pure functions behind the callbacks -------------------------------------------------------


def _position(store: dict[str, Any] | None, data: MarketData) -> Position:
    """Position of the stored terms on the session market; ValueError when either is invalid."""
    item = risk_panel.resolve(store, data)
    if not isinstance(item, Position):
        raise ValueError("This page has no option to price.")
    return item


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


def kpis(store: dict[str, Any] | None, data: MarketData) -> Any:
    """Price, premium and Greeks of the position, with its moneyness."""
    try:
        position = _position(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    price = position.price_pct()
    greeks = display_greeks(position.strategy, position.market)
    premium = price / 100.0 * position.notional
    paid = "paid" if premium >= 0 else "received"
    badge = position.moneyness
    head: list[Any] = [html.Span(position.name, className="position-name")]
    if badge is not None:
        head.append(dbc.Badge(badge.value, color=MONEYNESS_COLORS[badge], className="ms-2"))
    children: list[Any] = [
        html.Div(head, className="mb-2"),
        dbc.Row(
            [
                ui.kpi("Price", f"{price:.2f} %", f"{abs(premium):,.0f} {paid}"),
                ui.kpi("Delta", f"{greeks['Delta (%)']:+.1f} %", "per +1 % spot: Δ/100 %"),
                ui.kpi("Gamma", f"{greeks['Gamma (%)']:+.2f} %", "Δ change per +1 % spot"),
                ui.kpi("Vega", f"{greeks['Vega (%)']:+.3f} %", "per vol point"),
                ui.kpi("Theta", f"{greeks['Theta (%)']:+.4f} %", "per day"),
            ],
            className="g-2",
        ),
    ]
    children += [ui.sheet_button("option-sheet-button"), dcc.Download(id="option-sheet-download")]
    if position.barrier_crossed:
        children.append(
            ui.error(
                "The barrier is already crossed at today's spot: a knock-out option is dead, a "
                "knock-in option is a vanilla. Move the barrier to the other side of 100 %."
            )
        )
    return html.Div(children)


def payoff_view(
    store: dict[str, Any] | None, data: MarketData, net: bool | None, active: str | None
) -> tuple[Any, Any]:
    """Payoff (or P&L) chart and breakevens."""
    if active != "payoff":
        return dash.no_update, dash.no_update
    try:
        position = _position(store, data)
    except ValueError as exc:
        return go.Figure(), ui.error(str(exc))
    fig, points = figures.payoff_figure(position, net_of_premium=bool(net))
    notes: list[Any] = []
    if net:
        notes.append(
            ui.note(
                "Breakeven at " + " and ".join(f"{p:.1f} %" for p in points) + " of today's spot."
                if points
                else "No breakeven: the P&L keeps one sign, or stays at zero, wherever the spot "
                "ends."
            )
        )
    levels, rows = option_scenarios(position)
    notes += [
        html.H6("Paid at maturity", className="mt-2"),
        ui.scenario_table("Spot at maturity", levels, rows),
        ui.note(
            "Amounts in % of the notional, spots in % of today's; the P&L subtracts the "
            "premium compounded at the risk-free rate to maturity."
        ),
    ]
    return fig, html.Div(notes)


def decomposition_view(store: dict[str, Any] | None, data: MarketData, active: str | None) -> Any:
    """Price and Greeks leg by leg."""
    if active != "decomposition":
        return dash.no_update
    try:
        position = _position(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    rows = decomposition(position)
    skew = any(abs(r["Skew (%)"]) > 1e-12 for r in rows)
    header = ["Leg", "Terms", "σ (%)", "Price (%)", *(["of which skew"] if skew else []), *GREEKS]
    body = []
    for r in rows:
        line: list[Any] = [
            html.Span(r["Leg"], className=ui.side_class(r["Leg"])),
            r["Terms"],
            "" if r["Volatility (%)"] is None else f"{r['Volatility (%)']:.2f}",
            ui.fmt(r["Price (%)"], 3),
        ]
        if skew:
            line.append(ui.fmt(r["Skew (%)"], 3))
        line += [ui.fmt(r[g], 4 if g in ("Theta (%)", "Rho (%)") else 2) for g in GREEKS]
        body.append(line)
    notes: list[Any] = [
        ui.note(
            "Each leg is priced in closed form at the implied volatility of its own strike; the "
            "product is the sum of its legs. Greeks in % of notional: Delta per +1 % spot is "
            "Delta/100 %, Gamma the change of Delta (%) per +1 % spot, Vega per volatility "
            "point, Theta per day, Rho per basis point."
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
    return html.Div([ui.table(header, body, total=len(rows) > 1), *notes])


def time_spots(position: Position) -> list[tuple[str, float]]:
    """Spot levels (% of today) at which the Greeks are followed through time.

    A single option: in, at and out of the money (15 % from the strike), and today's spot when
    the strike is elsewhere. A strategy: today's spot and 15 % on either side.
    """
    if not position.single:
        return [("Spot 85 %", 85.0), (TODAY, 100.0), ("Spot 115 %", 115.0)]
    k = position.strikes_pct[0]
    up, down = (0.85, 1.15) if position.setup.kind == "put" else (1.15, 0.85)
    out = [
        (f"In the money (spot {k * up:.0f} %)", k * up),
        (f"At the money (spot {k:.0f} %)", k),
        (f"Out of the money (spot {k * down:.0f} %)", k * down),
    ]
    if abs(k - 100.0) < 0.5:
        out[1] = (TODAY, 100.0)
    else:
        out.append((TODAY, 100.0))
    return out


def _time_view(position: Position, greek: str | None) -> tuple[Any, go.Figure]:
    name = greek if greek in FIRST_ORDER[1:] else "Delta (%)"
    market, maturity = position.market, position.maturity
    times = np.linspace(maturity, min(0.02, 0.5 * maturity), 60)
    curves = {
        label: time_profile(position.instrument, market, times, pct / 100.0 * market.spot)[name]
        for label, pct in time_spots(position)
    }
    short = name.removesuffix(" (%)").lower()
    now = float(curves[TODAY][0])
    lines = [
        f"Each curve is the **{short}** of your position if the spot stayed at one level while "
        f"time passes: left, today ({maturity:g} years to maturity); right, expiry.",
        f"The **{TODAY.lower()}** curve starts at **{now:+.2f} %**: the {short} shown at the top "
        "of the page.",
    ]
    if position.single:
        lines.append(
            "In and out of the money: 15 % from the strike, on either side. Near expiry, at "
            "the money, gamma and theta grow without bound; in and out of the money they fade "
            "to zero and the delta goes to its final value (100 % or 0)."
        )
    else:
        lines.append(
            "The other curves: the spot 15 % lower or higher. Near expiry the Greeks of a "
            "strategy concentrate around its strikes."
        )
    box = html.Div([dcc.Markdown(line) for line in lines], className="explain-box")
    return box, figures.greek_time_figure(times, curves, name, highlight=TODAY)


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
    picker = ui.SHOWN if view == GREEK_VIEWS[1] else ui.HIDDEN
    try:
        position = _position(store, data)
    except ValueError as exc:
        return ui.error(str(exc)), go.Figure(), "", picker
    if view == GREEK_VIEWS[1]:
        box, fig = _time_view(position, greek)
        return box, fig, "", picker
    market, instrument, maturity = position.market, position.instrument, position.maturity
    profile = spot_profile(instrument, market, maturity, SPOTS_PCT / 100.0 * market.spot)
    now = {
        k: float(v)
        for k, v in (
            greeks_at(position.strategy, market) | second_order_at(instrument, market, maturity)
        ).items()
    }
    caption = (
        "Each panel: the Greek if the spot were elsewhere today (horizontal axis). Red dot: "
        "today's spot. Dashed lines: strikes. Shaded blue band: the at-the-money zone, spots "
        "within a quarter of a standard deviation of a strike (a delta between about 40 % and "
        "60 %). Green and red backgrounds: in and out of the money."
    )
    if view == GREEK_VIEWS[2]:
        intro = html.Div(
            dcc.Markdown(
                "Second-order Greeks tell **how your first-order Greeks move**: a hedge set "
                "today is no longer right after a volatility move (vanna, volga), a day "
                "(charm) or a spot move (speed)."
            ),
            className="explain-box",
        )
        reading = html.Div([intro, ui.greek_cards(ex.second_order(now))])
        fig = figures.greeks_grid_figure(position, SPOTS_PCT, profile, SECOND_ORDER)
        return reading, fig, caption, picker
    quarter = (
        float(greeks_at(instrument(maturity / 4.0), market)["Gamma (%)"])
        if maturity > 0.08
        else None
    )
    cards, notes = ex.first_order(position.situation, now, SPOTS_PCT, profile, quarter)
    fig = figures.greeks_grid_figure(position, SPOTS_PCT, profile, FIRST_ORDER)
    return ui.greek_cards(cards, notes), fig, caption, picker


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
        return ui.note(
            "The hedger values the option at its implied volatility and rebalances the delta; "
            "the spot moves with the realised volatility. Click Simulate."
        )
    try:
        position = _position(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    brick = position.strategy.legs[0].brick
    if isinstance(brick, BarrierOption) and brick.monitoring is not Monitoring.EUROPEAN:
        return ui.error(
            "The simulator settles the option on the final spot only: it is available for "
            "barriers observed at maturity. Set Observed to At maturity."
        )
    market, maturity = position.market, position.maturity
    implied = float(market.vol.vol(position.strikes_pct[0] / 100.0 * market.spot, maturity))
    realised = implied if realised_pct is None else float(realised_pct) / 100.0
    if not realised > 0.0:
        return ui.error("The realised volatility must be positive.")
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
            ui.kpi("Mean P&L", f"{sim.mean.value:+.3f} %", f"± {sim.mean.std_error:.3f} (1 s.e.)"),
            ui.kpi("Theory", f"{theory:+.3f} %", "value at realised vol − price paid"),
            ui.kpi("Std deviation", f"{sim.std:.3f} %", f"Derman-Kamal {dk:.3f} %"),
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
            ui.bullets(reading),
            dcc.Graph(figure=figures.hedge_histogram_figure(sim.pnl, theory)),
            dcc.Graph(
                figure=figures.hedge_path_figure(
                    sim.times, sim.spot, sim.delta, sim.gamma_pnl, sim.theta_pnl
                )
            ),
            ui.note(
                f"Implied volatility {100 * implied:.2f} % at the main strike; {rebalancings} "
                "rebalancing dates; P&L discounted, in % of notional. The path chart shows the "
                "first simulated path, undiscounted."
            ),
        ]
    )


def download_sheet(n_clicks: int | None, store: dict[str, Any] | None, data: MarketData) -> Any:
    """The product sheet of the position, as an HTML file."""
    try:
        position = _position(store, data)
    except ValueError:
        return dash.no_update
    return sheet.download(option_sheet(position))


# --- Callbacks ---------------------------------------------------------------------------------


def register(app: dash.Dash) -> None:
    """Register the callbacks of the option pages."""
    store = Input("product-store", "data")
    market = Input("market-data", "data")
    tab = Input("product-tabs", "active_tab")
    app.callback(Output({"type": "vol", "index": ALL}, "placeholder"), store, market)(placeholders)
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
        Output("option-sheet-download", "data"),
        Input("option-sheet-button", "n_clicks"),
        State("product-store", "data"),
        State("market-data", "data"),
        prevent_initial_call=True,
    )(download_sheet)
    app.callback(
        Output("sim-result", "children"),
        Input("sim-run", "n_clicks"),
        State("product-store", "data"),
        State("market-data", "data"),
        State("sim-realised", "value"),
        State("sim-frequency", "value"),
        State("sim-cost", "value"),
        State("sim-paths", "value"),
    )(simulate_view)


__all__ = ["option_page", "register"]
