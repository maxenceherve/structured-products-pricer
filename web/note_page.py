"""Product page of a note priced by bricks: capital protection, certificates, shark warrants.

Tabs: Payoff, Decomposition (brick by brick), Greeks (against the underlying, with their
reading), Risk & hedging (the issuer's hedge, the holder's P&L explain), Monte Carlo (a check
of the closed form under the local volatility) and Solve for. Each callback delegates to a pure
function of this module.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, dcc, html

from spricer.mc import price_note
from spricer.mc.paths import Z_95
from spricer.monitoring import Monitoring
from spricer.products.capital_protection import NoteSpec
from spricer.products.notes import LevelKind, Note
from spricer.products.participation import SHARK_WARRANTS
from spricer.solve import solve_note
from web import explain as ex
from web import figures, risk_panel, sheet, ui
from web.market import MarketData
from web.notes import (
    PROFILE,
    TOGGLED,
    NotePosition,
    decomposition,
    default_terms,
    desk_greeks,
    monte_carlo_rows,
    note_spec,
    shown_params,
    smile_levels,
    spot_profile,
)
from web.options import GREEKS
from web.registry import Product
from web.scenarios import note_scenarios
from web.sheets import note_sheet

MC_PATHS = (50_000, 100_000, 200_000)
GREEK_PERF = np.linspace(40.0, 180.0, 141)


# --- Layout ------------------------------------------------------------------------------------


def _terms_card(spec: NoteSpec) -> dbc.Card:
    terms = default_terms(spec)
    term = ui.term
    fields: list[Any] = [
        dbc.Row(
            [
                dbc.Col(
                    ui.field(
                        "Maturity (years)", ui.number(term("maturity"), terms["maturity"], 0.5)
                    )
                ),
                dbc.Col(ui.field("Notional", ui.number(term("notional"), terms["notional"], 1e5))),
            ],
            className="g-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    ui.field(
                        "Initial level",
                        ui.number(term("initial"), None, 1.0, placeholder="= spot today"),
                    )
                ),
                dbc.Col(ui.field("Issuer spread (bp)", ui.number(term("spread_bp"), 0.0, 10.0))),
            ],
            className="g-2",
        ),
        html.Hr(),
    ]
    toggles = {p.key: p for p in spec.params if p.kind == "toggle"}
    levels = []
    for p in spec.params:
        if p.kind == "toggle":
            continue
        control: Any = ui.number(term(p.key), p.default, 1.0)
        toggle = toggles.get(TOGGLED.get(p.key, ""))
        if toggle is not None:
            control = html.Div(
                [
                    dbc.Switch(id=term(toggle.key), label=toggle.label, value=bool(toggle.default)),
                    control,
                ]
            )
        levels.append(dbc.Col(ui.field(f"{p.label} (%)", control), xs=6))
    fields.append(dbc.Row(levels, className="g-2"))
    if spec.monitoring_choice:
        fields.append(ui.monitoring_field("Barrier observed", terms["monitoring"]))
    fields.append(
        ui.note(
            "Levels in % of the initial level, amounts in % of the notional. The issuer's "
            "credit spread discounts every cash flow of the note, not the underlying's drift."
        )
    )
    return dbc.Card(dbc.CardBody([html.H5("Terms"), *fields]), className="terms-card")


def _tab_monte_carlo() -> html.Div:
    return html.Div(
        [
            ui.note(
                "Simulates the underlying under the local volatility of the whole smile (Dupire) "
                "and prices every brick on the same paths."
            ),
            dbc.Row(
                [
                    dbc.Col(
                        ui.select("note-mc-paths", {n: f"{n:,} paths" for n in MC_PATHS}, 100_000),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="bi bi-shuffle me-1"), "Check with Monte Carlo"],
                            id="note-mc-run",
                            color="primary",
                            n_clicks=0,
                        ),
                        width="auto",
                    ),
                ],
                className="g-2 mb-3",
            ),
            dcc.Loading(html.Div(id="note-mc-result")),
        ]
    )


def _tab_solve(spec: NoteSpec) -> html.Div:
    # the participation first: the term a structurer most often solves for
    terms = sorted(
        (p for p in spec.params if p.kind != "toggle"),
        key=lambda p: not p.key.startswith("participation"),
    )
    return html.Div(
        [
            ui.note(
                "Fix every term but one and find the value that prices the note at your target: "
                "for instance the participation an issue price of 100 % minus a 2 % margin pays."
            ),
            dbc.Row(
                [
                    dbc.Col(
                        ui.field(
                            "Term to solve",
                            ui.select(
                                "note-solve-term", {p.key: p.label for p in terms}, terms[0].key
                            ),
                        )
                    ),
                    dbc.Col(
                        ui.field(
                            "Target price (%)",
                            ui.number(
                                "note-solve-target", 5.0 if spec in SHARK_WARRANTS else 98.0, 0.25
                            ),
                        )
                    ),
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="bi bi-bullseye me-1"), "Solve"],
                            id="note-solve-run",
                            color="primary",
                            n_clicks=0,
                            className="mt-4",
                        ),
                        width="auto",
                    ),
                ],
                className="g-2",
            ),
            dcc.Store(id="note-solution"),
            dcc.Loading(html.Div(id="note-solve-result")),
            ui.apply_button("note-solve-apply"),
        ]
    )


def note_page(product: Product) -> html.Div:
    """Product page of a note priced by bricks (``product`` must be one)."""
    spec = note_spec(product)
    assert spec is not None  # the router only sends notes here
    return ui.product_shell(
        product,
        spec.description,
        _terms_card(spec),
        {
            "Payoff": html.Div(
                [dcc.Loading([dcc.Graph(id="note-payoff-graph"), html.Div(id="note-payoff-notes")])]
            ),
            "Decomposition": dcc.Loading(html.Div(id="note-decomposition")),
            "Greeks": dcc.Loading(html.Div(id="note-greeks")),
            "Risk & hedging": risk_panel.risk_tab("Notional sold", 1_000_000),
            "Monte Carlo": _tab_monte_carlo(),
            "Solve for": _tab_solve(spec),
        },
    )


# --- Pure functions behind the callbacks -------------------------------------------------------


def _note(store: dict[str, Any] | None, data: MarketData) -> NotePosition:
    item = risk_panel.resolve(store, data)
    if not isinstance(item, NotePosition):
        raise ValueError("This page has no note to price.")
    return item


def kpis(store: dict[str, Any] | None, data: MarketData) -> Any:
    """Price and value of the note, its bond and option parts, the underlying today."""
    try:
        position = _note(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    price = position.value()
    bond = position.bond_pct()
    g = desk_greeks(position.greeks(), position.market.spot)
    top = [
        ui.kpi(
            "Price", f"{price:.2f} %", f"{price / 100.0 * position.notional:,.0f} for the notional"
        )
    ]
    if bond:
        top.append(ui.kpi("Bond / options", f"{bond:.2f} / {price - bond:+.2f}", "% of notional"))
    top.append(
        ui.kpi("Underlying", f"{position.performance_pct:.1f} %", "today, in % of initial level")
    )
    children: list[Any] = [
        html.Div(f"Holder of the {position.note.name.lower()}", className="position-name mb-2"),
        dbc.Row(top, className="g-2 mb-2"),
        dbc.Row(
            [
                ui.kpi("Delta", f"{g['Delta (%)']:+.1f} %", "per +1 % on the underlying"),
                ui.kpi("Gamma", f"{g['Gamma (%)']:+.2f} %", "Δ change per +1 %"),
                ui.kpi("Vega", f"{g['Vega (%)']:+.3f} %", "per vol point"),
                ui.kpi("Theta", f"{g['Theta (%)']:+.4f} %", "per day"),
            ],
            className="g-2",
        ),
        ui.sheet_button("note-sheet-button"),
        dcc.Download(id="note-sheet-download"),
    ]
    if position.monitoring is not Monitoring.EUROPEAN:
        children.append(
            dbc.Alert(
                "Barrier observed along the path: the closed form reads one volatility per "
                "strike. The Monte Carlo tab measures what the smile does to the barrier.",
                color="info",
                className="py-2",
            )
        )
    warning = position.warning()
    if warning:
        children.append(ui.error(warning))
    return html.Div(children)


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
        ui.scenario_table("Underlying at maturity", levels, rows),
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
    """Bricks of the note: price, skew and Greeks; where each strike reads the smile."""
    if active != "decomposition":
        return dash.no_update
    try:
        position = _note(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    rows = decomposition(position)
    skew = any(abs(r["Skew (%)"]) > 1e-12 for r in rows)
    header = ["Leg", "Terms", "Price (%)", *(["of which skew"] if skew else []), *GREEKS]
    body = []
    for r in rows:
        line: list[Any] = [
            html.Span(r["Leg"], className=ui.side_class(r["Leg"])),
            r["Terms"],
            ui.fmt(r["Price (%)"], 3),
        ]
        if skew:
            line.append(ui.fmt(r["Skew (%)"], 3))
        line += [ui.fmt(r[g], 4 if g in ("Theta (%)", "Rho (%)") else 2) for g in GREEKS]
        body.append(line)
    smile = [
        [
            r["Level"],
            f"{r['% of initial']:.1f}",
            f"{r['% of spot']:.1f}",
            f"{r['Volatility (%)']:.2f}",
        ]
        for r in smile_levels(position)
    ]
    children: list[Any] = [
        ui.table(header, body, total=True),
        ui.note(
            "The note is replicated exactly by these bricks, each priced in closed form; prices "
            "and Greeks in % of notional (Delta per +1 % spot is Delta/100 %)."
        ),
    ]
    if skew:
        total = rows[-1]
        children.append(
            dbc.Alert(
                f"The smile slope changes the digitals and barriers by {total['Skew (%)']:+.2f} % "
                f"of notional: with flat Black-Scholes prices the note would be worth "
                f"{total['Price (%)'] - total['Skew (%)']:.2f} % instead of "
                f"{total['Price (%)']:.2f} %.",
                color="info",
                className="py-2",
            )
        )
    children += [
        html.H6("Where each level reads the smile", className="mt-3"),
        ui.table(["Level", "% of initial", "% of spot", "Volatility (%)"], smile),
        ui.note(
            "A strike in % of the initial level sits elsewhere on today's smile once the "
            "underlying has moved: the volatility is read at its true moneyness."
        ),
    ]
    return html.Div(children)


def greeks_view(store: dict[str, Any] | None, data: MarketData, active: str | None) -> Any:
    """Greeks against the underlying's level, and what they mean for the holder and issuer."""
    if active != "greeks":
        return dash.no_update
    try:
        position = _note(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    today = position.performance_pct
    perf = np.union1d(GREEK_PERF, [round(today, 6)])
    profile = spot_profile(position, perf)
    now_index = int(np.argmin(np.abs(perf - today)))
    now = {k: float(profile[k][now_index]) for k in PROFILE}
    levels = {
        lv.name: lv.value_pct for lv in position.note.levels if lv.kind is LevelKind.PERFORMANCE
    }
    return html.Div(
        [
            ui.greek_cards(ex.note_greeks(now, perf, profile, levels)),
            dcc.Graph(figure=figures.note_greeks_figure(position.note, perf, profile, today)),
            ui.note(
                "Finite differences of the closed form: spot ±1 %, volatility ±1 point, implied "
                "volatilities fixed per strike. Dashed lines: the note's levels; red dot: today."
            ),
        ]
    )


def monte_carlo_view(
    n_clicks: int | None, store: dict[str, Any] | None, data: MarketData, paths: Any
) -> Any:
    """Monte Carlo price of the note against its closed form, leg by leg."""
    if not n_clicks:
        return ui.note("Click to simulate.")
    try:
        position = _note(store, data)
        mc = price_note(
            position.note, position.market, position.spread, n_paths=int(paths or MC_PATHS[1])
        )
    except ValueError as exc:
        return ui.error(str(exc))
    closed = position.value()
    gap = (mc.total.value - closed) / mc.total.std_error
    rows = []
    for r in monte_carlo_rows(position, mc):
        rows.append(
            [
                r["Leg"],
                f"{r['Closed form (%)']:.3f}",
                f"{r['Monte Carlo (%)']:.3f}",
                f"{r['Std error (%)']:.3f}",
                "" if r["Gap (s.e.)"] is None else ui.fmt(r["Gap (s.e.)"], 1),
            ]
        )
    if position.monitoring is Monitoring.EUROPEAN:
        reading = (
            "Both prices read the same smile: the gap is Monte Carlo noise, within ±2 standard "
            "errors 95 % of the time."
        )
    else:
        reading = (
            "The closed form reads one volatility, the strike's, for the whole path; the Monte "
            "Carlo diffuses on the local volatility of the whole smile. With a barrier observed "
            "along the path the gap is the smile's effect on the barrier, not noise."
        )
    return html.Div(
        [
            dbc.Row(
                [
                    ui.kpi(
                        "Monte Carlo",
                        f"{mc.total.value:.2f} %",
                        f"± {Z_95 * mc.total.std_error:.2f} % at 95 %",
                    ),
                    ui.kpi("Closed form", f"{closed:.2f} %", "sum of the bricks"),
                    ui.kpi(
                        "Gap", f"{mc.total.value - closed:+.2f} %", f"{gap:+.1f} standard errors"
                    ),
                ],
                className="g-2 mb-3",
            ),
            ui.table(
                ["Leg", "Closed form (%)", "Monte Carlo (%)", "Std error (%)", "Gap (s.e.)"],
                rows,
                total=True,
            ),
            ui.note(reading),
        ]
    )


def solve_view(
    n_clicks: int | None,
    store: dict[str, Any] | None,
    data: MarketData,
    key: str | None,
    target: float | None,
) -> tuple[Any, Any, dict[str, str]]:
    """Solve a term for a target price; the solution is kept to fill the term in."""
    if not n_clicks:
        return ui.note("Pick a term and a target, then Solve."), None, ui.HIDDEN
    try:
        position = _note(store, data)
        if target is None:
            raise ValueError("Type a target price.")
        param = next(p for p in position.spec.params if p.key == key)
        if param not in shown_params(position.spec, store["terms"]):  # type: ignore[index]
            raise ValueError(f"Switch the {param.label.lower()} on to solve for it.")

        def build(x: float) -> Note:
            return position.build({**position.values, param.key: x})

        solution = solve_note(
            build,
            position.market,
            float(target),
            position.solve_bounds(param.key),
            position.spread,
            param.label.lower(),
        )
    except (ValueError, StopIteration) as exc:
        message = str(exc) if isinstance(exc, ValueError) else "Pick a term to solve."
        return ui.error(message), None, ui.HIDDEN
    text = (
        f"**{param.label} = {solution.value:.2f} %** prices the note at {solution.price:.2f} % "
        "(closed form, exact)."
    )
    return (
        dbc.Alert(dcc.Markdown(text), color="success", className="py-2 mb-1"),
        {"key": param.key, "value": round(solution.value, 2)},
        ui.SHOWN,
    )


def apply_solution(
    n_clicks: int | None, solution: dict[str, Any] | None, ids: list[dict[str, str]]
) -> list[Any]:
    """Put the solved value in its term input."""
    if not n_clicks or not solution:
        return [dash.no_update] * len(ids)
    return [solution["value"] if i["key"] == solution["key"] else dash.no_update for i in ids]


def download_sheet(n_clicks: int | None, store: dict[str, Any] | None, data: MarketData) -> Any:
    """The product sheet of the note, as an HTML file."""
    try:
        position = _note(store, data)
    except ValueError:
        return dash.no_update
    return sheet.download(note_sheet(position))


# --- Callbacks ---------------------------------------------------------------------------------


def register(app: dash.Dash) -> None:
    """Register the callbacks of the note pages."""
    store, market = Input("product-store", "data"), Input("market-data", "data")
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
    app.callback(
        Output("note-mc-result", "children"),
        Input("note-mc-run", "n_clicks"),
        State("product-store", "data"),
        State("market-data", "data"),
        State("note-mc-paths", "value"),
    )(monte_carlo_view)
    app.callback(
        Output("note-solve-result", "children"),
        Output("note-solution", "data"),
        Output("note-solve-apply", "style"),
        Input("note-solve-run", "n_clicks"),
        State("product-store", "data"),
        State("market-data", "data"),
        State("note-solve-term", "value"),
        State("note-solve-target", "value"),
    )(solve_view)
    app.callback(
        Output(ui.term(ALL), "value", allow_duplicate=True),  # type: ignore[arg-type]
        Input("note-solve-apply", "n_clicks"),
        State("note-solution", "data"),
        State(ui.term(ALL), "id"),  # type: ignore[arg-type]
        prevent_initial_call=True,
    )(apply_solution)
    app.callback(
        Output("note-sheet-download", "data"),
        Input("note-sheet-button", "n_clicks"),
        State("product-store", "data"),
        State("market-data", "data"),
        prevent_initial_call=True,
    )(download_sheet)


__all__ = ["note_page", "register"]
