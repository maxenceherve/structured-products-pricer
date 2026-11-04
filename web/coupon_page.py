"""Product page of a coupon note: reverse convertibles and autocalls, on one or several stocks.

The note is priced by Monte Carlo under the local volatility of each underlying's smile, on the
Price button. Tabs: Payoff, Coupon (the observation schedule), Decomposition, Greeks and
Hedging, as on every product page, each number with its Monte Carlo error.
"""

from __future__ import annotations

import json
from typing import Any

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, ctx, dcc, html

from spricer.mc.coupon_notes import price_coupon_note
from spricer.mc.greeks import DELTA_BUMP, GAMMA_BUMP, coupon_note_greeks, coupon_note_profile
from spricer.mc.paths import Z_95
from spricer.products.coupon_notes import FREQUENCIES, CouponNoteSpec
from web import explain as ex
from web import figures, risk_panel, sheet_route, ui
from web.coupons import (
    UNDERLYING_FIELDS,
    Rows,
    add_underlying,
    basket_of,
    brick_rows,
    build_coupon,
    correlation_rows,
    coupon_spec,
    coupon_spec_of,
    default_terms,
    default_underlyings,
    greeks_json,
    greeks_now,
    nearest_rows,
    result_json,
    schedule_rows,
    setup_coupon,
)
from web.market import MarketData, default_data
from web.registry import Product
from web.scenarios import coupon_breakevens, coupon_scenarios

PATHS = (20_000, 50_000, 100_000)
PROFILE_PATHS = 20_000
HEADERS = ("Name", "Spot", "Initial", "Div. %", "ATM vol %")
PRICING = "Pricing by Monte Carlo..."
COMPUTING = "Computing the Greeks by Monte Carlo (about ten revaluations)..."
PROFILE_LEVELS = np.linspace(40.0, 160.0, 25)
AXIS = "today's level of the underlyings"
SCHEDULE_HEADER = [
    "Date (years)",
    "Coupon (%)",
    "Coupon barrier (%)",
    "Autocall barrier (%)",
    "Call probability (%)",
]


# --- Layout ------------------------------------------------------------------------------------


def _grid(id_: str, rows: Rows, columns: list[dict[str, Any]], height: str) -> dag.AgGrid:
    return dag.AgGrid(
        id=id_,
        rowData=rows,
        columnDefs=columns,
        defaultColDef={"editable": True, "resizable": True, "sortable": False, "minWidth": 56},
        dashGridOptions={"singleClickEdit": True, "stopEditingWhenCellsLoseFocus": True},
        columnSize="responsiveSizeToFit",
        style={"height": height},
        className="ag-theme-alpine",
    )


def _terms_card(spec: CouponNoteSpec, terms: dict[str, Any]) -> dbc.Card:
    term = ui.term
    params: list[Any] = []
    for p in spec.params:
        if p.kind == "toggle":
            control: Any = dbc.Switch(id=term(p.key), label=p.label, value=bool(p.default))
            params.append(dbc.Col(html.Div(control, className="pt-4"), xs=6))
        elif p.kind == "count":
            params.append(
                dbc.Col(ui.field(p.label, ui.number(term(p.key), terms[p.key], 1, min=0)), xs=6)
            )
        else:
            params.append(
                dbc.Col(
                    ui.field(
                        p.label if "%" in p.label else f"{p.label} (%)",
                        ui.number(term(p.key), terms[p.key], 0.5, min=0),
                    ),
                    xs=6,
                )
            )
    alive = dbc.Accordion(
        dbc.AccordionItem(
            [
                ui.note(
                    "For a note issued in the past: dates already observed are dropped, but the "
                    "non-call period and the step-down still count from issue."
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            ui.field(
                                "Years since issue", ui.number(term("elapsed"), 0.0, 0.25, min=0)
                            )
                        ),
                        dbc.Col(
                            ui.field(
                                "Coupons in memory (%)",
                                ui.number(term("memory_accrued"), 0.0, 0.5, min=0),
                            )
                        ),
                    ],
                    className="g-2",
                ),
                dbc.Switch(id=term("breached"), label="Barrier already breached", value=False),
            ],
            title="Note already alive (optional)",
        ),
        start_collapsed=True,
        className="mb-3",
    )
    underlyings = _grid(
        "cn-underlyings",
        default_underlyings(default_data()),
        [
            {"field": f, "headerName": h, "headerTooltip": f}
            | ({"type": "numericColumn"} if f != "Name" else {"minWidth": 96})
            for f, h in zip(UNDERLYING_FIELDS, HEADERS, strict=True)
        ],
        "150px",
    )
    fields: list[Any] = [
        dbc.Row(
            [
                dbc.Col(
                    ui.field(
                        "Maturity (years)", ui.number(term("maturity"), terms["maturity"], 0.25)
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
                        "Observations", ui.select(term("frequency"), list(FREQUENCIES), "Quarterly")
                    )
                ),
                dbc.Col(ui.field("Issue price (%)", ui.number(term("issue_price"), 100.0, 0.5))),
            ],
            className="g-2",
        ),
        ui.field(
            "Funding spread (bp)",
            ui.number(term("spread_bp"), None, 5.0, placeholder="= market"),
        ),
        html.Hr(),
        dbc.Row(params, className="g-2"),
    ]
    if spec.monitoring_choice:
        fields.append(ui.monitoring_field("Protection barrier observed", "european"))
    fields += [
        alive,
        html.Div("Underlyings (the note looks at the worst performance)", className="field-label"),
        underlyings,
        html.Div(
            [
                dbc.Button("Add an underlying", id="cn-add", color="link", size="sm", n_clicks=0),
                dbc.Button("Remove the last", id="cn-remove", color="link", size="sm", n_clicks=0),
            ]
        ),
        ui.note(
            "Each underlying uses the smile of the market panel, shifted to its own 1-year ATM "
            "volatility."
        ),
        html.Div(
            [
                html.Div("Correlation of the returns", className="field-label mt-2"),
                _grid("cn-correlation", [], [], "160px"),
                html.Div(id="cn-correlation-status"),
                dbc.Button(
                    "Use the nearest valid matrix",
                    id="cn-nearest",
                    color="link",
                    size="sm",
                    n_clicks=0,
                ),
            ],
            id="cn-correlation-box",
            style=ui.HIDDEN,
        ),
        html.Hr(),
        ui.field(
            "Monte Carlo paths", ui.select("cn-paths", {n: f"{n:,}" for n in PATHS}, PATHS[1])
        ),
        ui.note(
            "Levels in % of each underlying's initial level, coupons in % of the notional. "
            "Monte Carlo under the local volatility of each smile; every number comes with its "
            "standard error."
        ),
    ]
    solvable = {p.key: p.label for p in spec.params if p.kind == "pct"}
    solvable["spread_bp"] = "Funding spread"
    fields += [
        ui.price_block(),
        html.Div(id="cn-price-status", className="price-status"),
        ui.solver_block(solvable, 98.0, "cn-solve-run"),
    ]
    return dbc.Card(dbc.CardBody([html.H5("Terms"), *fields]), className="terms-card")


def coupon_page(product: Product, prefilled: dict[str, float] | None = None) -> html.Div:
    """Product page of a coupon note (``product`` must be one)."""
    spec = coupon_spec(product)
    assert spec is not None  # the router only sends coupon notes here
    terms = default_terms(spec)
    applied = ui.apply_prefill(terms, prefilled or {})
    return ui.product_shell(
        product,
        spec.description,
        _terms_card(spec, terms),
        {
            "Payoff": dcc.Loading(html.Div(id="cn-payoff")),
            "Coupon": dcc.Loading(html.Div(id="cn-coupon")),
            "Decomposition": dcc.Loading(html.Div(id="cn-decomposition")),
            "Greeks": dcc.Loading(html.Div(id="cn-greeks-result")),
            "Hedging": risk_panel.hedging_tab(),
        },
        kpis=html.Div(ui.note(ui.NOT_PRICED), id="cn-kpis"),
        prefilled=applied,
        stores=[
            dcc.Store(id="cn-result"),
            dcc.Store(id="cn-profile"),
            dcc.Store(id="cn-priced-rows"),
            dcc.Store(id="cn-priced-corr"),
            dcc.Store(id="cn-priced-paths"),
        ],
    )


# --- Pure functions behind the callbacks -------------------------------------------------------


def _apply(rows: Rows, changes: list[dict[str, Any]] | None) -> Rows:
    """Rows with the cells edited in a grid."""
    out = [dict(r) for r in rows]
    for change in changes or []:
        index, column = change.get("rowIndex"), change.get("colId")
        if isinstance(index, int) and 0 <= index < len(out) and column:
            out[index][column] = change.get("value")
    return out


def edit_underlyings(
    trigger: str | None,
    rows: Rows | None,
    data: MarketData,
    changes: list[dict[str, Any]] | None = None,
) -> Rows | Any:
    """Cells edited, an underlying added, or the last removed (at least one stays)."""
    rows = list(rows or [])
    if trigger == "cn-add":
        return add_underlying(rows, data)
    if trigger == "cn-remove" and len(rows) > 1:
        return rows[:-1]
    if trigger == "cn-underlyings" and changes:
        return _apply(rows, changes)
    return dash.no_update


def _edit_underlyings(
    _: Any, __: Any, changes: Any, rows: Rows | None, data: MarketData
) -> Any:  # pragma: no cover - reads the Dash context
    return edit_underlyings(ctx.triggered_id, rows, data, changes)


def correlation_view(
    trigger: str | None,
    rows: Rows | None,
    previous: Rows | None,
    changes: list[dict[str, Any]] | None = None,
) -> tuple[Any, Any, dict[str, str], Any]:
    """Correlation grid of the underlyings, its columns, whether it shows, and its status.

    A typed correlation is written on both sides of the diagonal.
    """
    names = [str(r.get("Name") or "") for r in rows or []]
    shown = ui.SHOWN if len(names) > 1 else ui.HIDDEN
    columns = [{"field": "Name", "editable": False}] + [
        {"field": n, "type": "numericColumn"} for n in names
    ]
    typed = [dict(r) for r in previous or []]
    status: Any = ""
    if trigger == "cn-correlation":
        for change in changes or []:
            index, column = change.get("rowIndex"), change.get("colId")
            if isinstance(index, int) and 0 <= index < len(typed) and column in names:
                typed[index][column] = change.get("value")
                mirror = next((r for r in typed if r.get("Name") == column), None)
                if mirror is not None:
                    mirror[str(typed[index]["Name"])] = change.get("value")
    current = correlation_rows(names, typed)
    if trigger == "cn-nearest" and len(names) > 1:
        try:
            current = nearest_rows(names, current)
            status = ui.note("Replaced by the nearest valid correlation matrix (Higham).")
        except ValueError as exc:
            status = ui.error(str(exc))
    return current, columns, shown, status


def _correlation_view(
    _: Any, __: Any, changes: Any, rows: Rows | None, previous: Rows | None
) -> Any:  # pragma: no cover - reads the Dash context
    return correlation_view(ctx.triggered_id, rows, previous, changes)


def fingerprint(
    store: dict[str, Any] | None, rows: Rows | None, corr: Rows | None, data: MarketData, paths: Any
) -> str:
    """What a price depends on: stale results are told apart from fresh ones."""
    terms = (store or {}).get("terms")
    return json.dumps(
        [(store or {}).get("slug"), terms, rows, corr, data, paths], sort_keys=True, default=str
    )


def priced_underlyings(
    n_clicks: int | None, rows: Rows | None, corr: Rows | None, paths: Any
) -> Any:
    """On the Price button, what is priced besides the terms: the grids and the paths."""
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    return rows, corr, paths


def underlyings_status(
    rows: Rows | None,
    corr: Rows | None,
    paths: Any,
    store: dict[str, Any] | None,
    priced_rows: Rows | None,
    priced_corr: Rows | None,
    priced_paths: Any,
) -> Any:
    """Tell when the numbers on screen are no longer those of the underlyings in the form."""
    if not store or (rows, corr, paths) == (priced_rows, priced_corr, priced_paths):
        return ""
    return ui.note(ui.STALE)


def price_compute(
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
) -> dict[str, Any] | None:
    """Monte Carlo price of the terms, kept in the page; an error is kept the same way."""
    if not store:
        return None
    try:
        terms, basket = setup_coupon(store, rows, corr, data)
        result = price_coupon_note(terms.note, basket, terms.spread, n_paths=int(paths or 50_000))
    except ValueError as exc:
        return {"error": str(exc)}
    return result_json(result) | {"fingerprint": fingerprint(store, rows, corr, data, paths)}


def _fresh(result: dict[str, Any] | None, current: str) -> bool:
    """The result was computed for these terms, and without an error."""
    return bool(result) and result.get("fingerprint") == current  # type: ignore[union-attr]


def greeks_compute(
    result: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
) -> dict[str, Any] | None:
    """After a price, the Greeks by bump and revalue, kept in the page."""
    current = fingerprint(store, rows, corr, data, paths)
    if not _fresh(result, current):
        return None
    try:
        terms, basket = setup_coupon(store, rows, corr, data)
        greeks = coupon_note_greeks(terms.note, basket, terms.spread, n_paths=int(paths or 50_000))
    except ValueError:
        return None
    return greeks_json(greeks, basket, rows or [], terms.notional) | {"fingerprint": current}


def kpis_view(
    result: dict[str, Any] | None,
    greeks: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
) -> Any:
    """Price and Greeks of the note, once they are computed."""
    if not store:
        return ui.note(ui.NOT_PRICED)
    if result and "error" in result:
        return ui.error(result["error"])
    current = fingerprint(store, rows, corr, data, paths)
    if not _fresh(result, current):
        return ui.note(PRICING)
    assert result is not None
    terms = build_coupon(coupon_spec_of(store or {}), (store or {})["terms"])
    price, se = result["total"]
    fresh = _fresh(greeks, current)
    now, errors = greeks_now(greeks) if fresh else (None, None)  # type: ignore[arg-type]
    state = {"store": store, "market": data, "rows": rows, "corr": corr, "result": result}
    if fresh:
        state["greeks"] = greeks
    return ui.headline(
        f"Holder of the {terms.note.name.lower()}",
        f"{price:.2f} %",
        f"± {Z_95 * se:.2f} % at 95 %",
        now,
        errors,
        facts=[
            f"Issuer margin {terms.issue_price - price:+.2f} %",
            f"Expected life {result['life'][0]:.2f} y",
            f"Probability of loss {100 * result['loss'][0]:.1f} %",
            f"Barrier breached at maturity on {100 * result['breach'][0]:.1f} % of the "
            f"{result['paths']:,} paths",
        ],
        sheet=ui.sheet_link(sheet_route.href(state)),
    )


def price_headline(
    result: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
) -> Any:
    """The price as soon as it is there; the Greeks follow (ten more revaluations)."""
    return kpis_view(result, None, store, rows, corr, data, paths)


def payoff_view(
    result: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
    active: str | None,
) -> Any:
    """What the note pays at maturity if never called."""
    if active != "payoff":
        return dash.no_update
    try:
        terms = build_coupon(coupon_spec_of(store or {}), (store or {})["terms"])
    except (ValueError, KeyError) as exc:
        return ui.error(
            str(exc) if isinstance(exc, ValueError) else "Set the terms of the product."
        )
    try:
        today = basket_of(list(rows or []), corr, data).today_worst_pct()
    except ValueError:
        today = 100.0
    note = terms.note
    fresh = _fresh(result, fingerprint(store, rows, corr, data, paths))
    price = result["total"][0] if fresh and result else None
    notes = [
        ui.note(
            "Without a call the note ends at maturity and pays this, against the worst performance "
            "of the underlyings. Coupons paid before maturity are not shown; the gain counts those "
            "that are certain, not those that depend on the underlying."
        )
    ]
    if note.observations[-1].coupon_level > 0.0:
        notes.append(
            ui.note(
                "The capital is flat above the barrier, as on every coupon note. The final coupon "
                f"is only paid at or above {note.observations[-1].coupon_level:g} %, hence the "
                "step of the dashed line; a reverse convertible's coupon is certain: no step."
            )
        )
    if note.memory:
        notes.append(
            ui.note(
                "With memory, the coupons missed before maturity are paid on top of the final "
                "coupon shown here when the coupon condition holds."
            )
        )
    levels, table_rows = coupon_scenarios(note, price)
    points = coupon_breakevens(note, price) if price is not None else []
    return html.Div(
        [
            dcc.Graph(
                figure=figures.coupon_payoff_figure(note, today), config={"displaylogo": False}
            ),
            html.H6("Paid at maturity if never called", className="mt-2"),
            ui.scenario_table("Worst performance", levels, table_rows, points),
            *notes,
        ]
    )


def coupon_view(
    result: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
    active: str | None,
) -> Any:
    """The observation schedule: dates, coupons, barriers, probability of a call."""
    if active != "coupon":
        return dash.no_update
    try:
        terms = build_coupon(coupon_spec_of(store or {}), (store or {})["terms"])
    except (ValueError, KeyError) as exc:
        return ui.error(
            str(exc) if isinstance(exc, ValueError) else "Set the terms of the product."
        )
    fresh = _fresh(result, fingerprint(store, rows, corr, data, paths))
    schedule = schedule_rows(terms.note, result if fresh else None)
    body = [
        [
            f"{t:.2f}",
            f"{c:.2f}",
            f"{cl:g}",
            "" if a is None else f"{a:g}",
            "" if p is None else f"{p:.1f}",
        ]
        for t, c, cl, a, p in schedule
    ]
    return html.Div([html.H6("Observation schedule"), ui.table(SCHEDULE_HEADER, body)])


def decomposition_view(
    result: dict[str, Any] | None,
    greeks: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
    active: str | None,
) -> Any:
    """The bricks of the note on common paths: price and Greeks."""
    if active != "decomposition":
        return dash.no_update
    if result and "error" in result:
        return ui.error(result["error"])
    current = fingerprint(store, rows, corr, data, paths)
    if not _fresh(result, current):
        return ui.note(PRICING)
    assert result is not None
    terms = build_coupon(coupon_spec_of(store or {}), (store or {})["terms"])
    fresh = _fresh(greeks, current)
    children: list[Any] = [
        ui.bricks_table(brick_rows(terms.note, result, greeks if fresh else None)),
        ui.note(
            "Every brick is valued on the same paths, so they add up to the price exactly; the "
            "price with its Monte Carlo standard error. The down-and-in put is what the holder "
            "has sold to pay for the coupon. In % of notional: " + ui.GREEK_UNITS_TEXT
        ),
    ]
    if not fresh:
        children.append(ui.note(COMPUTING))
    return html.Div(children)


def profile_compute(
    greeks: dict[str, Any] | None,
    previous: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    active: str | None,
) -> Any:
    """The Greeks against the level of the underlyings, when the Greeks tab is shown."""
    if active != "greeks" or not greeks:
        return dash.no_update
    if previous and previous["fingerprint"] == greeks["fingerprint"]:
        return dash.no_update
    terms, basket = setup_coupon(store, rows, corr, data)
    today = basket.today_worst_pct()
    levels = np.union1d(PROFILE_LEVELS, [round(today, 6)])
    profile = coupon_note_profile(
        terms.note,
        basket,
        100.0 * levels / today,
        terms.spread,
        n_paths=PROFILE_PATHS,
    )
    return {
        "fingerprint": greeks["fingerprint"],
        "levels": levels.tolist(),
        "today": today,
        "profile": {k: v.tolist() for k, v in profile.items()},
    }


def greeks_view(
    greeks: dict[str, Any] | None,
    profile: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
    active: str | None,
) -> Any:
    """Greeks today, what they mean, and against the level of the underlyings."""
    if active != "greeks":
        return dash.no_update
    current = fingerprint(store, rows, corr, data, paths)
    if not _fresh(greeks, current):
        return ui.note(COMPUTING)
    assert greeks is not None
    terms = build_coupon(coupon_spec_of(store or {}), (store or {})["terms"])
    now, errors = greeks_now(greeks)
    assets = greeks["assets"]
    children: list[Any] = []
    if len(assets) > 1:
        body = [
            [
                a["name"],
                ui.fmt(a["delta"][0], 2),
                f"{a['delta'][1]:.2f}",
                ui.fmt(a["gamma"][0], 3),
                f"{a['gamma'][1]:.3f}",
                ui.fmt(a["vega"][0], 3),
                f"{a['vega'][1]:.3f}",
            ]
            for a in assets
        ]
        children += [
            html.H6("Per underlying"),
            ui.table(["Underlying", "Delta (%)", "±", "Gamma (%)", "±", "Vega (%)", "±"], body),
        ]
    if not profile or profile["fingerprint"] != greeks["fingerprint"]:
        x = np.array([100.0])
        curves = {k: np.array([float(v or 0.0)]) for k, v in now.items()}
        cards, notes = ex.greek_cards("note", now, x, curves, {}, AXIS, errors)
        return html.Div([*children, ui.greek_cards(cards, notes), ui.note(COMPUTING)])
    x = np.array(profile["levels"])
    curves = {k: np.array(v) for k, v in profile["profile"].items()}
    levels = {lv.name: lv.value_pct for lv in terms.note.levels}
    cards, notes = ex.greek_cards("note", now, x, curves, levels, AXIS, errors)
    return html.Div(
        [
            *children,
            ui.greek_cards(cards, notes),
            dcc.Graph(
                figure=figures.greeks_figure(
                    x,
                    curves,
                    profile["today"],
                    list(levels.values()),
                    "Worst performance today (% of initial level)",
                )
            ),
            ui.note(
                f"{figures.GREEKS_CAPTION} Curves: every underlying moved together, the smile "
                "moving with them. Cards: each Greek with its Monte Carlo standard error, by bump "
                f"and revalue on common random numbers: spot ±{100 * DELTA_BUMP:.0f} % for delta "
                f"and ±{100 * GAMMA_BUMP:.0f} % for gamma (the payoff jumps at its barriers), ±1 "
                "volatility point, ±1 bp, one day, the implied volatility of each strike fixed."
            ),
        ]
    )


# --- Callbacks ---------------------------------------------------------------------------------


def register(app: dash.Dash) -> None:
    """Register the callbacks of the coupon note pages."""
    # the results read what the Price button has fixed, the form what is being typed
    store, data = State("product-store", "data"), State("priced-market", "data")
    rows, corr = State("cn-priced-rows", "data"), State("cn-priced-corr", "data")
    paths = State("cn-priced-paths", "data")
    form_rows, form_corr = State("cn-underlyings", "rowData"), State("cn-correlation", "rowData")
    inputs = (
        Input("product-store", "data"),
        Input("cn-priced-rows", "data"),
        Input("cn-priced-corr", "data"),
        Input("priced-market", "data"),
        Input("cn-priced-paths", "data"),
    )
    tab = Input("product-tabs", "active_tab")
    result = Input("cn-result", "data")
    greeks = Input("product-greeks", "data")
    app.callback(
        Output("cn-underlyings", "rowData"),
        Input("cn-add", "n_clicks"),
        Input("cn-remove", "n_clicks"),
        Input("cn-underlyings", "cellValueChanged"),
        form_rows,
        State("market-data", "data"),
        prevent_initial_call=True,
    )(_edit_underlyings)
    app.callback(
        Output("cn-correlation", "rowData"),
        Output("cn-correlation", "columnDefs"),
        Output("cn-correlation-box", "style"),
        Output("cn-correlation-status", "children"),
        Input("cn-underlyings", "rowData"),
        Input("cn-nearest", "n_clicks"),
        Input("cn-correlation", "cellValueChanged"),
        form_rows,
        form_corr,
    )(_correlation_view)
    app.callback(
        Output("cn-priced-rows", "data"),
        Output("cn-priced-corr", "data"),
        Output("cn-priced-paths", "data"),
        Input("price-run", "n_clicks"),
        form_rows,
        form_corr,
        State("cn-paths", "value"),
        prevent_initial_call=True,
    )(priced_underlyings)
    app.callback(
        Output("cn-price-status", "children"),
        Input("cn-underlyings", "rowData"),
        Input("cn-correlation", "rowData"),
        Input("cn-paths", "value"),
        Input("product-store", "data"),
        rows,
        corr,
        paths,
    )(underlyings_status)
    app.callback(
        Output("cn-result", "data"),
        *inputs,
        running=[(Output("price-run", "disabled"), True, False)],
    )(price_compute)
    app.callback(Output("product-greeks", "data"), result, store, rows, corr, data, paths)(
        greeks_compute
    )
    app.callback(Output("cn-kpis", "children"), result, *inputs)(price_headline)
    app.callback(
        Output("cn-kpis", "children", allow_duplicate=True),
        State("cn-result", "data"),
        greeks,
        store,
        rows,
        corr,
        data,
        paths,
        prevent_initial_call=True,
    )(kpis_view)
    app.callback(Output("cn-payoff", "children"), result, *inputs, tab)(payoff_view)
    app.callback(Output("cn-coupon", "children"), result, *inputs, tab)(coupon_view)
    app.callback(Output("cn-decomposition", "children"), result, greeks, *inputs, tab)(
        decomposition_view
    )
    app.callback(
        Output("cn-profile", "data"),
        greeks,
        State("cn-profile", "data"),
        store,
        rows,
        corr,
        data,
        tab,
    )(profile_compute)
    app.callback(
        Output("cn-greeks-result", "children"), greeks, Input("cn-profile", "data"), *inputs, tab
    )(greeks_view)


__all__ = ["coupon_page", "register"]
