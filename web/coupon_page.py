"""Product page of a coupon note: reverse convertibles and autocalls, on one or several stocks.

The note is priced by Monte Carlo under the local volatility of each underlying's smile, when
the user clicks Price. Tabs: Payoff (if never called), Legs (on common paths, with the closed
form when there is one), Life (when the note ends), Greeks (by bump and revalue) and Solve for.
"""

from __future__ import annotations

import json
from typing import Any

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, ctx, dcc, html

from spricer.mc.coupon_notes import price_coupon_note
from spricer.mc.greeks import DELTA_BUMP, GAMMA_BUMP, coupon_note_greeks
from spricer.mc.paths import Z_95
from spricer.monitoring import Monitoring
from spricer.products.coupon_notes import FREQUENCIES, CouponNoteSpec
from spricer.solve import solve_coupon_note
from web import explain as ex
from web import figures, sheet, ui
from web.coupons import (
    SOLVE_BOUNDS,
    UNDERLYING_FIELDS,
    CouponTerms,
    Rows,
    add_underlying,
    basket_of,
    build_coupon,
    correlation_rows,
    coupon_spec,
    default_terms,
    default_underlyings,
    legs_rows,
    nearest_rows,
    result_json,
    schedule_rows,
)
from web.market import MarketData, default_data, funding_bp
from web.registry import Product
from web.scenarios import coupon_scenarios
from web.sheets import coupon_sheet

PATHS = (20_000, 50_000, 100_000)
HEADERS = ("Name", "Spot", "Initial", "Div. %", "ATM vol %")
STALE = "The terms or the market changed since the last price: click Price again."
NOT_PRICED = "Click Price to run the Monte Carlo."


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
            continue
        label = p.label if p.kind == "count" else f"{p.label} (%)"
        step = 1 if p.kind == "count" else 0.5
        params.append(
            dbc.Col(ui.field(label, ui.number(term(p.key), terms[p.key], step, min=0)), xs=6)
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
                dbc.Col(
                    ui.field(
                        "Observations", ui.select(term("frequency"), list(FREQUENCIES), "Quarterly")
                    )
                ),
            ],
            className="g-2",
        ),
        dbc.Row(
            [
                dbc.Col(ui.field("Issue price (%)", ui.number(term("issue_price"), 100.0, 0.5))),
                dbc.Col(ui.field("Notional", ui.number(term("notional"), terms["notional"], 1e5))),
            ],
            className="g-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    ui.field(
                        "Funding spread (bp)",
                        ui.number(term("spread_bp"), None, 5.0, placeholder="= market"),
                    )
                ),
                *(
                    [dbc.Col(ui.monitoring_field("Protection barrier observed", "european"))]
                    if spec.monitoring_choice
                    else []
                ),
            ],
            className="g-2",
        ),
        html.Hr(),
        dbc.Row(params, className="g-2"),
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
        dbc.Row(
            [
                dbc.Col(ui.select("cn-paths", {n: f"{n:,} paths" for n in PATHS}, 50_000)),
                dbc.Col(
                    dbc.Button(
                        [html.I(className="bi bi-calculator me-1"), "Price"],
                        id="cn-price",
                        color="primary",
                        n_clicks=0,
                        className="w-100",
                    )
                ),
            ],
            className="g-2",
        ),
        ui.note(
            "Levels in % of each underlying's initial level, coupons in % of the notional. "
            "Monte Carlo under the local volatility of each smile; every number comes with its "
            "standard error."
        ),
    ]
    return dbc.Card(dbc.CardBody([html.H5("Terms"), *fields]), className="terms-card")


def _button(id_: str, icon: str, label: str) -> dbc.Button:
    return dbc.Button(
        [html.I(className=f"bi {icon} me-1"), label], id=id_, color="primary", n_clicks=0
    )


def coupon_page(product: Product, prefilled: dict[str, float] | None = None) -> html.Div:
    """Product page of a coupon note (``product`` must be one)."""
    spec = coupon_spec(product)
    assert spec is not None  # the router only sends coupon notes here
    terms = default_terms(spec)
    applied = ui.apply_prefill(terms, prefilled or {})
    solvable = [p for p in spec.params if p.solvable]
    solve = html.Div(
        [
            ui.note(
                "Fix every term but one and find the value that prices the note at the issue "
                "price minus your margin: what a structurer does before quoting. Solved on one "
                "set of paths, with the Monte Carlo error of the solution."
            ),
            dbc.Row(
                [
                    dbc.Col(
                        ui.field(
                            "Term to solve",
                            ui.select(
                                "cn-solve-term", {p.key: p.label for p in solvable}, solvable[0].key
                            ),
                        )
                    ),
                    dbc.Col(ui.field("Margin (%)", ui.number("cn-solve-margin", 2.0, 0.25))),
                    dbc.Col(
                        html.Div(_button("cn-solve-run", "bi-bullseye", "Solve"), className="pt-4"),
                        width="auto",
                    ),
                ],
                className="g-2",
            ),
            dcc.Store(id="cn-solution"),
            dcc.Loading(html.Div(id="cn-solve-result")),
            ui.apply_button("cn-solve-apply"),
        ]
    )
    greeks = dcc.Loading(html.Div(id="cn-greeks-result"))
    return ui.product_shell(
        product,
        spec.description,
        _terms_card(spec, terms),
        {
            "Payoff": dcc.Loading(html.Div(id="cn-payoff")),
            "Legs": dcc.Loading(html.Div(id="cn-legs")),
            "Greeks & hedging": greeks,
            "Life": dcc.Loading(html.Div(id="cn-life")),
            "Solve for": solve,
        },
        kpis=html.Div(
            [
                dcc.Store(id="cn-result"),
                dcc.Store(id="cn-greeks"),
                html.Div(ui.note(NOT_PRICED), id="cn-kpis"),
                dcc.Loading(html.Div(id="cn-greeks-kpis", className="mt-2")),
                dcc.Download(id="cn-sheet-download"),
            ]
        ),
        prefilled=applied,
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


def _setup(
    store: dict[str, Any] | None, rows: Rows | None, corr: Rows | None, data: MarketData
) -> tuple[CouponTerms, Any]:
    if not store:
        raise ValueError("Set the terms of the product.")
    product_spec = coupon_spec_of(store)
    terms = build_coupon(product_spec, store["terms"], funding_bp(data))
    return terms, basket_of(list(rows or []), corr, data)


def coupon_spec_of(store: dict[str, Any]) -> CouponNoteSpec:
    """The coupon note of a page's stored terms."""
    from web import registry

    product = registry.find(store.get("slug", ""))
    spec = coupon_spec(product) if product is not None else None
    if spec is None:
        raise ValueError("This page has no coupon note to price.")
    return spec


def price_view(
    n_clicks: int | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
) -> tuple[Any, Any]:
    """Monte Carlo price of the note: the result kept in the page, and the headline numbers."""
    if not n_clicks:
        return dash.no_update, dash.no_update
    try:
        terms, basket = _setup(store, rows, corr, data)
        result = price_coupon_note(terms.note, basket, terms.spread, n_paths=int(paths or 50_000))
    except ValueError as exc:
        return None, ui.error(str(exc))
    kept = result_json(result) | {"fingerprint": fingerprint(store, rows, corr, data, paths)}
    price, se = kept["total"]
    life, loss, breach = kept["life"][0], kept["loss"][0], kept["breach"][0]
    return kept, html.Div(
        [
            dbc.Row(
                [
                    ui.kpi(
                        "Price", f"{price:.2f} %", f"± {Z_95 * se:.2f} % at 95 %", featured=True
                    ),
                    ui.kpi(
                        "Issuer margin",
                        f"{terms.issue_price - price:+.2f} %",
                        "issue price − fair value",
                    ),
                    ui.kpi("Expected life", f"{life:.2f} y", "until called or matured"),
                    ui.kpi(
                        "Probability of loss",
                        f"{100 * loss:.1f} %",
                        "capital not paid back in full",
                    ),
                ],
                className="g-2",
            ),
            ui.note(
                f"Value of the notional {price / 100.0 * terms.notional:,.0f}. Barrier breached "
                f"and note alive at maturity on {100 * breach:.1f} % of the paths "
                f"({kept['paths']:,} paths)."
            ),
            ui.sheet_button("cn-sheet-button"),
        ]
    )


def _fresh(result: dict[str, Any] | None, current: str) -> str | None:
    """None when the result is up to date, else why it cannot be shown."""
    if not result:
        return NOT_PRICED
    if result["fingerprint"] != current:
        return STALE
    return None


def payoff_view(
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
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
    notes = [
        ui.note(
            "Coupons paid before maturity are not shown. Without a call the note ends at "
            "maturity and pays this, against the worst performance of the underlyings."
        )
    ]
    if terms.note.memory:
        notes.append(
            ui.note(
                "With memory, the coupons missed before maturity are paid on top of the final "
                "coupon shown here when the coupon condition holds."
            )
        )
    levels, table_rows = coupon_scenarios(terms.note)
    return html.Div(
        [
            dcc.Graph(figure=figures.coupon_payoff_figure(terms.note, today)),
            html.H6("Paid at maturity if never called", className="mt-2"),
            ui.scenario_table("Worst performance", levels, table_rows),
            *notes,
        ]
    )


def legs_view(
    result: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
    active: str | None,
) -> Any:
    """Legs on common paths, and the closed form when the note has one."""
    if active != "legs":
        return dash.no_update
    why = _fresh(result, fingerprint(store, rows, corr, data, paths))
    if why:
        return ui.note(why)
    assert result is not None
    body = [[name, f"{v:.3f}", f"{se:.3f}"] for name, v, se in legs_rows(result)]
    children: list[Any] = [
        ui.table(["Leg", "Value (%)", "Std error (%)"], body, total=True),
        ui.note(
            "Every leg is valued on the same paths, so they add up to the price exactly. The "
            "last leg is the down-and-in put on the worst performance the holder has sold: what "
            "pays for the coupon."
        ),
    ]
    terms, basket = _setup(store, rows, corr, data)
    replication = terms.note.replication(basket.initials[0]) if basket.size == 1 else None
    if replication is not None:
        closed = float(replication.value(basket.markets[0], terms.spread))
        price, se = result["total"]
        children.append(
            dbc.Row(
                [
                    ui.kpi(
                        "Closed form", f"{closed:.2f} %", "bond, coupons and a put on the smile"
                    ),
                    ui.kpi(
                        "Monte Carlo − closed form",
                        f"{price - closed:+.2f} %",
                        f"{(price - closed) / se:+.1f} standard errors",
                    ),
                ],
                className="g-2 mt-2",
            )
        )
        if terms.note.monitoring is Monitoring.EUROPEAN or terms.note.barrier_breached:
            children.append(
                ui.note(
                    "Same smile on both sides: the gap is Monte Carlo noise, within ±2 standard "
                    "errors 95 % of the time."
                )
            )
        else:
            children.append(
                ui.note(
                    "The closed form prices the down-and-in put at one volatility, the strike's; "
                    "the Monte Carlo diffuses on the local volatility of the whole smile. The gap "
                    "is the smile's effect on the barrier, not noise."
                )
            )
    return html.Div(children)


def life_view(
    result: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
    active: str | None,
) -> Any:
    """When the note ends, and its observation schedule."""
    if active != "life":
        return dash.no_update
    try:
        terms = build_coupon(coupon_spec_of(store or {}), (store or {})["terms"])
    except (ValueError, KeyError) as exc:
        return ui.error(
            str(exc) if isinstance(exc, ValueError) else "Set the terms of the product."
        )
    fresh = _fresh(result, fingerprint(store, rows, corr, data, paths)) is None
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
    children: list[Any] = []
    if fresh:
        assert result is not None
        calls = [100.0 * p for p, _ in result["calls"]]
        children.append(dcc.Graph(figure=figures.call_probability_figure(terms.note, calls)))
    else:
        children.append(ui.note(NOT_PRICED + " The probabilities of a call will show here."))
    children += [
        html.H6("Observation schedule", className="mt-2"),
        ui.table(
            [
                "Date (years)",
                "Coupon (%)",
                "Coupon barrier (%)",
                "Autocall barrier (%)",
                "Call probability (%)",
            ],
            body,
        ),
    ]
    return html.Div(children)


def _greeks_json(greeks: Any, basket: Any, notional: float, current: str) -> dict[str, Any]:
    def est(e: Any) -> list[float]:
        return [e.value, e.std_error]

    return {
        "fingerprint": current,
        "assets": [
            {"name": a.name, "delta": est(a.delta), "gamma": est(a.gamma), "vega": est(a.vega)}
            for a in greeks.assets
        ],
        "theta": None if greeks.theta is None else est(greeks.theta),
        "rho": est(greeks.rho),
        "spots": [m.spot for m in basket.markets],
        "notional": notional,
    }


def greeks_compute(
    result: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
) -> tuple[Any, Any]:
    """After a price, the Greeks by bump and revalue: kept in the page, and their headline."""
    current = fingerprint(store, rows, corr, data, paths)
    if _fresh(result, current):
        return None, ""
    try:
        terms, basket = _setup(store, rows, corr, data)
        greeks = coupon_note_greeks(terms.note, basket, terms.spread, n_paths=int(paths or 50_000))
    except ValueError as exc:
        return None, ui.error(str(exc))
    kept = _greeks_json(greeks, basket, terms.notional, current)

    def total(key: str) -> tuple[float, float]:
        values = [a[key] for a in kept["assets"]]
        return sum(v for v, _ in values), sum(se**2 for _, se in values) ** 0.5

    delta, gamma, vega = total("delta"), total("gamma"), total("vega")
    several = " (all underlyings)" if len(kept["assets"]) > 1 else ""
    theta = kept["theta"]
    cards = [
        ui.kpi(f"Delta{several}", f"{delta[0]:+.1f} %", f"± {delta[1]:.1f} · per +1 % spot"),
        ui.kpi(f"Gamma{several}", f"{gamma[0]:+.2f} %", f"± {gamma[1]:.2f} · Δ change per +1 %"),
        ui.kpi(f"Vega{several}", f"{vega[0]:+.3f} %", f"± {vega[1]:.3f} · per vol point"),
        ui.kpi(
            "Theta",
            "n/a" if theta is None else f"{theta[0]:+.4f} %",
            "observation within a day" if theta is None else f"± {theta[1]:.4f} · per day",
        ),
    ]
    return kept, dbc.Row(cards, className="g-2")


def greeks_view(
    greeks: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
    active: str | None,
) -> Any:
    """Greeks per underlying with their errors, the issuer's hedge, and their reading."""
    if active != "greeks-hedging":
        return dash.no_update
    why = _fresh(greeks, fingerprint(store, rows, corr, data, paths))
    if why:
        return ui.note(
            why + " The Greeks are computed right after the price (about ten revaluations)."
        )
    assert greeks is not None
    assets = greeks["assets"]
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
    hedge = []
    for a, spot in zip(assets, greeks["spots"], strict=True):
        shares = a["delta"][0] / 100.0 * greeks["notional"] / spot
        verb = "Buy" if shares >= 0 else "Sell"
        hedge.append(
            [
                html.Span(f"{verb} {abs(shares):,.0f} shares", className=ui.side_class(verb)),
                a["name"],
                f"{spot:,.2f}",
                f"{a['delta'][0] / 100.0 * greeks['notional']:,.0f}",
            ]
        )
    theta = greeks["theta"]
    theta_text = (
        "not computed: an observation falls within the next day"
        if theta is None
        else f"{theta[0]:+.4f} ± {theta[1]:.4f} % a day"
    )
    return html.Div(
        [
            html.H6("Greeks of the note, per underlying"),
            ui.table(["Underlying", "Delta (%)", "±", "Gamma (%)", "±", "Vega (%)", "±"], body),
            ui.note(
                f"Theta {theta_text}; rho {greeks['rho'][0]:+.4f} ± {greeks['rho'][1]:.4f} % per "
                "basis point. Each Greek with its Monte Carlo standard error."
            ),
            html.H6("The issuer's delta hedge", className="mt-3"),
            ui.table(["Trade", "Underlying", "Spot", "Exposure"], hedge),
            ui.note(
                "The issuer sold the note: it holds the note's delta in shares of each "
                "underlying, for the whole notional, and rebalances as the deltas move."
            ),
            ui.bullets(
                ex.coupon_greeks(
                    [a["name"] for a in assets],
                    [a["delta"][0] for a in assets],
                    [a["vega"][0] for a in assets],
                )
            ),
            ui.note(
                f"Bump and revalue on common random numbers: spot ±{100 * DELTA_BUMP:.0f} % for "
                f"delta and ±{100 * GAMMA_BUMP:.0f} % for gamma (the payoff jumps at its barriers: "
                "smaller bumps would be unbiased but far noisier), ±1 volatility point, ±1 bp, one "
                "day."
            ),
        ]
    )


def solve_view(
    n_clicks: int | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
    key: str | None,
    margin: float | None,
) -> tuple[Any, Any, dict[str, str]]:
    """Solve a term so that the note is worth the issue price minus the margin."""
    if not n_clicks:
        return ui.note("Pick a term and a margin, then Solve."), None, ui.HIDDEN
    try:
        terms, basket = _setup(store, rows, corr, data)
        if key not in SOLVE_BOUNDS:
            raise ValueError("Pick a term to solve.")
        target = terms.issue_price - float(margin or 0.0)
        label = next(p.label for p in terms.spec.params if p.key == key)

        def build(x: float) -> Any:
            return terms.build({**terms.values, key: x})

        solution, _ = solve_coupon_note(
            build,
            basket,
            target,
            SOLVE_BOUNDS[key],
            terms.spread,
            what=label.lower(),
            n_paths=int(paths or 50_000),
        )
    except ValueError as exc:
        return ui.error(str(exc)), None, ui.HIDDEN
    error = f" ± {solution.std_error:.2f}" if solution.std_error is not None else ""
    text = (
        f"**{label} = {solution.value:.2f} %**{error} prices the note at {solution.price:.2f} % "
        f"(target {target:.2f} %: issue price − margin)."
    )
    return (
        dbc.Alert(dcc.Markdown(text), color="success", className="py-2 mb-1"),
        {"key": key, "value": round(solution.value, 2)},
        ui.SHOWN,
    )


def download_sheet(
    n_clicks: int | None,
    result: dict[str, Any] | None,
    greeks: dict[str, Any] | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
) -> Any:
    """Product sheet of the priced note (with its Greeks once computed), as an HTML file."""
    current = fingerprint(store, rows, corr, data, paths)
    if not n_clicks or _fresh(result, current):
        return dash.no_update
    assert result is not None
    terms, basket = _setup(store, rows, corr, data)
    fresh_greeks = greeks if not _fresh(greeks, current) else None
    return sheet.download(coupon_sheet(terms, basket, result, fresh_greeks))


def apply_solution(
    n_clicks: int | None, solution: dict[str, Any] | None, ids: list[dict[str, str]]
) -> list[Any]:
    """Put the solved value in its term input."""
    if not n_clicks or not solution:
        return [dash.no_update] * len(ids)
    return [solution["value"] if i["key"] == solution["key"] else dash.no_update for i in ids]


# --- Callbacks ---------------------------------------------------------------------------------


def register(app: dash.Dash) -> None:
    """Register the callbacks of the coupon note pages."""
    store, data = State("product-store", "data"), State("market-data", "data")
    rows, corr = State("cn-underlyings", "rowData"), State("cn-correlation", "rowData")
    paths = State("cn-paths", "value")
    inputs = (
        Input("product-store", "data"),
        Input("cn-underlyings", "rowData"),
        Input("cn-correlation", "rowData"),
        Input("market-data", "data"),
    )
    tab = Input("product-tabs", "active_tab")
    app.callback(
        Output("cn-underlyings", "rowData"),
        Input("cn-add", "n_clicks"),
        Input("cn-remove", "n_clicks"),
        Input("cn-underlyings", "cellValueChanged"),
        rows,
        data,
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
        rows,
        corr,
    )(_correlation_view)
    app.callback(
        Output("cn-result", "data"),
        Output("cn-kpis", "children"),
        Input("cn-price", "n_clicks"),
        store,
        rows,
        corr,
        data,
        paths,
    )(price_view)
    app.callback(Output("cn-payoff", "children"), *inputs, tab)(payoff_view)
    app.callback(Output("cn-legs", "children"), Input("cn-result", "data"), *inputs, paths, tab)(
        legs_view
    )
    app.callback(Output("cn-life", "children"), Input("cn-result", "data"), *inputs, paths, tab)(
        life_view
    )
    app.callback(
        Output("cn-greeks", "data"),
        Output("cn-greeks-kpis", "children"),
        Input("cn-result", "data"),
        store,
        rows,
        corr,
        data,
        paths,
    )(greeks_compute)
    app.callback(
        Output("cn-greeks-result", "children"), Input("cn-greeks", "data"), *inputs, paths, tab
    )(greeks_view)
    app.callback(
        Output("cn-solve-result", "children"),
        Output("cn-solution", "data"),
        Output("cn-solve-apply", "style"),
        Input("cn-solve-run", "n_clicks"),
        store,
        rows,
        corr,
        data,
        paths,
        State("cn-solve-term", "value"),
        State("cn-solve-margin", "value"),
    )(solve_view)
    app.callback(
        Output(ui.term(ALL), "value", allow_duplicate=True),  # type: ignore[arg-type]
        Input("cn-solve-apply", "n_clicks"),
        State("cn-solution", "data"),
        State(ui.term(ALL), "id"),  # type: ignore[arg-type]
        prevent_initial_call=True,
    )(apply_solution)
    app.callback(
        Output("cn-sheet-download", "data"),
        Input("cn-sheet-button", "n_clicks"),
        State("cn-result", "data"),
        State("cn-greeks", "data"),
        store,
        rows,
        corr,
        data,
        paths,
        prevent_initial_call=True,
    )(download_sheet)


__all__ = ["coupon_page", "register"]
