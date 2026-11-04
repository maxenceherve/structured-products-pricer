"""Dash application: ``uv run python -m web.app``, or ``gunicorn web.app:server`` in production.

Every callback delegates to a pure function of this module, tested without a browser.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html

from web import (
    coupon_page,
    finder_page,
    layout,
    note_page,
    option_page,
    registry,
    risk_panel,
    sheet_route,
    solver,
    ui,
)
from web.coupons import coupon_spec
from web.market import (
    MATURITY,
    MarketData,
    default_data,
    funding_bp,
    market_from_data,
)
from web.notes import NotePosition

TITLE = "Structured Products Pricer"
FONTS = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600"
    "&family=IBM+Plex+Serif:ital,wght@0,500;0,600;1,400&display=swap"
)


# --- Pure functions behind the callbacks -------------------------------------------------------


def route(pathname: str | None, query: str | None, search: str | None = None) -> Any:
    """Page for an address; a search shows the matching products wherever you are.

    ``search`` is the address's query string: terms set by the finder for a product page.
    """
    path = (pathname or "/").rstrip("/") or "/"
    if query and query.strip():
        return layout.gallery(query)
    if path == "/":
        return layout.gallery()
    if path == "/finder":
        return finder_page.finder_page()
    if path.startswith("/product/"):
        product = registry.find(path.removeprefix("/product/"))
        if product is not None:
            return layout.product_page(product, ui.prefill(search))
    return layout.not_found(path)


def toggle(clicked: list[dict[str, Any]], is_open: bool) -> bool:
    """Open or close the market panel when the gear or a link to it was just clicked.

    ``clicked`` is what triggered the callback: only a click (``n_clicks`` above zero on the
    element that fired) counts. A button kept from an earlier click, or a page that adds a link
    to the panel, must not open it.
    """
    fired = any(t.get("prop_id", "").endswith(".n_clicks") and t.get("value") for t in clicked)
    return not is_open if fired else is_open


def _toggle(*args: Any) -> bool:  # pragma: no cover - reads the Dash context
    return toggle(list(ctx.triggered), args[-1])


def clear_search(_: str | None) -> str:
    """Leave the search when the user opens a page: the page, not the results, is shown."""
    return ""


def apply_edits(
    spot: float | None,
    rate_pct: float | None,
    dividend_pct: float | None,
    funding: float | None,
    edits: list[dict[str, Any]] | None,
    data: MarketData,
) -> MarketData:
    """Session market data after the panel's inputs and grid edits (others unchanged)."""
    out = dict(data)
    out["rows"] = [dict(r) for r in data["rows"]]
    values = (
        ("spot", spot),
        ("rate_pct", rate_pct),
        ("dividend_pct", dividend_pct),
        ("funding_bp", funding),
    )
    for key, value in values:
        if value is not None:
            out[key] = value
    for edit in edits or []:
        index, column = edit.get("rowIndex"), edit.get("colId")
        if isinstance(index, int) and 0 <= index < len(out["rows"]) and column:
            out["rows"][index][column] = edit.get("value")
    return out


def market_view(data: MarketData) -> tuple[Any, go.Figure, go.Figure]:
    """Status message, skew and volatility surface of the session market."""
    try:
        market = market_from_data(data)
    except ValueError as exc:
        return layout.market_status(None, str(exc)), go.Figure(), go.Figure()
    return (
        layout.market_status(market, None),
        layout.smile_figure(market),
        layout.surface_figure(market),
    )


PanelValues = tuple[float, float, float, float, list[dict[str, Any]]]


def panel_values(data: MarketData) -> PanelValues:
    """Values shown by the panel for the session market."""
    return data["spot"], data["rate_pct"], data["dividend_pct"], funding_bp(data), data["rows"]


def sync_panel(_: bool, data: MarketData) -> PanelValues:
    """Fill the panel from the session market when it opens."""
    return panel_values(data)


def market_summary(data: MarketData) -> Any:
    """The session market, shown on every product page: each figure with its name."""
    try:
        market = market_from_data(data)
    except ValueError as exc:
        return html.Span(f"The market data are invalid: {exc}", className="neg")
    atm = 100.0 * float(market.vol.vol(market.spot, 1.0))
    figures = (
        ("Spot", f"{market.spot:,.2f}"),
        ("Rate", f"{100 * market.rate:.2f} %"),
        ("Dividend", f"{100 * market.dividend_yield:.2f} %"),
        ("Funding spread", f"{funding_bp(data):g} bp"),
        ("ATM volatility, 1 year", f"{atm:.2f} %"),
    )
    return [
        html.Span(
            [
                html.Span(label, className="market-label"),
                html.Span(value, className="market-value"),
            ],
            className="market-item",
        )
        for label, value in figures
    ]


def kpis(store: dict[str, Any] | None, data: MarketData) -> Any:
    """Headline numbers of the product page shown (option or note), once it has been priced.

    A coupon note writes its own, from its Monte Carlo price.
    """
    if not store:
        return dash.no_update
    product = registry.find(store.get("slug", ""))
    if product is not None and coupon_spec(product) is not None:
        return dash.no_update
    try:
        item = risk_panel.resolve(store, data)
    except ValueError as exc:
        return ui.error(str(exc))
    page = note_page if isinstance(item, NotePosition) else option_page
    return page.kpis(store, data)


def reset(_: int) -> tuple[Any, ...]:
    """The default market, and the panel's values for it."""
    fresh = default_data()
    return (fresh, *panel_values(fresh))


# --- Application -------------------------------------------------------------------------------


def create_app() -> dash.Dash:
    """The Dash application with its layout and callbacks."""
    app = dash.Dash(
        __name__,
        title=TITLE,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP, FONTS],
        suppress_callback_exceptions=True,
        update_title=None,  # type: ignore[arg-type]  # None keeps the tab title still
    )
    data = default_data()
    app.layout = html.Div(
        [
            dcc.Location(id="url"),
            dcc.Store(id="market-data", storage_type="session", data=data),
            layout.top_bar(),
            layout.market_panel(data),
            dbc.Container(html.Div(id="page"), fluid="xl", className="page-container"),
        ]
    )

    app.callback(
        Output("page", "children"),
        Input("url", "pathname"),
        Input("search", "value"),
        Input("url", "search"),
    )(route)
    app.callback(
        Output("market-panel", "is_open"),
        Input("market-button", "n_clicks"),
        Input({"type": "open-market", "where": ALL}, "n_clicks"),
        State("market-panel", "is_open"),
        prevent_initial_call=True,
    )(_toggle)
    app.callback(Output("search", "value"), Input("url", "pathname"), prevent_initial_call=True)(
        clear_search
    )
    app.callback(Output("market-summary", "children"), Input("market-data", "data"))(market_summary)
    app.callback(
        Output("market-data", "data"),
        Input("market-spot", "value"),
        Input("market-rate", "value"),
        Input("market-dividend", "value"),
        Input("market-funding", "value"),
        Input("vol-grid", "cellValueChanged"),
        State("market-data", "data"),
        prevent_initial_call=True,
    )(apply_edits)
    app.callback(
        Output("market-data", "data", allow_duplicate=True),
        Output("market-spot", "value"),
        Output("market-rate", "value"),
        Output("market-dividend", "value"),
        Output("market-funding", "value"),
        Output("vol-grid", "rowData"),
        Input("market-reset", "n_clicks"),
        prevent_initial_call=True,
    )(reset)
    app.callback(
        Output("market-spot", "value", allow_duplicate=True),
        Output("market-rate", "value", allow_duplicate=True),
        Output("market-dividend", "value", allow_duplicate=True),
        Output("market-funding", "value", allow_duplicate=True),
        Output("vol-grid", "rowData", allow_duplicate=True),
        Input("market-panel", "is_open"),
        State("market-data", "data"),
        prevent_initial_call=True,
    )(sync_panel)
    app.callback(
        Output("market-status", "children"),
        Output("smile-graph", "figure"),
        Output("surface-graph", "figure"),
        Input("market-data", "data"),
    )(market_view)
    # every product page: its terms, and one tab shown at a time
    app.callback(
        Output("product-store", "data"),
        Output("priced-market", "data"),
        Input("price-run", "n_clicks"),
        State(ui.term(ALL), "value"),  # type: ignore[arg-type]
        State(ui.term(ALL), "id"),  # type: ignore[arg-type]
        State("url", "pathname"),
        State("market-data", "data"),
        prevent_initial_call=True,
    )(ui.price_click)
    app.callback(Output("product-results", "style"), Input("product-store", "data"))(
        ui.show_results
    )
    app.callback(
        Output("price-status", "children"),
        Input(ui.term(ALL), "value"),  # type: ignore[arg-type]
        Input("market-data", "data"),
        Input("product-store", "data"),
        State(ui.term(ALL), "id"),  # type: ignore[arg-type]
        State("priced-market", "data"),
    )(ui.price_status)
    app.callback(
        Output({"type": "tab-block", "tab": ALL}, "style"),
        Input("product-tabs", "active_tab"),
        State({"type": "tab-block", "tab": ALL}, "id"),
    )(ui.show_tab)
    app.callback(
        Output("product-kpis", "children"),
        Input("product-store", "data"),
        State("priced-market", "data"),
    )(kpis)
    risk_panel.register(app)
    solver.register(app)
    option_page.register(app)
    note_page.register(app)
    coupon_page.register(app)
    finder_page.register(app)
    sheet_route.register(app.server)
    return app


app = create_app()
server = app.server

__all__ = ["MATURITY", "app", "create_app", "server"]

if __name__ == "__main__":  # pragma: no cover - development server
    # Import lazily loaded modules (plotly validators, scipy) before serving: the development
    # reloader would otherwise take their first import for a code change and drop the request.
    market_view(default_data())
    route("/", None)
    # hot reload while developing, without Dash's debug menu on the page
    app.run(debug=True, dev_tools_ui=False, port=8050)
