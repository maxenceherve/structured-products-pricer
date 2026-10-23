"""Dash application: ``uv run python -m web.app``, or ``gunicorn web.app:server`` in production.

Every callback delegates to a pure function of this module, tested without a browser.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from web import layout, option_page, registry
from web.market import (
    MATURITY,
    MarketData,
    default_data,
    market_from_data,
)

TITLE = "Structured Products Pricer"


# --- Pure functions behind the callbacks -------------------------------------------------------


def route(pathname: str | None, query: str | None) -> Any:
    """Page for an address; a search shows the matching products wherever you are."""
    path = (pathname or "/").rstrip("/") or "/"
    if query and query.strip():
        return layout.gallery(query)
    if path == "/":
        return layout.gallery()
    if path == "/finder":
        return layout.finder_page()
    if path.startswith("/product/"):
        product = registry.find(path.removeprefix("/product/"))
        if product is not None:
            return layout.product_page(product)
    return layout.not_found(path)


def toggle(n_clicks: int | None, is_open: bool) -> bool:
    """Open or close the market panel."""
    return not is_open if n_clicks else is_open


def apply_edits(
    spot: float | None,
    rate_pct: float | None,
    dividend_pct: float | None,
    edits: list[dict[str, Any]] | None,
    data: MarketData,
) -> MarketData:
    """Session market data after the panel's inputs and grid edits (others unchanged)."""
    out = dict(data)
    out["rows"] = [dict(r) for r in data["rows"]]
    for key, value in (("spot", spot), ("rate_pct", rate_pct), ("dividend_pct", dividend_pct)):
        if value is not None:
            out[key] = value
    for edit in edits or []:
        index, column = edit.get("rowIndex"), edit.get("colId")
        if isinstance(index, int) and 0 <= index < len(out["rows"]) and column:
            out["rows"][index][column] = edit.get("value")
    return out


def market_view(data: MarketData) -> tuple[Any, go.Figure]:
    """Status message and smile of the session market."""
    try:
        market = market_from_data(data)
    except ValueError as exc:
        return layout.market_status(None, str(exc)), go.Figure()
    return layout.market_status(market, None), layout.smile_figure(market)


def panel_values(data: MarketData) -> tuple[float, float, float, list[dict[str, Any]]]:
    """Values shown by the panel for the session market."""
    return data["spot"], data["rate_pct"], data["dividend_pct"], data["rows"]


def sync_panel(_: bool, data: MarketData) -> tuple[float, float, float, list[dict[str, Any]]]:
    """Fill the panel from the session market when it opens."""
    return panel_values(data)


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
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
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

    app.callback(Output("page", "children"), Input("url", "pathname"), Input("search", "value"))(
        route
    )
    app.callback(
        Output("market-panel", "is_open"),
        Input("market-button", "n_clicks"),
        State("market-panel", "is_open"),
        prevent_initial_call=True,
    )(toggle)
    app.callback(
        Output("market-data", "data"),
        Input("market-spot", "value"),
        Input("market-rate", "value"),
        Input("market-dividend", "value"),
        Input("vol-grid", "cellValueChanged"),
        State("market-data", "data"),
        prevent_initial_call=True,
    )(apply_edits)
    app.callback(
        Output("market-data", "data", allow_duplicate=True),
        Output("market-spot", "value"),
        Output("market-rate", "value"),
        Output("market-dividend", "value"),
        Output("vol-grid", "rowData"),
        Input("market-reset", "n_clicks"),
        prevent_initial_call=True,
    )(reset)
    app.callback(
        Output("market-spot", "value", allow_duplicate=True),
        Output("market-rate", "value", allow_duplicate=True),
        Output("market-dividend", "value", allow_duplicate=True),
        Output("vol-grid", "rowData", allow_duplicate=True),
        Input("market-panel", "is_open"),
        State("market-data", "data"),
        prevent_initial_call=True,
    )(sync_panel)
    app.callback(
        Output("market-status", "children"),
        Output("smile-graph", "figure"),
        Input("market-data", "data"),
    )(market_view)
    option_page.register(app)
    return app


app = create_app()
server = app.server

__all__ = ["MATURITY", "app", "create_app", "server"]

if __name__ == "__main__":  # pragma: no cover - development server
    # Import lazily loaded modules (plotly validators, scipy) before serving: the development
    # reloader would otherwise take their first import for a code change and drop the request.
    market_view(default_data())
    route("/", None)
    app.run(debug=True, port=8050)
