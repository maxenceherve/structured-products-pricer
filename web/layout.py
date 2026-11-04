"""Layout pieces: top bar, market data panel, gallery, product page skeleton."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from spricer.market import InterpolatedVolSurface, Market
from web import registry
from web.coupon_page import coupon_page
from web.market import MATURITY, MarketData, strike_field
from web.note_page import note_page
from web.notes import note_spec
from web.option_page import option_page
from web.options import setup_of
from web.registry import Product
from web.thumbnails import payoff_data_uri

# --- Top bar and market panel ------------------------------------------------------------------


def top_bar() -> dbc.Navbar:
    """Logo, product search and the market data button."""
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    html.Img(src="/assets/logo.svg", height="40px", alt="Structured Products"),
                    href="/",
                    className="navbar-brand",
                ),
                dbc.Input(
                    id="search",
                    type="search",
                    placeholder="Search a product: call spread, autocall, shark...",
                    debounce=False,
                    className="search-box",
                ),
                dbc.Button(
                    html.I(className="bi bi-gear-fill"),
                    id="market-button",
                    color="light",
                    className="gear-button",
                    title="Market data and volatility surface",
                    n_clicks=0,
                ),
            ],
            fluid=True,
            className="top-bar-inner",
        ),
        className="top-bar",
        color="white",
        sticky="top",
    )


def _grid_columns(strikes_pct: Sequence[float]) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = [
        {"field": MATURITY, "editable": True, "pinned": "left", "width": 110}
    ]
    columns += [
        {"field": strike_field(k), "editable": True, "width": 78, "type": "numericColumn"}
        for k in strikes_pct
    ]
    return columns


def market_panel(data: MarketData) -> dbc.Offcanvas:
    """Side panel to edit the market of the whole site."""
    return dbc.Offcanvas(
        [
            html.P(
                "One market for the whole site: today's spot, rates, the issuer's funding "
                "spread over the risk-free rate (it discounts the notes' cash flows) and the "
                "implied volatility grid (strikes in % of spot, volatilities in points).",
                className="text-muted small",
            ),
            dbc.Row(
                [
                    dbc.Col(_number_input("Spot", "market-spot", data["spot"], 1.0)),
                    dbc.Col(_number_input("Rate (%)", "market-rate", data["rate_pct"], 0.25)),
                    dbc.Col(
                        _number_input("Dividend (%)", "market-dividend", data["dividend_pct"], 0.25)
                    ),
                    dbc.Col(
                        _number_input(
                            "Funding (bp)", "market-funding", data.get("funding_bp", 0.0), 5.0
                        )
                    ),
                ],
                className="g-2 mb-3",
            ),
            html.H6("Implied volatility (%)"),
            dag.AgGrid(
                id="vol-grid",
                rowData=data["rows"],
                columnDefs=_grid_columns(data["strikes_pct"]),
                defaultColDef={"resizable": True, "sortable": False},
                dashGridOptions={"singleClickEdit": True, "stopEditingWhenCellsLoseFocus": True},
                style={"height": "250px"},
                className="ag-theme-alpine",
            ),
            html.Div(id="market-status", className="mt-2"),
            dcc.Graph(id="smile-graph", config={"displayModeBar": False}),
            dcc.Graph(id="surface-graph", config={"displaylogo": False}),
            dbc.Button("Reset to the default market", id="market-reset", color="link", n_clicks=0),
        ],
        id="market-panel",
        title="Market data",
        placement="end",
        is_open=False,
        style={"width": "min(720px, 100vw)"},
    )


def _number_input(label: str, id_: str, value: float, step: float) -> html.Div:
    return html.Div(
        [
            dbc.Label(label, html_for=id_, className="small mb-1"),
            dbc.Input(id=id_, type="number", value=value, step=step, debounce=True),
        ]
    )


def _vols(surface: InterpolatedVolSurface, strikes: np.ndarray, t: float) -> np.ndarray:
    # a grid with arbitrage can interpolate a negative total variance: leave a gap there
    with np.errstate(invalid="ignore"):
        return np.asarray(surface.vol(strikes, t))


def smile_figure(market: Market) -> go.Figure:
    """Smile of the panel at a few maturities."""
    fig = go.Figure()
    surface = market.vol
    strikes = np.linspace(0.5 * market.spot, 1.6 * market.spot, 111)
    if isinstance(surface, InterpolatedVolSurface):
        for t, color in zip((0.25, 1.0, 3.0), ("#b3261e", "#1b2a41", "#1e6b57"), strict=True):
            fig.add_trace(
                go.Scatter(
                    x=100.0 * strikes / market.spot,
                    y=100.0 * _vols(surface, strikes, t),
                    name=f"{t:g} y",
                    line={"color": color, "width": 2},
                )
            )
    fig.update_layout(
        height=260,
        margin={"t": 30, "l": 40, "r": 10, "b": 40},
        title={"text": "Skew: implied volatility by strike", "font": {"size": 14}},
        xaxis_title="Strike (% of spot)",
        yaxis_title="Vol (%)",
        legend={"orientation": "h", "y": -0.3},
        template="plotly_white",
        font={"family": "IBM Plex Sans, Segoe UI, sans-serif", "size": 12, "color": "#1b2a41"},
        paper_bgcolor="#ffffff",
    )
    fig.update_xaxes(gridcolor="#e6efe2")
    fig.update_yaxes(gridcolor="#e6efe2")
    return fig


def surface_figure(market: Market) -> go.Figure:
    """Implied volatility against moneyness and maturity, in three dimensions."""
    fig = go.Figure()
    surface = market.vol
    if isinstance(surface, InterpolatedVolSurface):
        moneyness = np.linspace(50.0, 160.0, 45)
        maturities = np.geomspace(0.1, max(surface.maturities), 30)
        strikes = moneyness / 100.0 * market.spot
        vols = np.array([100.0 * _vols(surface, strikes, float(t)) for t in maturities])
        fig.add_trace(
            go.Surface(
                x=moneyness,
                y=maturities,
                z=vols,
                colorscale=[[0.0, "#1e6b57"], [0.5, "#edf3ea"], [1.0, "#b3261e"]],
                colorbar={"title": "Vol (%)", "thickness": 12, "len": 0.7},
                hovertemplate=(
                    "moneyness %{x:.0f} %<br>maturity %{y:.2f} y<br>vol %{z:.2f} %<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=460,
        margin={"t": 30, "l": 0, "r": 0, "b": 0},
        title={"text": "Volatility surface", "font": {"size": 14}},
        scene={
            "xaxis_title": "Moneyness (strike, % of spot)",
            "yaxis_title": "Maturity (years)",
            "zaxis_title": "Vol (%)",
            "camera": {"eye": {"x": -1.6, "y": -1.6, "z": 0.9}},
        },
        font={"family": "IBM Plex Sans, Segoe UI, sans-serif", "size": 11, "color": "#1b2a41"},
        paper_bgcolor="#ffffff",
    )
    return fig


def market_status(market: Market | None, error: str | None) -> dbc.Alert:
    """Arbitrage status of the market, or the input error."""
    if error is not None or market is None:
        return dbc.Alert(error or "Invalid market", color="danger", className="py-2 small")
    surface = market.vol
    if (
        isinstance(surface, InterpolatedVolSurface)
        and not surface.check_arbitrage().is_arbitrage_free
    ):
        return dbc.Alert(
            "The surface has static arbitrage (negative density or decreasing total "
            "variance): prices still compute, but check the grid.",
            color="warning",
            className="py-2 small",
        )
    return dbc.Alert("No static arbitrage.", color="success", className="py-2 small")


# --- Gallery -----------------------------------------------------------------------------------


def card(product: Product) -> dbc.Col:
    """A gallery card: the shape of the payoff and the product's name."""
    return dbc.Col(
        dcc.Link(
            dbc.Card(
                [
                    html.Img(src=payoff_data_uri(product), className="card-payoff", alt=""),
                    dbc.CardBody(html.Div(product.name, className="card-name")),
                ],
                className="product-card",
            ),
            href=f"/product/{product.slug}",
            className="card-link",
        ),
        xs=6,
        md=3,
        className="mb-4",
    )


def _family_shape(family: str) -> html.A:
    """A family drawn by the payoff of its first product: a link to its section."""
    first = next(p for p in registry.PRODUCTS if p.family == family)
    return html.A(
        [html.Img(src=payoff_data_uri(first), alt=""), family],
        href=f"#{registry.slugify(family)}",
        className="hero-shape",
    )


def gallery(query: str | None = None) -> html.Div:
    """Every product matching ``query``, by family and group, four per row."""
    products = registry.matching(query)
    sections: list[Any] = []
    if not query:
        sections.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Structured Products Pricer", className="hero-title"),
                            html.P(
                                "Price options and structured products on your own volatility "
                                "smile: the payoff, the bricks, the Greeks and what they mean, the "
                                "hedge and a product sheet for each of them.",
                                className="hero-text",
                            ),
                        ]
                    ),
                    html.Nav(
                        [_family_shape(f) for f in registry.FAMILIES],
                        className="hero-shapes",
                    ),
                ],
                className="hero",
            )
        )
        sections.append(
            html.Div(
                [
                    "Not sure which product fits your view? ",
                    dcc.Link("Answer five questions in the product finder", href="/finder"),
                    ".",
                ],
                className="finder-banner mb-2",
            )
        )
    if not products:
        sections.append(html.P("No product matches this search.", className="text-muted"))
    for family in registry.FAMILIES:
        members = [p for p in products if p.family == family]
        if not members:
            continue
        section: list[Any] = [
            html.H2(family, className="family-title"),
            html.P(registry.FAMILY_BLURBS[family], className="family-blurb"),
        ]
        groups = list(dict.fromkeys(p.group for p in members))
        for group in groups:
            if group:
                section.append(html.H5(group, className="group-title"))
            section.append(dbc.Row([card(p) for p in members if p.group == group]))
        sections.append(
            html.Section(section, className="family-section", id=registry.slugify(family))
        )
    return html.Div(sections, className="gallery")


# --- Product page ------------------------------------------------------------------------------


def product_page(product: Product, prefilled: dict[str, float] | None = None) -> html.Div:
    """Product page: terms on the left, results in tabs on the right.

    ``prefilled`` holds terms set by the finder (maturity, protection...).
    """
    if setup_of(product) is not None:
        return option_page(product, prefilled)
    if note_spec(product) is not None:
        return note_page(product, prefilled)
    return coupon_page(product, prefilled)


def not_found(path: str) -> html.Div:
    """Unknown address."""
    return html.Div(
        [
            html.H1("Page not found", className="product-title"),
            html.P(f"Nothing lives at {path}."),
            dcc.Link("Back to the products", href="/"),
        ]
    )
