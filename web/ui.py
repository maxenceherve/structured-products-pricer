"""Small layout pieces shared by the product pages."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any
from urllib.parse import parse_qs

import dash_bootstrap_components as dbc
from dash import dcc, html

from web import registry
from web.explain import Card
from web.options import MONITORING_HELP, MONITORING_LABELS
from web.registry import Product

HIDDEN, SHOWN = {"display": "none"}, {"display": "block"}


def field(label: str, control: Any) -> html.Div:
    """A labelled input."""
    return html.Div([dbc.Label(label, className="field-label"), control], className="mb-3")


def number(id_: Any, value: Any, step: float, **kwargs: Any) -> dbc.Input:
    """A number input that reports its value when the user leaves it or presses Enter."""
    return dbc.Input(id=id_, type="number", value=value, step=step, debounce=True, **kwargs)


def segmented(id_: Any, options: Any, value: str) -> dbc.RadioItems:
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


def select(id_: Any, options: Any, value: Any) -> dbc.Select:
    """A drop-down list; ``options`` maps values to labels, or lists values shown as such."""
    pairs = options.items() if isinstance(options, dict) else ((o, o) for o in options)
    return dbc.Select(id=id_, options=[{"label": lab, "value": v} for v, lab in pairs], value=value)


def term(key: str) -> dict[str, str]:
    """Pattern-matching id of a term input."""
    return {"type": "term", "key": key}


def fmt(value: float, digits: int = 2, signed: bool = True) -> html.Span:
    """A number, green when positive and red when negative (zero within rounding is plain)."""
    if abs(value) < 0.5 * 10.0**-digits:
        value = 0.0
    text = f"{value:+.{digits}f}" if signed and value else f"{value:.{digits}f}"
    tone = "pos" if value > 5e-5 else "neg" if value < -5e-5 else ""
    return html.Span(text, className=tone if signed else "")


def bullets(lines: list[str]) -> html.Div:
    """Sentences in Markdown, as a list."""
    return html.Div([dcc.Markdown(f"- {line}") for line in lines], className="reading")


def error(message: str) -> dbc.Alert:
    """A warning box."""
    return dbc.Alert(message, color="warning", className="py-2")


def note(text: str) -> html.P:
    """A small explanatory paragraph."""
    return html.P(text, className="field-help")


def table(header: list[str], rows: list[list[Any]], total: bool = False) -> dbc.Table:
    """A compact results table; ``total`` highlights the last row."""
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
        className="result-table",
    )


def kpi(label: str, value: str, sub: str = "") -> dbc.Col:
    """A headline number with its label and a short explanation."""
    return dbc.Col(
        html.Div(
            [
                html.Div(label, className="kpi-label"),
                html.Div(value, className="kpi-value"),
                html.Div(sub, className="kpi-sub"),
            ],
            className="kpi",
        ),
        xs=6,
        md=True,
    )


def side_class(name: str) -> str:
    """Green for a long leg, red for a short one, bold for a total."""
    if name.startswith(("Long", "Buy")):
        return "pos fw-semibold"
    if name.startswith(("Short", "Sell")):
        return "neg fw-semibold"
    return "fw-bold"


def breadcrumb(product: Product) -> dbc.Breadcrumb:
    """Home / family / product."""
    return dbc.Breadcrumb(
        items=[
            {"label": "Home", "href": "/"},
            {"label": product.family, "href": f"/#{registry.slugify(product.family)}"},
            {"label": product.name, "active": True},
        ]
    )


def tab_bar(tabs: tuple[str, ...], first: str) -> dbc.Tabs:
    """Tabs used as a navigation bar: their contents are separate blocks shown one at a time."""
    return dbc.Tabs(
        [dbc.Tab(label=t, tab_id=registry.slugify(t)) for t in tabs],
        id="product-tabs",
        active_tab=first,
        className="mb-3",
    )


def tab_block(tab: str, content: Any, shown: bool) -> html.Div:
    """Content of one tab; :func:`show_tab` shows the active one."""
    return html.Div(content, id={"type": "tab-block", "tab": tab}, style=SHOWN if shown else HIDDEN)


def show_tab(active: str | None, ids: list[dict[str, str]]) -> list[dict[str, str]]:
    """Show the active tab's content, hide the others."""
    return [SHOWN if i["tab"] == active else HIDDEN for i in ids]


def product_shell(
    product: Product,
    description: str,
    terms_card: Any,
    tabs: dict[str, Any],
    kpis: Any = None,
    prefilled: dict[str, float] | None = None,
) -> html.Div:
    """A product page: terms on the left; headline numbers and tabs on the right.

    ``tabs`` maps tab titles to their contents; the first tab is shown. ``kpis`` is the
    initial content of the headline numbers; ``prefilled`` the terms taken from the finder.
    """
    ids = [registry.slugify(t) for t in tabs]
    return html.Div(
        [
            dcc.Store(id="product-store"),
            breadcrumb(product),
            html.H1(product.name, className="product-title mb-1"),
            html.P(description, className="product-description"),
            html.Div(
                [
                    html.I(className="bi bi-graph-up me-2"),
                    html.Span(id="market-summary"),
                    dbc.Button(
                        [html.I(className="bi bi-gear-fill me-1"), "Edit the market"],
                        id={"type": "open-market", "where": "product"},
                        color="link",
                        size="sm",
                        n_clicks=0,
                        className="ms-2 py-0",
                    ),
                ],
                className="market-strip",
            ),
            prefill_notice(prefilled or {}),
            dbc.Row(
                [
                    dbc.Col(terms_card, lg=4, className="mb-4"),
                    dbc.Col(
                        [
                            html.Div(kpis, id="product-kpis", className="mb-3"),
                            tab_bar(tuple(tabs), ids[0]),
                            *[
                                tab_block(i, content, i == ids[0])
                                for i, content in zip(ids, tabs.values(), strict=True)
                            ],
                        ],
                        lg=8,
                    ),
                ]
            ),
        ],
        className="product-page",
    )


def collect(
    values: list[Any], ids: list[dict[str, str]], vols: list[Any], pathname: str | None
) -> dict[str, Any]:
    """A product page's terms as stored JSON: product slug, terms, volatility overrides."""
    slug = (pathname or "").rstrip("/").removeprefix("/product/")
    terms = {i["key"]: v for i, v in zip(ids, values, strict=True)}
    return {"slug": slug, "terms": terms, "vols": list(vols)}


def greek_cards(cards: Sequence[Card], notes: Sequence[str] = ()) -> html.Div:
    """Each Greek in its own card: value, what it means, and remarks; then general notes."""
    cols = [
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(c.title, className="greek-title"),
                            html.Span(c.subtitle, className="greek-subtitle"),
                        ]
                    ),
                    html.Div(c.value, className="greek-value"),
                    dcc.Markdown(c.headline, className="greek-headline"),
                    *[dcc.Markdown(d, className="greek-detail") for d in c.details],
                ],
                className="greek-card",
            ),
            md=6,
            className="mb-3",
        )
        for c in cards
    ]
    alerts = [
        dbc.Alert(dcc.Markdown(n), color="secondary", className="py-2 greek-note") for n in notes
    ]
    return html.Div([dbc.Row(cols, className="g-3"), *alerts])


def monitoring_field(label: str, value: str) -> html.Div:
    """Choice of the barrier observation, with what each convention means."""
    return html.Div(
        [
            dbc.Label(label, className="field-label"),
            select(term("monitoring"), MONITORING_LABELS, value),
            html.Div(MONITORING_HELP, className="field-help mt-1"),
        ],
        className="mb-3",
    )


def scenario_table(
    level_label: str,
    levels: Sequence[float],
    rows: Sequence[tuple[str, Sequence[float]]],
    breakevens: Sequence[float] = (),
) -> html.Div:
    """What is paid at a few final levels: one line per level, one column per amount.

    The breakevens are given above the table; the line of today's level (100 %) is marked.
    """
    head = html.Tr([html.Th(level_label), *[html.Th(name) for name, _ in rows]])
    body = []
    for i, level in enumerate(levels):
        cells = [html.Td(f"{level:g} %", className="scenario-level")]
        for name, values in rows:
            v = values[i]
            if v != v:  # nan: this case cannot happen at this level
                cells.append(html.Td("n/a", className="text-muted"))
            elif name.startswith(("P&L", "Gain")):
                cells.append(html.Td(fmt(v, 2)))
            else:
                cells.append(html.Td(f"{v:.2f}"))
        body.append(html.Tr(cells, className="scenario-today" if level == 100.0 else ""))
    badges: list[Any] = []
    if breakevens:
        badges = [
            html.Span("Breakeven", className="scenario-badge-label"),
            *[dbc.Badge(f"{b:.1f} %", color="primary", className="ms-2") for b in breakevens],
        ]
    return html.Div(
        [
            html.Div(badges, className="mb-2") if badges else None,
            html.Div(
                dbc.Table(
                    [html.Thead(head), html.Tbody(body)],
                    bordered=False,
                    hover=True,
                    size="sm",
                    className="result-table scenario-table",
                ),
                className="scenario-wrap",
            ),
        ]
    )


def apply_button(id_: str) -> dbc.Button:
    """Button that writes a solved value into the terms, hidden until there is one."""
    return dbc.Button(
        [html.I(className="bi bi-box-arrow-in-left me-2"), "Use this value in the terms"],
        id=id_,
        color="success",
        outline=True,
        size="sm",
        n_clicks=0,
        className="mt-2",
        style=HIDDEN,
    )


def sheet_button(id_: str) -> dbc.Button:
    """Button that downloads the product sheet."""
    return dbc.Button(
        [html.I(className="bi bi-file-earmark-richtext me-2"), "Product sheet"],
        id=id_,
        color="primary",
        outline=True,
        size="sm",
        n_clicks=0,
        className="mt-2",
    )


PREFILL_LABELS = {
    "maturity": ("maturity", "{:g} years"),
    "protection": ("capital protection", "{:g} %"),
    "buffer": ("buffer level", "{:g} %"),
}


def prefill(search: str | None) -> dict[str, float]:
    """Terms passed in the address by the finder (``?maturity=2&protection=80``)."""
    out: dict[str, float] = {}
    for key, values in parse_qs((search or "").lstrip("?")).items():
        try:
            value = float(values[0])
        except ValueError:
            continue
        if key in PREFILL_LABELS and math.isfinite(value) and value >= 0.0:
            out[key] = value
    return out


def prefill_notice(values: dict[str, float]) -> Any:
    """What the finder has set on this page, if anything."""
    if not values:
        return None
    parts = [f"{PREFILL_LABELS[k][0]} {PREFILL_LABELS[k][1].format(v)}" for k, v in values.items()]
    return dbc.Alert(
        [
            html.I(className="bi bi-magic me-2"),
            "Set from your answers in the finder: " + ", ".join(parts) + ". Change them freely.",
        ],
        color="info",
        className="py-2",
    )


def apply_prefill(terms: dict[str, Any], values: dict[str, float]) -> dict[str, float]:
    """Put the prefilled values in the terms the product has; return those applied."""
    applied = {k: v for k, v in values.items() if k in terms}
    terms.update(applied)
    return applied
