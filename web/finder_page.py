"""Product finder: say what you expect and what you accept, get the products that fit.

Every product of the site is scored with transparent rules (:mod:`spricer.finder`), each point
explained; equivalent ways to build the same payoff (put-call parity) show what changes when
one buys options or sells others with cash aside. An educational tool, not investment advice.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from spricer.finder import Objective, Profile, Protection, Recommendation, View, VolView, recommend
from spricer.packaging import Equivalence, equivalences
from web import figures, registry, ui
from web.market import MarketData, market_from_data
from web.options import display_greeks
from web.thumbnails import payoff_data_uri

TOP = 3
MAX_SCORE = 7.5  # 4 (view) + 1.5 (objective) + 1 (volatility) + 0.5 (horizon) + headroom


def _choice(id_: str, options: Any, value: str) -> dbc.RadioItems:
    """Answers as pills that wrap on small screens."""
    return dbc.RadioItems(
        id=id_,
        options=[{"label": o.value, "value": o.value} for o in options],
        value=value,
        className="pill-choice",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary btn-sm",
        labelCheckedClassName="active",
    )


def finder_page() -> html.Div:
    """The questions on the left, the products that fit on the right."""
    questions = dbc.Card(
        dbc.CardBody(
            [
                html.H5("What do you expect?"),
                ui.field("Market view", _choice("finder-view", View, View.MILD_BULL.value)),
                ui.field("Volatility", _choice("finder-vol", VolView, VolView.NEUTRAL.value)),
                ui.field(
                    "Horizon (years)",
                    dcc.Slider(
                        id="finder-horizon",
                        min=0.25,
                        max=10.0,
                        step=0.25,
                        value=3.0,
                        marks={t: f"{t:g}" for t in (0.25, 1, 2, 3, 5, 7, 10)},
                        tooltip={"placement": "bottom"},
                    ),
                ),
                html.Hr(),
                html.H5("What can you accept?"),
                ui.field(
                    "Capital protection",
                    _choice("finder-protection", Protection, Protection.FULL.value),
                ),
                ui.field(
                    "You want", _choice("finder-objective", Objective, Objective.GROWTH.value)
                ),
            ]
        ),
        className="terms-card",
    )
    return html.Div(
        [
            dbc.Breadcrumb(
                items=[{"label": "Home", "href": "/"}, {"label": "Find a product", "active": True}]
            ),
            html.H1("Find a product", className="product-title mb-1"),
            html.P(
                "Say what you expect and what you can accept: every product of the site is "
                "ranked against your answers, each point explained, with equivalent ways to "
                "build the payoff. An educational tool, not investment advice.",
                className="product-description",
            ),
            dbc.Row(
                [
                    dbc.Col(questions, lg=4, className="mb-4"),
                    dbc.Col(dcc.Loading(html.Div(id="finder-results")), lg=8),
                ]
            ),
            html.Div(id="finder-packaging", className="mt-2"),
        ],
        className="product-page",
    )


# --- Pure functions behind the callback ----------------------------------------------------------


def _recommendation(recommendation: Recommendation, rank: int) -> dbc.Card:
    card = recommendation.card
    product = registry.find(registry.slugify(card.name))
    fit = max(0.0, min(1.0, recommendation.score / MAX_SCORE))
    image = (
        html.Img(src=payoff_data_uri(product), className="finder-payoff", alt="")
        if product is not None
        else None
    )
    lines = [
        html.Div(["✓ ", dcc.Markdown(r, className="d-inline")], className="pos")
        for r in recommendation.reasons
    ]
    lines += [
        html.Div(["⚠ ", dcc.Markdown(w, className="d-inline")], className="finder-warning")
        for w in recommendation.warnings
    ]
    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(image, md=3, className="mb-2"),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Span(f"{rank}. {card.name}", className="finder-name"),
                                    dbc.Badge(
                                        card.page,
                                        color="light",
                                        text_color="primary",
                                        className="ms-2",
                                    ),
                                ]
                            ),
                            dbc.Progress(
                                value=round(100 * fit),
                                label=f"Fit {100 * fit:.0f} %",
                                className="my-2",
                                style={"height": "18px"},
                            ),
                            html.Div(lines, className="finder-lines"),
                            dcc.Link(
                                dbc.Button(
                                    [
                                        f"Price the {card.name.lower()}",
                                        html.I(className="bi bi-arrow-right ms-2"),
                                    ],
                                    color="primary",
                                    size="sm",
                                    className="mt-2",
                                ),
                                href=f"/product/{registry.slugify(card.name)}",
                            ),
                        ],
                        md=9,
                    ),
                ]
            )
        ),
        className="finder-card mb-3",
    )


def _packaging(equivalence: Equivalence, data: MarketData, maturity: float) -> dbc.Card:
    market = market_from_data(data)
    rows = []
    for construction in (equivalence.first, equivalence.second):
        g = display_greeks(construction.strategy, market)
        rows.append(
            [
                construction.name,
                ui.fmt(construction.cash_today(market), 2),
                ui.fmt(construction.option_premium(market), 2),
                f"{construction.bond() / market.spot * 100.0:.2f}",
                f"{construction.collateral(market):.2f}",
                ui.fmt(g["Delta (%)"], 2),
                ui.fmt(g["Gamma (%)"], 2),
                ui.fmt(g["Vega (%)"], 3),
            ]
        )
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(equivalence.title),
                ui.table(
                    [
                        "Construction",
                        "Value today (%)",
                        "Options (%)",
                        "Bond at maturity (%)",
                        "Collateral (%)",
                        "Delta (%)",
                        "Gamma (%)",
                        "Vega (%)",
                    ],
                    rows,
                ),
                dcc.Graph(figure=figures.packaging_figure(equivalence, market.spot)),
                ui.bullets(
                    [
                        f"**{equivalence.first.name}**: {equivalence.first.obligation}",
                        f"**{equivalence.second.name}**: {equivalence.second.obligation}",
                    ]
                ),
                dbc.Alert(equivalence.lesson, color="info", className="py-2"),
                ui.note(
                    f"Strikes around today's spot, {maturity:g} years. Same payoff, same value and "
                    "same Greeks: only the cash flows and the obligations differ. Options: paid "
                    "(+) or received (−). Collateral: the worst amount the option and stock legs "
                    "can owe at maturity, covered by the bond leg when there is one."
                ),
            ]
        ),
        className="mb-3",
    )


def results(
    view: str | None,
    vol: str | None,
    horizon: float | None,
    protection: str | None,
    objective: str | None,
    data: MarketData,
) -> tuple[Any, Any]:
    """The products that fit the answers, the full ranking, and equivalent constructions."""
    profile = Profile(
        View(view or View.MILD_BULL.value),
        Protection(protection or Protection.FULL.value),
        Objective(objective or Objective.GROWTH.value),
        VolView(vol or VolView.NEUTRAL.value),
        float(horizon or 3.0),
    )
    ranked = recommend(profile)
    table = ui.table(
        ["Product", "Family", "Points", "Protection", "Pays"],
        [
            [
                dcc.Link(r.card.name, href=f"/product/{registry.slugify(r.card.name)}"),
                r.card.page,
                f"{r.score:+.1f}",
                r.card.protection.value,
                r.card.objective.value,
            ]
            for r in ranked
        ],
    )
    top = html.Div(
        [
            html.H5("Products that fit", className="mb-3"),
            *[_recommendation(r, i) for i, r in enumerate(ranked[:TOP], start=1)],
            dbc.Accordion(
                dbc.AccordionItem(
                    [
                        table,
                        ui.note(
                            "Points: up to 4 for the view, 1.5 for the objective, ±1 for "
                            "volatility, +0.5/−1 for the horizon, −10 when the product protects "
                            "less than asked."
                        ),
                    ],
                    title=f"Every product, ranked ({len(ranked)})",
                ),
                start_collapsed=True,
            ),
        ]
    )
    try:
        pairs = equivalences(profile.view, market_from_data(data).spot, profile.horizon)
    except ValueError as exc:
        return top, ui.error(f"The market data are invalid: {exc}")
    packaging = html.Div(
        [
            html.H4("Same payoff, different packaging", className="family-title mt-3"),
            html.P(
                "Put-call parity lets the same payoff be built by buying options or by selling "
                "others with cash aside: the risk is identical, who pays and who must post "
                "collateral is not.",
                className="family-blurb",
            ),
            *[_packaging(e, data, profile.horizon) for e in pairs],
        ]
    )
    return top, packaging


def register(app: dash.Dash) -> None:
    """Register the callback of the finder."""
    app.callback(
        Output("finder-results", "children"),
        Output("finder-packaging", "children"),
        Input("finder-view", "value"),
        Input("finder-vol", "value"),
        Input("finder-horizon", "value"),
        Input("finder-protection", "value"),
        Input("finder-objective", "value"),
        Input("market-data", "data"),
    )(results)
