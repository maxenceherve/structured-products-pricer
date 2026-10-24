"""Product finder: say what you expect and what you accept, get the products that fit.

Every product of the site is scored with transparent rules (:mod:`spricer.finder`), each point
explained. An educational tool, not investment advice.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from spricer.finder import Objective, Profile, Recommendation, View, VolView, recommend
from web import registry, ui
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
                    "Largest loss of capital you accept at maturity",
                    dcc.Slider(
                        id="finder-max-loss",
                        min=0,
                        max=100,
                        step=5,
                        value=0,
                        marks={v: f"{v} %" for v in (0, 20, 40, 60, 80, 100)},
                        tooltip={"placement": "bottom", "template": "{value} %"},
                    ),
                ),
                dbc.Switch(
                    id="finder-barrier",
                    label="I accept a barrier: protected unless it is breached",
                    value=False,
                    className="mb-3",
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
                "ranked against your answers, each point explained. An educational tool, not "
                "investment advice.",
                className="product-description",
            ),
            dbc.Row(
                [
                    dbc.Col(questions, lg=4, className="mb-4"),
                    dbc.Col(dcc.Loading(html.Div(id="finder-results")), lg=8),
                ]
            ),
        ],
        className="product-page",
    )


# --- Pure functions behind the callback ----------------------------------------------------------


def product_link(name: str, profile: Profile) -> str:
    """Address of a product's page with the answers that set its terms.

    The horizon sets the maturity; the largest loss accepted sets the capital protection of a
    protected note (100 % minus the loss) and the level of a buffer (the loss itself). A page
    applies only the terms its product has.
    """
    terms: dict[str, float] = {"maturity": profile.horizon}
    if profile.max_loss < 100.0:
        terms["protection"] = 100.0 - profile.max_loss
    if 0.0 < profile.max_loss < 100.0:
        terms["buffer"] = profile.max_loss
    return f"/product/{registry.slugify(name)}?{urlencode(terms)}"


def _recommendation(recommendation: Recommendation, rank: int, href: str) -> dbc.Card:
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
                                href=href,
                            ),
                        ],
                        md=9,
                    ),
                ]
            )
        ),
        className="finder-card mb-3",
    )


def _ranking(ranked: list[Recommendation], profile: Profile) -> html.Div:
    """Every product with its points, as a table that scrolls sideways on small screens."""
    head = html.Tr(
        [html.Th(h) for h in ("#", "Product", "Family", "Fit", "Points", "Protection", "Pays")]
    )
    body = []
    for rank, r in enumerate(ranked, start=1):
        fit = max(0.0, min(1.0, r.score / MAX_SCORE))
        body.append(
            html.Tr(
                [
                    html.Td(rank, className="ranking-rank"),
                    html.Td(
                        dcc.Link(r.card.name, href=product_link(r.card.name, profile)),
                        className="ranking-name",
                    ),
                    html.Td(r.card.page),
                    html.Td(
                        dbc.Progress(value=round(100 * fit), style={"height": "8px"}),
                        className="ranking-fit",
                    ),
                    html.Td(ui.fmt(r.score, 1)),
                    html.Td(r.card.protection_text),
                    html.Td(r.card.objective.value),
                ],
                className="ranking-out" if r.score < 0 else "",
            )
        )
    return html.Div(
        dbc.Table(
            [html.Thead(head), html.Tbody(body)],
            hover=True,
            size="sm",
            className="ranking-table",
        ),
        className="ranking-wrap",
    )


def results(
    view: str | None,
    vol: str | None,
    horizon: float | None,
    max_loss: float | None,
    accept_barrier: bool | None,
    objective: str | None,
) -> Any:
    """The products that fit the answers, and every product ranked."""
    profile = Profile(
        View(view or View.MILD_BULL.value),
        float(max_loss or 0.0),
        Objective(objective or Objective.GROWTH.value),
        VolView(vol or VolView.NEUTRAL.value),
        float(horizon or 3.0),
        bool(accept_barrier),
    )
    ranked = recommend(profile)
    return html.Div(
        [
            html.H5("Products that fit", className="mb-3"),
            *[
                _recommendation(r, i, product_link(r.card.name, profile))
                for i, r in enumerate(ranked[:TOP], start=1)
            ],
            html.H5(f"Every product, ranked ({len(ranked)})", className="mt-4 mb-2"),
            _ranking(ranked, profile),
            ui.note(
                "Points: up to 4 for the view, 1.5 for the objective, ±1 for volatility, "
                "+0.5/−1 for the horizon, −0.5 for much more protection than needed, −10 when "
                "the product can lose more than you accept (greyed out)."
            ),
        ]
    )


def register(app: dash.Dash) -> None:
    """Register the callback of the finder."""
    app.callback(
        Output("finder-results", "children"),
        Input("finder-view", "value"),
        Input("finder-vol", "value"),
        Input("finder-horizon", "value"),
        Input("finder-max-loss", "value"),
        Input("finder-barrier", "value"),
        Input("finder-objective", "value"),
    )(results)
