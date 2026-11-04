"""The solver at the bottom of the terms form: one term, for a target price.

It finds the value of the chosen term that gives the target price, the other terms fixed, and
writes it in the form. The three families share the same block; only their price function
differs (closed form, or the Monte Carlo on one set of paths).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import ALL, Input, Output, State, dcc

from spricer.mc.coupon_notes import evaluate_note, simulate_note
from spricer.solve import Solution, solve_on_grid
from web import risk_panel, ui
from web.coupons import SOLVE_BOUNDS, Rows, setup_coupon
from web.market import MarketData
from web.notes import NotePosition, shown_params
from web.options import Position, build_position

LEVELS = np.linspace(0.5, 400.0, 400)  # a level or an amount, in %
MATURITIES = np.geomspace(0.02, 30.0, 60)
SPREADS = np.linspace(0.0, 2000.0, 81)  # bp
COUPON_POINTS = 41
Priced = tuple[Callable[[float], float], Any, float, str, str]


def _option_terms(position: Position, terms: dict[str, Any], key: str) -> Priced:
    labels = dict(
        zip([f"strike_{i}" for i in range(9)], position.setup.strike_labels, strict=False)
    )
    labels |= {"maturity": "Maturity", "barrier": "Barrier", "payout": "Amount paid"}
    if key not in labels:
        raise ValueError("This term cannot be solved.")

    def price(x: float) -> float:
        changed = {**terms, key: x}
        return build_position(position.setup, changed, position.market).price_pct()

    unit = " years" if key == "maturity" else " %"
    grid = MATURITIES if key == "maturity" else LEVELS
    return price, grid, float(terms[key]), labels[key], unit


def _note_terms(position: NotePosition, store: dict[str, Any], key: str) -> Priced:
    market, spread = position.market, position.spread
    if key == "maturity":
        return (
            lambda x: float(position.at(x).value(market, spread)),
            MATURITIES,
            position.maturity,
            "Maturity",
            " years",
        )
    if key == "spread_bp":
        return (
            lambda x: float(position.note.value(market, x / 1e4)),
            SPREADS,
            1e4 * spread,
            "Funding spread",
            " bp",
        )
    param = next((p for p in position.spec.params if p.key == key and p.kind != "toggle"), None)
    if param is None:
        raise ValueError("This term cannot be solved.")
    if param not in shown_params(position.spec, store["terms"]):
        raise ValueError(f"Switch the {param.label.lower()} on to solve for it.")
    return (
        lambda x: float(position.build({**position.values, key: x}).value(market, spread)),
        LEVELS,
        position.values[key],
        param.label,
        " %",
    )


def _message(label: str, value: float, unit: str, solution: Solution, error: float | None) -> Any:
    noise = "" if error is None else f" ± {error:.2f}"
    text = (
        f"**{label} = {value:g}{unit}** gives a price of **{solution.price:.2f}{noise} %**. "
        "Click Price to see the product with it."
    )
    return dbc.Alert(dcc.Markdown(text), color="success", className="py-2 mb-0")


def _written(ids: list[dict[str, str]], key: str, value: float) -> list[Any]:
    return [value if i["key"] == key else dash.no_update for i in ids]


def solve_view(
    n_clicks: int | None,
    key: str | None,
    target: float | None,
    store: dict[str, Any] | None,
    data: MarketData,
    ids: list[dict[str, str]],
) -> tuple[Any, list[Any]]:
    """Solve the term of an option or a note; the solution goes in its input."""
    unchanged = [dash.no_update] * len(ids)
    if not n_clicks or key is None:
        return dash.no_update, unchanged
    try:
        if target is None:
            raise ValueError("Type a target price.")
        item = risk_panel.resolve(store, data)
        assert store is not None  # resolve refuses a page without terms
        if isinstance(item, NotePosition):
            price, grid, current, label, unit = _note_terms(item, store, key)
        else:
            price, grid, current, label, unit = _option_terms(item, store["terms"], key)
        solution = solve_on_grid(price, float(target), grid, current, label.lower())
    except ValueError as exc:
        return ui.error(str(exc)), unchanged
    value = round(solution.value, 3 if key == "maturity" else 2)
    return _message(label, value, unit, solution, None), _written(ids, key, value)


def solve_coupon_view(
    n_clicks: int | None,
    key: str | None,
    target: float | None,
    store: dict[str, Any] | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
    ids: list[dict[str, str]],
) -> tuple[Any, list[Any]]:
    """Solve the term of a coupon note on one set of paths (common random numbers)."""
    unchanged = [dash.no_update] * len(ids)
    if not n_clicks or key is None:
        return dash.no_update, unchanged
    try:
        if target is None:
            raise ValueError("Type a target price.")
        terms, basket = setup_coupon(store, rows, corr, data)
        simulated = simulate_note(terms.note, basket, n_paths=int(paths or 50_000))

        def result(x: float) -> Any:
            if key == "spread_bp":
                return evaluate_note(terms.note, basket, simulated, x / 1e4)
            return evaluate_note(
                terms.build({**terms.values, key: x}), basket, simulated, terms.spread
            )

        if key == "spread_bp":
            label, unit, grid, current = "Funding spread", " bp", SPREADS, 1e4 * terms.spread
        else:
            label = terms.spec.param(key).label
            unit = " %"
            low, high = SOLVE_BOUNDS.get(key, (1.0, 200.0))
            grid, current = np.linspace(low, high, COUPON_POINTS), terms.values[key]
        base_error = result(current).total.std_error
        solution = solve_on_grid(
            lambda x: result(x).total.value,
            float(target),
            grid,
            current,
            label.lower(),
            tolerance=max(0.01, 2.0 * base_error),
        )
        error = result(solution.value).total.std_error
    except ValueError as exc:
        return ui.error(str(exc)), unchanged
    value = round(solution.value, 2)
    return _message(label, value, unit, solution, error), _written(ids, key, value)


def solve_form(
    n_clicks: int | None,
    key: str | None,
    target: float | None,
    values: list[Any],
    ids: list[dict[str, str]],
    pathname: str | None,
    data: MarketData,
) -> tuple[Any, list[Any]]:
    """Solve from the form as it stands, whatever was last priced."""
    return solve_view(n_clicks, key, target, ui.collect(values, ids, pathname), data, ids)


def solve_coupon_form(
    n_clicks: int | None,
    key: str | None,
    target: float | None,
    values: list[Any],
    ids: list[dict[str, str]],
    pathname: str | None,
    rows: Rows | None,
    corr: Rows | None,
    data: MarketData,
    paths: Any,
) -> tuple[Any, list[Any]]:
    """Solve a coupon note from the form as it stands, whatever was last priced."""
    store = ui.collect(values, ids, pathname)
    return solve_coupon_view(n_clicks, key, target, store, rows, corr, data, paths, ids)


def register(app: dash.Dash) -> None:
    """Register the solver of the three families."""
    app.callback(
        Output("solve-result", "children"),
        Output(ui.term(ALL), "value", allow_duplicate=True),  # type: ignore[arg-type]
        Input("solve-run", "n_clicks"),
        State("solve-term", "value"),
        State("solve-target", "value"),
        State(ui.term(ALL), "value"),  # type: ignore[arg-type]
        State(ui.term(ALL), "id"),  # type: ignore[arg-type]
        State("url", "pathname"),
        State("market-data", "data"),
        prevent_initial_call=True,
    )(solve_form)
    app.callback(
        Output("solve-result", "children", allow_duplicate=True),
        Output(ui.term(ALL), "value", allow_duplicate=True),  # type: ignore[arg-type]
        Input("cn-solve-run", "n_clicks"),
        State("solve-term", "value"),
        State("solve-target", "value"),
        State(ui.term(ALL), "value"),  # type: ignore[arg-type]
        State(ui.term(ALL), "id"),  # type: ignore[arg-type]
        State("url", "pathname"),
        State("cn-underlyings", "rowData"),
        State("cn-correlation", "rowData"),
        State("market-data", "data"),
        State("cn-paths", "value"),
        prevent_initial_call=True,
    )(solve_coupon_form)
