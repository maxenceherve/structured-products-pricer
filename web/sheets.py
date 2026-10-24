"""Content of the product sheets of the three families: options, notes, coupon notes."""

from __future__ import annotations

from typing import Any

import numpy as np

from spricer.bricks import Underlying
from spricer.market import Basket
from spricer.monitoring import Monitoring
from spricer.products.notes import LevelKind
from spricer.products.participation import SHARK_WARRANTS
from web import registry
from web.coupons import CouponTerms, legs_rows, schedule_rows
from web.notes import NotePosition, decomposition, desk_greeks, shown_params
from web.options import Position, display_greeks
from web.options import decomposition as option_legs
from web.scenarios import breakevens, coupon_scenarios, note_scenarios, option_scenarios
from web.sheet import ACCENT, GUIDE, PRIMARY, Chart, Sheet

SCHEDULE_HEADER = [
    "Date (years)",
    "Coupon (%)",
    "Coupon barrier (%)",
    "Autocall barrier (%)",
    "Call probability (%)",
]
_OBSERVATION = {
    Monitoring.EUROPEAN: "European (at maturity)",
    Monitoring.CONTINUOUS: "American (continuous)",
    Monitoring.DISCRETE: "Discrete (daily closes)",
}


def _years(t: float) -> str:
    return f"{t:g} year" + ("" if t == 1 else "s")


def _greek_rows(g: dict[str, float]) -> list[tuple[str, object]]:
    return [
        ("Delta (per +1 % spot)", f"{g['Delta (%)']:+.2f} %"),
        ("Gamma (Δ change per +1 %)", f"{g['Gamma (%)']:+.2f} %"),
        ("Vega (per vol point)", f"{g['Vega (%)']:+.3f} %"),
        ("Theta (per day)", f"{g['Theta (%)']:+.4f} %"),
        ("Rho (per bp)", f"{g['Rho (%)']:+.4f} %"),
    ]


def option_sheet(position: Position) -> Sheet:
    """Sheet of an option position."""
    market, strategy = position.market, position.strategy
    levels, rows = option_scenarios(position)
    pnl = rows[-1] if len(rows) == 2 else rows[2]
    price = position.price_pct()
    x = np.linspace(50.0, 150.0, 401)
    spots = x / 100.0 * market.spot
    growth = float(np.exp(market.rate * position.maturity))
    paid = 100.0 * np.asarray(strategy.payoff(spots), dtype=float) / market.spot
    holds = any(isinstance(leg.brick, Underlying) for leg in strategy.legs)
    if len(rows) == 2:  # on a fine grid: a payoff that jumps crosses zero at its jump
        fine = np.linspace(50.0, 150.0, 20001)
        fine_paid = np.asarray(strategy.payoff(fine / 100.0 * market.spot), dtype=float)
        crossings = breakevens(list(fine), list(100.0 * fine_paid / market.spot - price * growth))
    else:  # barrier observed along the path: if never touched
        crossings = breakevens(levels, pnl[1])
    chart = Chart(
        x,
        [
            ("Paid at maturity", paid, PRIMARY),
            ("P&L at maturity", paid - price * growth, ACCENT),
            ("Delta one", x if holds else x - 100.0, GUIDE),
        ],
        [(k, f"K {k:g}%") for k in position.strikes_pct]
        + ([(position.barrier_pct, "barrier")] if position.barrier_pct is not None else []),
        "Spot at maturity (% of today)",
        "% of notional",
    )
    terms: list[tuple[str, object]] = [
        ("Position", "Short" if position.short else "Long"),
        ("Maturity", _years(position.maturity)),
        ("Notional", f"{position.notional:,.0f}"),
    ]
    terms += [
        (f"{label} (% of spot)", f"{k:g}")
        for label, k in zip(position.setup.strike_labels, position.strikes_pct, strict=True)
    ]
    if position.barrier_pct is not None:
        brick = strategy.legs[0].brick
        terms += [
            ("Barrier (% of spot)", f"{position.barrier_pct:g}"),
            ("Barrier observation", _OBSERVATION[brick.monitoring]),  # type: ignore[attr-defined]
        ]
    risks = (
        [
            "Short options: losses can be far larger than the premium received; the position "
            "must be hedged (delta, gamma, vega).",
        ]
        if position.short
        else ["The most you can lose is the premium paid."]
    )
    risks.append(
        "Before maturity the value moves with the spot, the volatility and time: see the Greeks."
    )
    legs = [[r["Leg"], r["Terms"], r["Price (%)"]] for r in option_legs(position)]
    return Sheet(
        position.setup.product.family,
        position.name.capitalize(),
        position.setup.description,
        [
            ("Price", f"{price:.2f} %", "of notional"),
            (
                "Premium",
                f"{abs(price) / 100 * position.notional:,.0f}",
                "paid" if price >= 0 else "received",
            ),
            ("Maturity", f"{position.maturity:g} y", ""),
        ],
        chart,
        ("Spot at maturity", levels, rows),
        crossings,
        terms,
        _greek_rows(display_greeks(strategy, market)),
        [("Legs", (["Leg", "Terms", "Price (%)"], legs))],
        risks,
    )


def note_sheet(position: NotePosition) -> Sheet:
    """Sheet of a note priced by bricks."""
    note, spec = position.note, position.spec
    levels, rows = note_scenarios(position)
    price, bond = position.value(), position.bond_pct()
    top = max(160.0, 1.2 * max(lv.value_pct for lv in note.levels))
    x = np.linspace(0.0, top, 481)
    fine = np.linspace(0.0, top, 40001)
    paid = np.asarray(note.payoff(x), dtype=float)
    series = [("Paid at maturity", paid, PRIMARY)]
    if spec not in SHARK_WARRANTS:
        series.append(("Underlying", x, GUIDE))
    chart = Chart(
        x,
        series,
        [(lv.value_pct, lv.name) for lv in note.levels if lv.kind is LevelKind.PERFORMANCE],
        "Underlying at maturity (% of initial level)",
        "% of notional",
    )
    terms: list[tuple[str, object]] = [
        (f"{p.label} (%)", f"{position.values[p.key]:g}")
        for p in shown_params(spec, _terms(position))
    ]
    terms += [
        ("Maturity", _years(position.maturity)),
        ("Initial level", f"{note.initial:,.2f}"),
        ("Barrier observation", _OBSERVATION[position.monitoring]),
        ("Issuer spread", f"{1e4 * position.spread:g} bp"),
    ]
    figures = [("Price", f"{price:.2f} %", "of notional")]
    if bond:
        figures.append(("Bond part", f"{bond:.2f} %", "pays the protection"))
        figures.append(("Options", f"{price - bond:+.2f} %", "the bricks"))
    figures.append(("Maturity", f"{position.maturity:g} y", ""))
    risks = ["Issuer credit risk: the note is a debt of the issuer, lost if the issuer defaults."]
    risks.append(
        "The capital protection applies at maturity only; sold earlier, the note is worth its "
        "market value."
        if bond
        else "The capital is at risk: you can lose part or all of the investment."
    )
    if any("barrier" in lv.name.lower() for lv in note.levels):
        risks.append(
            "Barrier: once it is breached the payoff changes for good; a barrier observed "
            "along the path is breached more often than one observed at maturity."
        )
    legs = [[r["Leg"], r["Terms"], r["Price (%)"]] for r in decomposition(position)]
    return Sheet(
        _family(position),
        note.name,
        spec.description,
        figures,
        chart,
        ("Underlying at maturity", levels, rows),
        breakevens(list(fine), list(np.asarray(note.payoff(fine), dtype=float) - price)),
        terms,
        _greek_rows(desk_greeks(position.greeks(), position.market.spot)),
        [("Brick decomposition", (["Leg", "Terms", "Price (%)"], legs))],
        risks,
    )


def _terms(position: NotePosition) -> dict[str, Any]:
    return {k: bool(v) for k, v in position.values.items()}


def _family(position: NotePosition) -> str:
    product = registry.find(registry.slugify(position.spec.name))
    return product.family if product is not None else "Structured note"


def coupon_sheet(
    terms: CouponTerms,
    basket: Basket,
    result: dict[str, Any],
    greeks: dict[str, Any] | None,
) -> Sheet:
    """Sheet of a coupon note priced by Monte Carlo (Greeks when computed)."""
    note, spec = terms.note, terms.spec
    levels, rows = coupon_scenarios(note)
    price, se = result["total"]
    x = np.linspace(0.0, 160.0, 481)
    series = [("Paid, barrier never breached", np.asarray(note.maturity_payoff(x, False)), PRIMARY)]
    if len(rows) == 2:
        series.append(("Paid, barrier breached", np.asarray(note.maturity_payoff(x, True)), ACCENT))
    series.append(("Underlying", x, GUIDE))
    chart = Chart(
        x,
        series,
        [(lv.value_pct, lv.name) for lv in note.levels],
        "Worst performance at maturity (% of initial level)",
        "% of notional",
        zero_line=False,
    )
    labels = {p.key: p.label if p.kind != "pct" else f"{p.label} (%)" for p in spec.params}
    term_rows: list[tuple[str, object]] = [
        (labels[k], ("yes" if v else "no") if k == "memory" else f"{v:g}")
        for k, v in terms.values.items()
    ]
    term_rows += [
        ("Maturity", _years(terms.maturity)),
        ("Observations a year", terms.frequency),
        ("Protection barrier observation", _OBSERVATION[note.monitoring]),
        ("Issue price", f"{terms.issue_price:g} %"),
    ]
    greek_rows: list[tuple[str, object]] = []
    for a in (greeks or {}).get("assets", []):
        greek_rows += [
            (f"Delta {a['name']}", f"{a['delta'][0]:+.2f} ± {a['delta'][1]:.2f} %"),
            (f"Gamma {a['name']}", f"{a['gamma'][0]:+.2f} ± {a['gamma'][1]:.2f} %"),
            (f"Vega {a['name']}", f"{a['vega'][0]:+.3f} ± {a['vega'][1]:.3f} %"),
        ]
    underlyings = [
        [n, m.spot, i, 100.0 * m.dividend_yield]
        for n, m, i in zip(basket.names, basket.markets, basket.initials, strict=True)
    ]
    risks = [
        "Capital at risk: below the protection barrier the worst performance is lost one for one.",
        "Issuer credit risk: the note is a debt of the issuer, lost if the issuer defaults.",
    ]
    if note.callable:
        risks.append("Early redemption: once the note is called, the coupons stop.")
    if basket.size > 1:
        risks.append("Worst-of: the weakest underlying alone decides the coupons and the capital.")
    return Sheet(
        "Autocallables" if note.callable else "Yield enhancement",
        note.name,
        spec.description,
        [
            ("Fair value", f"{price:.2f} %", f"± {se:.2f} (Monte Carlo)"),
            ("Issuer margin", f"{terms.issue_price - price:+.2f} %", "issue price − value"),
            ("Expected life", f"{result['life'][0]:.2f} y", "until called or matured"),
            ("Probability of loss", f"{100 * result['loss'][0]:.1f} %", "at maturity"),
        ],
        chart,
        ("Worst performance", levels, rows),
        (),
        term_rows,
        greek_rows,
        [
            (
                "Underlyings (worst performance)",
                (["Underlying", "Spot", "Initial level", "Dividend (%)"], underlyings),
            ),
            (
                "Observation schedule",
                (
                    [
                        "Date (years)",
                        "Coupon (%)",
                        "Coupon barrier (%)",
                        "Autocall barrier (%)",
                        "Call probability (%)",
                    ],
                    schedule_rows(note, result),
                ),
            ),
            (
                "Decomposition on common paths",
                (["Leg", "Value (%)", "Std error (%)"], legs_rows(result)),
            ),
        ],
        risks,
    )
