"""Coupon notes (reverse convertibles, autocalls): terms, underlyings and Monte Carlo results.

A coupon note looks at the worst performance of one or several underlyings. Each underlying
uses the smile of the market panel, shifted in parallel to its own 1-year ATM volatility, with
its own spot and dividend yield. Monte Carlo results are kept in the page as plain JSON.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from spricer.market import Basket, Market
from spricer.mc import check_correlation, nearest_correlation
from spricer.mc.coupon_notes import LEG_NAMES, CouponNoteMonteCarlo
from spricer.mc.greeks import NoteGreeks
from spricer.mc.paths import MCEstimate
from spricer.monitoring import Monitoring
from spricer.products.coupon_notes import (
    COUPON_NOTE_CATALOG,
    FREQUENCIES,
    CouponNote,
    CouponNoteSpec,
    LifeState,
)
from web import registry
from web.market import MATURITY, MarketData, funding_bp, market_from_data
from web.options import GREEKS, MONITORING, Terms
from web.registry import Product

CATALOG = {s.name: s for s in COUPON_NOTE_CATALOG}
UNDERLYING_FIELDS = ("Name", "Spot", "Initial level", "Dividend (%)", "ATM vol 1y (%)")
MAX_UNDERLYINGS = 5
DEFAULT_CORRELATION = 0.6
SOLVE_BOUNDS = {
    "coupon_pa": (0.0, 60.0),
    "barrier": (1.0, 150.0),
    "autocall_level": (40.0, 200.0),
    "coupon_level": (0.0, 150.0),
    "strike": (10.0, 200.0),
    "step_down": (0.0, 50.0),
}
Rows = list[dict[str, Any]]


def coupon_spec(product: Product) -> CouponNoteSpec | None:
    """The coupon note behind a gallery product, None for other products."""
    return CATALOG.get(product.name)


def default_terms(spec: CouponNoteSpec) -> Terms:
    """Terms of a new page."""
    terms: Terms = {
        "maturity": 3.0,
        "frequency": "Quarterly",
        "issue_price": 100.0,
        "notional": 1_000_000.0,
        "spread_bp": None,  # the market's funding spread
        "monitoring": "european",
        "elapsed": 0.0,
        "memory_accrued": 0.0,
        "breached": False,
    }
    for p in spec.params:
        terms[p.key] = bool(p.default) if p.kind == "toggle" else p.default
    return terms


def _number(terms: Terms, key: str, label: str, *, positive: bool = True) -> float:
    try:
        number = float(terms.get(key))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number") from None
    if not math.isfinite(number) or (positive and number <= 0.0) or number < 0.0:
        raise ValueError(f"{label} must be {'positive' if positive else 'non-negative'}")
    return number


@dataclass(frozen=True)
class CouponTerms:
    """A coupon note and the terms around it."""

    spec: CouponNoteSpec
    note: CouponNote
    values: dict[str, float]
    maturity: float
    frequency: int
    issue_price: float
    notional: float
    spread: float
    monitoring: Monitoring
    state: LifeState

    def build(self, values: dict[str, float]) -> CouponNote:
        """The note with other term values (same dates and barrier observation)."""
        return self.spec.build(values, self.maturity, self.frequency, self.monitoring, self.state)


def build_coupon(spec: CouponNoteSpec, terms: Terms, funding: float = 0.0) -> CouponTerms:
    """The coupon note the terms describe; ValueError with a readable message.

    The funding spread (bp) discounts the note's cash flows unless the terms set their own.
    """
    maturity = _number(terms, "maturity", "The maturity")
    frequency = FREQUENCIES.get(str(terms.get("frequency")), 4)
    issue_price = _number(terms, "issue_price", "The issue price")
    notional = _number(terms, "notional", "The notional")
    typed = terms.get("spread_bp")
    spread = (
        funding
        if typed in (None, "")
        else _number(terms, "spread_bp", "The funding spread", positive=False)
    ) / 1e4
    values: dict[str, float] = {}
    for p in spec.params:
        if p.kind == "toggle":
            values[p.key] = 1.0 if terms.get(p.key) else 0.0
        else:
            values[p.key] = _number(terms, p.key, p.label, positive=False)
    monitoring = (
        MONITORING.get(str(terms.get("monitoring")), Monitoring.EUROPEAN)
        if spec.monitoring_choice
        else Monitoring.EUROPEAN
    )
    state = LifeState(
        _number(terms, "elapsed", "The years since issue", positive=False),
        _number(terms, "memory_accrued", "The coupons in memory", positive=False),
        bool(terms.get("breached")),
    )
    note = spec.build(values, maturity, frequency, monitoring, state)
    return CouponTerms(
        spec, note, values, maturity, frequency, issue_price, notional, spread, monitoring, state
    )


# --- Underlyings -----------------------------------------------------------------------------


def _atm(data: MarketData) -> float:
    market = market_from_data(data)
    return 100.0 * float(market.vol.vol(market.spot, 1.0))


def default_underlyings(data: MarketData) -> Rows:
    """One underlying: the market panel's."""
    try:
        atm = round(_atm(data), 2)
    except ValueError:
        atm = 20.0
    spot = data.get("spot") or 100.0
    return [
        {
            "Name": "Underlying 1",
            "Spot": spot,
            "Initial level": spot,
            "Dividend (%)": data.get("dividend_pct", 0.0),
            "ATM vol 1y (%)": atm,
        }
    ]


def add_underlying(rows: Rows, data: MarketData) -> Rows:
    """One more underlying, a copy of the market panel's under a new name."""
    if len(rows) >= MAX_UNDERLYINGS:
        return rows
    names = {str(r.get("Name")) for r in rows}
    n = len(rows) + 1
    while f"Underlying {n}" in names:
        n += 1
    new = default_underlyings(data)[0] | {"Name": f"Underlying {n}"}
    return [*rows, new]


def underlying_market(row: dict[str, Any], data: MarketData) -> Market:
    """Market of one underlying: the panel's smile shifted to its ATM volatility."""
    try:
        spot, dividend, atm = (
            float(row.get(f))  # type: ignore[arg-type]
            for f in ("Spot", "Dividend (%)", "ATM vol 1y (%)")
        )
    except (TypeError, ValueError):
        raise ValueError(
            "Every underlying needs a spot, an initial level, a dividend yield and an ATM "
            "volatility."
        ) from None
    if not (spot > 0.0 and atm > 0.0):
        raise ValueError("Spots and volatilities must be positive.")
    shift = atm - _atm(data)
    rows = [
        {k: (v if k == MATURITY or v is None else float(v) + shift) for k, v in r.items()}
        for r in data["rows"]
    ]
    return market_from_data(data | {"spot": spot, "dividend_pct": dividend, "rows": rows})


def correlation_rows(names: list[str], previous: Rows | None) -> Rows:
    """Correlation grid for ``names``: pairs already typed are kept, others at the default."""
    known: dict[tuple[str, str], float] = {}
    for typed in previous or []:
        for other, value in typed.items():
            if other != "Name" and value is not None:
                known[(str(typed["Name"]), other)] = value
    out = []
    for a in names:
        row: dict[str, Any] = {"Name": a}
        for b in names:
            row[b] = 1.0 if a == b else known.get((a, b), DEFAULT_CORRELATION)
        out.append(row)
    return out


def correlation_matrix(names: list[str], rows: Rows) -> np.ndarray:
    """The matrix typed in the grid (not yet checked)."""
    by_name = {str(r.get("Name")): r for r in rows}
    try:
        return np.array([[float(by_name.get(a, {}).get(b)) for b in names] for a in names])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError("Every correlation must be a number.") from None


def nearest_rows(names: list[str], rows: Rows) -> Rows:
    """The nearest valid correlation matrix to the one typed, as grid rows."""
    matrix = correlation_matrix(names, rows)
    fixed = nearest_correlation(0.5 * (matrix + matrix.T))
    return [
        {"Name": a, **{b: round(float(fixed[i, j]), 4) for j, b in enumerate(names)}}
        for i, a in enumerate(names)
    ]


def basket_of(rows: Rows, correlation: Rows | None, data: MarketData) -> Basket:
    """The underlyings typed on the page; ValueError with a readable message."""
    if not rows:
        raise ValueError("Add at least one underlying.")
    if len(rows) > MAX_UNDERLYINGS:
        raise ValueError(f"At most {MAX_UNDERLYINGS} underlyings.")
    names = [str(r.get("Name") or "").strip() for r in rows]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("Underlyings need distinct names.")
    markets = [underlying_market(r, data) for r in rows]
    try:
        initials = [float(r["Initial level"]) for r in rows]
    except (TypeError, ValueError):
        raise ValueError("Every underlying needs an initial level.") from None
    matrix = None
    if len(rows) > 1:
        matrix = check_correlation(correlation_matrix(names, correlation or []), len(names))
    return Basket.of(markets, initials, names, matrix)


# --- Monte Carlo results as JSON ---------------------------------------------------------------


def _estimate(e: MCEstimate) -> list[float]:
    return [e.value, e.std_error]


def result_json(result: CouponNoteMonteCarlo) -> dict[str, Any]:
    """The numbers of a Monte Carlo price, to keep in the page."""
    return {
        "total": _estimate(result.total),
        "legs": [_estimate(e) for e in result.legs],
        "calls": [_estimate(e) for e in result.call_probability],
        "life": _estimate(result.expected_life),
        "loss": _estimate(result.loss_probability),
        "breach": _estimate(result.breach_probability),
        "paths": result.total.n_paths,
    }


def _brick_terms(note: CouponNote) -> dict[str, str]:
    first = note.observations[0]
    condition = f" if at or above {first.coupon_level:g} %" if first.coupon_level > 0.0 else ""
    memory = " · memory" if note.memory else ""
    calls = [o.autocall_level for o in note.observations if o.autocall_level is not None]
    call = f"par if at or above {calls[0]:g} %" if calls else ""
    return dict(
        zip(
            LEG_NAMES,
            (
                f"{first.coupon:g} % a date{condition}{memory}",
                call,
                "par",
                f"K {note.put_strike:g}% · H {note.protection_barrier:g}%",
            ),
            strict=True,
        )
    )


def brick_rows(
    note: CouponNote, result: dict[str, Any], greeks: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """The bricks on common paths then the total: price with its error, and Greeks if computed."""
    terms = _brick_terms(note)
    legs = (greeks or {}).get("legs", [{}] * len(LEG_NAMES))
    rows = []
    for name, (value, se), leg in zip(LEG_NAMES, result["legs"], legs, strict=True):
        if name == LEG_NAMES[1] and not note.callable:
            continue
        rows.append(
            {
                "Brick": name,
                "Terms": terms[name],
                "Price (%)": value,
                "Error (%)": se,
                **{key: leg.get(key) for key in GREEKS},
            }
        )
    now = greeks_now(greeks)[0] if greeks else dict.fromkeys(GREEKS)
    rows.append(
        {
            "Brick": "Total",
            "Terms": "",
            "Price (%)": result["total"][0],
            "Error (%)": result["total"][1],
            **now,
        }
    )
    return rows


def greeks_json(greeks: NoteGreeks, basket: Basket, rows: Rows, notional: float) -> dict[str, Any]:
    """The numbers of the Monte Carlo Greeks, to keep in the page."""
    return {
        "assets": [
            {
                "name": a.name,
                "delta": _estimate(a.delta),
                "gamma": _estimate(a.gamma),
                "vega": _estimate(a.vega),
            }
            for a in greeks.assets
        ],
        "legs": [
            {
                "Delta (%)": leg.delta.value,
                "Gamma (%)": leg.gamma.value,
                "Vega (%)": leg.vega.value,
                "Theta (%)": None if leg.theta is None else leg.theta.value,
                "Rho (%)": leg.rho.value,
            }
            for leg in greeks.legs
        ],
        "theta": None if greeks.theta is None else _estimate(greeks.theta),
        "rho": _estimate(greeks.rho),
        "spots": [m.spot for m in basket.markets],
        "rows": rows,
        "notional": notional,
    }


def greeks_now(greeks: dict[str, Any]) -> tuple[dict[str, float | None], dict[str, float]]:
    """The Greeks of the note, all underlyings together, and their standard errors."""

    def total(key: str) -> tuple[float, float]:
        values = [a[key] for a in greeks["assets"]]
        return sum(v for v, _ in values), sum(se**2 for _, se in values) ** 0.5

    delta, gamma, vega = total("delta"), total("gamma"), total("vega")
    theta, rho = greeks["theta"], greeks["rho"]
    return (
        {
            "Delta (%)": delta[0],
            "Gamma (%)": gamma[0],
            "Vega (%)": vega[0],
            "Theta (%)": None if theta is None else theta[0],
            "Rho (%)": rho[0],
        },
        {
            "Delta (%)": delta[1],
            "Gamma (%)": gamma[1],
            "Vega (%)": vega[1],
            "Theta (%)": 0.0 if theta is None else theta[1],
            "Rho (%)": rho[1],
        },
    )


def schedule_rows(note: CouponNote, result: dict[str, Any] | None) -> list[list[Any]]:
    """Date, coupon, coupon barrier, autocall barrier and call probability of each date."""
    calls = result["calls"] if result else [[None, None]] * len(note.observations)
    return [
        [
            o.time,
            o.coupon,
            o.coupon_level,
            o.autocall_level,
            None if p is None else 100.0 * p,
        ]
        for o, (p, _) in zip(note.observations, calls, strict=True)
    ]


def coupon_spec_of(store: dict[str, Any]) -> CouponNoteSpec:
    """The coupon note of a page's stored terms."""
    product = registry.find(store.get("slug", ""))
    spec = coupon_spec(product) if product is not None else None
    if spec is None:
        raise ValueError("This page has no coupon note to price.")
    return spec


def setup_coupon(
    store: dict[str, Any] | None, rows: Rows | None, corr: Rows | None, data: MarketData
) -> tuple[CouponTerms, Basket]:
    """The coupon note and the underlyings of a page; ValueError when they cannot be priced."""
    if not store:
        raise ValueError("Set the terms of the product.")
    terms = build_coupon(coupon_spec_of(store), store["terms"], funding_bp(data))
    return terms, basket_of(list(rows or []), corr, data)
