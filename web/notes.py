"""Notes priced by bricks (capital protection, certificates, shark warrants): terms and pricing.

A note page keeps its terms as plain JSON; :func:`build_note` turns them into a note on the
session market. Levels are in % of the note's initial level, amounts in % of the notional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from spricer.analytic.barrier import bgk_warning
from spricer.bricks import BarrierOption, DigitalOption, OneTouch, ZeroCoupon
from spricer.conventions import DAYS_PER_YEAR, Greeks, to_desk
from spricer.market import Market
from spricer.monitoring import Monitoring
from spricer.products.capital_protection import NOTE_CATALOG, NoteParam, NoteSpec
from spricer.products.notes import NOTIONAL, Note, NoteLeg
from spricer.products.participation import (
    PARTICIPATION_CATALOG,
    SHARK_WARRANTS,
    YIELD_CERTIFICATES,
)
from web.options import DAILY, GREEKS, MONITORING, Terms
from web.registry import Product

CATALOG: dict[str, NoteSpec] = {
    s.name: s for s in NOTE_CATALOG + PARTICIPATION_CATALOG + YIELD_CERTIFICATES + SHARK_WARRANTS
}
# a toggle switches a level on or off: the level is asked only when its toggle is on
TOGGLED = {"cap": "capped", "floor": "limited"}


def note_spec(product: Product) -> NoteSpec | None:
    """The note behind a gallery product, None for other products."""
    return CATALOG.get(product.name)


def default_terms(spec: NoteSpec) -> Terms:
    """Terms of a new page (the initial level empty: today's spot)."""
    terms: Terms = {
        "maturity": 3.0 if spec not in SHARK_WARRANTS else 1.0,
        "notional": 1_000_000.0,
        "initial": None,
        "spread_bp": None,  # the market's funding spread
        "monitoring": "european",
    }
    for p in spec.params:
        terms[p.key] = bool(p.default) if p.kind == "toggle" else p.default
    return terms


def shown_params(spec: NoteSpec, terms: Terms) -> list[NoteParam]:
    """Level and amount terms in use (a capped level only when the cap is on)."""
    return [
        p
        for p in spec.params
        if p.kind != "toggle" and (p.key not in TOGGLED or terms.get(TOGGLED[p.key], True))
    ]


def _number(terms: Terms, key: str, label: str, *, positive: bool = True) -> float:
    try:
        number = float(terms.get(key))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number") from None
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"{label} must be positive")
    return number


@dataclass(frozen=True)
class NotePosition:
    """A note on the session market, with what is needed to rebuild it at another maturity."""

    spec: NoteSpec
    note: Note
    market: Market
    spread: float  # issuer's funding spread over the risk-free rate, a rate
    notional: float
    values: dict[str, float]
    monitoring: Monitoring

    @property
    def maturity(self) -> float:
        """Years to maturity."""
        return self.note.maturity

    @property
    def performance_pct(self) -> float:
        """Today's level of the underlying in % of the initial level."""
        return 100.0 * self.market.spot / self.note.initial

    def build(self, values: dict[str, float], maturity: float | None = None) -> Note:
        """The note with other term values, or ``maturity`` years left (daily dates rescaled)."""
        t = self.maturity if maturity is None else maturity
        observations = max(1, round(DAILY * t)) if self.monitoring is Monitoring.DISCRETE else None
        return self.spec.build(values, t, self.note.initial, self.monitoring, observations)

    def at(self, maturity: float) -> Note:
        """The same note with ``maturity`` years left."""
        return self.build(self.values, maturity)

    def value(self, market: Market | None = None, maturity: float | None = None) -> float:
        """Value in % of notional (today's market and maturity by default)."""
        note = self.note if maturity is None else self.at(maturity)
        return float(note.value(market or self.market, self.spread))

    def bond_pct(self) -> float:
        """Value of the zero-coupon legs."""
        values = self.note.leg_values(self.market, self.spread)
        return sum(
            v
            for leg, v in zip(self.note.legs, values, strict=True)
            if isinstance(leg.brick, ZeroCoupon)
        )

    def greeks(self) -> Greeks:
        """Raw Greeks of the note (value in % of notional, per unit of the underlying)."""
        return self.note.greeks(self.market, self.spread)

    def warning(self) -> str | None:
        """Caveat on the barrier observation, None when there is none."""
        if self.monitoring is Monitoring.DISCRETE:
            return bgk_warning(max(1, round(DAILY * self.maturity)))
        return None


def build_note(spec: NoteSpec, terms: Terms, market: Market, funding: float = 0.0) -> NotePosition:
    """The note the terms describe, on ``market``; ValueError with a readable message.

    The funding spread (bp) discounts the note's cash flows unless the terms set their own.
    """
    maturity = _number(terms, "maturity", "The maturity")
    notional = _number(terms, "notional", "The notional")
    initial_term = terms.get("initial")
    initial = (
        market.spot
        if initial_term in (None, "")
        else _number(terms, "initial", "The initial level")
    )
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
    for p in shown_params(spec, terms):
        values[p.key] = _number(terms, p.key, p.label, positive=False)
    monitoring = (
        MONITORING.get(str(terms.get("monitoring")), Monitoring.EUROPEAN)
        if spec.monitoring_choice
        else Monitoring.EUROPEAN
    )
    observations = max(1, round(DAILY * maturity)) if monitoring is Monitoring.DISCRETE else None
    note = spec.build(values, maturity, initial, monitoring, observations)
    return NotePosition(spec, note, market, spread, notional, values, monitoring)


# --- Tables ------------------------------------------------------------------------------------


def desk_greeks(g: Greeks, spot: float) -> dict[str, float]:
    """Raw Greeks of a value in % of notional, in the desk units of the site."""
    desk = to_desk(g, spot=spot, notional=NOTIONAL)
    return {
        "Delta (%)": 100.0 * float(desk.delta),
        "Gamma (%)": 100.0 * float(desk.gamma),
        "Vega (%)": float(desk.vega),
        "Theta (%)": float(desk.theta),
        "Rho (%)": float(desk.rho),
    }


def leg_terms(note: Note, leg: NoteLeg, market: Market) -> str:
    """Terms of a leg: strike, barrier, amount or participation, volatility."""
    brick = leg.brick
    parts = []
    if isinstance(brick, ZeroCoupon | DigitalOption | OneTouch):
        if brick.strike is not None:
            parts.append(f"K {100.0 * brick.strike / note.initial:g}%")
        if isinstance(brick, OneTouch):
            parts.append(f"H {100.0 * brick.barrier / note.initial:g}%")
        parts.append(f"pays {abs(leg.quantity):g}%")
    else:
        participation = 100.0 * abs(leg.quantity) * note.initial / NOTIONAL
        if not math.isclose(participation, 100.0):
            parts.append(f"× {participation:g}%")
        if brick.strike is not None:
            parts.append(f"K {100.0 * brick.strike / note.initial:g}%")
    if isinstance(brick, BarrierOption):
        parts.append(f"H {100.0 * brick.barrier / note.initial:g}%")
    vol = brick.implied_vol(market)
    if vol is not None:
        parts.append(f"σ {100.0 * vol:.2f}%")
    return " · ".join(parts)


def decomposition(position: NotePosition) -> list[dict[str, Any]]:
    """Every brick then the total, in % of notional: price and Greeks."""
    note, market, spread = position.note, position.market, position.spread
    rows: list[dict[str, Any]] = [
        {
            "Brick": leg.role,
            "Terms": leg_terms(note, leg, market),
            "Price (%)": value,
            **desk_greeks(g, market.spot),
        }
        for leg, value, g in zip(
            note.legs, note.leg_values(market, spread), note.leg_greeks(market, spread), strict=True
        )
    ]
    total: dict[str, Any] = {"Brick": "Total", "Terms": ""}
    for key in ("Price (%)", *GREEKS):
        total[key] = sum(r[key] for r in rows)
    rows.append(total)
    return rows


# --- Greeks against the spot -------------------------------------------------------------------

PROFILE = ("Price (%)", "Delta (%)", "Gamma (%)", "Vega (%)", "Theta (%)", "Rho (%)")
_H = 0.01  # spot bump, relative
_H_VOL = 0.01  # volatility bump
_H_RATE = 1e-4  # rate bump


def spot_profile(position: NotePosition, perf_pct: np.ndarray) -> dict[str, np.ndarray]:
    """Price and Greeks against the underlying's level (% of initial).

    Finite differences of the closed form: spot ±1 %, volatility ±1 point (sticky strike), one
    calendar day, rate ±1 basis point.
    """
    note, market, spread = position.note, position.market, position.spread
    s = perf_pct / 100.0 * note.initial

    def v(m: Market, x: np.ndarray, n: Note = note) -> np.ndarray:
        return np.asarray(n.value(m, spread, x), dtype=np.float64)

    mid, up, down = v(market, s), v(market, s * (1 + _H)), v(market, s * (1 - _H))
    vol_up, vol_down = v(market.with_vol_shift(_H_VOL), s), v(market.with_vol_shift(-_H_VOL), s)
    rho_up = v(market.with_rate(market.rate + _H_RATE), s)
    rho_down = v(market.with_rate(market.rate - _H_RATE), s)
    day = min(1.0 / DAYS_PER_YEAR, 0.5 * position.maturity)
    later = v(market, s, position.at(position.maturity - day))
    delta = (up - down) / (2 * _H)  # % of notional per 100 % move: the desk Delta (%)
    gamma = (up - 2 * mid + down) / _H**2 / 100.0  # change of Delta (%) for +1 % spot
    return {
        "Price (%)": mid,
        "Delta (%)": delta,
        "Gamma (%)": gamma,
        "Vega (%)": (vol_up - vol_down) / 2.0,
        "Theta (%)": (later - mid) / (day * DAYS_PER_YEAR),
        "Rho (%)": (rho_up - rho_down) / 2.0,
    }
