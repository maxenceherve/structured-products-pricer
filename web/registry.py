"""Every product of the site: where it sits in the gallery and the shape of its payoff."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spricer._typing import FloatArray
from spricer.bricks import BARRIER_TYPES, BarrierOption, EuropeanOption
from spricer.monitoring import Monitoring
from spricer.products.capital_protection import NOTE_CATALOG
from spricer.products.coupon_notes import COUPON_NOTE_CATALOG, CouponNote
from spricer.strategies import CATALOG

OPTIONS = "Options & strategies"
BARRIERS = "Barrier options"
CAPITAL_PROTECTION = "Capital protection"
YIELD = "Yield enhancement"
AUTOCALLS = "Autocallables"
FAMILIES = (OPTIONS, BARRIERS, CAPITAL_PROTECTION, YIELD, AUTOCALLS)
FAMILY_BLURBS = {
    OPTIONS: "Calls, puts and their combinations: spreads, volatility plays, butterflies, "
    "digitals.",
    BARRIERS: "Options activated or cancelled when the underlying touches a level.",
    CAPITAL_PROTECTION: "A bond that pays the capital back, plus options for the upside.",
    YIELD: "A high coupon, paid for by a put sold on the underlying.",
    AUTOCALLS: "Coupons and an early redemption as soon as the underlying is back up.",
}

SPOT = 100.0  # every payoff is drawn for an initial level of 100
Payoff = Callable[[FloatArray], FloatArray]  # underlying at maturity (% of initial) -> paid


@dataclass(frozen=True)
class Product:
    """A product of the gallery.

    Attributes
    ----------
    slug
        URL part, unique.
    name
        Name shown on the card.
    family, group
        Gallery section and, inside it, sub-group (empty when the section has none).
    payoff
        Payoff at maturity against the underlying in % of its initial level.
    notional_line
        True when the payoff reads in % of a notional (notes): the thumbnail then shows the
        delta-one line; false for options, whose payoff starts at zero.
    marker
        Level (% of initial) drawn as a dashed vertical line, e.g. a barrier.
    """

    slug: str
    name: str
    family: str
    group: str
    payoff: Payoff
    notional_line: bool = False
    marker: float | None = None


def slugify(text: str) -> str:
    """Lower-case URL slug of a name."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _of(payoff: Callable[[FloatArray], FloatArray]) -> Payoff:
    def at(s: FloatArray) -> FloatArray:
        return np.asarray(payoff(s), dtype=np.float64)

    return at


def _never_breached(note: CouponNote) -> Payoff:
    def at(p: FloatArray) -> FloatArray:
        return np.asarray(note.maturity_payoff(p, False), dtype=np.float64)

    return at


def _strategies() -> list[Product]:
    out = []
    for spec in CATALOG:
        strategy = spec.build([SPOT * k / 100.0 for k in spec.default_strikes_pct], 1.0, 0.1 * SPOT)
        out.append(
            Product(
                slugify(spec.name),
                spec.name,
                OPTIONS,
                spec.family.value,
                _of(strategy.payoff),
            )
        )
    return out


def _barriers() -> list[Product]:
    out = []
    for kind in ("call", "put"):
        for barrier_type in BARRIER_TYPES:
            level = 130.0 if barrier_type.startswith("up") else 70.0
            option = BarrierOption(kind, barrier_type, SPOT, level, 1.0, Monitoring.CONTINUOUS)
            # a knock-out is drawn as paid when never touched, a knock-in once activated
            shape = option.payoff if not option.is_in else EuropeanOption(kind, SPOT, 1.0).payoff
            name = f"{barrier_type.capitalize()} {kind}"
            out.append(
                Product(
                    slugify(name),
                    name,
                    BARRIERS,
                    "Calls" if kind == "call" else "Puts",
                    _of(shape),
                    marker=level,
                )
            )
    return out


def _capital_protection() -> list[Product]:
    out = []
    for spec in NOTE_CATALOG:
        note = spec.build({}, 3.0, SPOT)
        out.append(
            Product(
                slugify(spec.name),
                spec.name,
                CAPITAL_PROTECTION,
                "",
                _of(note.payoff),
                notional_line=True,
            )
        )
    return out


def _coupon_notes() -> list[Product]:
    out = []
    for spec in COUPON_NOTE_CATALOG:
        note = spec.build({}, 3.0, 4)
        out.append(
            Product(
                slugify(spec.name),
                spec.name,
                YIELD if spec.family == YIELD else AUTOCALLS,
                "",
                _never_breached(note),
                notional_line=True,
            )
        )
    return out


PRODUCTS: tuple[Product, ...] = tuple(
    _strategies() + _barriers() + _capital_protection() + _coupon_notes()
)
BY_SLUG = {p.slug: p for p in PRODUCTS}


def find(slug: str) -> Product | None:
    """The product with this slug, None if there is none."""
    return BY_SLUG.get(slug)


def matching(query: str | None) -> list[Product]:
    """Products whose name, family or group contains every word of ``query``."""
    words = (query or "").lower().split()
    return [
        p for p in PRODUCTS if all(w in f"{p.name} {p.family} {p.group}".lower() for w in words)
    ]
