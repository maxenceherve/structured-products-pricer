"""Plain-English reading of the Greeks and hedges, from the numbers on screen.

Every sentence is built from values the page has just computed (signs, peaks, sign changes),
so the explanation always matches what the charts and tables show. Sentences use Markdown bold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from spricer._typing import FloatArray
from spricer.greeks_lab import peak, sign_changes
from spricer.hedging import StaticHedge


@dataclass(frozen=True)
class Card:
    """One Greek explained: what it is worth today and what that means.

    Attributes
    ----------
    title
        Name of the Greek ("Delta").
    subtitle
        What it measures, in a few words.
    value
        Its value today, formatted.
    headline
        What the value means, in one sentence (Markdown).
    details
        Further remarks (Markdown), possibly none.
    """

    title: str
    subtitle: str
    value: str
    headline: str
    details: tuple[str, ...] = ()


def _gains(x: float) -> str:
    return "gains" if x >= 0 else "loses"


def _spots(xs: list[float]) -> str:
    return ", ".join(f"{x:.0f} %" for x in xs)


def _value(x: float, digits: int, errors: Mapping[str, float] | None, name: str) -> str:
    error = f" ± {errors[name]:.{digits}f}" if errors else ""
    return f"{x:+.{digits}f}{error} %"


def greek_cards(
    noun: str,
    now: Mapping[str, float | None],
    x_pct: FloatArray,
    profile: Mapping[str, FloatArray],
    levels: Mapping[str, float],
    axis: str,
    errors: Mapping[str, float] | None = None,
) -> tuple[list[Card], list[str]]:
    """Delta, gamma, vega, theta and rho today, the same reading for every product.

    ``noun`` names what is held ("position", "note"); ``x_pct`` and ``profile`` give the Greeks
    along the underlying, ``axis`` what its levels are a percentage of; ``levels`` the product's
    own levels, to say where the risk concentrates; ``errors`` the Monte Carlo standard errors.
    Greeks in % of notional: Delta and Gamma per +1 % on the underlying (the "1 % gamma"
    convention), Vega per volatility point, Theta per day, Rho per basis point. A theta of None
    could not be computed: an observation falls within the next day.
    """
    delta, gamma, vega = (
        float(now["Delta (%)"] or 0.0),
        float(now["Gamma (%)"] or 0.0),
        float(now["Vega (%)"] or 0.0),
    )
    theta, rho = now["Theta (%)"], float(now["Rho (%)"] or 0.0)
    if max(abs(delta), abs(gamma), abs(vega), abs(theta or 0.0), abs(rho)) < 1e-6:
        return [], [
            f"The {noun} is worth nothing here and none of its Greeks moves: a barrier is "
            "probably already crossed (knocked out) or out of reach."
        ]
    where = peak(x_pct, profile["Gamma (%)"])
    gamma_details = [f"It is largest at **{where:.0f} %** of {axis}"]
    if levels:
        name, level = min(levels.items(), key=lambda kv: abs(kv[1] - where))
        gamma_details[0] += f", next to the {name.lower()} ({level:g} %)"
    gamma_details[0] += ": there the hedge must be adjusted most often."
    # a Monte Carlo gamma crosses zero through noise alone
    flips = [] if errors else sign_changes(x_pct, profile["Gamma (%)"])
    if flips:
        gamma_details.append(
            f"It changes sign at {_spots(flips)}: convex on one side, concave on the other."
        )
    vega_detail = (
        f"More volatility costs the {noun}: the options it has sold gain more value than the "
        "options it holds."
        if vega < 0
        else f"More volatility helps the {noun}, which owns more optionality than it sells."
    )
    cards = [
        Card(
            "Delta",
            "exposure to the underlying",
            _value(delta, 1, errors, "Delta (%)"),
            f"If the underlying rises by 1 %, the {noun} {_gains(delta)} "
            f"**{abs(delta) / 100:.3f} %** of notional.",
            (
                f"It moves like **{abs(delta) / 100:.2f} share** of the underlying: what a hedger "
                "holds against it.",
            ),
        ),
        Card(
            "Gamma",
            "how fast the delta moves",
            _value(gamma, 2, errors, "Gamma (%)"),
            f"If the underlying rises by 1 %, the delta changes by **{gamma:+.2f} %**.",
            tuple(gamma_details),
        ),
        Card(
            "Vega",
            "exposure to volatility",
            _value(vega, 3, errors, "Vega (%)"),
            f"If volatility rises by 1 point, the {noun} {_gains(vega)} **{abs(vega):.3f} %** of "
            "notional.",
            (vega_detail,),
        ),
    ]
    if theta is None:
        cards.append(
            Card(
                "Theta",
                "the cost of time",
                "n/a",
                "Not computed: an observation falls within the next day.",
            )
        )
    else:
        cards.append(
            Card(
                "Theta",
                "the cost of time",
                _value(theta, 4, errors, "Theta (%)"),
                f"Each day that passes, the {noun} {_gains(theta)} **{abs(theta):.4f} %** of "
                "notional, the market unchanged.",
                (
                    "Theta pays for gamma: being long convexity costs time value, being short it "
                    "earns it."
                    if gamma * theta < 0
                    else "Here it comes mostly from the carry: rates, dividends and funding.",
                ),
            )
        )
    cards.append(
        Card(
            "Rho",
            "exposure to the rate",
            _value(rho, 4, errors, "Rho (%)"),
            f"If the rate rises by 1 basis point, the {noun} {_gains(rho)} "
            f"**{abs(rho):.4f} %** of notional.",
            ("The rate moves both the discounting of what is paid and the underlying's forward.",),
        )
    )
    return cards, []


# --- Hedging -----------------------------------------------------------------------------------


def _verb(quantity: float) -> str:
    return "Buy" if quantity > 0 else "Sell"


def static_reading(hedge: StaticHedge, options: int, short: bool) -> list[str]:
    """What each leg of the hedge does, and who would run it."""
    out = []
    gamma = float(hedge.position.gamma)
    if abs(gamma) > 1e-12:
        side = "long" if gamma > 0 else "short"
        effect = (
            "earns on large moves either way" if gamma > 0 else "loses on large moves either way"
        )
        out.append(f"The position is **{side} gamma**: once delta-hedged, it {effect}.")
    for leg in hedge.legs:
        units = leg.quantity * options
        if leg.name == "Underlying":
            out.append(
                f"**{_verb(units)} {abs(units):,.0f} shares**: they offset the remaining delta, so "
                "a small move of the spot no longer changes the value."
            )
        else:
            out.append(
                f"**{_verb(units)} {abs(units):,.0f} {leg.name}**: "
                + (
                    "they bring the convexity the position lacks."
                    if float(leg.greeks.gamma) * leg.quantity > 0
                    else "they sell the convexity the position has in excess."
                )
            )
    if len(hedge.legs) == 3:
        out.append(
            "Two options are needed for gamma and vega: short-dated options carry mostly gamma, "
            "long-dated ones mostly vega, so a mix of the two can neutralise both."
        )
    if short:
        out.append(
            "A **market maker** who sold this product runs the hedge: shares every day, listed "
            "options for the gamma and the vega, and keeps the margin charged at inception."
        )
    elif gamma > 0:
        out.append(
            "A **buyer** who wants to keep the convexity hedges only the delta: each rebalancing "
            "locks in part of the large moves (gamma scalping), paid for by the time decay."
        )
    else:
        out.append(
            "A **buyer** of this position is short gamma at today's spot: hedging only the delta "
            "earns the time decay but loses on large moves, so a desk buys options to cap it."
        )
    return out
