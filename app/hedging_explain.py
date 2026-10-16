"""Plain-English reading of the hedges, simulations and P&L explains, from the numbers."""

from __future__ import annotations

from spricer.hedging import StaticHedge
from spricer.risk import Attribution


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
    out.append(
        "A **market maker** who sold this option runs the hedge: shares every day, listed options "
        "for the gamma and the vega, and earns the margin charged at inception."
        if short
        else "A **buyer** who wants to keep the convexity hedges only the delta: each rebalancing "
        "locks in part of the large moves (gamma scalping), paid for by the time decay."
    )
    return out


def simulation_reading(
    mean: float,
    theory: float,
    std: float,
    derman_kamal: float,
    realised_pct: float,
    implied_pct: float,
    long_gamma: bool,
    cost_bp: float,
) -> list[str]:
    """What the simulated hedged P&L shows."""
    out = []
    if abs(realised_pct - implied_pct) < 1e-9:
        out.append(
            f"Realised volatility equals the implied one: on average the hedge breaks even "
            f"(mean {mean:+.3f} %, theory {theory:+.3f} %). What remains is noise from hedging "
            "at discrete dates."
        )
    else:
        better = (realised_pct > implied_pct) == long_gamma
        out.append(
            f"The spot moves with {realised_pct:.1f} % volatility while the option was priced at "
            f"{implied_pct:.1f} %: the {'long' if long_gamma else 'short'} gamma position "
            f"{'earns' if better else 'loses'} {abs(mean):.3f} % on average, close to the "
            f"theoretical {theory:+.3f} % (the price gap between the two volatilities)."
        )
    out.append(
        f"Standard deviation {std:.3f} %, against {derman_kamal:.3f} % by the Derman-Kamal "
        "formula: hedging more often divides the noise by the square root of the frequency "
        "(daily instead of weekly: about 2.2 times less)."
    )
    if cost_bp > 0:
        out.append(
            f"Transaction costs of {cost_bp:g} bp are paid on every rebalancing: hedging more "
            "often reduces the noise but costs more. A desk picks the frequency that balances both."
        )
    return out


def attribution_reading(attribution: Attribution) -> list[str]:
    """How much of the P&L the Greeks explain."""
    full, taylor, residual = attribution.full, attribution.taylor, attribution.residual
    share = 100.0 * taylor / full if abs(full) > 1e-9 else 100.0
    out = [
        f"The Greeks at today's market explain **{taylor:+.3f} %** of a full revaluation of "
        f"**{full:+.3f} %** ({share:.0f} %). The unexplained residual is {residual:+.3f} %."
    ]
    if abs(residual) > max(0.05, 0.1 * abs(full)):
        out.append(
            "The residual is large: the move is too big for a second-order expansion (the third "
            "order grows like the cube of the move), or the payoff jumps nearby. Stress tests "
            "must be revalued, as in the matrix below."
        )
    else:
        out.append(
            "The residual is small: for moves of this size the Greeks are a reliable P&L explain."
        )
    biggest = max(attribution.terms, key=lambda k: abs(attribution.terms[k]))
    out.append(
        f"The largest contribution comes from **{biggest}** ({attribution.terms[biggest]:+.3f} %)."
    )
    return out
