"""Plain-English reading of the Greeks, hedges and P&L explains, from the numbers on screen.

Every sentence is built from values the page has just computed (signs, peaks, sign changes,
ratios), so the explanation always matches what the charts and tables show. Sentences use
Markdown bold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from spricer._typing import FloatArray
from spricer.greeks_lab import Moneyness, peak, sign_changes
from spricer.hedging import StaticHedge
from spricer.risk import Attribution


@dataclass(frozen=True)
class Situation:
    """What the explanations need to know about the instrument on screen.

    Attributes
    ----------
    instrument
        Name shown to the user ("Call", "Barrier option"...).
    family
        "vanilla", "digital", "spread", "volatility" or "barrier".
    short
        True for a short position.
    moneyness
        ATM/ITM/OTM of the main strike, None for several strikes.
    knock_out
        True for a knock-out barrier option.
    """

    instrument: str
    family: str
    short: bool
    moneyness: Moneyness | None = None
    knock_out: bool = False


def _fmt_spots(xs: Sequence[float]) -> str:
    return ", ".join(f"{x:.0f} %" for x in xs)


def first_order(
    situation: Situation,
    now: Mapping[str, float],
    spots_pct: FloatArray,
    profile: Mapping[str, FloatArray],
    quarter_time_gamma: float | None,
) -> list[str]:
    """Sentences on price, delta, gamma, vega and theta at today's spot."""
    out: list[str] = []
    if max(abs(float(v)) for v in now.values()) < 1e-6:
        return [
            "The position is worth nothing here and none of its Greeks moves: for a barrier "
            "option, the barrier is probably already crossed (knocked out) or out of reach."
        ]
    delta, gamma = now["Delta (%)"], now["Gamma (%)"]
    vega, theta = now["Vega (%)"], now["Theta (%)"]
    side = "short" if situation.short else "long"

    if situation.moneyness is not None:
        what = situation.instrument.lower()
        out.append(f"**{situation.moneyness.value}.** The position is {side} {what}.")

    shares = abs(delta) / 100.0
    direction = "rises" if delta > 0 else "falls"
    line = (
        f"**Delta {delta:+.1f} %**: the position moves like {shares:.2f} share, and gains when "
        f"the spot {direction}."
    )
    if situation.family == "vanilla":
        line += (
            " Deep in the money a vanilla's delta tends to ±100 % (it behaves like the stock), "
            "far out of the money to 0, and at the money it is close to 50 %: the Black-Scholes "
            "delta is close to the probability of ending in the money."
        )
    elif situation.family == "digital":
        line += (
            f" A digital's delta is largest at {peak(spots_pct, profile['Delta (%)']):.0f} % of "
            "today's spot, around the strike, where a small move decides whether it pays."
        )
    out.append(line)

    gamma_peak = peak(spots_pct, profile["Gamma (%)"])
    line = (
        f"**Gamma {gamma:+.2f}**: delta changes by {gamma:+.2f} points for +1 % spot. "
        f"Gamma is largest at {gamma_peak:.0f} % of today's spot"
    )
    gamma_flips = sign_changes(spots_pct, profile["Gamma (%)"])
    if gamma_flips:
        line += (
            f" and changes sign at {_fmt_spots(gamma_flips)}: the position is convex on one side "
            "and concave on the other, which a single vanilla never is."
        )
    else:
        line += "."
    if quarter_time_gamma is not None and abs(gamma) > 1e-6:
        ratio = quarter_time_gamma / gamma
        if ratio > 1.2:
            line += (
                f" With a quarter of the time left it would be {quarter_time_gamma:+.2f} "
                f"({ratio:.1f}x): at the money gamma grows like 1/sqrt(T), so short-dated "
                "options are the hardest to hedge."
            )
    out.append(line)

    vega_flips = sign_changes(spots_pct, profile["Vega (%)"])
    line = f"**Vega {vega:+.3f} %** per volatility point: "
    if situation.knock_out and vega < 0:
        line += (
            "negative, because more volatility mostly adds chances of touching the barrier and "
            "losing the option."
        )
    elif vega_flips:
        line += (
            f"it changes sign at {_fmt_spots(vega_flips)}. Below that level more volatility "
            "helps one side of the position, above it the other: volatility does not simply add "
            "value here."
        )
    else:
        line += f"largest at {peak(spots_pct, profile['Vega (%)']):.0f} % of today's spot."
        if situation.family in ("vanilla", "volatility"):
            line += (
                " A vanilla's vega is largest at the money and grows with the square root of "
                "the maturity."
            )
    out.append(line)

    line = f"**Theta {theta:+.4f} %** a day. "
    if (gamma > 1e-6 and theta < 0) or (gamma < -1e-6 and theta > 0):
        line += (
            "Theta pays for gamma: being long convexity costs time value every day, being "
            "short it earns that time value."
        )
    elif abs(gamma) > 1e-6:
        line += (
            "Here theta and gamma have the same sign: far from the strike the carry of the "
            "position (rates and dividends) outweighs the time value."
        )
    else:
        line += "Without convexity, theta is only the carry of the position."
    out.append(line)

    if situation.short:
        out.append(
            "**Short position**: every Greek flips sign. You receive the premium and the time "
            "value, and you lose when the spot moves sharply (short gamma) or volatility rises "
            "(short vega): this is the market maker's side, which must be hedged."
        )
    return out


def second_order(now: Mapping[str, float]) -> list[str]:
    """Sentences on vanna, volga, charm and speed at today's spot."""
    vanna, volga, charm, speed = now["Vanna"], now["Volga"], now["Charm"], now["Speed"]
    return [
        f"**Vanna {vanna:+.2f}**: delta moves by {vanna:+.2f} points for +1 volatility point. "
        + (
            "More volatility raises the delta here: the payoff becomes likelier to be reached "
            "from below, so the position leans more on the spot."
            if vanna > 0
            else "More volatility lowers the delta here: the outcome becomes less certain, so "
            "the position leans less on the spot."
        ),
        f"**Volga {volga:+.4f}**: vega changes by {volga:+.4f} for +1 volatility point. "
        + (
            "Positive: the position gains more and more as volatility rises (typical of options "
            "away from the strike)."
            if volga > 0
            else "Negative or small: vega shrinks as volatility rises."
        ),
        f"**Charm {charm:+.3f}**: change of delta after one day, with nothing else moving. "
        "Out of the money the delta drifts toward 0 as time passes, in the money toward ±100 %: "
        "a delta hedge goes stale even if the spot does not move.",
        f"**Speed {speed:+.3f}**: change of gamma for +1 % spot. Gamma is not constant: near "
        "the strike it changes fast, so the hedge must be rebalanced more often there.",
    ]


# --- Hedging and P&L ---------------------------------------------------------------------------


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


def simulation_reading(
    mean: float,
    theory: float,
    std: float,
    derman_kamal: float,
    realised_pct: float,
    implied_pct: tuple[float, float],
    long_gamma: bool,
    cost_bp: float,
) -> list[str]:
    """What the simulated hedged P&L shows.

    ``implied_pct`` is the range of the implied volatilities of the legs: one value when the
    whole position is priced at one volatility, a range on a smile.
    """
    out = []
    low, high = implied_pct
    one_vol = abs(high - low) < 1e-9
    if one_vol and abs(realised_pct - low) < 1e-9:
        out.append(
            f"Realised volatility equals the implied one: on average the hedge breaks even "
            f"(mean {mean:+.3f} %, theory {theory:+.3f} %). What remains is noise from hedging "
            "at discrete dates."
        )
    elif one_vol:
        better = (realised_pct > low) == long_gamma
        out.append(
            f"The spot moves with {realised_pct:.1f} % volatility while the option was priced at "
            f"{low:.1f} %: the {'long' if long_gamma else 'short'} gamma position "
            f"{'earns' if better else 'loses'} {abs(mean):.3f} % on average, close to the "
            f"theoretical {theory:+.3f} % (the price gap between the two volatilities)."
        )
    else:
        out.append(
            f"The legs are priced on the smile, from {low:.1f} % to {high:.1f} % volatility, while "
            f"the spot moves with {realised_pct:.1f} %: no single volatility matches every leg. "
            f"On average the hedge earns the value at the realised volatility minus the price "
            f"paid, {theory:+.3f} % in theory against {mean:+.3f} % simulated."
        )
    if one_vol:
        out.append(
            f"Standard deviation {std:.3f} %, against {derman_kamal:.3f} % by the Derman-Kamal "
            "formula: hedging more often divides the noise by the square root of the frequency "
            "(daily instead of weekly: about 2.2 times less)."
        )
    else:
        out.append(
            f"Standard deviation {std:.3f} %. The Derman-Kamal formula ({derman_kamal:.3f} %) "
            "only covers the noise of hedging at discrete dates; here most of it comes from the "
            "gap between the realised volatility and each leg's implied one, which pays "
            "differently on every path (it depends on where the spot spends its time)."
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
