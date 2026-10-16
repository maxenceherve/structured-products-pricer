"""Plain-English reading of the Greeks, computed from the numbers on screen.

Every sentence is built from values the page has just computed (signs, peaks, sign changes,
ratios), so the explanation always matches what the charts show.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from spricer._typing import FloatArray
from spricer.greeks_lab import Moneyness, peak, sign_changes


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


def note_reading(
    names: Sequence[str], deltas: Sequence[float], vegas: Sequence[float]
) -> list[str]:
    """How the investor's position in a coupon note reacts, underlying by underlying."""
    out = []
    for name, delta, vega in zip(names, deltas, vegas, strict=True):
        direction = "gains" if delta > 0 else "loses"
        line = f"**{name}**: the investor {direction} {abs(delta) / 100:.2f} % per +1 % move"
        line += " (long the underlying through the put sold)." if delta > 0 else "."
        if vega < 0:
            line += " Volatility hurts: more of it means more chances of breaching the barrier."
        elif vega > 0:
            line += " Volatility helps here (more chances of reaching the autocall level)."
        out.append(line)
    if len(names) > 1:
        weakest = names[int(np.argmax(np.abs(np.asarray(deltas))))]
        out.append(
            f"The note is most exposed to **{weakest}**: in a worst-of the delta concentrates on "
            "the underlying most likely to end as the worst."
        )
    return out
