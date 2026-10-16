"""Product finder: from an investor's view and constraints to products that fit, and why.

The logic is the one taught in structuring courses and set out by the EUSIPA product map: a
product is chosen from a market view, a level of capital protection, an objective (income or
participation) and an opinion on volatility. Each product of the app carries those features
(:class:`ProductCard`); a profile is matched against them with transparent, additive points, and
every point comes with the sentence that justifies it. Nothing is hidden in the score.

Points:

* view: up to 4 points, how well the payoff fits the expected move;
* protection: -10 when the product protects less than required (and a warning), which ranks
  it after every product that respects the constraint (the other points add up to at most 7),
  -0.5 when it protects more (the extra protection is paid for with return);
* objective: 1.5 when the product pays the way the investor wants (coupons or participation);
* volatility: ±1 when the product is long or short volatility and the investor has a view;
* horizon: 0.5 inside the product's usual range of maturities, -1 outside (and a warning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class View(StrEnum):
    """Expected move of the underlying over the horizon."""

    BULLISH = "Strongly bullish"
    MILD_BULL = "Moderately bullish"
    RANGE = "Range-bound"
    MILD_BEAR = "Moderately bearish"
    BEARISH = "Strongly bearish"
    BIG_MOVE = "Big move, either way"


class Protection(StrEnum):
    """Capital protection, from the safest to the riskiest."""

    FULL = "Full capital protection"
    PARTIAL = "Partial protection (90 %)"
    CONDITIONAL = "Protected unless a barrier is breached"
    NONE = "Capital at risk"


PROTECTION_PENALTY = 10.0
PROTECTION_RANK = {
    Protection.FULL: 3,
    Protection.PARTIAL: 2,
    Protection.CONDITIONAL: 1,
    Protection.NONE: 0,
}


class Objective(StrEnum):
    """How the investor wants to be paid."""

    INCOME = "Regular income"
    GROWTH = "Participation in the move"


class VolView(StrEnum):
    """Opinion on implied volatility."""

    RISE = "Volatility will rise"
    NEUTRAL = "No view on volatility"
    FALL = "Volatility will fall"


@dataclass(frozen=True)
class Profile:
    """What the investor wants."""

    view: View
    protection: Protection
    objective: Objective
    vol_view: VolView = VolView.NEUTRAL
    horizon: float = 3.0


@dataclass(frozen=True)
class ProductCard:
    """Features of a product of the app.

    Attributes
    ----------
    name
        Product name, as on its page.
    page
        App page where it is priced ("Vanilla options", "Capital protection"...).
    views
        Fit of the payoff to each view, from 0 to 1.
    protection
        Protection the product offers.
    objective
        How it pays.
    vega
        +1 long volatility, -1 short volatility, 0 roughly neutral.
    horizon
        Usual range of maturities in years.
    how
        How the product makes money, in one sentence.
    risks
        What can go wrong, in one sentence.
    """

    name: str
    page: str
    views: dict[View, float]
    protection: Protection
    objective: Objective
    vega: int
    horizon: tuple[float, float]
    how: str
    risks: str


@dataclass(frozen=True)
class Recommendation:
    """A product, its points and the sentences behind them."""

    card: ProductCard
    score: float
    reasons: tuple[str, ...] = field(default=())
    warnings: tuple[str, ...] = field(default=())


V = View
CARDS: tuple[ProductCard, ...] = (
    ProductCard(
        "Call",
        "Vanilla options",
        {V.BULLISH: 1.0, V.MILD_BULL: 0.5},
        Protection.NONE,
        Objective.GROWTH,
        +1,
        (0.1, 3.0),
        "pays all the rise above the strike for a premium paid upfront",
        "the whole premium is lost if the spot ends below the strike",
    ),
    ProductCard(
        "Bull call spread",
        "Vanilla options",
        {V.MILD_BULL: 1.0, V.BULLISH: 0.6},
        Protection.NONE,
        Objective.GROWTH,
        0,
        (0.1, 3.0),
        "pays the rise between two strikes, cheaper than a call because the upside is sold",
        "the premium is lost below the lower strike; gains are capped at the upper one",
    ),
    ProductCard(
        "Put",
        "Vanilla options",
        {V.BEARISH: 1.0, V.MILD_BEAR: 0.5},
        Protection.NONE,
        Objective.GROWTH,
        +1,
        (0.1, 3.0),
        "pays all the fall below the strike for a premium paid upfront",
        "the whole premium is lost if the spot ends above the strike",
    ),
    ProductCard(
        "Bear put spread",
        "Vanilla options",
        {V.MILD_BEAR: 1.0, V.BEARISH: 0.6},
        Protection.NONE,
        Objective.GROWTH,
        0,
        (0.1, 3.0),
        "pays the fall between two strikes, cheaper than a put",
        "the premium is lost above the upper strike; gains are capped at the lower one",
    ),
    ProductCard(
        "Straddle",
        "Vanilla options",
        {V.BIG_MOVE: 1.0},
        Protection.NONE,
        Objective.GROWTH,
        +1,
        (0.1, 2.0),
        "a call and a put at the same strike: pays a large move either way",
        "both premiums are lost if the spot stays near the strike; time decay is fast",
    ),
    ProductCard(
        "Iron condor",
        "Vanilla options",
        {V.RANGE: 1.0},
        Protection.NONE,
        Objective.INCOME,
        -1,
        (0.1, 1.0),
        "sells options around the spot and buys wings further out: earns the premium if the "
        "spot stays in a range",
        "loses up to the width of the wings minus the premium if the spot leaves the range",
    ),
    ProductCard(
        "Covered call",
        "Vanilla options",
        {V.RANGE: 0.8, V.MILD_BULL: 0.7},
        Protection.NONE,
        Objective.INCOME,
        -1,
        (0.1, 2.0),
        "holds the stock and sells a call: the premium is an income, the upside above the "
        "strike is given away",
        "the stock can fall, and the premium only cushions the loss",
    ),
    ProductCard(
        "Participation note",
        "Capital protection",
        {V.BULLISH: 0.9, V.MILD_BULL: 0.7},
        Protection.FULL,
        Objective.GROWTH,
        +1,
        (1.0, 10.0),
        "a bond pays the capital back, the interest it saves buys a call spread",
        "the capital comes back only at maturity, and if the issuer is solvent; the upside is "
        "capped and partial",
    ),
    ProductCard(
        "Bearish participation note",
        "Capital protection",
        {V.BEARISH: 0.9, V.MILD_BEAR: 0.7},
        Protection.FULL,
        Objective.GROWTH,
        +1,
        (1.0, 10.0),
        "a bond pays the capital back and a put spread pays part of the fall",
        "nothing beyond the capital if the market rises; issuer credit risk",
    ),
    ProductCard(
        "Shark note (up)",
        "Capital protection",
        {V.MILD_BULL: 1.0, V.BULLISH: 0.4},
        Protection.FULL,
        Objective.GROWTH,
        -1,
        (1.0, 5.0),
        "a higher participation than a plain note, because the call is knocked out above a "
        "barrier (a rebate is paid instead)",
        "a strong rally above the barrier only earns the rebate",
    ),
    ProductCard(
        "Twin-win note",
        "Capital protection",
        {V.BIG_MOVE: 0.9},
        Protection.FULL,
        Objective.GROWTH,
        +1,
        (1.0, 7.0),
        "a bond plus a call spread and a put spread: gains on a large move up or down",
        "nothing beyond the capital if the spot ends near its initial level",
    ),
    ProductCard(
        "Digital coupon note",
        "Capital protection",
        {V.MILD_BULL: 0.8, V.RANGE: 0.5},
        Protection.FULL,
        Objective.INCOME,
        0,
        (1.0, 5.0),
        "a bond plus a digital: a fixed coupon if the underlying ends above a level",
        "no coupon at all if it ends below, however close",
    ),
    ProductCard(
        "Bonus note",
        "Capital protection",
        {V.MILD_BULL: 0.8, V.RANGE: 0.6},
        Protection.PARTIAL,
        Objective.GROWTH,
        -1,
        (1.0, 5.0),
        "pays at least a bonus level if the barrier holds, plus a share of the rise",
        "below the barrier only the partial protection is left",
    ),
    ProductCard(
        "Reverse convertible",
        "Yield enhancement",
        {V.RANGE: 0.8, V.MILD_BULL: 0.8},
        Protection.NONE,
        Objective.INCOME,
        -1,
        (0.5, 3.0),
        "a high certain coupon, paid for by a put sold on the underlying",
        "below the strike at maturity the capital falls with the underlying",
    ),
    ProductCard(
        "Barrier reverse convertible",
        "Yield enhancement",
        {V.RANGE: 0.9, V.MILD_BULL: 0.8, V.MILD_BEAR: 0.3},
        Protection.CONDITIONAL,
        Objective.INCOME,
        -1,
        (0.5, 3.0),
        "a certain coupon, paid for by a down-and-in put: the capital is safe while the "
        "barrier holds",
        "after a breach the capital falls with the underlying below the strike",
    ),
    ProductCard(
        "Athena autocall",
        "Autocallables",
        {V.MILD_BULL: 1.0, V.RANGE: 0.7},
        Protection.CONDITIONAL,
        Objective.INCOME,
        -1,
        (1.0, 10.0),
        "called early with all the coupons as soon as the underlying is back at its initial level",
        "no coupon while it stays below, and capital at risk below the barrier at maturity",
    ),
    ProductCard(
        "Phoenix autocall",
        "Autocallables",
        {V.RANGE: 1.0, V.MILD_BULL: 0.8, V.MILD_BEAR: 0.4},
        Protection.CONDITIONAL,
        Objective.INCOME,
        -1,
        (1.0, 10.0),
        "a coupon at each date above a low coupon barrier, called early above the initial level",
        "coupons stop below the coupon barrier, and the capital is at risk below the "
        "protection barrier",
    ),
)

_VOL_SIGN = {VolView.RISE: 1, VolView.NEUTRAL: 0, VolView.FALL: -1}


def _score(card: ProductCard, profile: Profile) -> Recommendation:
    reasons: list[str] = []
    warnings: list[str] = []
    fit = card.views.get(profile.view, 0.0)
    score = 4.0 * fit
    if fit >= 0.75:
        reasons.append(f"Fits a {profile.view.value.lower()} view: it {card.how}.")
    elif fit > 0.0:
        reasons.append(f"Partly fits a {profile.view.value.lower()} view: it {card.how}.")

    wanted, offered = PROTECTION_RANK[profile.protection], PROTECTION_RANK[card.protection]
    if offered < wanted:
        score -= PROTECTION_PENALTY
        warnings.append(
            f"Protects less than you asked: {card.protection.value.lower()} instead of "
            f"{profile.protection.value.lower()}."
        )
    elif offered > wanted:
        score -= 0.5
        reasons.append(f"{card.protection.value}: more than you need, paid for with some return.")
    else:
        reasons.append(f"{card.protection.value}, as you asked.")

    if card.objective is profile.objective:
        score += 1.5
        reasons.append(
            "Pays regular income."
            if card.objective is Objective.INCOME
            else "Pays a share of the move."
        )

    sign = _VOL_SIGN[profile.vol_view]
    if sign and card.vega:
        score += sign * card.vega
        if sign == card.vega:
            reasons.append(
                "Long volatility: it gains if volatility rises, as you expect."
                if card.vega > 0
                else "Short volatility: it is cheap to build when volatility is high and "
                "gains if it falls, as you expect."
            )
        else:
            warnings.append(
                "Long volatility, against your view: it loses value if volatility falls."
                if card.vega > 0
                else "Short volatility, against your view: it loses value if volatility rises."
            )

    low, high = card.horizon
    if low <= profile.horizon <= high:
        score += 0.5
    else:
        score -= 1.0
        warnings.append(f"Usually built for {low:g} to {high:g} years, not {profile.horizon:g}.")
    warnings.append(f"Risk: {card.risks}.")
    return Recommendation(card, score, tuple(reasons), tuple(warnings))


def recommend(profile: Profile, top: int | None = None) -> list[Recommendation]:
    """Every product scored against ``profile``, best first (ties keep the catalog order)."""
    if not profile.horizon > 0.0:
        raise ValueError("the horizon must be positive")
    ranked = sorted((_score(c, profile) for c in CARDS), key=lambda r: -r.score)
    return ranked if top is None else ranked[:top]


def find_card(name: str) -> ProductCard:
    """The card with this name."""
    for card in CARDS:
        if card.name == name:
            return card
    raise KeyError(name)
