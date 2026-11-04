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
    """How a product protects the capital."""

    FULL = "Full capital protection"
    PARTIAL = "Partial protection"
    CONDITIONAL = "Protected unless a barrier is breached"
    NONE = "Capital at risk"


PROTECTION_PENALTY = 10.0
MORE_THAN_NEEDED = 25.0  # % of capital: protection this much beyond the need costs return


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
    """What the investor expects and accepts.

    Attributes
    ----------
    view, objective, vol_view, horizon
        Expected move, how to be paid, view on volatility, years.
    max_loss
        Largest loss of capital accepted at maturity, in % (0: full protection).
    accept_barrier
        A product protected only while a barrier holds is acceptable.
    """

    view: View
    max_loss: float
    objective: Objective
    vol_view: VolView = VolView.NEUTRAL
    horizon: float = 3.0
    accept_barrier: bool = False


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
    partial_loss
        For a partial protection, the largest loss of capital at maturity in % (with the
        product's usual terms).
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
    partial_loss: float = 100.0

    @property
    def worst_loss(self) -> float:
        """Largest loss of capital at maturity, in %, with the product's usual terms."""
        if self.protection is Protection.FULL:
            return 0.0
        if self.protection is Protection.PARTIAL:
            return self.partial_loss
        return 100.0

    @property
    def protection_text(self) -> str:
        """The protection in words, with its level."""
        if self.protection is Protection.PARTIAL:
            return f"Loss limited to {self.partial_loss:g} %"
        return self.protection.value


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
        partial_loss=10.0,
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
    ProductCard(
        "Shark note (down)",
        "Capital protection",
        {V.MILD_BEAR: 1.0, V.BEARISH: 0.4},
        Protection.FULL,
        Objective.GROWTH,
        -1,
        (1.0, 5.0),
        "a higher participation in the fall than a plain bearish note, because the put is "
        "knocked out below a barrier (a rebate is paid instead)",
        "a crash below the barrier only earns the rebate",
    ),
    ProductCard(
        "Outperformance certificate",
        "Participation certificates",
        {V.BULLISH: 1.0, V.MILD_BULL: 0.6},
        Protection.NONE,
        Objective.GROWTH,
        +1,
        (1.0, 5.0),
        "holds the underlying and more than its rise above the strike, paid for with the dividends",
        "all of the fall is borne, as with the underlying",
    ),
    ProductCard(
        "Barrier outperformance certificate",
        "Participation certificates",
        {V.BULLISH: 0.9, V.MILD_BULL: 0.7},
        Protection.CONDITIONAL,
        Objective.GROWTH,
        0,
        (1.0, 5.0),
        "more than the rise above the strike, and the capital back as long as the barrier holds",
        "after a breach the capital falls with the underlying",
    ),
    ProductCard(
        "Bonus certificate",
        "Participation certificates",
        {V.MILD_BULL: 0.9, V.RANGE: 0.8},
        Protection.CONDITIONAL,
        Objective.GROWTH,
        -1,
        (1.0, 5.0),
        "pays at least the bonus level while the barrier holds, and a share of the rise above",
        "once the barrier is breached the certificate is the underlying, losses included",
    ),
    ProductCard(
        "Twin-win certificate",
        "Participation certificates",
        {V.BIG_MOVE: 0.6, V.MILD_BULL: 0.6, V.MILD_BEAR: 0.5},
        Protection.CONDITIONAL,
        Objective.GROWTH,
        0,
        (1.0, 5.0),
        "gains on a move either way as long as the barrier holds",
        "a fall through the barrier turns the gains on the way down into losses",
    ),
    ProductCard(
        "Airbag certificate",
        "Participation certificates",
        {V.MILD_BULL: 0.9, V.BULLISH: 0.6, V.MILD_BEAR: 0.3},
        Protection.PARTIAL,
        Objective.GROWTH,
        0,
        (1.0, 5.0),
        "the rise of the underlying, the capital back down to the airbag level, then a "
        "cushioned loss",
        "below the airbag level the loss grows, and the upside can be capped",
        partial_loss=100.0,
    ),
    ProductCard(
        "Buffer certificate",
        "Participation certificates",
        {V.MILD_BULL: 0.9, V.BULLISH: 0.6, V.MILD_BEAR: 0.3},
        Protection.PARTIAL,
        Objective.GROWTH,
        0,
        (1.0, 5.0),
        "the rise of the underlying, and the first losses absorbed down to the buffer",
        "below the buffer the loss is one for one",
        partial_loss=90.0,
    ),
    ProductCard(
        "Sprint certificate",
        "Participation certificates",
        {V.MILD_BULL: 1.0, V.BULLISH: 0.5},
        Protection.NONE,
        Objective.GROWTH,
        0,
        (0.5, 3.0),
        "a leveraged share of the rise up to a cap, paid for by giving up the gains above it",
        "all of the fall is borne, and the gain stops at the cap",
    ),
    ProductCard(
        "Reverse bonus certificate",
        "Participation certificates",
        {V.MILD_BEAR: 1.0, V.RANGE: 0.6, V.BEARISH: 0.4},
        Protection.CONDITIONAL,
        Objective.GROWTH,
        -1,
        (0.5, 3.0),
        "gains as the underlying falls, with a bonus as long as it never rises to the barrier",
        "a rally through the barrier removes the bonus, and losses grow as the underlying rises",
    ),
    ProductCard(
        "Discount certificate",
        "Yield enhancement",
        {V.RANGE: 0.9, V.MILD_BULL: 0.7},
        Protection.NONE,
        Objective.INCOME,
        -1,
        (0.5, 2.0),
        "buys the underlying below its price by selling the upside above a cap",
        "a fall below the purchase price is borne; the gain stops at the cap",
    ),
    ProductCard(
        "Shark warrant (call)",
        "Barrier options",
        {V.MILD_BULL: 0.9, V.BULLISH: 0.4},
        Protection.NONE,
        Objective.GROWTH,
        -1,
        (0.25, 2.0),
        "a call cheaper than a vanilla because it is cancelled above a barrier, with a rebate",
        "the premium is lost below the strike, and a rally through the barrier pays only the "
        "rebate",
    ),
    ProductCard(
        "Shark warrant (put)",
        "Barrier options",
        {V.MILD_BEAR: 0.9, V.BEARISH: 0.4},
        Protection.NONE,
        Objective.GROWTH,
        -1,
        (0.25, 2.0),
        "a put cheaper than a vanilla because it is cancelled below a barrier, with a rebate",
        "the premium is lost above the strike, and a crash through the barrier pays only the "
        "rebate",
    ),
    ProductCard(
        "Autocallable reverse convertible",
        "Autocallables",
        {V.MILD_BULL: 1.0, V.RANGE: 0.8},
        Protection.NONE,
        Objective.INCOME,
        -1,
        (1.0, 5.0),
        "a certain coupon until the note is called back at par, as soon as the underlying is "
        "back at its initial level",
        "if never called, the capital falls with the underlying below the strike",
    ),
    ProductCard(
        "Autocallable barrier reverse convertible",
        "Autocallables",
        {V.MILD_BULL: 1.0, V.RANGE: 0.9, V.MILD_BEAR: 0.3},
        Protection.CONDITIONAL,
        Objective.INCOME,
        -1,
        (1.0, 5.0),
        "a certain coupon until the note is called back at par; the capital is safe while the "
        "barrier holds",
        "after a breach, and without a call, the capital falls with the underlying",
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

    loss, accepted = card.worst_loss, profile.max_loss
    if loss <= accepted + 1e-9:
        if loss < accepted - MORE_THAN_NEEDED:
            score -= 0.5
            reasons.append(
                f"Loses at most {loss:g} % of the capital: more protection than you need, paid "
                "for with some return."
            )
        elif loss >= 100.0:
            reasons.append("Capital at risk, as you accept.")
        else:
            reasons.append(
                f"Loses at most {loss:g} % of the capital, within the {accepted:g} % you accept."
            )
    elif card.protection is Protection.CONDITIONAL and profile.accept_barrier:
        reasons.append("Capital protected unless the barrier is breached, as you accept.")
    else:
        score -= PROTECTION_PENALTY
        warnings.append(
            f"Can lose up to {loss:g} % of the capital, more than the {accepted:g} % you accept"
            + (
                " (protected only while the barrier holds)."
                if card.protection is Protection.CONDITIONAL
                else "."
            )
        )

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
    if not 0.0 <= profile.max_loss <= 100.0:
        raise ValueError("the largest loss accepted must be between 0 and 100 %")
    ranked = sorted((_score(c, profile) for c in CARDS), key=lambda r: -r.score)
    return ranked if top is None else ranked[:top]


def find_card(name: str) -> ProductCard:
    """The card with this name."""
    for card in CARDS:
        if card.name == name:
            return card
    raise KeyError(name)
