"""Participation certificates and shark warrants, priced exactly by bricks.

Certificates (EUSIPA 12xx and 13xx) hold the underlying itself and reshape its payoff with
options; the capital is not protected, at most cushioned. Levels are in % of the initial level,
amounts in % of the notional. ``U`` is the underlying (a prepaid forward), ``C``/``P`` calls and
puts, ``PDI`` a down-and-in put and ``UOC`` an up-and-out call, observed as the product says.

Most certificates share an *upside*: the underlying, with ``part`` % of its rise above the
strike ``K``, possibly capped at ``cap``::

    upside = U + (part - 100 %) C(K) - part x C(cap)

* outperformance: ``upside``: more than the rise above the strike, all the fall;
* barrier outperformance: ``upside + P(100) - PDI(100, B)``: the capital back while the barrier
  holds;
* bonus certificate: ``upside + (bonus - upside)+`` while the barrier holds, replicated by puts
  at the strike and where the upside reaches the bonus, less the same puts knocked in at ``B``;
* twin-win certificate: ``upside + (100 % + down) (P(K) - PDI(K, B))``: gains on a move either
  way while the barrier holds;
* airbag certificate: ``ZC(100) + part x [C(K) - C(cap)] - (100 / A) P(A)``: the capital back
  down to the airbag level ``A``, then a loss that reaches the whole notional only at zero;
* buffer certificate: ``ZC(100) + part x [C(K) - C(cap)] - P(B)``: the capital back down to the
  buffer ``B``, then a one-for-one loss;
* sprint certificate: ``upside`` with a participation above 100 % and a cap;
* reverse bonus certificate: ``part x [P(R) + UOC(R - bonus / part, B)] [- part x P(R - cap /
  part)]``: ``part x (R - performance)``, at least the bonus while the upper barrier holds;
* discount certificate: ``U - C(cap)``: the underlying bought at a discount, capped (EUSIPA
  1200, yield enhancement);
* shark warrant: ``part x CallUpAndOut(K, B) + rebate x OneTouch(B)`` (or the put version):
  a capped, knock-out option with a rebate, without capital protection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from spricer._typing import FloatArray
from spricer.monitoring import Monitoring, require_european
from spricer.products.capital_protection import (
    NoteParam,
    NoteSpec,
    _Builder,
    _cap,
    _non_negative,
    _positive,
)
from spricer.products.notes import Level, LevelKind, Note


def _check_upside(participation: float, strike: float, cap: float | None) -> None:
    _positive(participation=participation, strike=strike, cap=cap)
    if cap is not None and cap <= strike:
        raise ValueError("the cap must be above the strike")


def _add_upside(b: _Builder, participation: float, strike: float, cap: float | None) -> None:
    """The underlying, ``participation`` % of its rise above ``strike``, up to ``cap``."""
    b.underlying("Underlying")
    b.option("Long call (participation above the strike)", "call", strike, participation - 100.0)
    if cap is not None:
        b.option("Short call (cap)", "call", cap, -participation)


def _upside(p: FloatArray, participation: float, strike: float, cap: float | None) -> FloatArray:
    top = np.inf if cap is None else cap
    return np.asarray(
        p
        + participation / 100.0 * np.clip(p - strike, 0.0, top - strike)
        - np.clip(p - strike, 0.0, None)
    )


def _levels(strike: float, cap: float | None, *extra: Level) -> tuple[Level, ...]:
    levels = [Level("Strike", strike, LevelKind.PERFORMANCE), *extra]
    if cap is not None:
        levels.append(Level("Cap", cap, LevelKind.PERFORMANCE))
    return tuple(levels)


def outperformance_certificate(
    *,
    participation: float,
    maturity: float,
    initial: float,
    strike: float = 100.0,
    cap: float | None = None,
) -> Note:
    """The underlying, with ``participation`` % of its rise above ``strike``, up to ``cap``."""
    _check_upside(participation, strike, cap)
    b = _Builder(initial, maturity)
    _add_upside(b, participation, strike, cap)

    def formula(p: FloatArray) -> FloatArray:
        return _upside(p, participation, strike, cap)

    return b.note("Outperformance certificate", formula, _levels(strike, cap))


def barrier_outperformance_certificate(
    *,
    participation: float,
    barrier: float,
    maturity: float,
    initial: float,
    strike: float = 100.0,
    cap: float | None = None,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    observations: int | None = None,
) -> Note:
    """Outperformance above ``strike``, and the capital back while ``barrier`` holds."""
    _check_upside(participation, strike, cap)
    _positive(barrier=barrier)
    if barrier >= 100.0:
        raise ValueError("the barrier must be below the initial level")
    b = _Builder(initial, maturity)
    _add_upside(b, participation, strike, cap)
    b.option("Long put (capital back)", "put", 100.0, 100.0)
    b.barrier_option(
        "Short put down-and-in",
        "put",
        "down-and-in",
        100.0,
        barrier,
        -100.0,
        monitoring,
        observations,
    )

    def formula(p: FloatArray) -> FloatArray:
        kept = _upside(p, participation, strike, cap) + np.maximum(100.0 - p, 0.0)
        return np.asarray(np.where(p >= barrier, kept, _upside(p, participation, strike, cap)))

    barrier_level = Level("Barrier", barrier, LevelKind.PERFORMANCE)
    return b.note(
        "Barrier outperformance certificate",
        formula,
        _levels(strike, cap, barrier_level),
        monitoring,
    )


def bonus_certificate(
    *,
    bonus: float,
    barrier: float,
    cap: float | None,
    maturity: float,
    initial: float,
    participation: float = 100.0,
    strike: float = 100.0,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    observations: int | None = None,
) -> Note:
    """At least ``bonus`` while ``barrier`` holds, the upside otherwise.

    The upside is the underlying with ``participation`` % of its rise above ``strike``, up to
    ``cap``.
    """
    _check_upside(participation, strike, cap)
    _positive(bonus=bonus, barrier=barrier)
    if not barrier < bonus:
        raise ValueError("the barrier must be below the bonus level")
    # level where the upside reaches the bonus: below it the bonus is paid
    reach = bonus if bonus <= strike else strike + (bonus - strike) * 100.0 / participation
    if cap is not None and cap <= reach:
        raise ValueError(f"the cap must be above {reach:g} %, where the upside reaches the bonus")
    b = _Builder(initial, maturity)
    _add_upside(b, participation, strike, cap)
    # (bonus - upside)+: puts at the strike and where the upside reaches the bonus
    puts = (
        [(reach, 100.0)]
        if bonus <= strike
        else [(reach, participation), (strike, 100.0 - participation)]
    )
    for level, quantity in puts:
        b.option(f"Long put (bonus) K {level:g}%", "put", level, quantity)
        b.barrier_option(
            f"Short put down-and-in (bonus lost) K {level:g}%",
            "put",
            "down-and-in",
            level,
            barrier,
            -quantity,
            monitoring,
            observations,
        )

    def formula(p: FloatArray) -> FloatArray:
        up = _upside(p, participation, strike, cap)
        return np.asarray(np.where(p >= barrier, np.maximum(up, bonus), up))

    levels = _levels(
        strike,
        cap,
        Level("Barrier", barrier, LevelKind.PERFORMANCE),
        Level("Bonus level", bonus, LevelKind.PAYOFF),
    )
    return b.note("Bonus certificate", formula, levels, monitoring)


def twin_win_certificate(
    *,
    participation_up: float,
    participation_down: float,
    barrier: float,
    maturity: float,
    initial: float,
    strike: float = 100.0,
    cap: float | None = None,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    observations: int | None = None,
) -> Note:
    """Gains on a move either way from ``strike`` while ``barrier`` holds.

    Once the barrier is breached, the upside only, capped at ``cap``.
    """
    _check_upside(participation_up, strike, cap)
    _positive(barrier=barrier)
    _non_negative(participation_down=participation_down)
    if barrier >= strike:
        raise ValueError("the barrier must be below the strike")
    b = _Builder(initial, maturity)
    _add_upside(b, participation_up, strike, cap)
    b.option("Long put (downside turned into a gain)", "put", strike, 100.0 + participation_down)
    b.barrier_option(
        "Short put down-and-in",
        "put",
        "down-and-in",
        strike,
        barrier,
        -(100.0 + participation_down),
        monitoring,
        observations,
    )

    def formula(p: FloatArray) -> FloatArray:
        up = _upside(p, participation_up, strike, cap)
        alive = up + (1.0 + participation_down / 100.0) * np.maximum(strike - p, 0.0)
        return np.asarray(np.where(p >= barrier, alive, up))

    barrier_level = Level("Barrier", barrier, LevelKind.PERFORMANCE)
    return b.note("Twin-win certificate", formula, _levels(strike, cap, barrier_level), monitoring)


def _cushioned(
    name: str,
    participation: float,
    strike: float,
    cap: float | None,
    maturity: float,
    initial: float,
    loss: tuple[str, float, float],
    payoff_below: Callable[[FloatArray], FloatArray],
) -> Note:
    """Zero-coupon at 100 %, the rise above the strike, and a put sold for the cushion."""
    _check_upside(participation, strike, cap)
    role, level, quantity = loss
    b = _Builder(initial, maturity)
    b.bond("Zero-coupon bond", 100.0)
    b.option("Long call (upside)", "call", strike, participation)
    if cap is not None:
        b.option("Short call (cap)", "call", cap, -participation)
    b.option(role, "put", level, quantity)

    def formula(p: FloatArray) -> FloatArray:
        top = np.inf if cap is None else cap
        gain = participation / 100.0 * np.clip(p - strike, 0.0, top - strike)
        return np.asarray(100.0 + gain - payoff_below(p))

    level_name = "Airbag level" if "Airbag" in name else "Buffer level"
    return b.note(
        name, formula, _levels(strike, cap, Level(level_name, level, LevelKind.PERFORMANCE))
    )


def airbag_certificate(
    *,
    participation: float,
    airbag: float,
    cap: float | None,
    maturity: float,
    initial: float,
    strike: float = 100.0,
    monitoring: Monitoring = Monitoring.EUROPEAN,
) -> Note:
    """The rise above ``strike``; the capital back down to ``airbag``, then cushioned."""
    require_european(monitoring, "airbag certificate")
    _positive(airbag=airbag)
    if airbag >= 100.0:
        raise ValueError("the airbag level must be below the initial level")

    def below(p: FloatArray) -> FloatArray:
        return np.asarray(100.0 / airbag * np.maximum(airbag - p, 0.0))

    return _cushioned(
        "Airbag certificate",
        participation,
        strike,
        cap,
        maturity,
        initial,
        ("Short put (airbag)", airbag, -100.0 * 100.0 / airbag),
        below,
    )


def buffer_certificate(
    *,
    participation: float,
    buffer: float,
    cap: float | None,
    maturity: float,
    initial: float,
    strike: float = 100.0,
    monitoring: Monitoring = Monitoring.EUROPEAN,
) -> Note:
    """The rise above ``strike``; the capital back down to ``buffer``, then one for one."""
    require_european(monitoring, "buffer certificate")
    _positive(buffer=buffer)
    if buffer >= 100.0:
        raise ValueError("the buffer level must be below the initial level")

    def below(p: FloatArray) -> FloatArray:
        return np.asarray(np.maximum(buffer - p, 0.0))

    return _cushioned(
        "Buffer certificate",
        participation,
        strike,
        cap,
        maturity,
        initial,
        ("Short put (beyond the buffer)", buffer, -100.0),
        below,
    )


def sprint_certificate(
    *, participation: float, cap: float, maturity: float, initial: float, strike: float = 100.0
) -> Note:
    """``participation`` % of the rise above ``strike`` up to ``cap``, all of the fall."""
    _check_upside(participation, strike, cap)
    b = _Builder(initial, maturity)
    _add_upside(b, participation, strike, cap)

    def formula(p: FloatArray) -> FloatArray:
        return _upside(p, participation, strike, cap)

    return b.note("Sprint certificate", formula, _levels(strike, cap))


def reverse_bonus_certificate(
    *,
    reverse_level: float,
    bonus: float,
    barrier: float,
    cap: float | None,
    maturity: float,
    initial: float,
    participation: float = 100.0,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    observations: int | None = None,
) -> Note:
    """``participation`` % of ``reverse_level - performance``, up to ``cap``.

    At least ``bonus`` as long as the underlying never reaches ``barrier``.
    """
    _positive(reverse_level=reverse_level, bonus=bonus, barrier=barrier, cap=cap)
    _positive(participation=participation)
    gearing = participation / 100.0
    # the call that pays the bonus: struck where the reverse payoff falls to the bonus
    bonus_strike = reverse_level - bonus / gearing
    if bonus_strike <= 0.0:
        raise ValueError("the bonus must be below participation x reverse level")
    if not bonus_strike < barrier < reverse_level:
        raise ValueError(
            f"the barrier must be between {bonus_strike:g} % (where the bonus starts) and the "
            "reverse level"
        )
    if cap is not None and not bonus < cap < gearing * reverse_level:
        raise ValueError("the cap must be between the bonus and participation x reverse level")
    b = _Builder(initial, maturity)
    b.option("Long put (reverse)", "put", reverse_level, participation)
    b.barrier_option(
        "Long call up-and-out (bonus)",
        "call",
        "up-and-out",
        bonus_strike,
        barrier,
        participation,
        monitoring,
        observations,
    )
    if cap is not None:
        b.option("Short put (cap)", "put", reverse_level - cap / gearing, -participation)

    def formula(p: FloatArray) -> FloatArray:
        reverse = gearing * np.maximum(reverse_level - p, 0.0)
        out = np.where(p < barrier, np.maximum(reverse, bonus), reverse)
        return np.asarray(out if cap is None else np.minimum(out, cap))

    levels = [
        Level("Reverse level", reverse_level, LevelKind.PERFORMANCE),
        Level("Bonus level", bonus, LevelKind.PAYOFF),
        Level("Barrier", barrier, LevelKind.PERFORMANCE),
    ]
    if cap is not None:
        levels.append(Level("Cap", cap, LevelKind.PAYOFF))
    return b.note("Reverse bonus certificate", formula, tuple(levels), monitoring)


def discount_certificate(*, cap: float, maturity: float, initial: float) -> Note:
    """The underlying bought at a discount, paid at most ``cap``."""
    _positive(cap=cap)
    b = _Builder(initial, maturity)
    b.underlying("Underlying")
    b.option("Short call (cap)", "call", cap, -100.0)

    def formula(p: FloatArray) -> FloatArray:
        return np.asarray(np.minimum(p, cap))

    return b.note("Discount certificate", formula, (Level("Cap", cap, LevelKind.PERFORMANCE),))


def shark_warrant(
    *,
    direction: str,
    strike: float,
    barrier: float,
    participation: float,
    rebate: float,
    maturity: float,
    initial: float,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    observations: int | None = None,
) -> Note:
    """A knock-out call (up) or put (down) with a rebate when the barrier is touched."""
    _positive(strike=strike, barrier=barrier)
    _non_negative(participation=participation, rebate=rebate)
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    up = direction == "up"
    if (up and barrier <= strike) or (not up and barrier >= strike):
        raise ValueError("the barrier must be beyond the strike, on the side of the bet")
    b = _Builder(initial, maturity)
    kind = "call" if up else "put"
    b.barrier_option(
        f"Long {kind} {'up' if up else 'down'}-and-out",
        kind,
        "up-and-out" if up else "down-and-out",
        strike,
        barrier,
        participation,
        monitoring,
        observations,
    )
    b.one_touch(
        "Rebate if the barrier is touched",
        "up" if up else "down",
        barrier,
        rebate,
        monitoring,
        observations,
    )

    def formula(p: FloatArray) -> FloatArray:
        gain = participation / 100.0 * np.maximum((p - strike) if up else (strike - p), 0.0)
        knocked = p >= barrier if up else p < barrier
        return np.asarray(np.where(knocked, rebate, gain))

    levels = (
        Level("Strike", strike, LevelKind.PERFORMANCE),
        Level("Knock-out barrier", barrier, LevelKind.PERFORMANCE),
        Level("Rebate", rebate, LevelKind.PAYOFF),
    )
    return b.note(f"Shark warrant ({'call' if up else 'put'})", formula, levels, monitoring)


# --- Catalog ----------------------------------------------------------------------------------


def _shark_builder(
    direction: str,
) -> Callable[[Mapping[str, float], float, float, Monitoring, int | None], Note]:
    def build(
        v: Mapping[str, float], t: float, s0: float, mon: Monitoring, obs: int | None
    ) -> Note:
        return shark_warrant(
            direction=direction,
            strike=v["strike"],
            barrier=v["barrier"],
            participation=v["participation"],
            rebate=v["rebate"],
            maturity=t,
            initial=s0,
            monitoring=mon,
            observations=obs,
        )

    return build


BARRIER = NoteParam("barrier", "Barrier", 70.0, "Protection lost once breached.")
CAPPED = NoteParam("capped", "Capped", 0.0, kind="toggle")
CAP = NoteParam("cap", "Cap", 140.0)
PARTICIPATION = NoteParam("participation", "Participation", 100.0, "Share of the rise.")
STRIKE = NoteParam("strike", "Strike", 100.0, "Where the participation starts.")
_BARRIER_OPTIONS = {"has_barrier": True, "monitoring_choice": True}

PARTICIPATION_CATALOG: tuple[NoteSpec, ...] = (
    NoteSpec(
        "Outperformance certificate",
        "The underlying, with more than its rise above the strike, and all of its fall; the "
        "gain can be capped.",
        (replace(PARTICIPATION, default=150.0), STRIKE, CAPPED, CAP),
        lambda v, t, s0, mon, obs: outperformance_certificate(
            participation=v["participation"],
            strike=v["strike"],
            cap=_cap(v),
            maturity=t,
            initial=s0,
        ),
    ),
    NoteSpec(
        "Barrier outperformance certificate",
        "More than the rise of the underlying above the strike, and the capital back as long "
        "as the barrier holds.",
        (replace(PARTICIPATION, default=130.0), STRIKE, BARRIER, CAPPED, CAP),
        lambda v, t, s0, mon, obs: barrier_outperformance_certificate(
            participation=v["participation"],
            strike=v["strike"],
            barrier=v["barrier"],
            cap=_cap(v),
            maturity=t,
            initial=s0,
            monitoring=mon,
            observations=obs,
        ),
        **_BARRIER_OPTIONS,
    ),
    NoteSpec(
        "Bonus certificate",
        "At least the bonus level as long as the barrier holds, a share of the rise above the "
        "strike, and the underlying once the barrier is breached.",
        (
            NoteParam("bonus", "Bonus level", 115.0),
            BARRIER,
            PARTICIPATION,
            STRIKE,
            CAPPED,
            CAP,
        ),
        lambda v, t, s0, mon, obs: bonus_certificate(
            bonus=v["bonus"],
            barrier=v["barrier"],
            participation=v["participation"],
            strike=v["strike"],
            cap=_cap(v),
            maturity=t,
            initial=s0,
            monitoring=mon,
            observations=obs,
        ),
        **_BARRIER_OPTIONS,
    ),
    NoteSpec(
        "Twin-win certificate",
        "Gains on a move either way from the strike as long as the barrier holds; the upside "
        "once it is breached.",
        (
            NoteParam("participation_up", "Upside participation", 100.0),
            NoteParam("participation_down", "Downside participation", 100.0),
            STRIKE,
            BARRIER,
            CAPPED,
            CAP,
        ),
        lambda v, t, s0, mon, obs: twin_win_certificate(
            participation_up=v["participation_up"],
            participation_down=v["participation_down"],
            strike=v["strike"],
            barrier=v["barrier"],
            cap=_cap(v),
            maturity=t,
            initial=s0,
            monitoring=mon,
            observations=obs,
        ),
        **_BARRIER_OPTIONS,
    ),
    NoteSpec(
        "Airbag certificate",
        "A share of the rise above the strike, the capital back down to the airbag level, then "
        "a loss cushioned all the way to zero.",
        (PARTICIPATION, STRIKE, NoteParam("airbag", "Airbag level", 80.0), CAPPED, CAP),
        lambda v, t, s0, mon, obs: airbag_certificate(
            participation=v["participation"],
            strike=v["strike"],
            airbag=v["airbag"],
            cap=_cap(v),
            maturity=t,
            initial=s0,
            monitoring=mon,
        ),
    ),
    NoteSpec(
        "Buffer certificate",
        "A share of the rise above the strike, the capital back down to the buffer level, then "
        "a one-for-one loss.",
        (PARTICIPATION, STRIKE, NoteParam("buffer", "Buffer level", 90.0), CAPPED, CAP),
        lambda v, t, s0, mon, obs: buffer_certificate(
            participation=v["participation"],
            strike=v["strike"],
            buffer=v["buffer"],
            cap=_cap(v),
            maturity=t,
            initial=s0,
            monitoring=mon,
        ),
    ),
    NoteSpec(
        "Sprint certificate",
        "A leveraged share of the rise above the strike up to the cap, and all of the fall.",
        (replace(PARTICIPATION, default=200.0), STRIKE, replace(CAP, default=120.0)),
        lambda v, t, s0, mon, obs: sprint_certificate(
            participation=v["participation"],
            strike=v["strike"],
            cap=v["cap"],
            maturity=t,
            initial=s0,
        ),
    ),
    NoteSpec(
        "Reverse bonus certificate",
        "Gains as the underlying falls, and at least the bonus as long as it never rises to "
        "the barrier.",
        (
            NoteParam(
                "reverse_level", "Reverse level", 200.0, "Paid: reverse level - performance."
            ),
            NoteParam("bonus", "Bonus level", 115.0),
            NoteParam("barrier", "Barrier", 130.0, "Bonus lost once reached."),
            replace(PARTICIPATION, help="Share of the fall."),
            CAPPED,
            replace(CAP, default=150.0),
        ),
        lambda v, t, s0, mon, obs: reverse_bonus_certificate(
            reverse_level=v["reverse_level"],
            bonus=v["bonus"],
            barrier=v["barrier"],
            participation=v["participation"],
            cap=_cap(v),
            maturity=t,
            initial=s0,
            monitoring=mon,
            observations=obs,
        ),
        **_BARRIER_OPTIONS,
    ),
)

YIELD_CERTIFICATES: tuple[NoteSpec, ...] = (
    NoteSpec(
        "Discount certificate",
        "The underlying bought below its price, in exchange for a cap: the investor has sold a "
        "call.",
        (replace(CAP, default=105.0),),
        lambda v, t, s0, mon, obs: discount_certificate(cap=v["cap"], maturity=t, initial=s0),
    ),
)

SHARK_WARRANTS: tuple[NoteSpec, ...] = tuple(
    NoteSpec(
        f"Shark warrant ({kind})",
        f"A {kind} cancelled beyond a barrier, which then pays a rebate; leverage without "
        "capital protection.",
        (
            NoteParam("strike", "Strike", 100.0),
            NoteParam("barrier", "Knock-out barrier", barrier),
            NoteParam("participation", "Participation", 100.0),
            NoteParam("rebate", "Rebate", 5.0),
        ),
        _shark_builder(direction),
        has_barrier=True,
        monitoring_choice=True,
    )
    for kind, direction, barrier in (("call", "up", 130.0), ("put", "down", 70.0))
)


def find_certificate(name: str) -> NoteSpec:
    """The certificate or warrant with this name."""
    for spec in PARTICIPATION_CATALOG + YIELD_CERTIFICATES + SHARK_WARRANTS:
        if spec.name == name:
            return spec
    raise KeyError(name)
