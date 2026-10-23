"""Participation certificates and shark warrants, priced exactly by bricks.

Certificates (EUSIPA 13xx) hold the underlying itself and reshape its payoff with options; the
capital is not protected, at most cushioned. Levels are in % of the initial level, amounts in %
of the notional. ``U`` is the underlying (a prepaid forward), ``C``/``P`` calls and puts,
``PDI`` a down-and-in put observed as the product says.

* outperformance: ``U + (part - 100 %) x C(100)``: more than the rise, all the fall;
* barrier outperformance: ``U + (part - 100 %) C(100) + P(100) - PDI(100, B)``: more than the
  rise, and the capital back while the barrier holds;
* bonus certificate: ``U + P(bonus) - PDI(bonus, B) [- C(cap)]``: at least the bonus while the
  barrier holds, the underlying otherwise;
* twin-win certificate: ``U + (up - 100 %) C(100) + (100 % + down) (P(100) - PDI(100, B))``:
  gains on a move either way while the barrier holds;
* airbag certificate: ``ZC(100) + part x C(100) - (100 / A) P(A)``: the capital back down to the
  airbag level ``A``, then a loss that reaches the whole notional only at zero;
* buffer certificate: ``ZC(100) + part x [C(100) - C(cap)] - P(B)``: the capital back down to
  the buffer ``B``, then a one-for-one loss;
* shark warrant: ``part x CallUpAndOut(K, B) + rebate x OneTouch(B)`` (or the put version):
  a capped, knock-out option with a rebate, without capital protection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

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


def outperformance_certificate(*, participation: float, maturity: float, initial: float) -> Note:
    """The underlying, with ``participation`` % of its rise above the initial level."""
    _positive(participation=participation)
    b = _Builder(initial, maturity)
    b.underlying("Underlying")
    b.option("Long call (extra upside)", "call", 100.0, participation - 100.0)

    def formula(p: FloatArray) -> FloatArray:
        return np.asarray(p + (participation - 100.0) / 100.0 * np.maximum(p - 100.0, 0.0))

    levels = (Level("Initial level", 100.0, LevelKind.PERFORMANCE),)
    return b.note("Outperformance certificate", formula, levels)


def barrier_outperformance_certificate(
    *,
    participation: float,
    barrier: float,
    maturity: float,
    initial: float,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    observations: int | None = None,
) -> Note:
    """Outperformance above the initial level, the capital back while ``barrier`` holds."""
    _positive(participation=participation, barrier=barrier)
    if barrier >= 100.0:
        raise ValueError("the barrier must be below the initial level")
    b = _Builder(initial, maturity)
    b.underlying("Underlying")
    b.option("Long call (extra upside)", "call", 100.0, participation - 100.0)
    b.option("Long put (protection)", "put", 100.0, 100.0)
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
        upside = p + (participation - 100.0) / 100.0 * np.maximum(p - 100.0, 0.0)
        return np.asarray(np.where(p >= barrier, np.maximum(upside, 100.0), p))

    levels = (
        Level("Initial level", 100.0, LevelKind.PERFORMANCE),
        Level("Barrier", barrier, LevelKind.PERFORMANCE),
    )
    return b.note("Barrier outperformance certificate", formula, levels, monitoring)


def bonus_certificate(
    *,
    bonus: float,
    barrier: float,
    cap: float | None,
    maturity: float,
    initial: float,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    observations: int | None = None,
) -> Note:
    """At least ``bonus`` while ``barrier`` holds, the underlying otherwise, up to ``cap``."""
    _positive(bonus=bonus, barrier=barrier, cap=cap)
    if not barrier < bonus:
        raise ValueError("the barrier must be below the bonus level")
    if cap is not None and cap <= bonus:
        raise ValueError("the cap must be above the bonus level")
    b = _Builder(initial, maturity)
    b.underlying("Underlying")
    b.option("Long put (bonus)", "put", bonus, 100.0)
    b.barrier_option(
        "Short put down-and-in",
        "put",
        "down-and-in",
        bonus,
        barrier,
        -100.0,
        monitoring,
        observations,
    )
    if cap is not None:
        b.option("Short call (cap)", "call", cap, -100.0)

    def formula(p: FloatArray) -> FloatArray:
        out = np.where(p >= barrier, np.maximum(p, bonus), p)
        return out if cap is None else np.asarray(np.minimum(out, cap))

    levels = [
        Level("Bonus level", bonus, LevelKind.PAYOFF),
        Level("Barrier", barrier, LevelKind.PERFORMANCE),
    ]
    if cap is not None:
        levels.append(Level("Cap", cap, LevelKind.PERFORMANCE))
    return b.note("Bonus certificate", formula, tuple(levels), monitoring)


def twin_win_certificate(
    *,
    participation_up: float,
    participation_down: float,
    barrier: float,
    maturity: float,
    initial: float,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    observations: int | None = None,
) -> Note:
    """Gains on a move either way while ``barrier`` holds; the underlying once it is breached."""
    _positive(barrier=barrier)
    _non_negative(participation_up=participation_up, participation_down=participation_down)
    if barrier >= 100.0:
        raise ValueError("the barrier must be below the initial level")
    b = _Builder(initial, maturity)
    b.underlying("Underlying")
    b.option("Long call (upside)", "call", 100.0, participation_up - 100.0)
    b.option("Long put (downside turned into a gain)", "put", 100.0, 100.0 + participation_down)
    b.barrier_option(
        "Short put down-and-in",
        "put",
        "down-and-in",
        100.0,
        barrier,
        -(100.0 + participation_down),
        monitoring,
        observations,
    )

    def formula(p: FloatArray) -> FloatArray:
        up = participation_up / 100.0 * np.maximum(p - 100.0, 0.0)
        down = participation_down / 100.0 * np.maximum(100.0 - p, 0.0)
        breached = p + (participation_up - 100.0) / 100.0 * np.maximum(p - 100.0, 0.0)
        return np.asarray(np.where(p >= barrier, 100.0 + up + down, breached))

    levels = (
        Level("Initial level", 100.0, LevelKind.PERFORMANCE),
        Level("Barrier", barrier, LevelKind.PERFORMANCE),
    )
    return b.note("Twin-win certificate", formula, levels, monitoring)


def airbag_certificate(
    *,
    participation: float,
    airbag: float,
    cap: float | None,
    maturity: float,
    initial: float,
    monitoring: Monitoring = Monitoring.EUROPEAN,
) -> Note:
    """The rise above the initial level; the capital back down to ``airbag``, then cushioned."""
    require_european(monitoring, "airbag certificate")
    _positive(participation=participation, airbag=airbag, cap=cap)
    if airbag >= 100.0:
        raise ValueError("the airbag level must be below the initial level")
    if cap is not None and cap <= 100.0:
        raise ValueError("the cap must be above the initial level")
    b = _Builder(initial, maturity)
    b.bond("Zero-coupon bond", 100.0)
    b.option("Long call (upside)", "call", 100.0, participation)
    if cap is not None:
        b.option("Short call (cap)", "call", cap, -participation)
    b.option("Short put (airbag)", "put", airbag, -100.0 * 100.0 / airbag)

    def formula(p: FloatArray) -> FloatArray:
        gain = participation / 100.0 * np.maximum(p - 100.0, 0.0)
        if cap is not None:
            gain = np.minimum(gain, participation / 100.0 * (cap - 100.0))
        return np.asarray(np.where(p >= airbag, 100.0 + gain, 100.0 * p / airbag))

    levels = [
        Level("Initial level", 100.0, LevelKind.PERFORMANCE),
        Level("Airbag level", airbag, LevelKind.PERFORMANCE),
    ]
    if cap is not None:
        levels.append(Level("Cap", cap, LevelKind.PERFORMANCE))
    return b.note("Airbag certificate", formula, tuple(levels))


def buffer_certificate(
    *,
    participation: float,
    buffer: float,
    cap: float | None,
    maturity: float,
    initial: float,
    monitoring: Monitoring = Monitoring.EUROPEAN,
) -> Note:
    """The rise above the initial level; the capital back down to ``buffer``, then one for one."""
    require_european(monitoring, "buffer certificate")
    _positive(participation=participation, buffer=buffer, cap=cap)
    if buffer >= 100.0:
        raise ValueError("the buffer level must be below the initial level")
    if cap is not None and cap <= 100.0:
        raise ValueError("the cap must be above the initial level")
    b = _Builder(initial, maturity)
    b.bond("Zero-coupon bond", 100.0)
    b.option("Long call (upside)", "call", 100.0, participation)
    if cap is not None:
        b.option("Short call (cap)", "call", cap, -participation)
    b.option("Short put (beyond the buffer)", "put", buffer, -100.0)

    def formula(p: FloatArray) -> FloatArray:
        gain = participation / 100.0 * np.maximum(p - 100.0, 0.0)
        if cap is not None:
            gain = np.minimum(gain, participation / 100.0 * (cap - 100.0))
        return np.asarray(100.0 + gain - np.maximum(buffer - p, 0.0))

    levels = [
        Level("Initial level", 100.0, LevelKind.PERFORMANCE),
        Level("Buffer level", buffer, LevelKind.PERFORMANCE),
    ]
    if cap is not None:
        levels.append(Level("Cap", cap, LevelKind.PERFORMANCE))
    return b.note("Buffer certificate", formula, tuple(levels))


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

PARTICIPATION_CATALOG: tuple[NoteSpec, ...] = (
    NoteSpec(
        "Outperformance certificate",
        "The underlying, with more than its rise above the initial level, and all of its fall.",
        (NoteParam("participation", "Upside participation", 150.0),),
        lambda v, t, s0, mon, obs: outperformance_certificate(
            participation=v["participation"], maturity=t, initial=s0
        ),
    ),
    NoteSpec(
        "Barrier outperformance certificate",
        "More than the rise of the underlying, and the capital back as long as the barrier holds.",
        (NoteParam("participation", "Upside participation", 130.0), BARRIER),
        lambda v, t, s0, mon, obs: barrier_outperformance_certificate(
            participation=v["participation"],
            barrier=v["barrier"],
            maturity=t,
            initial=s0,
            monitoring=mon,
            observations=obs,
        ),
        has_barrier=True,
        monitoring_choice=True,
    ),
    NoteSpec(
        "Bonus certificate",
        "At least the bonus level as long as the barrier holds, the underlying otherwise.",
        (NoteParam("bonus", "Bonus level", 115.0), BARRIER, CAPPED, CAP),
        lambda v, t, s0, mon, obs: bonus_certificate(
            bonus=v["bonus"],
            barrier=v["barrier"],
            cap=_cap(v),
            maturity=t,
            initial=s0,
            monitoring=mon,
            observations=obs,
        ),
        has_barrier=True,
        monitoring_choice=True,
    ),
    NoteSpec(
        "Twin-win certificate",
        "Gains on a move either way as long as the barrier holds; the underlying once it is "
        "breached.",
        (
            NoteParam("participation_up", "Upside participation", 100.0),
            NoteParam("participation_down", "Downside participation", 100.0),
            BARRIER,
        ),
        lambda v, t, s0, mon, obs: twin_win_certificate(
            participation_up=v["participation_up"],
            participation_down=v["participation_down"],
            barrier=v["barrier"],
            maturity=t,
            initial=s0,
            monitoring=mon,
            observations=obs,
        ),
        has_barrier=True,
        monitoring_choice=True,
    ),
    NoteSpec(
        "Airbag certificate",
        "The rise of the underlying, the capital back down to the airbag level, then a loss "
        "cushioned all the way to zero.",
        (
            NoteParam("participation", "Upside participation", 100.0),
            NoteParam("airbag", "Airbag level", 80.0),
            CAPPED,
            CAP,
        ),
        lambda v, t, s0, mon, obs: airbag_certificate(
            participation=v["participation"],
            airbag=v["airbag"],
            cap=_cap(v),
            maturity=t,
            initial=s0,
            monitoring=mon,
        ),
    ),
    NoteSpec(
        "Buffer certificate",
        "The rise of the underlying, the capital back down to the buffer level, then a "
        "one-for-one loss.",
        (
            NoteParam("participation", "Upside participation", 100.0),
            NoteParam("buffer", "Buffer level", 90.0),
            CAPPED,
            CAP,
        ),
        lambda v, t, s0, mon, obs: buffer_certificate(
            participation=v["participation"],
            buffer=v["buffer"],
            cap=_cap(v),
            maturity=t,
            initial=s0,
            monitoring=mon,
        ),
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
    for spec in PARTICIPATION_CATALOG + SHARK_WARRANTS:
        if spec.name == name:
            return spec
    raise KeyError(name)
