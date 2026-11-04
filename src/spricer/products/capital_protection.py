"""Capital protected notes, priced exactly by closed-form bricks.

Every level is in percent of the underlying's initial level and every amount in percent of the
notional, as on a term sheet: ``protection = 90`` means 90 % of the notional is paid back at
least, ``barrier = 130`` means 130 % of the initial level. Participations are in percent too.

Barriers are observed at maturity (European): the products then replicate exactly with vanilla
options, digitals and a zero-coupon. Continuous or discrete observation comes with the barrier
module and is refused explicitly until then (:class:`~spricer.products.notes.Monitoring`).

Replications (``C``/``P`` calls and puts, ``D`` digitals, ``ZC`` the bond part):

* participation note: ``ZC(prot) + part x [C(K) - C(cap)]``;
* bearish participation note: ``ZC(prot) + part x [P(K) - P(floor)]``;
* shark note, up: ``ZC + part x CallUpAndOut(K, B) + rebate x D_call(B)``;
* shark note, down: ``ZC + part x PutDownAndOut(K, B) + rebate x D_put(B)``;
* bonus note (protection at or below the barrier): ``ZC(B) + P(prot) - PutDownAndIn(B, B)
  + (bonus - B) x D_call(B) + part x [C(K) - C(cap)]``. When the protection equals the barrier,
  the protection put and the down-and-in put are the same option and cancel out;
* twin-win note: ``ZC + up x [C(K) - C(cap)] + down x [P(K) - P(floor)]``;
* digital coupon note: ``ZC + coupon x D_call(B)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from spricer._typing import FloatArray
from spricer.bricks import (
    BarrierOption,
    BarrierType,
    Brick,
    DigitalOption,
    EuropeanOption,
    OneTouch,
    Underlying,
    ZeroCoupon,
)
from spricer.monitoring import Monitoring, require_european
from spricer.products.notes import NOTIONAL, Level, LevelKind, Note, NoteLeg

UNCAPPED = None


def _positive(**values: float | None) -> None:
    for name, value in values.items():
        if value is not None and not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive")


def _non_negative(**values: float) -> None:
    for name, value in values.items():
        if not (np.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be non-negative")


class _Builder:
    """Bricks of a note on a given initial level and maturity."""

    def __init__(self, initial: float, maturity: float) -> None:
        _positive(initial=initial, maturity=maturity)
        self.initial, self.maturity = initial, maturity
        self.legs: list[NoteLeg] = []

    def _strike(self, pct: float) -> float:
        return pct / 100.0 * self.initial

    def bond(self, role: str, amount_pct: float) -> None:
        self.legs.append(NoteLeg(role, ZeroCoupon(self.maturity), amount_pct))

    def underlying(self, role: str, participation_pct: float = 100.0) -> None:
        """The underlying itself (prepaid forward), ``participation_pct`` % of its move."""
        self.legs.append(NoteLeg(role, Underlying(self.maturity), self._units(participation_pct)))

    def _units(self, participation_pct: float) -> float:
        return participation_pct / 100.0 * NOTIONAL / self.initial

    def barrier_option(
        self,
        role: str,
        kind: str,
        barrier_type: BarrierType,
        strike_pct: float,
        barrier_pct: float,
        participation_pct: float,
        monitoring: Monitoring,
        observations: int | None = None,
    ) -> None:
        if participation_pct:
            brick = BarrierOption(
                kind,  # type: ignore[arg-type]
                barrier_type,
                self._strike(strike_pct),
                self._strike(barrier_pct),
                self.maturity,
                monitoring,
                observations,
            )
            self.legs.append(NoteLeg(role, brick, self._units(participation_pct)))

    def option(self, role: str, kind: str, strike_pct: float, participation_pct: float) -> None:
        if participation_pct:
            brick: Brick = EuropeanOption(kind, self._strike(strike_pct), self.maturity)  # type: ignore[arg-type]
            self.legs.append(
                NoteLeg(role, brick, participation_pct / 100.0 * NOTIONAL / self.initial)
            )

    def one_touch(
        self,
        role: str,
        direction: str,
        barrier_pct: float,
        amount_pct: float,
        monitoring: Monitoring,
        observations: int | None,
    ) -> None:
        if amount_pct:
            brick = OneTouch(
                direction,  # type: ignore[arg-type]
                self._strike(barrier_pct),
                self.maturity,
                monitoring,
                observations,
            )
            self.legs.append(NoteLeg(role, brick, amount_pct))

    def digital(self, role: str, kind: str, strike_pct: float, amount_pct: float) -> None:
        if amount_pct:
            brick = DigitalOption(kind, self._strike(strike_pct), self.maturity)  # type: ignore[arg-type]
            self.legs.append(NoteLeg(role, brick, amount_pct))

    def note(
        self,
        name: str,
        formula: Callable[[FloatArray], FloatArray],
        levels: tuple[Level, ...],
        monitoring: Monitoring | None = None,
    ) -> Note:
        return Note(
            name, self.maturity, self.initial, tuple(self.legs), formula, levels, monitoring
        )


def _capped(gain: FloatArray, width: float | None) -> FloatArray:
    return gain if width is None else np.minimum(gain, width)


def participation_note(
    *,
    protection: float,
    strike: float,
    participation: float,
    cap: float | None,
    maturity: float,
    initial: float,
) -> Note:
    """Capital protection plus a share of the rise above ``strike``, optionally capped."""
    _positive(protection=protection, strike=strike, cap=cap)
    _non_negative(participation=participation)
    if cap is not None and cap <= strike:
        raise ValueError("the cap must be above the participation strike")
    b = _Builder(initial, maturity)
    b.bond("Zero-coupon bond", protection)
    b.option("Long call (participation)", "call", strike, participation)
    if cap is not None:
        b.option("Short call (cap)", "call", cap, -participation)

    def formula(p: FloatArray) -> FloatArray:
        width = None if cap is None else cap - strike
        return protection + participation / 100.0 * _capped(np.maximum(p - strike, 0.0), width)

    levels = (
        Level("Capital protection", protection, LevelKind.PAYOFF),
        Level("Long call strike", strike, LevelKind.PERFORMANCE),
        *(() if cap is None else (Level("Short call strike (cap)", cap, LevelKind.PERFORMANCE),)),
    )
    return b.note("Participation note", formula, levels)


def bearish_participation_note(
    *,
    protection: float,
    strike: float,
    participation: float,
    floor: float | None,
    maturity: float,
    initial: float,
) -> Note:
    """Capital protection plus a share of the fall below ``strike``, down to ``floor``."""
    _positive(protection=protection, strike=strike, floor=floor)
    _non_negative(participation=participation)
    if floor is not None and floor >= strike:
        raise ValueError("the floor must be below the participation strike")
    b = _Builder(initial, maturity)
    b.bond("Zero-coupon bond", protection)
    b.option("Long put (participation)", "put", strike, participation)
    if floor is not None:
        b.option("Short put (gain limit)", "put", floor, -participation)

    def formula(p: FloatArray) -> FloatArray:
        width = None if floor is None else strike - floor
        return protection + participation / 100.0 * _capped(np.maximum(strike - p, 0.0), width)

    levels = (
        Level("Capital protection", protection, LevelKind.PAYOFF),
        Level("Long put strike", strike, LevelKind.PERFORMANCE),
        *(
            ()
            if floor is None
            else (Level("Short put strike (gain limit)", floor, LevelKind.PERFORMANCE),)
        ),
    )
    return b.note("Bearish participation note", formula, levels)


def shark_note(
    *,
    protection: float,
    strike: float,
    participation: float,
    barrier: float,
    rebate: float,
    direction: str,
    maturity: float,
    initial: float,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    observations: int | None = None,
) -> Note:
    """Participation that stops at a barrier: once touched, only ``rebate`` is paid on top.

    ``direction`` is ``"up"`` (call participation, barrier above the strike) or ``"down"``
    (put participation, barrier below the strike). The barrier is observed at maturity
    (European), continuously, or at ``observations`` equally spaced dates; the rebate is paid
    at maturity. The term-sheet formula is the payoff when the barrier is observed at maturity
    (with a path-dependent barrier, it is the payoff when it was not touched before).
    """
    _positive(protection=protection, strike=strike, barrier=barrier)
    _non_negative(participation=participation, rebate=rebate)
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    up = direction == "up"
    if (barrier <= strike) if up else (barrier >= strike):
        raise ValueError(
            "the barrier must be above the strike" if up else "the barrier must be below the strike"
        )
    kind = "call" if up else "put"
    b = _Builder(initial, maturity)
    b.bond("Zero-coupon bond", protection)
    barrier_type: BarrierType = "up-and-out" if up else "down-and-out"
    b.barrier_option(
        f"Long {kind} {barrier_type} (participation)",
        kind,
        barrier_type,
        strike,
        barrier,
        participation,
        monitoring,
        observations,
    )
    european = monitoring is Monitoring.EUROPEAN
    b.one_touch(
        f"Long digital {kind} (rebate)" if european else "Long one-touch (rebate)",
        direction,
        barrier,
        rebate,
        monitoring,
        observations,
    )

    def formula(p: FloatArray) -> FloatArray:
        part = participation / 100.0
        if up:
            return np.where(
                p >= barrier, protection + rebate, protection + part * np.maximum(p - strike, 0.0)
            )
        return np.where(
            p < barrier, protection + rebate, protection + part * np.maximum(strike - p, 0.0)
        )

    levels = (
        Level("Capital protection", protection, LevelKind.PAYOFF),
        Level(f"Long {kind} strike", strike, LevelKind.PERFORMANCE),
        Level(f"{barrier_type.capitalize()} barrier", barrier, LevelKind.PERFORMANCE),
        *(() if not rebate else (Level("Rebate", protection + rebate, LevelKind.PAYOFF),)),
    )
    return b.note(f"Shark note ({direction})", formula, levels, monitoring)


def bonus_note(
    *,
    protection: float,
    barrier: float,
    bonus: float,
    strike: float,
    participation: float,
    cap: float | None,
    maturity: float,
    initial: float,
    monitoring: Monitoring = Monitoring.EUROPEAN,
) -> Note:
    """Bonus if the underlying ends at or above ``barrier``, its performance otherwise.

    The payoff never falls below ``protection``, and a participation above ``strike`` (capped
    at ``cap`` if given) is added in every case.
    """
    require_european(monitoring, "bonus note")
    _positive(protection=protection, barrier=barrier, bonus=bonus, strike=strike, cap=cap)
    _non_negative(participation=participation)
    if bonus < protection:
        raise ValueError("the bonus must be at least the capital protection")
    if cap is not None and cap <= strike:
        raise ValueError("the cap must be above the participation strike")
    b = _Builder(initial, maturity)
    if protection <= barrier:
        # Above the barrier the note pays the bonus, below it the underlying, floored at the
        # protection: a bond paying the barrier level, a protection put, and a short
        # down-and-in put that hands the underlying's fall below the barrier to the investor.
        b.bond("Zero-coupon bond", barrier)
        b.option("Long put (capital protection)", "put", protection, 100.0)
        b.barrier_option(
            "Short put down-and-in", "put", "down-and-in", barrier, barrier, -100.0, monitoring
        )
        b.digital("Long digital call (bonus)", "call", barrier, bonus - barrier)
    else:
        b.bond("Zero-coupon bond", protection)
        b.digital("Long digital call (bonus)", "call", barrier, bonus - protection)
    b.option("Long call (participation)", "call", strike, participation)
    if cap is not None:
        b.option("Short call (cap)", "call", cap, -participation)

    def formula(p: FloatArray) -> FloatArray:
        width = None if cap is None else cap - strike
        upside = participation / 100.0 * _capped(np.maximum(p - strike, 0.0), width)
        return np.where(p >= barrier, bonus, np.maximum(p, protection)) + upside

    levels = (
        Level("Capital protection", protection, LevelKind.PAYOFF),
        Level("Bonus level", bonus, LevelKind.PAYOFF),
        Level("Bonus barrier", barrier, LevelKind.PERFORMANCE),
        Level("Long call strike", strike, LevelKind.PERFORMANCE),
        *(() if cap is None else (Level("Short call strike (cap)", cap, LevelKind.PERFORMANCE),)),
    )
    return b.note("Bonus note", formula, levels, monitoring)


def twin_win_note(
    *,
    protection: float,
    call_strike: float,
    participation_up: float,
    cap: float,
    put_strike: float,
    participation_down: float,
    floor: float,
    maturity: float,
    initial: float,
) -> Note:
    """Gains above ``call_strike`` up to ``cap``, and below ``put_strike`` down to ``floor``.

    The two strikes can differ: between them the note pays only the protection.
    """
    _positive(
        protection=protection, call_strike=call_strike, put_strike=put_strike, cap=cap, floor=floor
    )
    _non_negative(participation_up=participation_up, participation_down=participation_down)
    if not call_strike < cap:
        raise ValueError("the cap must be above the call strike")
    if not floor < put_strike:
        raise ValueError("the downside limit (floor) must be below the put strike")
    b = _Builder(initial, maturity)
    b.bond("Zero-coupon bond", protection)
    b.option("Long call (upside)", "call", call_strike, participation_up)
    b.option("Short call (upside cap)", "call", cap, -participation_up)
    b.option("Long put (downside)", "put", put_strike, participation_down)
    b.option("Short put (downside limit)", "put", floor, -participation_down)

    def formula(p: FloatArray) -> FloatArray:
        up = participation_up / 100.0 * np.clip(p - call_strike, 0.0, cap - call_strike)
        down = participation_down / 100.0 * np.clip(put_strike - p, 0.0, put_strike - floor)
        return protection + up + down

    levels = (
        Level("Capital protection", protection, LevelKind.PAYOFF),
        Level("Long call strike", call_strike, LevelKind.PERFORMANCE),
        Level("Long put strike", put_strike, LevelKind.PERFORMANCE),
        Level("Short call strike (cap)", cap, LevelKind.PERFORMANCE),
        Level("Short put strike (downside limit)", floor, LevelKind.PERFORMANCE),
    )
    return b.note("Twin-win note", formula, levels)


def digital_coupon_note(
    *,
    protection: float,
    coupon: float,
    barrier: float,
    maturity: float,
    initial: float,
    monitoring: Monitoring = Monitoring.EUROPEAN,
) -> Note:
    """Pays ``protection`` plus ``coupon`` if the underlying ends at or above ``barrier``."""
    require_european(monitoring, "digital coupon note")
    _positive(protection=protection, barrier=barrier)
    _non_negative(coupon=coupon)
    b = _Builder(initial, maturity)
    b.bond("Zero-coupon bond", protection)
    b.digital("Long digital call (coupon)", "call", barrier, coupon)

    def formula(p: FloatArray) -> FloatArray:
        return protection + np.where(p >= barrier, coupon, 0.0)

    levels = (
        Level("Capital protection", protection, LevelKind.PAYOFF),
        Level("With coupon", protection + coupon, LevelKind.PAYOFF),
        Level("Coupon barrier", barrier, LevelKind.PERFORMANCE),
    )
    return b.note("Digital coupon note", formula, levels, monitoring)


# --- Catalog ------------------------------------------------------------------------------


@dataclass(frozen=True)
class NoteParam:
    """An input of a note, in percent unless ``kind`` says otherwise."""

    key: str
    label: str
    default: float
    help: str = ""
    kind: str = "pct"  # "pct" or "toggle" (1.0 = on)


@dataclass(frozen=True)
class NoteSpec:
    """A capital protected note as offered by the app."""

    name: str
    description: str
    params: tuple[NoteParam, ...]
    builder: Callable[[Mapping[str, float], float, float, Monitoring, int | None], Note]
    has_barrier: bool = False
    monitoring_choice: bool = False  # True when the barrier can be observed along the path

    def build(
        self,
        values: Mapping[str, float],
        maturity: float,
        initial: float,
        monitoring: Monitoring = Monitoring.EUROPEAN,
        observations: int | None = None,
    ) -> Note:
        """Build the note from parameter values (defaults for missing ones)."""
        full = {p.key: values.get(p.key, p.default) for p in self.params}
        return self.builder(full, maturity, initial, monitoring, observations)


def _cap(v: Mapping[str, float], key: str = "cap", toggle: str = "capped") -> float | None:
    return v[key] if v.get(toggle, 1.0) else UNCAPPED


PROTECTION = NoteParam("protection", "Capital protection", 100.0, "Minimum paid back at maturity.")
PARTICIPATION = NoteParam("participation", "Participation", 100.0, "Share of the move received.")

NOTE_CATALOG: tuple[NoteSpec, ...] = (
    NoteSpec(
        "Participation note",
        "Capital protected, plus a share of the rise above the strike, up to a cap.",
        (
            PROTECTION,
            NoteParam("strike", "Long call strike", 100.0),
            PARTICIPATION,
            NoteParam("capped", "Capped", 1.0, kind="toggle"),
            NoteParam("cap", "Cap (short call strike)", 130.0),
        ),
        lambda v, t, s0, mon, obs: participation_note(
            protection=v["protection"],
            strike=v["strike"],
            participation=v["participation"],
            cap=_cap(v),
            maturity=t,
            initial=s0,
        ),
    ),
    NoteSpec(
        "Bearish participation note",
        "Capital protected, plus a share of the fall below the strike, down to a limit.",
        (
            PROTECTION,
            NoteParam("strike", "Long put strike", 100.0),
            PARTICIPATION,
            NoteParam("limited", "Gain limited", 1.0, kind="toggle"),
            NoteParam("floor", "Gain limit (short put strike)", 70.0),
        ),
        lambda v, t, s0, mon, obs: bearish_participation_note(
            protection=v["protection"],
            strike=v["strike"],
            participation=v["participation"],
            floor=_cap(v, "floor", "limited"),
            maturity=t,
            initial=s0,
        ),
    ),
    NoteSpec(
        "Shark note (up)",
        "Participation in the rise until a barrier; at or above it, only a rebate is paid.",
        (
            PROTECTION,
            NoteParam("strike", "Long call strike", 100.0),
            PARTICIPATION,
            NoteParam("barrier", "Up-and-out barrier", 130.0),
            NoteParam("rebate", "Rebate if the barrier is touched", 5.0),
        ),
        lambda v, t, s0, mon, obs: shark_note(
            protection=v["protection"],
            strike=v["strike"],
            participation=v["participation"],
            barrier=v["barrier"],
            rebate=v["rebate"],
            direction="up",
            maturity=t,
            initial=s0,
            monitoring=mon,
            observations=obs,
        ),
        has_barrier=True,
        monitoring_choice=True,
    ),
    NoteSpec(
        "Shark note (down)",
        "Participation in the fall until a barrier; below it, only a rebate is paid.",
        (
            PROTECTION,
            NoteParam("strike", "Long put strike", 100.0),
            PARTICIPATION,
            NoteParam("barrier", "Down-and-out barrier", 70.0),
            NoteParam("rebate", "Rebate if the barrier is touched", 5.0),
        ),
        lambda v, t, s0, mon, obs: shark_note(
            protection=v["protection"],
            strike=v["strike"],
            participation=v["participation"],
            barrier=v["barrier"],
            rebate=v["rebate"],
            direction="down",
            maturity=t,
            initial=s0,
            monitoring=mon,
            observations=obs,
        ),
        has_barrier=True,
        monitoring_choice=True,
    ),
    NoteSpec(
        "Bonus note",
        "At least the bonus if the underlying ends above the barrier, its performance otherwise "
        "(never below the protection), plus a participation above the strike.",
        (
            NoteParam("protection", "Capital protection", 90.0, "Minimum paid back at maturity."),
            NoteParam("barrier", "Bonus barrier", 90.0),
            NoteParam("bonus", "Bonus level", 108.0),
            NoteParam("strike", "Long call strike", 108.0),
            PARTICIPATION,
            NoteParam("capped", "Capped", 1.0, kind="toggle"),
            NoteParam("cap", "Cap (short call strike)", 122.0),
        ),
        lambda v, t, s0, mon, obs: bonus_note(
            protection=v["protection"],
            barrier=v["barrier"],
            bonus=v["bonus"],
            strike=v["strike"],
            participation=v["participation"],
            cap=_cap(v),
            maturity=t,
            initial=s0,
            monitoring=mon,
        ),
        has_barrier=True,
    ),
    NoteSpec(
        "Twin-win note",
        "Capital protected, gains when the underlying rises above the call strike or falls "
        "below the put strike.",
        (
            PROTECTION,
            NoteParam("call_strike", "Call strike", 100.0, "The upside starts above it."),
            NoteParam("participation_up", "Upside participation", 100.0),
            NoteParam("cap", "Upside cap (short call strike)", 130.0),
            NoteParam("put_strike", "Put strike", 100.0, "The downside gain starts below it."),
            NoteParam("participation_down", "Downside participation", 100.0),
            NoteParam("floor", "Downside limit (short put strike)", 70.0),
        ),
        lambda v, t, s0, mon, obs: twin_win_note(
            protection=v["protection"],
            call_strike=v["call_strike"],
            put_strike=v["put_strike"],
            participation_up=v["participation_up"],
            cap=v["cap"],
            participation_down=v["participation_down"],
            floor=v["floor"],
            maturity=t,
            initial=s0,
        ),
    ),
    NoteSpec(
        "Digital coupon note",
        "Capital protected, plus a fixed coupon if the underlying ends above the barrier.",
        (
            PROTECTION,
            NoteParam("coupon", "Coupon", 8.0),
            NoteParam("barrier", "Coupon barrier", 100.0),
        ),
        lambda v, t, s0, mon, obs: digital_coupon_note(
            protection=v["protection"],
            coupon=v["coupon"],
            barrier=v["barrier"],
            maturity=t,
            initial=s0,
            monitoring=mon,
        ),
        has_barrier=True,
    ),
)


def find_note(name: str) -> NoteSpec:
    """Catalog entry with the given name."""
    for spec in NOTE_CATALOG:
        if spec.name == name:
            return spec
    raise KeyError(name)
