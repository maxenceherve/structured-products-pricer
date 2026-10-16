"""Yield enhancement and autocallable notes: coupons, early redemption, capital at risk.

One class, :class:`CouponNote`, describes them all with an explicit observation schedule: at
each date a coupon may be paid (if the worst performance is at or above the coupon level) and
the note may be called (at or above the autocall level), paying back par. At maturity, par is
paid back unless the protection barrier was breached and the worst performance ends below the
put strike; then the investor receives ``performance / strike`` of the notional.

Conventions, shared by every product:

* levels are in % of each underlying's initial level and apply to the worst performance;
* "at or above" is ``>=`` (coupon, autocall); a barrier is breached strictly below (``<``);
* redemption is at par (100 % of notional); the issue price only measures the margin;
* coupons are in % of notional per observation; with memory, missed coupons are paid at the
  next date where the coupon condition holds;
* the protection barrier is observed as the term sheet says (at maturity, continuously or
  daily), never on a simulation grid.

Products already alive carry their state: observations already past are dropped but the
schedule (non-call period, step-down) still counts from issue; coupons in memory and a barrier
already breached are inputs.

Named constructors build the usual products (EUSIPA families 12xx and 13xx): reverse
convertible, barrier reverse convertible, Athena (autocall paying the accumulated coupons when
called) and Phoenix (conditional coupons below the autocall level, with or without memory).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray
from spricer.bricks import BarrierOption, EuropeanOption, ZeroCoupon
from spricer.monitoring import Monitoring, observation_dates
from spricer.products.notes import NOTIONAL, Level, LevelKind, Note, NoteLeg

TRADING_DAYS = 252
FREQUENCIES: dict[str, int] = {"Monthly": 12, "Quarterly": 4, "Semi-annual": 2, "Annual": 1}
_TIME_TOL = 1e-9


@dataclass(frozen=True)
class Observation:
    """One observation date of a coupon note.

    Attributes
    ----------
    time
        Years from today (positive).
    coupon
        Coupon paid at this date, in % of notional, if the coupon condition holds.
    coupon_level
        Worst performance (% of initial) at or above which the coupon is paid; 0 for an
        unconditional coupon.
    autocall_level
        Worst performance at or above which the note is called (par paid back, note ends),
        or None when the note cannot be called at this date.
    """

    time: float
    coupon: float
    coupon_level: float
    autocall_level: float | None = None

    def __post_init__(self) -> None:
        if not (np.isfinite(self.time) and self.time > 0.0):
            raise ValueError("observation times must be positive")
        if not (np.isfinite(self.coupon) and self.coupon >= 0.0):
            raise ValueError("coupons must be non-negative")
        if not (np.isfinite(self.coupon_level) and self.coupon_level >= 0.0):
            raise ValueError("coupon levels must be non-negative")
        if self.autocall_level is not None and not (
            np.isfinite(self.autocall_level) and self.autocall_level > 0.0
        ):
            raise ValueError("autocall levels must be positive")


@dataclass(frozen=True)
class CouponNote:
    """A note paying coupons, possibly callable, with capital at risk below a barrier.

    Attributes
    ----------
    name
        Product name.
    observations
        Observation dates, increasing; the last one is maturity (no call there: par is paid
        back anyway when the capital is not at risk).
    protection_barrier
        Worst performance (% of initial) below which the capital is at risk.
    put_strike
        Strike of the put the investor has sold (% of initial): below it, after a breach, the
        investor receives ``performance / put_strike`` of the notional.
    monitoring
        How the protection barrier is observed.
    memory
        Missed coupons are paid later, at the next date where the coupon condition holds.
    memory_accrued
        Coupons already missed and kept in memory (% of notional), for a note alive.
    barrier_breached
        The protection barrier was already breached (note alive).
    """

    name: str
    observations: tuple[Observation, ...]
    protection_barrier: float
    put_strike: float = 100.0
    monitoring: Monitoring = Monitoring.EUROPEAN
    memory: bool = False
    memory_accrued: float = 0.0
    barrier_breached: bool = False

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("a coupon note needs at least one observation date (maturity)")
        times = np.array([o.time for o in self.observations])
        if np.any(np.diff(times) <= _TIME_TOL):
            raise ValueError("observation dates must be increasing")
        if not (np.isfinite(self.protection_barrier) and self.protection_barrier > 0.0):
            raise ValueError("the protection barrier must be positive")
        if not (np.isfinite(self.put_strike) and self.put_strike > 0.0):
            raise ValueError("the put strike must be positive")
        if not (np.isfinite(self.memory_accrued) and self.memory_accrued >= 0.0):
            raise ValueError("coupons in memory must be non-negative")
        if self.memory_accrued > 0.0 and not self.memory:
            raise ValueError("coupons in memory need a note with memory")

    def aged(self, years: float) -> CouponNote:
        """The same note ``years`` later: every date comes closer (for theta).

        Raises
        ------
        ValueError
            If an observation would fall today or in the past within that time.
        """
        if self.observations[0].time <= years + _TIME_TOL:
            raise ValueError("an observation date falls within this period")
        return replace(
            self, observations=tuple(replace(o, time=o.time - years) for o in self.observations)
        )

    @property
    def maturity(self) -> float:
        """Years to maturity."""
        return self.observations[-1].time

    @property
    def times(self) -> tuple[float, ...]:
        """Observation times."""
        return tuple(o.time for o in self.observations)

    @property
    def callable(self) -> bool:
        """True when the note can be called before maturity."""
        return any(o.autocall_level is not None for o in self.observations[:-1])

    @property
    def daily_observations(self) -> int:
        """Number of daily observation dates of the barrier from today to maturity."""
        return max(1, round(TRADING_DAYS * self.maturity))

    def barrier_dates(self) -> list[float] | None:
        """Observation dates of a daily barrier, None otherwise."""
        if self.monitoring is not Monitoring.DISCRETE:
            return None
        return observation_dates(self.maturity, self.daily_observations)

    @property
    def levels(self) -> tuple[Level, ...]:
        """Named levels for the payoff chart."""
        out = [
            Level("Protection barrier", self.protection_barrier, LevelKind.PERFORMANCE),
        ]
        if not np.isclose(self.put_strike, 100.0):
            out.append(Level("Put strike", self.put_strike, LevelKind.PERFORMANCE))
        final = self.observations[-1]
        if final.coupon_level > 0.0:
            out.append(Level("Coupon barrier", final.coupon_level, LevelKind.PERFORMANCE))
        calls = [o.autocall_level for o in self.observations if o.autocall_level is not None]
        if calls:
            out.append(Level("Autocall barrier", calls[0], LevelKind.PERFORMANCE))
        return tuple(out)

    def maturity_payoff(self, worst_pct: ArrayLike, breached: bool) -> FloatArray:
        """Payoff at maturity of a note never called, in % of notional (base coupon only).

        ``breached`` says whether the barrier was breached before maturity (path observation);
        a final performance below the barrier is a breach in any case.
        """
        p = np.asarray(worst_pct, dtype=np.float64)
        final = self.observations[-1]
        coupon = np.where(p >= final.coupon_level, final.coupon, 0.0)
        # ending below the barrier is a breach whatever the observation (maturity is observed)
        hit = (p < self.protection_barrier) | breached | self.barrier_breached
        loss = np.where(hit, np.maximum(self.put_strike - p, 0.0) / self.put_strike, 0.0)
        return np.asarray(NOTIONAL * (1.0 - loss) + coupon)

    def replication(self, initial: float) -> Note | None:
        """Exact bricks of a note that cannot be called and pays certain coupons, else None.

        Applies to a reverse convertible or a barrier reverse convertible on one underlying.

        ``ZC(coupons) + 100 ZC(T) - (100 / K) x PutDownAndIn(K, B)``, or a vanilla put once the
        barrier is breached or when it sits at or above the strike with European observation.
        """
        if self.callable or any(o.coupon_level > 0.0 for o in self.observations):
            return None
        t = self.maturity
        k, h = self.put_strike / 100.0 * initial, self.protection_barrier / 100.0 * initial
        legs = [
            NoteLeg(f"Coupon {o.time:g}y", ZeroCoupon(o.time), o.coupon)
            for o in self.observations
            if o.coupon > 0.0
        ]
        legs.append(NoteLeg("Zero-coupon bond", ZeroCoupon(t), NOTIONAL))
        put: BarrierOption | EuropeanOption
        if self.barrier_breached:
            put = EuropeanOption("put", k, t)
        else:
            observations = (
                self.daily_observations if self.monitoring is Monitoring.DISCRETE else None
            )
            put = BarrierOption("put", "down-and-in", k, h, t, self.monitoring, observations)
        legs.append(NoteLeg("Short put", put, -NOTIONAL / k))

        def formula(p: FloatArray) -> FloatArray:
            return self.maturity_payoff(p, breached=False) + sum(
                o.coupon for o in self.observations[:-1]
            )

        return Note(self.name, t, initial, tuple(legs), formula, self.levels, self.monitoring)


# --- Schedules ----------------------------------------------------------------------------


def schedule(maturity: float, frequency: int, elapsed: float = 0.0) -> list[tuple[int, float]]:
    """Observation dates of a note issued ``elapsed`` years ago.

    Returns ``(k, t)`` pairs: ``k`` counts observations from issue (1 for the first) and ``t``
    is the time from today, for the dates still to come.
    """
    if not (np.isfinite(maturity) and maturity > 0.0):
        raise ValueError("maturity must be positive")
    if frequency not in FREQUENCIES.values():
        raise ValueError(f"frequency must be one of {sorted(FREQUENCIES.values())} a year")
    n = round(maturity * frequency)
    if n < 1 or not np.isclose(n / frequency, maturity):
        raise ValueError("maturity must be a whole number of observation periods")
    if not (np.isfinite(elapsed) and 0.0 <= elapsed < maturity):
        raise ValueError("elapsed time must be between 0 and the maturity")
    out = [(k, k / frequency - elapsed) for k in range(1, n + 1)]
    return [(k, t) for k, t in out if t > _TIME_TOL]


def _autocall_level(k: int, n: int, level: float, non_call: int, step_down: float) -> float | None:
    """Autocall level at observation ``k`` of ``n`` (counted from issue)."""
    if k <= non_call or k == n:
        return None
    return max(level - (k - non_call - 1) * step_down, 1e-6)


@dataclass(frozen=True)
class LifeState:
    """State of a note already alive."""

    elapsed: float = 0.0
    memory_accrued: float = 0.0
    barrier_breached: bool = False


NEW_NOTE = LifeState()  # a note priced at issue


def reverse_convertible(
    *,
    maturity: float,
    frequency: int,
    coupon_pa: float,
    strike: float = 100.0,
    state: LifeState = NEW_NOTE,
) -> CouponNote:
    """Certain coupons; at maturity the notional, or ``performance / strike`` below the strike."""
    obs = tuple(
        Observation(t, coupon_pa / frequency, 0.0)
        for _, t in schedule(maturity, frequency, state.elapsed)
    )
    return CouponNote(
        "Reverse convertible",
        obs,
        protection_barrier=strike,
        put_strike=strike,
        barrier_breached=state.barrier_breached,
    )


def barrier_reverse_convertible(
    *,
    maturity: float,
    frequency: int,
    coupon_pa: float,
    barrier: float,
    strike: float = 100.0,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    state: LifeState = NEW_NOTE,
) -> CouponNote:
    """Certain coupons; the capital is at risk only once the barrier has been breached."""
    obs = tuple(
        Observation(t, coupon_pa / frequency, 0.0)
        for _, t in schedule(maturity, frequency, state.elapsed)
    )
    return CouponNote(
        "Barrier reverse convertible",
        obs,
        protection_barrier=barrier,
        put_strike=strike,
        monitoring=monitoring,
        barrier_breached=state.barrier_breached,
    )


def athena(
    *,
    maturity: float,
    frequency: int,
    coupon_pa: float,
    autocall_level: float = 100.0,
    barrier: float = 60.0,
    final_coupon_level: float | None = None,
    step_down: float = 0.0,
    non_call: int = 0,
    strike: float = 100.0,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    state: LifeState = NEW_NOTE,
) -> CouponNote:
    """Autocall paying, when called, every coupon since issue (coupons accumulate).

    The coupon condition is the autocall condition, so the memory pays the accumulated coupons
    exactly at the call. At maturity they are paid at or above ``final_coupon_level`` (by
    default the last autocall level).
    """
    n = round(maturity * frequency)
    dates = schedule(maturity, frequency, state.elapsed)
    last_call = _autocall_level(n - 1, n, autocall_level, non_call, step_down) if n > 1 else None
    final_level = (
        final_coupon_level if final_coupon_level is not None else (last_call or autocall_level)
    )
    obs = []
    for k, t in dates:
        call = _autocall_level(k, n, autocall_level, non_call, step_down)
        level = final_level if k == n else (call if call is not None else autocall_level)
        obs.append(Observation(t, coupon_pa / frequency, level, call))
    return CouponNote(
        "Athena autocall",
        tuple(obs),
        protection_barrier=barrier,
        put_strike=strike,
        monitoring=monitoring,
        memory=True,
        memory_accrued=state.memory_accrued,
        barrier_breached=state.barrier_breached,
    )


def phoenix(
    *,
    maturity: float,
    frequency: int,
    coupon_pa: float,
    coupon_level: float = 70.0,
    autocall_level: float = 100.0,
    barrier: float = 60.0,
    memory: bool = True,
    step_down: float = 0.0,
    non_call: int = 0,
    strike: float = 100.0,
    monitoring: Monitoring = Monitoring.EUROPEAN,
    state: LifeState = NEW_NOTE,
) -> CouponNote:
    """Autocall with conditional coupons paid at or above a lower coupon level."""
    n = round(maturity * frequency)
    obs = tuple(
        Observation(
            t,
            coupon_pa / frequency,
            coupon_level,
            _autocall_level(k, n, autocall_level, non_call, step_down),
        )
        for k, t in schedule(maturity, frequency, state.elapsed)
    )
    return CouponNote(
        "Phoenix autocall",
        obs,
        protection_barrier=barrier,
        put_strike=strike,
        monitoring=monitoring,
        memory=memory,
        memory_accrued=state.memory_accrued if memory else 0.0,
        barrier_breached=state.barrier_breached,
    )


# --- Catalog for the app ------------------------------------------------------------------


@dataclass(frozen=True)
class CouponParam:
    """An input of a coupon note (% unless ``kind`` says otherwise)."""

    key: str
    label: str
    default: float
    help: str = ""
    kind: str = "pct"  # "pct", "count" (whole number) or "toggle" (1.0 = on)
    solvable: bool = False  # offered by the solver


Builder = Callable[[Mapping[str, float], float, int, Monitoring, LifeState], CouponNote]


@dataclass(frozen=True)
class CouponNoteSpec:
    """A coupon note as offered by the app."""

    name: str
    family: str  # "Yield enhancement" or "Autocallables"
    description: str
    params: tuple[CouponParam, ...]
    builder: Builder
    monitoring_choice: bool = True

    def build(
        self,
        values: Mapping[str, float],
        maturity: float,
        frequency: int,
        monitoring: Monitoring = Monitoring.EUROPEAN,
        state: LifeState = NEW_NOTE,
    ) -> CouponNote:
        """Build the note from parameter values (defaults for missing ones)."""
        full = {p.key: values.get(p.key, p.default) for p in self.params}
        return self.builder(full, maturity, frequency, monitoring, state)

    def param(self, key: str) -> CouponParam:
        """The parameter with this key."""
        return next(p for p in self.params if p.key == key)


COUPON = CouponParam(
    "coupon_pa", "Coupon (% a year)", 8.0, "Paid in equal parts at each date.", solvable=True
)
STRIKE = CouponParam("strike", "Put strike", 100.0, "Below it, after a breach, the loss starts.")
BARRIER = CouponParam(
    "barrier", "Protection barrier", 60.0, "Capital at risk once breached.", solvable=True
)
AUTOCALL = CouponParam(
    "autocall_level", "Autocall barrier", 100.0, "Called at or above it.", solvable=True
)
STEP_DOWN = CouponParam(
    "step_down", "Step-down per date", 0.0, "Autocall barrier lowered each date."
)
NON_CALL = CouponParam(
    "non_call", "Dates without call", 0.0, "First dates where no call is possible.", kind="count"
)

COUPON_NOTE_CATALOG: tuple[CouponNoteSpec, ...] = (
    CouponNoteSpec(
        "Reverse convertible",
        "Yield enhancement",
        "A high certain coupon; at maturity the notional, or the performance if it ends below "
        "the strike: the investor has sold a put.",
        (COUPON, STRIKE),
        lambda v, t, f, mon, st: reverse_convertible(
            maturity=t, frequency=f, coupon_pa=v["coupon_pa"], strike=v["strike"], state=st
        ),
        monitoring_choice=False,
    ),
    CouponNoteSpec(
        "Barrier reverse convertible",
        "Yield enhancement",
        "A certain coupon; the capital is at risk only if the barrier is breached: the investor "
        "has sold a down-and-in put.",
        (replace(COUPON, default=6.0), BARRIER, STRIKE),
        lambda v, t, f, mon, st: barrier_reverse_convertible(
            maturity=t,
            frequency=f,
            coupon_pa=v["coupon_pa"],
            barrier=v["barrier"],
            strike=v["strike"],
            monitoring=mon,
            state=st,
        ),
    ),
    CouponNoteSpec(
        "Athena autocall",
        "Autocallables",
        "Called back at par as soon as the worst performance is at or above the autocall "
        "barrier, with every coupon since issue; capital at risk below the barrier at maturity.",
        (COUPON, AUTOCALL, BARRIER, STEP_DOWN, NON_CALL, STRIKE),
        lambda v, t, f, mon, st: athena(
            maturity=t,
            frequency=f,
            coupon_pa=v["coupon_pa"],
            autocall_level=v["autocall_level"],
            barrier=v["barrier"],
            step_down=v["step_down"],
            non_call=round(v["non_call"]),
            strike=v["strike"],
            monitoring=mon,
            state=st,
        ),
    ),
    CouponNoteSpec(
        "Phoenix autocall",
        "Autocallables",
        "A coupon at each date where the worst performance is above the coupon barrier (with "
        "memory of missed ones), called back at par above the autocall barrier.",
        (
            COUPON,
            CouponParam(
                "coupon_level", "Coupon barrier", 70.0, "Coupon paid at or above it.", solvable=True
            ),
            AUTOCALL,
            BARRIER,
            CouponParam("memory", "Memory of missed coupons", 1.0, kind="toggle"),
            STEP_DOWN,
            NON_CALL,
            STRIKE,
        ),
        lambda v, t, f, mon, st: phoenix(
            maturity=t,
            frequency=f,
            coupon_pa=v["coupon_pa"],
            coupon_level=v["coupon_level"],
            autocall_level=v["autocall_level"],
            barrier=v["barrier"],
            memory=bool(v["memory"]),
            step_down=v["step_down"],
            non_call=round(v["non_call"]),
            strike=v["strike"],
            monitoring=mon,
            state=st,
        ),
    ),
)


def find_coupon_note(name: str) -> CouponNoteSpec:
    """The catalog entry with this name."""
    for spec in COUPON_NOTE_CATALOG:
        if spec.name == name:
            return spec
    raise KeyError(name)
