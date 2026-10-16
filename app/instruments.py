"""Option positions shared by the Greeks lab and the Hedging page: inputs and construction."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

import state
from spricer import strategies as strat
from spricer.bricks import BARRIER_TYPES, BarrierOption
from spricer.greeks_lab import Instrument
from spricer.market import FlatVol, Market
from spricer.monitoring import Monitoring
from spricer.strategies import Family, Leg, Strategy

INSTRUMENTS = {
    "Call": "vanilla",
    "Put": "vanilla",
    "Digital call": "digital",
    "Digital put": "digital",
    "Bull call spread": "spread",
    "Straddle": "volatility",
    "Call butterfly": "spread",
    "Barrier option": "barrier",
}
DIGITAL_PAYOUT_PCT = 10.0  # % of the spot paid by a digital
FLAT, SMILE = "Flat", "Smile of the Volatility surface page"


@dataclass(frozen=True)
class Selection:
    """The position chosen on screen."""

    name: str
    family: str
    values: dict[str, float | str]
    maturity: float
    vol_source: str
    flat_vol_pct: float

    @property
    def short(self) -> bool:
        """True for a short position."""
        return self.values["side"] == "Short"

    @property
    def strikes_pct(self) -> tuple[float, ...]:
        """Strikes in % of today's spot."""
        k, w = float(self.values["strike"]), float(self.values["width"])
        if self.name == "Bull call spread":
            return (k, k + w)
        if self.name == "Call butterfly":
            return (k - w, k, k + w)
        return (k,)

    @property
    def kind(self) -> str:
        """Call or put side of the main option."""
        if self.name == "Barrier option":
            return str(self.values["kind"])
        return "put" if self.name in ("Put", "Digital put") else "call"

    @property
    def barrier_crossed(self) -> bool:
        """True when a barrier option's barrier is already on the wrong side of today's spot."""
        if self.family != "barrier":
            return False
        level = float(self.values["barrier"])
        return (
            level <= 100.0 if str(self.values["barrier_type"]).startswith("up") else level > 100.0
        )

    def market(self, base: Market) -> Market:
        """The page's market, with a flat volatility if chosen."""
        if self.vol_source == FLAT:
            return Market(
                base.spot, base.rate, base.dividend_yield, FlatVol(self.flat_vol_pct / 100.0)
            )
        return base

    def instrument(self, spot: float) -> Instrument:
        """Strategy as a function of the maturity."""
        values = self.values
        name = self.name
        k = float(values["strike"]) / 100.0 * spot
        w = float(values["width"]) / 100.0 * spot

        def build(t: float) -> Strategy:
            if name == "Call":
                s = strat.call(k, t)
            elif name == "Put":
                s = strat.put(k, t)
            elif name == "Digital call":
                s = strat.digital_call(k, t, DIGITAL_PAYOUT_PCT / 100.0 * spot)
            elif name == "Digital put":
                s = strat.digital_put(k, t, DIGITAL_PAYOUT_PCT / 100.0 * spot)
            elif name == "Bull call spread":
                s = strat.bull_call_spread(k, k + w, t)
            elif name == "Straddle":
                s = strat.straddle(k, t)
            elif name == "Call butterfly":
                s = strat.call_butterfly(max(k - w, 1e-6), k, k + w, t)
            else:
                option = BarrierOption(
                    str(values["kind"]),  # type: ignore[arg-type]
                    str(values["barrier_type"]),  # type: ignore[arg-type]
                    k,
                    float(values["barrier"]) / 100.0 * spot,
                    t,
                    Monitoring.CONTINUOUS
                    if values["monitoring"] == "Continuously"
                    else Monitoring.EUROPEAN,
                )
                s = Strategy(option.label, Family.CUSTOM, (Leg(option, 1.0),))
            return s.short() if values["side"] == "Short" else s

        return build


def draw_inputs(prefix: str) -> Selection:
    """Draw the instrument, position, strike, maturity and volatility inputs."""
    name = state.choice(st.pills, "Instrument", f"{prefix}_instrument", list(INSTRUMENTS))
    family = INSTRUMENTS[name]
    values: dict[str, float | str] = {}
    top = st.columns(3)
    with top[0]:
        values["side"] = state.choice(
            st.radio, "Position", f"{prefix}_side", ["Long", "Short"], horizontal=True
        )
    with top[1]:
        vol_source = state.choice(
            st.radio, "Volatility", f"{prefix}_vol_source", [FLAT, SMILE], horizontal=True
        )
    sliders = st.columns(3)
    with sliders[0]:
        values["strike"] = state.persistent(
            st.slider,
            "Strike (% of spot)",
            f"{prefix}_strike",
            100.0,
            min_value=50.0,
            max_value=150.0,
            step=1.0,
        )
    with sliders[1]:
        maturity = state.persistent(
            st.slider,
            "Time to maturity (years)",
            f"{prefix}_maturity",
            1.0,
            min_value=0.02,
            max_value=5.0,
            step=0.01,
        )
    with sliders[2]:
        flat_vol = state.persistent(
            st.slider,
            "Flat volatility (%)",
            f"{prefix}_vol",
            20.0,
            min_value=5.0,
            max_value=80.0,
            step=0.5,
            disabled=vol_source != FLAT,
        )
    values["width"] = 10.0
    if family == "spread":
        values["width"] = state.persistent(
            st.slider,
            "Distance between strikes (% of spot)",
            f"{prefix}_width",
            10.0,
            min_value=2.0,
            max_value=40.0,
            step=1.0,
        )
    if family == "barrier":
        b = st.columns(4)
        with b[0]:
            values["kind"] = state.choice(st.radio, "Option", f"{prefix}_kind", ["call", "put"])
        with b[1]:
            values["barrier_type"] = state.choice(
                st.selectbox,
                "Barrier",
                f"{prefix}_barrier_type",
                list(BARRIER_TYPES),
                "down-and-out",
            )
        with b[2]:
            values["barrier"] = state.persistent(
                st.slider,
                "Barrier (% of spot)",
                f"{prefix}_barrier",
                80.0,
                min_value=40.0,
                max_value=160.0,
                step=1.0,
            )
        with b[3]:
            values["monitoring"] = state.choice(
                st.radio, "Observed", f"{prefix}_monitoring", ["Continuously", "At maturity"]
            )
    return Selection(name, family, values, float(maturity), vol_source, float(flat_vol))


def crossed_barrier_warning(selection: Selection) -> None:
    """Warn when a barrier option is already knocked at today's spot."""
    if selection.barrier_crossed:
        st.warning(
            "The barrier is already crossed at today's spot: a knock-out option is dead, a "
            "knock-in option is a vanilla. Move the barrier to the other side of 100 %."
        )
