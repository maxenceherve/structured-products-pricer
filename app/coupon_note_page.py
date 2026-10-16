"""Page shared by the coupon note families (Yield enhancement, Autocallables).

Product, underlyings and market data, Price button, then the Monte Carlo result: price and
margin, legs on common paths, payoff at maturity, when the note ends, and the solver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import charts
import presentation as pres
import state
from spricer.market import Basket, Market
from spricer.mc import check_correlation, nearest_correlation
from spricer.mc.coupon_notes import LEG_NAMES, CouponNoteMonteCarlo, price_coupon_note
from spricer.mc.paths import Z_95
from spricer.monitoring import Monitoring
from spricer.products.coupon_notes import (
    COUPON_NOTE_CATALOG,
    FREQUENCIES,
    CouponNote,
    CouponNoteSpec,
    LifeState,
)
from spricer.solve import solve_coupon_note

MONITORING_CHOICES = {
    "At maturity": Monitoring.EUROPEAN,
    "Continuously": Monitoring.CONTINUOUS,
    "Daily (252 dates a year)": Monitoring.DISCRETE,
}
PATHS = {"20,000": 20_000, "50,000": 50_000, "100,000": 100_000}
UNDERLYING_COLUMNS = ["Name", "Spot", "Initial level", "Dividend yield (%)", "ATM vol 1y (%)"]
MAX_UNDERLYINGS = 5
DEFAULT_CORRELATION = 0.6
SOLVE_BOUNDS = {
    "coupon_pa": (0.0, 60.0),
    "barrier": (1.0, 150.0),
    "autocall_level": (40.0, 200.0),
    "coupon_level": (0.0, 150.0),
}


def _key(family: str, name: str) -> str:
    return f"cn_{family.split()[0].lower()}_{name}"


def _editor(data: pd.DataFrame, key: str, **kwargs: Any) -> pd.DataFrame:
    """Data editor whose edits survive page switches.

    Streamlit drops a widget's state when a run does not draw it. The last edited table is
    kept under ``<key>_last`` and becomes the editor's data whenever the widget state is gone.
    """
    base, last = f"{key}_base", f"{key}_last"
    st.session_state.setdefault(base, data)
    if key not in st.session_state:
        st.session_state[base] = st.session_state.get(last, st.session_state[base])
    edited: pd.DataFrame = st.data_editor(st.session_state[base], key=key, **kwargs)
    st.session_state[last] = edited
    return edited


def _reset_editor(key: str) -> None:
    for suffix in ("", "_base", "_last"):
        st.session_state.pop(f"{key}{suffix}", None)


# --- Inputs -------------------------------------------------------------------------------------


def _terms(spec: CouponNoteSpec, family: str) -> dict[str, float]:
    values: dict[str, float] = {}
    cols = st.columns(4)
    for i, param in enumerate(spec.params):
        key = _key(family, f"{spec.name}_{param.key}")
        with cols[i % 4]:
            if param.kind == "toggle":
                on = state.persistent(st.checkbox, param.label, key, bool(param.default))
                values[param.key] = 1.0 if on else 0.0
            elif param.kind == "count":
                values[param.key] = float(
                    state.persistent(
                        st.number_input,
                        param.label,
                        key,
                        int(param.default),
                        min_value=0,
                        step=1,
                        help=param.help or None,
                    )
                )
            else:
                values[param.key] = state.persistent(
                    st.number_input,
                    f"{param.label} (%)",
                    key,
                    param.default,
                    min_value=0.0,
                    step=1.0,
                    format="%.2f",
                    help=param.help or None,
                )
    return values


def _product(family: str) -> tuple[CouponNote | None, CouponNoteSpec, dict[str, Any]]:
    specs = [s for s in COUPON_NOTE_CATALOG if s.family == family]
    name = state.choice(st.pills, "Product", _key(family, "product"), [s.name for s in specs])
    spec = next(s for s in specs if s.name == name)
    st.markdown(f"**{spec.name}** — {spec.description}")
    top = st.columns(4)
    with top[0]:
        maturity = state.persistent(
            st.number_input,
            "Maturity (years)",
            _key(family, "maturity"),
            3.0,
            min_value=0.25,
            max_value=15.0,
            step=0.25,
            format="%.2f",
        )
    with top[1]:
        frequency_label = state.choice(
            st.selectbox,
            "Observation frequency",
            _key(family, "frequency"),
            list(FREQUENCIES),
            "Quarterly",
        )
    with top[2]:
        issue_price = state.persistent(
            st.number_input,
            "Issue price (%)",
            _key(family, "issue_price"),
            100.0,
            min_value=1.0,
            step=0.5,
            format="%.2f",
            help="What the investor pays. Redemption is always at par; the gap between the "
            "issue price and the fair value is the issuer's margin.",
        )
    with top[3]:
        notional = state.persistent(
            st.number_input,
            "Notional",
            _key(family, "notional"),
            1_000_000.0,
            min_value=1.0,
            step=100_000.0,
            format="%.0f",
        )
    st.markdown("**Terms** (levels in % of the initial level, coupons in % of the notional)")
    values = _terms(spec, family)
    monitoring = Monitoring.EUROPEAN
    if spec.monitoring_choice:
        choice = state.choice(
            st.radio,
            "Protection barrier observed",
            _key(family, "monitoring"),
            list(MONITORING_CHOICES),
            horizontal=True,
            help="As written in the term sheet: at maturity only, continuously, or at each "
            "daily close.",
        )
        monitoring = MONITORING_CHOICES[choice]
    with st.expander("Note already alive (optional)"):
        st.caption(
            "For a note issued in the past: dates already observed are dropped, but the "
            "non-call period and the step-down still count from issue."
        )
        alive = st.columns(3)
        with alive[0]:
            elapsed = state.persistent(
                st.number_input,
                "Years since issue",
                _key(family, "elapsed"),
                0.0,
                min_value=0.0,
                step=0.25,
                format="%.2f",
            )
        with alive[1]:
            memory = state.persistent(
                st.number_input,
                "Coupons in memory (%)",
                _key(family, "memory_accrued"),
                0.0,
                min_value=0.0,
                step=0.5,
                format="%.2f",
                help="Coupons already missed and still owed (notes with memory only).",
            )
        with alive[2]:
            breached = state.persistent(
                st.checkbox, "Barrier already breached", _key(family, "breached"), False
            )
    frequency = FREQUENCIES[frequency_label]
    context: dict[str, Any] = {
        "maturity": maturity,
        "frequency": frequency,
        "monitoring": monitoring,
        "issue_price": issue_price,
        "notional": notional,
        "values": values,
        "state": LifeState(elapsed, memory, breached),
    }
    try:
        note = spec.build(values, maturity, frequency, monitoring, context["state"])
    except ValueError as exc:
        st.error(str(exc))
        return None, spec, context
    return note, spec, context


def _default_underlyings(base: Market) -> pd.DataFrame:
    atm = 100.0 * float(base.vol.vol(base.spot, 1.0))
    return pd.DataFrame(
        [["Underlying 1", base.spot, base.spot, 100.0 * base.dividend_yield, round(atm, 2)]],
        columns=UNDERLYING_COLUMNS,
    )


def _markets(table: pd.DataFrame, base: Market) -> list[Market]:
    """One market per underlying: the page's smile, shifted to its ATM volatility."""
    if table.isna().to_numpy().any():
        raise ValueError(
            "Every underlying needs a name, a spot, an initial level, a dividend "
            "yield and an ATM volatility."
        )
    base_atm = 100.0 * float(base.vol.vol(base.spot, 1.0))
    grid = st.session_state[state.GRID]
    out = []
    for _, row in table.iterrows():
        shifted = grid.copy()
        columns = [c for c in shifted.columns if c != state.MATURITY_COLUMN]
        shifted[columns] = shifted[columns] + (float(row["ATM vol 1y (%)"]) - base_atm)
        if float(row["Spot"]) <= 0.0:
            raise ValueError("Spots must be positive.")
        out.append(
            state.build_market(
                float(row["Spot"]),
                100.0 * base.rate,
                float(row["Dividend yield (%)"]),
                shifted,
            )
        )
    return out


def _correlation(family: str, names: list[str]) -> np.ndarray | None:
    """Correlation editor for several underlyings; None when it is not valid."""
    key = _key(family, "correlation")
    stored = st.session_state.get(key)
    if stored is None or list(stored.index) != names:
        size = len(names)
        matrix = np.full((size, size), DEFAULT_CORRELATION)
        np.fill_diagonal(matrix, 1.0)
        if stored is not None:  # keep the pairs the user already typed
            for i, a in enumerate(names):
                for j, b in enumerate(names):
                    if a in stored.index and b in stored.columns:
                        matrix[i, j] = float(stored.loc[a, b])
        stored = pd.DataFrame(matrix, index=names, columns=names)
        st.session_state[key] = stored
        _reset_editor(f"{key}_editor")
    st.markdown("**Correlation** of the underlyings' returns (symmetric, one on the diagonal)")
    edited = _editor(
        stored,
        f"{key}_editor",
        column_config={n: st.column_config.NumberColumn(n, format="%.2f") for n in names},
    )
    st.session_state[key] = edited
    try:
        return check_correlation(edited.to_numpy(dtype=float), len(names))
    except ValueError as exc:
        st.error(str(exc))
        if st.button("Use the nearest valid matrix", key=f"{key}_fix"):
            fixed = nearest_correlation(0.5 * (edited.to_numpy(float) + edited.to_numpy(float).T))
            st.session_state[key] = pd.DataFrame(fixed, index=names, columns=names)
            _reset_editor(f"{key}_editor")
            st.rerun()
        return None


def _basket(family: str, base: Market) -> Basket | None:
    st.markdown("**Underlyings** — the note looks at the worst performance")
    table = _editor(
        _default_underlyings(base),
        _key(family, "underlyings"),
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Spot": st.column_config.NumberColumn(min_value=0.01, format="%.2f"),
            "Initial level": st.column_config.NumberColumn(min_value=0.01, format="%.2f"),
            "Dividend yield (%)": st.column_config.NumberColumn(format="%.2f"),
            "ATM vol 1y (%)": st.column_config.NumberColumn(
                min_value=1.0,
                format="%.2f",
                help="Shifts the smile of the Volatility surface page to this level.",
            ),
        },
    )
    st.caption(
        "Each underlying uses the smile of the Volatility surface page, shifted in parallel to "
        "its own 1-year ATM volatility. Add rows for a worst-of."
    )
    if len(table) > MAX_UNDERLYINGS:
        st.error(f"At most {MAX_UNDERLYINGS} underlyings.")
        return None
    if table.empty:
        st.error("Add at least one underlying.")
        return None
    names = [str(n) for n in table["Name"]]
    if len(set(names)) != len(names):
        st.error("Underlyings need distinct names.")
        return None
    try:
        markets = _markets(table, base)
    except ValueError as exc:
        st.error(str(exc))
        return None
    correlation = _correlation(family, names) if len(names) > 1 else None
    if len(names) > 1 and correlation is None:
        return None
    try:
        return Basket.of(markets, table["Initial level"].astype(float).tolist(), names, correlation)
    except ValueError as exc:
        st.error(str(exc))
        return None


# --- Results ------------------------------------------------------------------------------------


def _legs_table(result: CouponNoteMonteCarlo) -> pd.DataFrame:
    rows = [
        {"Leg": name, "Price (%)": est.value, "± 1 std error (%)": est.std_error}
        for name, est in zip(LEG_NAMES, result.legs, strict=True)
    ]
    rows.append(
        {
            "Leg": "Total",
            "Price (%)": result.total.value,
            "± 1 std error (%)": result.total.std_error,
        }
    )
    return pd.DataFrame(rows)


def _schedule_table(note: CouponNote, result: CouponNoteMonteCarlo) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date (years)": [o.time for o in note.observations],
            "Coupon (%)": [o.coupon for o in note.observations],
            "Coupon barrier (%)": [o.coupon_level for o in note.observations],
            "Autocall barrier (%)": [
                o.autocall_level if o.autocall_level is not None else np.nan
                for o in note.observations
            ],
            "Call probability (%)": [100.0 * p.value for p in result.call_probability],
        }
    )


def _closed_form_check(
    note: CouponNote, basket: Basket, spread: float, mc: CouponNoteMonteCarlo
) -> None:
    if basket.size != 1:
        return
    replication = note.replication(basket.initials[0])
    if replication is None:
        return
    closed = float(replication.value(basket.markets[0], spread))
    gap = (mc.total.value - closed) / mc.total.std_error
    cols = st.columns(3)
    cols[0].metric("Closed form (bricks)", f"{closed:.2f} %")
    cols[1].metric("MC − closed form", f"{mc.total.value - closed:+.2f} %")
    cols[2].metric("Gap", f"{gap:+.1f} std errors")
    if note.monitoring is Monitoring.EUROPEAN or note.barrier_breached:
        st.caption(
            "Bond, coupons and a put read on the smile: the gap is Monte Carlo noise, within "
            "±2 standard errors 95 % of the time."
        )
    else:
        st.info(
            "The closed form prices the down-and-in put at one volatility, the strike's; the "
            "Monte Carlo diffuses on the local volatility of the whole smile, where the vol at "
            "the barrier is higher. The gap is the smile's effect on the barrier, not noise."
        )


def _solver(
    family: str,
    spec: CouponNoteSpec,
    context: dict[str, Any],
    basket: Basket,
    spread: float,
    n: int,
) -> None:
    solvable = [p for p in spec.params if p.solvable]
    st.subheader("Solve for")
    st.caption(
        "Fix every term but one and find the value that prices the note at the issue price "
        "minus your margin: what a structurer does before quoting."
    )
    cols = st.columns([2, 1, 1], vertical_alignment="bottom")
    with cols[0]:
        label = state.choice(
            st.selectbox, "Term to solve", _key(family, "solve_term"), [p.label for p in solvable]
        )
    param = next(p for p in solvable if p.label == label)
    with cols[1]:
        margin = state.persistent(
            st.number_input,
            "Margin (%)",
            _key(family, "solve_margin"),
            2.0,
            step=0.25,
            format="%.2f",
        )
    target = float(context["issue_price"]) - margin
    with cols[2]:
        run = st.button("Solve", icon=":material/target:", key=_key(family, "solve_button"))
    solve_key = _key(family, "solution")
    fingerprint = pres.signature(spec.name, param.key, target, context, spread, n)
    if run:
        values = dict(context["values"])

        def build(x: float) -> CouponNote:
            return spec.build(
                {**values, param.key: x},
                context["maturity"],
                context["frequency"],
                context["monitoring"],
                context["state"],
            )

        with st.spinner("Solving on common paths..."):
            try:
                solution, _ = solve_coupon_note(
                    build,
                    basket,
                    target,
                    SOLVE_BOUNDS[param.key],
                    spread,
                    what=label.lower(),
                    n_paths=n,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
        st.session_state[solve_key] = (fingerprint, solution)
    stored = st.session_state.get(solve_key)
    if stored is None or stored[0] != fingerprint:
        st.caption(f"Target price: {target:.2f} % (issue price − margin).")
        return
    solution = stored[1]
    error = f" ± {solution.std_error:.2f}" if solution.std_error is not None else ""
    st.success(
        f"**{label} = {solution.value:.2f} %**{error} prices the note at {solution.price:.2f} % "
        f"(target {target:.2f} %)."
    )
    if st.button("Use this value", key=_key(family, "solve_apply")):
        st.session_state[_key(family, f"{spec.name}_{param.key}")] = round(solution.value, 2)
        st.rerun()


def render(family: str, title: str) -> None:
    """Draw the page of one family of coupon notes."""
    st.title(title)
    with st.container(border=True):
        st.subheader("1 · Product")
        note, spec, context = _product(family)

    try:
        base = state.current_market()
    except ValueError as exc:
        st.error(f"The volatility surface is invalid: {exc}")
        st.page_link("views/volatility_surface.py", label="Fix the volatility surface")
        st.stop()

    with st.container(border=True):
        st.subheader("2 · Underlyings and market data")
        m = st.columns(3)
        with m[0]:
            state.persistent(
                st.number_input,
                "Interest rate (%)",
                state.RATE_PCT,
                3.0,
                step=0.25,
                format="%.2f",
                help="Continuously compounded risk-free rate: drift of the underlyings.",
            )
        with m[1]:
            spread_bp = state.persistent(
                st.number_input,
                "Issuer credit spread (bp)",
                _key(family, "spread_bp"),
                0.0,
                step=10.0,
                format="%.0f",
                help="Discounts every cash flow of the note, not the drift.",
            )
        with m[2]:
            paths_label = state.choice(
                st.radio,
                "Monte Carlo paths",
                _key(family, "paths"),
                list(PATHS),
                "50,000",
                horizontal=True,
            )
        base = state.current_market()
        basket = _basket(family, base)

    if note is None or basket is None:
        st.stop()
    spread = spread_bp / 1e4
    n_paths = PATHS[paths_label]
    fingerprint = pres.signature(
        spec.name,
        context,
        spread,
        n_paths,
        basket.initials,
        basket.names,
        [(m.spot, m.rate, m.dividend_yield) for m in basket.markets],
        None if basket.correlation is None else basket.correlation.round(6).tolist(),
        st.session_state[state.GRID],
    )
    result_key = _key(family, "result")
    if st.button("Price", type="primary", icon=":material/calculate:", key=_key(family, "price")):
        with st.spinner("Simulating paths under the local volatility..."):
            try:
                priced = price_coupon_note(note, basket, spread, n_paths=n_paths)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
        st.session_state[result_key] = (fingerprint, priced)
    stored = st.session_state.get(result_key)
    if stored is None or stored[0] != fingerprint:
        st.info("Set the product and the underlyings, then click **Price**.")
        st.stop()
    result: CouponNoteMonteCarlo = stored[1]
    price = result.total.value
    issue_price = float(context["issue_price"])

    with st.container(border=True):
        st.subheader(f"Result · {note.name}")
        top, bottom = st.columns(2), st.columns(2)
        top[0].metric(
            "Price",
            f"{price:.2f} %",
            f"± {Z_95 * result.total.std_error:.2f} % at 95 %",
            delta_color="off",
            delta_arrow="off",
            help="Monte Carlo value under the local volatility of the smile, in % of notional.",
        )
        top[1].metric(
            "Issuer margin",
            f"{issue_price - price:+.2f} %",
            help="Issue price minus fair value: what the issuer earns at launch.",
        )
        bottom[0].metric(
            "Expected life",
            f"{result.expected_life.value:.2f} y",
            help="Average time until the note is called or matures.",
        )
        bottom[1].metric(
            "Probability of loss",
            f"{100.0 * result.loss_probability.value:.1f} %",
            help="Probability of getting back less than the notional at maturity.",
        )
        st.caption(
            f"Value of the notional: {price / 100.0 * float(context['notional']):,.0f}. "
            f"Barrier breached and note alive at maturity: "
            f"{100.0 * result.breach_probability.value:.1f} % of the paths."
        )

        st.markdown("#### Decomposition on common paths")
        st.table(
            pres.style_decomposition(
                _legs_table(result), {"Price (%)": "{:.3f}", "± 1 std error (%)": "{:.3f}"}
            )
        )
        st.caption(
            "Every leg is valued on the same paths, so they add up to the price exactly. The "
            "last leg is the down-and-in put on the worst performance the investor has sold: "
            "what pays for the coupon."
        )
        _closed_form_check(note, basket, spread, result)

        payoff_tab, life_tab, schedule_tab = st.tabs(
            ["Payoff at maturity", "When does the note end?", "Observation schedule"]
        )
        with payoff_tab:
            st.plotly_chart(charts.coupon_payoff_figure(note, basket.today_worst_pct()))
            if note.memory:
                st.caption(
                    "With memory, the coupons missed before maturity are paid on top of the "
                    "final coupon shown here when the coupon condition holds."
                )
        with life_tab:
            st.plotly_chart(charts.call_probability_figure(note, result))
        with schedule_tab:
            schedule = _schedule_table(note, result)
            st.dataframe(
                schedule,
                hide_index=True,
                column_config={
                    c: st.column_config.NumberColumn(format="%.2f") for c in schedule.columns
                },
            )

    with st.container(border=True):
        _solver(family, spec, context, basket, spread, n_paths)

    with st.expander("How is this priced?"):
        st.markdown(
            """
- Each underlying follows the **Dupire local volatility** of its smile, so every vanilla and
  digital option of the surface is repriced; several underlyings are correlated.
- At each date: coupon first (with memory if any), then the call. At maturity, par is paid
  back unless the protection barrier was breached and the worst performance ends below the
  strike, in which case the investor receives `performance / strike` of the notional.
- The barrier is observed as written in the term sheet: at maturity, continuously (Brownian
  bridge between steps) or at each daily close.
- Cash flows are discounted at the risk-free rate plus the issuer's credit spread.
- Every number comes with its Monte Carlo standard error.
"""
        )
