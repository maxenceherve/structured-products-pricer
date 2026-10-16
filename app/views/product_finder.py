"""Home page: describe what you want, get products that fit, and see how to build them."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import presentation as pres
import state
from spricer.finder import Objective, Profile, Protection, Recommendation, View, VolView, recommend
from spricer.market import Market
from spricer.packaging import Equivalence, equivalences
from spricer.strategies import CATALOG

PAGES = {
    "Vanilla options": "views/vanilla_options.py",
    "Capital protection": "views/capital_protection.py",
    "Yield enhancement": "views/yield_enhancement.py",
    "Autocallables": "views/autocallables.py",
}
TOP = 3
MAX_SCORE = 7.5  # 4 (view) + 1.5 (objective) + 1 (volatility) + 0.5 (horizon) + headroom


def _open(recommendation: Recommendation) -> None:
    """Preselect the product on its page, then go there."""
    card = recommendation.card
    if card.page == "Vanilla options":
        spec = next(s for s in CATALOG if s.name == card.name)
        st.session_state["vanilla_family"] = spec.family.value
        st.session_state[f"vanilla_product_{spec.family.name}"] = spec.name
        st.session_state.pop("vanilla_product_search", None)
    elif card.page == "Capital protection":
        st.session_state["cp_product"] = card.name
    elif card.page == "Yield enhancement":
        st.session_state["cn_yield_product"] = card.name
    else:
        st.session_state["cn_autocallables_product"] = card.name
    st.switch_page(PAGES[card.page])


def _card(recommendation: Recommendation, rank: int) -> None:
    card = recommendation.card
    with st.container(border=True):
        head = st.columns([3, 2], vertical_alignment="center")
        head[0].markdown(f"### {rank}. {card.name}")
        head[1].markdown(f":blue-badge[{card.page}]")
        fit = max(0.0, min(1.0, recommendation.score / MAX_SCORE))
        st.progress(fit, text=f"Fit {100 * fit:.0f} %")
        for reason in recommendation.reasons:
            st.markdown(f":green[✓] {reason}")
        for warning in recommendation.warnings:
            st.markdown(f":orange[⚠] {warning}")
        if st.button(
            f"Price the {card.name.lower()}",
            icon=":material/arrow_forward:",
            key=f"finder_open_{rank}",
        ):
            _open(recommendation)


def _packaging(equivalence: Equivalence, market: Market, maturity: float) -> None:
    rows = []
    for construction in (equivalence.first, equivalence.second):
        greeks = pres.display_greeks(construction.strategy, market)
        rows.append(
            {
                "Construction": construction.name,
                "Value today (%)": construction.cash_today(market),
                "Options: paid + / received − (%)": construction.option_premium(market),
                "Bond at maturity (%)": construction.bond() / market.spot * 100.0,
                "Collateral (%)": construction.collateral(market),
                "Delta (%)": greeks["Delta (%)"],
                "Gamma (%)": greeks["Gamma (%)"],
                "Vega (%)": greeks["Vega (%)"],
            }
        )
    with st.container(border=True):
        st.markdown(f"#### {equivalence.title}")
        table = pd.DataFrame(rows)
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                c: st.column_config.NumberColumn(format="%+.2f")
                for c in table.columns
                if c != "Construction"
            },
        )
        st.plotly_chart(charts.packaging_figure(equivalence, market.spot))
        st.markdown(f"- **{equivalence.first.name}**: {equivalence.first.obligation}")
        st.markdown(f"- **{equivalence.second.name}**: {equivalence.second.obligation}")
        st.info(equivalence.lesson)
        st.caption(
            f"Strikes around today's spot, {maturity:g} years. Same payoff, same value and same "
            "Greeks: only the cash flows and the obligations differ. Collateral: the worst "
            "amount the option and stock legs can owe at maturity, covered by the bond leg when "
            "there is one."
        )


st.title("Find a product")
st.caption(
    "Say what you expect and what you can accept: the finder ranks the products of this app "
    "against your answers, explains every point, and shows equivalent ways to build the payoff. "
    "An educational tool, not investment advice."
)

with st.container(border=True):
    st.subheader("1 · What do you expect?")
    view = View(
        state.choice(
            st.pills, "Market view", "finder_view", [v.value for v in View], View.MILD_BULL.value
        )
    )
    vol_view = VolView(
        state.choice(
            st.radio,
            "Volatility",
            "finder_vol",
            [v.value for v in VolView],
            VolView.NEUTRAL.value,
            horizontal=True,
        )
    )
    horizon = state.persistent(
        st.slider,
        "Horizon (years)",
        "finder_horizon",
        3.0,
        min_value=0.25,
        max_value=10.0,
        step=0.25,
    )
    st.subheader("2 · What can you accept?")
    protection = Protection(
        state.choice(
            st.pills,
            "Capital protection",
            "finder_protection",
            [p.value for p in Protection],
            Protection.FULL.value,
        )
    )
    objective = Objective(
        state.choice(
            st.radio, "You want", "finder_objective", [o.value for o in Objective], horizontal=True
        )
    )

profile = Profile(view, protection, objective, vol_view, float(horizon))
ranked = recommend(profile)

st.subheader("Products that fit")
for rank, recommendation in enumerate(ranked[:TOP], start=1):
    _card(recommendation, rank)
with st.expander("Every product, ranked"):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Product": r.card.name,
                    "Page": r.card.page,
                    "Points": r.score,
                    "Protection": r.card.protection.value,
                    "Pays": r.card.objective.value,
                }
                for r in ranked
            ]
        ),
        hide_index=True,
        column_config={"Points": st.column_config.NumberColumn(format="%.1f")},
    )
    st.caption(
        "Points: up to 4 for the view, 1.5 for the objective, ±1 for volatility, ±0.5/−1 for "
        "the horizon, −10 when the product protects less than asked."
    )

try:
    market = state.current_market()
except ValueError as exc:
    st.error(f"The volatility surface is invalid: {exc}")
    st.stop()
pairs = equivalences(view, market.spot, float(horizon))
if pairs:
    st.subheader("Same payoff, different packaging")
    st.caption(
        "Put-call parity lets the same payoff be built by buying options or by selling others "
        "with cash aside: the risk is identical, who pays and who must post collateral is not."
    )
    for equivalence in pairs:
        _packaging(equivalence, market, float(horizon))
