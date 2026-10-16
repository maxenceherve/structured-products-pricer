"""Streamlit entry point: ``uv run streamlit run app/streamlit_app.py``."""

import streamlit as st

import spricer
import state

st.set_page_config(
    page_title="Structured Products Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
state.init()

page = st.navigation(
    {
        "Pricing": [
            st.Page(
                "views/vanilla_options.py",
                title="Vanilla options",
                icon=":material/stacked_line_chart:",
                default=True,
            ),
        ],
        "Structured products": [
            st.Page(
                "views/capital_protection.py",
                title="Capital protection",
                icon=":material/shield:",
            ),
            st.Page(
                "views/yield_enhancement.py",
                title="Yield enhancement",
                icon=":material/savings:",
            ),
            st.Page(
                "views/autocallables.py",
                title="Autocallables",
                icon=":material/autorenew:",
            ),
        ],
        "Market data": [
            st.Page(
                "views/volatility_surface.py", title="Volatility surface", icon=":material/tune:"
            ),
        ],
    },
    position="sidebar",
    expanded=True,
)

with st.sidebar:
    st.markdown("**Coming next**")
    st.markdown(":gray[Greeks lab]  \n:gray[Hedging]  \n:gray[Product finder]")
    st.divider()
    st.caption(f"Structured Products Pricer · spricer {spricer.__version__}")

page.run()
