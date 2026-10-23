"""Streamlit entry point: ``uv run streamlit run app/streamlit_app.py``."""

from pathlib import Path

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
ASSETS = Path(__file__).parent / "assets"
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))

page = st.navigation(
    {
        "Start": [
            st.Page(
                "views/product_finder.py",
                title="Product finder",
                icon=":material/explore:",
                default=True,
            ),
        ],
        "Pricing": [
            st.Page(
                "views/vanilla_options.py",
                title="Vanilla options",
                icon=":material/stacked_line_chart:",
            ),
            st.Page(
                "views/greeks_lab.py",
                title="Greeks lab",
                icon=":material/science:",
            ),
            st.Page(
                "views/hedging.py",
                title="Hedging",
                icon=":material/balance:",
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
    st.caption(f"Structured Products Pricer · spricer {spricer.__version__}")

page.run()
