"""Product sheets: HTML content, formatting, and the download buttons of the note pages."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import termsheet

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
ENTRY = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_the_sheet_holds_every_section_and_the_notice():
    table = pd.DataFrame({"Leg": ["Bond", "Call <spread>"], "Price (%)": [91.39, 3.25]})
    html = termsheet.product_sheet(
        "Bonus note",
        "A note & its bricks",
        [("Terms", table), ("Valuation", "Fair value 95.11 %.")],
        valuation_date=date(2026, 10, 17),
    )
    assert html.startswith("<!DOCTYPE html>")
    assert "<h1>Bonus note</h1>" in html
    assert "A note &amp; its bricks" in html
    assert "Call &lt;spread&gt;" in html
    assert "not investment advice" in html
    assert "2026-10-17" in html
    assert "<h2>Valuation</h2><p>Fair value 95.11 %.</p>" in html


def test_numbers_are_formatted_for_reading():
    frame = pd.DataFrame({"x": [1_234_567.0, 98.031, 0.01234, np.nan]})
    html = termsheet.product_sheet("t", "s", [("Numbers", frame)])
    for text in ("1,234,567", "98.03", "0.0123"):
        assert text in html
    assert '<td class="num"></td>' in html


def test_helpers():
    terms = termsheet.terms_table(
        {"coupon_pa": 8.0, "Maturity (years)": 3.0}, {"coupon_pa": "Coupon"}
    )
    assert list(terms["Term"]) == ["Coupon", "Maturity (years)"]
    scenarios = termsheet.maturity_scenarios(np.array([100.0, 110.0]), np.array([90.0, 120.0]))
    assert scenarios.shape == (2, 2)
    assert termsheet.file_name("Shark note (up)") == "shark-note-up-product-sheet.html"


@pytest.mark.parametrize(
    ("page", "price_key", "paths_key"),
    [
        ("views/capital_protection.py", "cp_price_button", None),
        ("views/yield_enhancement.py", "cn_yield_price", "widget_cn_yield_paths"),
    ],
)
def test_note_pages_offer_the_sheet(page, price_key, paths_key):
    at = AppTest.from_file(str(ENTRY), default_timeout=120)
    at.run()
    at.switch_page(page).run()
    if paths_key is not None:
        at.radio(key=paths_key).set_value("20,000").run()
    at.button(key=price_key).click().run()
    assert not at.exception
    assert len(at.get("download_button")) == 1
