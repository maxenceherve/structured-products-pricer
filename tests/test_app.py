from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP_ENTRY_POINT = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_starts_without_exception() -> None:
    at = AppTest.from_file(str(APP_ENTRY_POINT)).run()
    assert not at.exception
    assert at.title[0].value == "Structured Products Pricer"
