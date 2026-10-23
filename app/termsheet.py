"""One-page product sheet, as a self-contained HTML file to download and print."""

from __future__ import annotations

import html
from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

Section = tuple[str, "pd.DataFrame | str"]

STYLE = """
body { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; color: #1b2333; margin: 32px;
       max-width: 860px; }
h1 { color: #1f4e79; margin-bottom: 0; }
.subtitle { color: #5b6577; margin-top: 4px; }
.notice { background: #fff4e5; border-left: 4px solid #f46d43; padding: 8px 12px;
          font-size: 0.9em; margin: 16px 0; }
h2 { color: #1f4e79; border-bottom: 2px solid #dde3ec; padding-bottom: 4px; margin-top: 28px;
     font-size: 1.15em; }
table { border-collapse: collapse; width: 100%; font-size: 0.92em; }
th { background: #eef2f8; text-align: left; padding: 6px 8px; }
td { border-top: 1px solid #dde3ec; padding: 6px 8px; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
p { line-height: 1.45; }
.footer { color: #7f7f7f; font-size: 0.8em; margin-top: 32px; }
"""


def _number(x: float) -> str:
    if np.isnan(x):
        return ""
    if abs(x) >= 1000.0:
        return f"{x:,.0f}"
    return f"{x:.2f}" if abs(x) >= 1.0 else f"{x:.4f}"


def _table(frame: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float | np.floating):
                text = _number(float(value))
                cells.append(f'<td class="num">{text}</td>')
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def product_sheet(
    title: str, subtitle: str, sections: Sequence[Section], valuation_date: date | None = None
) -> str:
    """HTML page: title, notice, then each section (a table or a paragraph)."""
    day = (valuation_date or date.today()).isoformat()
    body = [
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="subtitle">{html.escape(subtitle)}</p>',
        '<div class="notice">Indicative terms computed on synthetic, user-entered market data. '
        "Educational document: not an offer, not investment advice.</div>",
    ]
    for heading, content in sections:
        body.append(f"<h2>{html.escape(heading)}</h2>")
        if isinstance(content, pd.DataFrame):
            body.append(_table(content))
        else:
            body.append(f"<p>{html.escape(content)}</p>")
    body.append(
        f'<p class="footer">Generated on {day} by the Structured Products Pricer '
        "(open-source, MIT licence). Prices in % of notional.</p>"
    )
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        f"<body>{''.join(body)}</body></html>"
    )


def terms_table(values: dict[str, float], labels: dict[str, str]) -> pd.DataFrame:
    """Terms of a product as a two-column table."""
    return pd.DataFrame([{"Term": labels.get(k, k), "Value": f"{v:g}"} for k, v in values.items()])


def maturity_scenarios(payoff: np.ndarray, performances: np.ndarray) -> pd.DataFrame:
    """Paid at maturity for a few final performances."""
    return pd.DataFrame(
        {
            "Underlying at maturity (% of initial)": performances,
            "Paid at maturity (% of notional)": payoff,
        }
    )


def file_name(title: str) -> str:
    """A safe file name for the sheet."""
    slug = "".join(c.lower() if c.isalnum() else "-" for c in title).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"{slug}-product-sheet.html"
