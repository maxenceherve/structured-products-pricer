"""Product sheet: a self-contained, printable HTML page that presents a product.

Name and description, key figures, the payoff chart (inline SVG, no script), what is paid and
the P&L at maturity with the breakevens, the terms, the Greeks, the detailed tables and the
main risks. Everything is centred on the page.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from numpy.typing import ArrayLike

Table = tuple[list[str], Sequence[Sequence[object]]]  # header, rows
Series = tuple[str, ArrayLike, str]  # name, values, colour
PRIMARY, GUIDE, ACCENT = "#1f4e79", "#9aa5b5", "#f46d43"

STYLE = """
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; color: #1b2333; margin: 0;
       background: #eef2f8; }
.page { max-width: 920px; margin: 32px auto; background: #fff; padding: 40px 48px;
        border-radius: 14px; box-shadow: 0 8px 28px rgba(27, 35, 51, 0.08); }
.family { text-transform: uppercase; letter-spacing: 0.12em; color: #6b7587; font-size: 0.78em;
          text-align: center; }
h1 { color: #1f4e79; margin: 6px 0 4px; text-align: center; font-size: 2.1em; }
.subtitle { color: #5b6577; margin: 0 auto 24px; text-align: center; max-width: 680px; }
.figures { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin: 8px 0 20px; }
.figure { border: 1px solid #dde3ec; border-radius: 10px; padding: 10px 16px; min-width: 150px;
          text-align: center; }
.figure .label { font-size: 0.72em; color: #6b7587; text-transform: uppercase;
                 letter-spacing: 0.05em; }
.figure .value { font-size: 1.35em; font-weight: 700; color: #1b2333; }
.figure .sub { font-size: 0.75em; color: #6b7587; }
h2 { color: #1f4e79; border-bottom: 2px solid #dde3ec; padding-bottom: 4px; margin: 30px 0 12px;
     font-size: 1.1em; text-align: center; }
.chart { text-align: center; }
table { border-collapse: collapse; margin: 0 auto; font-size: 0.88em; }
th { background: #eef2f8; padding: 6px 9px; text-align: center; }
td { border-top: 1px solid #dde3ec; padding: 6px 9px; text-align: center; }
td.num { font-variant-numeric: tabular-nums; }
td.pos { color: #1a9850; } td.neg { color: #d73027; }
th.row { text-align: left; background: #f7f9fc; white-space: nowrap; }
table.scenario { margin-bottom: 10px; }
.columns { display: flex; gap: 28px; justify-content: center; flex-wrap: wrap; }
.columns > div { flex: 1; min-width: 300px; }
.note { color: #5b6577; font-size: 0.85em; text-align: center; margin: 8px auto 0;
        max-width: 720px; }
ul.risks { max-width: 720px; margin: 0 auto; line-height: 1.5; }
.notice { background: #fff4e5; border-left: 4px solid #f46d43; padding: 8px 12px;
          font-size: 0.85em; margin: 28px auto 0; max-width: 720px; }
.footer { color: #7f7f7f; font-size: 0.78em; margin-top: 18px; text-align: center; }
@media print { body { background: #fff; } .page { box-shadow: none; margin: 0 auto; } }
"""


def _number(x: float, digits: int = 2) -> str:
    if math.isnan(x):
        return "n/a"
    if abs(x) >= 1000.0:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}" if abs(x) >= 0.01 or x == 0.0 else f"{x:.4f}"


def _cell(value: object, signed: bool = False) -> str:
    if isinstance(value, float | int) and not isinstance(value, bool):
        v = float(value)
        tone = "" if not signed or math.isnan(v) or abs(v) < 5e-3 else ("pos" if v > 0 else "neg")
        text = _number(v) if not signed or math.isnan(v) else f"{v:+.2f}"
        return f'<td class="num {tone}">{text}</td>'
    return f"<td>{html.escape('' if value is None else str(value))}</td>"


def table_html(header: list[str], rows: Sequence[Sequence[object]]) -> str:
    """A centred table; numbers right-aligned."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join(f"<tr>{''.join(_cell(v) for v in row)}</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def scenario_html(
    level_label: str,
    levels: Sequence[float],
    rows: Sequence[tuple[str, Sequence[float]]],
    columns: int = 9,
) -> str:
    """What is paid at a few final levels: one column per level (``columns`` per table)."""
    out = []
    for start in range(0, len(levels), columns):
        chunk = slice(start, start + columns)
        head = f"<th>{html.escape(level_label)}</th>" + "".join(
            f"<th>{x:g} %</th>" for x in levels[chunk]
        )
        body = ""
        for name, values in rows:
            signed = name.startswith(("P&L", "Gain"))
            cells = "".join(_cell(float(v), signed) for v in list(values)[chunk])
            body += f'<tr><th class="row">{html.escape(name)}</th>{cells}</tr>'
        out.append(
            f'<table class="scenario"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
        )
    return "".join(out)


# --- Chart -------------------------------------------------------------------------------------


def _ticks(low: float, high: float, n: int = 6) -> list[float]:
    span = max(high - low, 1e-9)
    raw = span / n
    unit = 10 ** math.floor(math.log10(raw))
    step = next(m * unit for m in (1, 2, 2.5, 5, 10) if m * unit >= raw)
    first = math.ceil(low / step) * step
    return [round(first + i * step, 10) for i in range(int((high - first) / step) + 1)]


@dataclass(frozen=True)
class Chart:
    """A line chart of the payoff: series against the underlying, with vertical levels."""

    x: ArrayLike
    series: Sequence[Series]
    levels: Sequence[tuple[float, str]] = ()
    x_label: str = ""
    y_label: str = ""
    zero_line: bool = True


def chart_svg(chart: Chart, width: int = 780, height: int = 330) -> str:
    """The chart as inline SVG: axes, ticks, levels, lines (a NaN breaks a line), legend."""
    left, right, top, bottom = 58, 16, 34, 46
    x = np.asarray(chart.x, dtype=float)
    ys = [np.asarray(s[1], dtype=float) for s in chart.series]
    finite = np.concatenate([y[np.isfinite(y)] for y in ys])
    y_low, y_high = float(finite.min()), float(finite.max())
    if chart.zero_line:
        y_low, y_high = min(y_low, 0.0), max(y_high, 0.0)
    pad = 0.08 * max(y_high - y_low, 1.0)
    y_low, y_high = y_low - pad, y_high + pad
    x_low, x_high = float(x.min()), float(x.max())

    def px(v: float) -> float:
        return left + (v - x_low) / (x_high - x_low) * (width - left - right)

    def py(v: float) -> float:
        return top + (y_high - v) / (y_high - y_low) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Arial" font-size="11">'
    ]
    for t in _ticks(y_low, y_high):
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{py(t):.1f}" y2="{py(t):.1f}" '
            f'stroke="#eef2f8"/><text x="{left - 6}" y="{py(t) + 4:.1f}" text-anchor="end" '
            f'fill="#6b7587">{t:g}</text>'
        )
    for t in _ticks(x_low, x_high, 8):
        parts.append(
            f'<text x="{px(t):.1f}" y="{height - bottom + 16}" text-anchor="middle" '
            f'fill="#6b7587">{t:g}</text>'
        )
    if chart.zero_line:
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{py(0):.1f}" y2="{py(0):.1f}" '
            'stroke="#1b2333" stroke-width="1"/>'
        )
    for level, name in chart.levels:
        if x_low <= level <= x_high:
            parts.append(
                f'<line x1="{px(level):.1f}" x2="{px(level):.1f}" y1="{top}" '
                f'y2="{height - bottom}" stroke="{ACCENT}" stroke-dasharray="5 4"/>'
                f'<text x="{px(level) + 3:.1f}" y="{top + 10}" fill="{ACCENT}">'
                f"{html.escape(name)}</text>"
            )
    legend_x = float(left)
    for (name, _, color), y in zip(chart.series, ys, strict=True):
        dash = ' stroke-dasharray="3 3"' if color == GUIDE else ""
        segment: list[str] = []
        for xv, yv in zip(x, y, strict=True):
            if np.isfinite(yv):
                segment.append(f"{px(xv):.1f},{py(yv):.1f}")
            elif segment:
                parts.append(_polyline(segment, color, dash))
                segment = []
        if segment:
            parts.append(_polyline(segment, color, dash))
        parts.append(
            f'<rect x="{legend_x}" y="8" width="14" height="4" fill="{color}"/>'
            f'<text x="{legend_x + 19}" y="14" fill="#1b2333">{html.escape(name)}</text>'
        )
        legend_x += 30 + 6.5 * len(name)
    parts.append(
        f'<text x="{(left + width - right) / 2}" y="{height - 8}" text-anchor="middle" '
        f'fill="#3a4558">{html.escape(chart.x_label)}</text>'
        f'<text x="14" y="{(top + height - bottom) / 2}" text-anchor="middle" fill="#3a4558" '
        f'transform="rotate(-90 14 {(top + height - bottom) / 2})">'
        f"{html.escape(chart.y_label)}</text></svg>"
    )
    return "".join(parts)


def _polyline(points: list[str], color: str, dash: str) -> str:
    width = 1.5 if dash else 3
    return (
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
        f'stroke-width="{width}"{dash}/>'
    )


# --- Sheet -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sheet:
    """Content of a product sheet.

    Attributes
    ----------
    family, title, description
        What the product is.
    figures
        Key figures: label, value, small print.
    chart
        The payoff chart.
    payoff
        Level label, levels and rows of what is paid at maturity (and the P&L).
    breakevens
        Levels where the P&L at maturity crosses zero.
    terms, greeks
        Key features and Greeks: label, value.
    tables
        Further tables with their headings.
    risks
        Main risks, one sentence each.
    """

    family: str
    title: str
    description: str
    figures: Sequence[tuple[str, str, str]]
    chart: Chart
    payoff: tuple[str, Sequence[float], Sequence[tuple[str, Sequence[float]]]]
    breakevens: Sequence[float] = ()
    terms: Sequence[tuple[str, object]] = ()
    greeks: Sequence[tuple[str, object]] = ()
    tables: Sequence[tuple[str, Table]] = field(default_factory=tuple)
    risks: Sequence[str] = ()


def render(sheet: Sheet, valuation_date: date | None = None) -> str:
    """The sheet as a self-contained HTML page."""
    day = (valuation_date or date.today()).isoformat()
    figures = "".join(
        f'<div class="figure"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="sub">{html.escape(sub)}</div></div>'
        for label, value, sub in sheet.figures
    )
    breakeven = (
        "Breakeven at " + " and ".join(f"{b:.1f} %" for b in sheet.breakevens) + "."
        if sheet.breakevens
        else ""
    )
    body = [
        f'<div class="family">{html.escape(sheet.family)}</div>',
        f"<h1>{html.escape(sheet.title)}</h1>",
        f'<p class="subtitle">{html.escape(sheet.description)}</p>',
        f'<div class="figures">{figures}</div>',
        "<h2>Payoff at maturity</h2>",
        f'<div class="chart">{chart_svg(sheet.chart)}</div>',
        scenario_html(*sheet.payoff),
        f'<p class="note">{html.escape(breakeven)} Amounts in % of the notional.</p>',
    ]
    columns = []
    if sheet.terms:
        columns.append(
            "<div><h2>Key features</h2>" + table_html(["Term", "Value"], sheet.terms) + "</div>"
        )
    if sheet.greeks:
        columns.append(
            "<div><h2>Greeks</h2>" + table_html(["Greek", "Value"], sheet.greeks) + "</div>"
        )
    if columns:
        body.append(f'<div class="columns">{"".join(columns)}</div>')
    for heading, (header, rows) in sheet.tables:
        body.append(f"<h2>{html.escape(heading)}</h2>{table_html(header, rows)}")
    if sheet.risks:
        items = "".join(f"<li>{html.escape(r)}</li>" for r in sheet.risks)
        body.append(f'<h2>Main risks</h2><ul class="risks">{items}</ul>')
    body.append(
        '<div class="notice">Indicative terms computed on synthetic, user-entered market data. '
        "Educational document: not an offer, not investment advice.</div>"
        f'<p class="footer">Generated on {day} by the Structured Products Pricer '
        "(open-source, MIT licence).</p>"
    )
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(sheet.title)}</title><style>{STYLE}</style></head>"
        f'<body><div class="page">{"".join(body)}</div></body></html>'
    )


def file_name(title: str) -> str:
    """A safe file name for the sheet."""
    slug = "".join(c.lower() if c.isalnum() else "-" for c in title).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"{slug}-product-sheet.html"


def download(sheet: Sheet) -> dict[str, str]:
    """What dcc.Download needs to save the sheet."""
    return {"content": render(sheet), "filename": file_name(sheet.title), "type": "text/html"}
