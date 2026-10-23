"""Payoff thumbnails: the shape of a payoff as a small inline SVG, without axes or numbers."""

from __future__ import annotations

import base64

import numpy as np

from web.registry import Product

WIDTH, HEIGHT, PAD = 240, 140, 14
LINE = "#1f4e79"
GUIDE = "#c4ccd8"
MARKER = "#f46d43"
_GRID = np.linspace(30.0, 170.0, 351)  # underlying at maturity, % of initial


def _points(xs: np.ndarray, ys: np.ndarray, low: float, high: float) -> str:
    span = high - low if high > low else 1.0
    px = PAD + (xs - xs[0]) / (xs[-1] - xs[0]) * (WIDTH - 2 * PAD)
    py = HEIGHT - PAD - (ys - low) / span * (HEIGHT - 2 * PAD)
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(px, py, strict=True))


def payoff_svg(product: Product) -> str:
    """SVG of the payoff at maturity, scaled to fill the box.

    Notes (payoffs in % of notional) show the delta-one line dashed for scale; options show the
    zero line. Steep wings are clipped to the box so the shape of the body stays readable.
    """
    payoff = np.asarray(product.payoff(_GRID), dtype=np.float64)
    reference = _GRID if product.notional_line else np.zeros_like(_GRID)
    values = np.concatenate([payoff, reference]) if product.notional_line else payoff
    low, high = float(np.min(values)), float(np.max(values))
    if np.isclose(high, low):
        low, high = low - 1.0, high + 1.0
    margin = 0.08 * (high - low)
    low, high = low - margin, high + margin
    clipped_ref = np.clip(reference, low, high)
    guide = _points(_GRID, clipped_ref, low, high)
    line = _points(_GRID, np.clip(payoff, low, high), low, high)
    dash = ' stroke-dasharray="5 4"' if product.notional_line else ""
    marker = ""
    if product.marker is not None:
        x = PAD + (product.marker - _GRID[0]) / (_GRID[-1] - _GRID[0]) * (WIDTH - 2 * PAD)
        marker = (
            f'<line x1="{x:.1f}" y1="{PAD / 2:.1f}" x2="{x:.1f}" y2="{HEIGHT - PAD / 2:.1f}" '
            f'stroke="{MARKER}" stroke-width="1.5" stroke-dasharray="4 4"/>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">{marker}'
        f'<polyline points="{guide}" fill="none" stroke="{GUIDE}" stroke-width="1.5"{dash}/>'
        f'<polyline points="{line}" fill="none" stroke="{LINE}" stroke-width="3" '
        'stroke-linejoin="round" stroke-linecap="round"/></svg>'
    )


def payoff_data_uri(product: Product) -> str:
    """The thumbnail as a data URI, for an ``<img>``."""
    encoded = base64.b64encode(payoff_svg(product).encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"
