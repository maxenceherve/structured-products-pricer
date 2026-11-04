"""Product sheets served at ``/sheet``, to open in a new tab.

A page builds a link carrying its state (terms, market, and for a coupon note its Monte Carlo
results), compressed into the address; the route rebuilds the product and renders the sheet.
Nothing is stored on the server.
"""

from __future__ import annotations

import base64
import binascii
import html
import json
import zlib
from typing import Any

from flask import Flask, Response, request

from web import risk_panel
from web.coupons import setup_coupon
from web.notes import NotePosition
from web.sheet import render
from web.sheets import coupon_sheet, note_sheet, option_sheet

ROUTE = "/sheet"


def encode(state: dict[str, Any]) -> str:
    """The state as a short, address-safe token."""
    raw = json.dumps(state, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii").rstrip("=")


def decode(token: str) -> dict[str, Any]:
    """The state of a token; ValueError when it is not one."""
    try:
        raw = zlib.decompress(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
        state = json.loads(raw)
    except (binascii.Error, zlib.error, ValueError) as exc:
        raise ValueError("this link does not describe a product") from exc
    if not isinstance(state, dict):
        raise ValueError("this link does not describe a product")
    return state


def href(state: dict[str, Any]) -> str:
    """Address of the sheet of a page's state."""
    return f"{ROUTE}?s={encode(state)}"


def sheet_html(state: dict[str, Any]) -> str:
    """The sheet of a page's state; ValueError when the product cannot be priced."""
    store, data = state.get("store"), state.get("market") or {}
    if "result" in state:
        terms, basket = setup_coupon(store, state.get("rows"), state.get("corr"), data)
        return render(coupon_sheet(terms, basket, state["result"], state.get("greeks")), True)
    item = risk_panel.resolve(store, data)
    sheet = note_sheet(item) if isinstance(item, NotePosition) else option_sheet(item)
    return render(sheet, True)


def _error(message: str) -> str:
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Product sheet</title>"
        "</head><body style=\"font-family: 'IBM Plex Sans', sans-serif; color: #1b2a41; "
        'max-width: 60ch; margin: 64px auto;">'
        "<h1 style=\"font-family: 'IBM Plex Serif', serif\">No sheet to show</h1>"
        f"<p>{html.escape(message.capitalize())}. Open the product page again and use its "
        "link to the product sheet.</p></body></html>"
    )


def view() -> Response:
    """The page served at ``/sheet``."""
    try:
        body = sheet_html(decode(request.args.get("s", "")))
    except (ValueError, KeyError, TypeError) as exc:
        return Response(_error(str(exc)), status=400, mimetype="text/html")
    return Response(body, mimetype="text/html")


def register(server: Flask) -> None:
    """Serve the sheets on the app's server."""
    server.add_url_rule(ROUTE, "product_sheet", view)
