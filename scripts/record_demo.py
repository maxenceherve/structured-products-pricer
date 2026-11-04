"""Record the GIF of the README: a walk through the running web app.

The browser is scripted and every frame is a screenshot, so the recording does not depend on the
speed of the machine. Start the app, then run this (Playwright stays out of the project's
dependencies, and it drives the Edge installed on the machine):

    uv run python -m web.app
    uv run --with playwright python scripts/record_demo.py
"""

from __future__ import annotations

import argparse
import io
import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from playwright.sync_api import Locator, Page, sync_playwright

WIDTH, HEIGHT = 1280, 960
GIF_WIDTH = 960
COLOURS = 96
FRAME_MS = 90
READING_TOP = 240
OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "demo.gif"

# A drawn pointer: headless browsers show none.
CURSOR = """
window.addEventListener('DOMContentLoaded', () => {
  const c = document.createElement('div');
  c.id = 'demo-cursor';
  c.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24"><path d="M5 2l14 13h-8l-4 7z" '
    + 'fill="#111" stroke="#fff" stroke-width="1.5"/></svg>';
  c.style.cssText = 'position:fixed;left:0;top:0;z-index:99999;pointer-events:none;';
  document.documentElement.appendChild(c);
  window.demoCursor = (x, y, down) => {
    c.style.transform = `translate(${x - 5}px, ${y - 2}px) scale(${down ? 0.8 : 1})`;
  };
});
"""


def ease(t: float) -> float:
    """Slow at both ends."""
    return (1 - math.cos(math.pi * t)) / 2


@dataclass
class Recorder:
    """The browser page, the pointer and the frames taken so far."""

    page: Page
    x: float = WIDTH * 0.7
    y: float = HEIGHT * 0.35
    frames: list[tuple[Image.Image, int]] = field(default_factory=list)

    def snap(self, ms: int = FRAME_MS) -> None:
        """Take one frame, shown for ``ms``."""
        png = self.page.screenshot(type="png")
        self.frames.append((Image.open(io.BytesIO(png)).convert("RGB"), ms))

    def settle(self, ms: int = 500) -> None:
        """Let the page finish drawing, without a frame."""
        self.page.wait_for_timeout(ms)

    def pointer(self, down: bool = False) -> None:
        """Draw the pointer where it is."""
        self.page.evaluate("([x, y, d]) => window.demoCursor(x, y, d)", [self.x, self.y, down])

    def move_to(self, x: float, y: float, steps: int = 9) -> None:
        """Glide the pointer to a point."""
        x0, y0 = self.x, self.y
        for i in range(1, steps + 1):
            t = ease(i / steps)
            self.x, self.y = x0 + (x - x0) * t, y0 + (y - y0) * t
            self.pointer()
            self.page.mouse.move(self.x, self.y)
            self.snap()

    def scroll_to(self, top: float, speed: float = 55.0) -> None:
        """Scroll the page to ``top``, ``speed`` pixels per frame on average."""
        start = self.page.evaluate("window.scrollY")
        steps = max(2, round(abs(top - start) / speed))
        for i in range(1, steps + 1):
            y = start + (top - start) * ease(i / steps)
            self.page.evaluate("y => window.scrollTo({top: y, behavior: 'instant'})", y)
            self.snap()

    def reveal(self, target: Locator) -> None:
        """Scroll until ``target`` is well inside the window, below the top bar."""
        box = target.bounding_box()
        assert box is not None
        if box["y"] < 90 or box["y"] + box["height"] > HEIGHT - 30:
            scroll = self.page.evaluate("window.scrollY")
            self.scroll_to(scroll + box["y"] + box["height"] / 2 - HEIGHT / 2)

    def click(self, target: Locator, hold: int = 200) -> None:
        """Glide to ``target`` and click it."""
        target.wait_for(state="visible")
        self.reveal(target)
        box = target.bounding_box()
        assert box is not None
        self.move_to(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        self.pointer(down=True)
        self.snap(120)
        self.page.mouse.click(self.x, self.y)
        self.pointer()
        self.snap(hold)

    def tab(self, name: str, hold: int = 3000, wait_for: str | None = None) -> None:
        """Open a tab of the product page and show it."""
        self.click(self.page.locator("#product-tabs a.nav-link", has_text=name))
        if wait_for:
            self.page.locator(wait_for).first.wait_for(state="visible", timeout=60_000)
        self.settle(900)
        self.snap(hold)


def tour(rec: Recorder, url: str) -> None:
    """The walk: the products, one note priced tab by tab, the solver, the market."""
    page = rec.page
    page.goto(url, wait_until="networkidle")
    page.locator(".product-card").first.wait_for()
    rec.pointer()
    rec.settle()
    rec.snap(1800)

    # Every family, top to bottom.
    bottom = page.evaluate("document.documentElement.scrollHeight") - HEIGHT
    rec.scroll_to(bottom, speed=60)
    rec.snap(500)

    # Back to the bonus note, in the capital protection family.
    link = page.locator('a[href="/product/bonus-note"]').first
    box = link.bounding_box()
    assert box is not None
    rec.scroll_to(page.evaluate("window.scrollY") + box["y"] - HEIGHT * 0.35, speed=110)
    rec.snap(600)
    rec.click(link)

    # The form alone, then the Price button.
    page.locator("#price-run").wait_for()
    rec.settle(900)
    rec.snap(2200)
    rec.click(page.locator("#price-run"))
    plot = "#product-results .js-plotly-plot:visible"
    page.locator(plot).first.wait_for(timeout=60_000)
    rec.settle(1200)
    rec.snap(1500)

    # The price, the tabs and the solver in one window, the same for every tab.
    rec.scroll_to(READING_TOP, speed=30)
    rec.snap(3500)
    rec.tab("Decomposition", hold=4500)
    rec.tab("Greeks", hold=2500, wait_for=plot)
    rec.scroll_to(READING_TOP + 620, speed=40)
    rec.snap(2500)
    rec.scroll_to(READING_TOP, speed=60)
    rec.tab("Hedging", hold=3000)

    # The solver: the bonus level that makes the note worth 99 %, then priced.
    rec.tab("Payoff", hold=800)
    rec.click(page.locator("#solve-term"))
    page.locator("#solve-term").select_option(label="Bonus level")
    rec.snap(700)
    rec.click(page.locator("#solve-target"))
    page.keyboard.press("Control+A")
    for key in "99":
        page.keyboard.type(key)
        rec.snap(220)
    rec.click(page.locator("#solve-run"))
    page.locator("#solve-result", has_text="gives a price").wait_for(timeout=60_000)
    rec.settle(600)
    rec.scroll_to(READING_TOP + 130, speed=25)
    rec.snap(3000)
    rec.click(page.locator("#price-run"))
    rec.settle(1800)
    rec.scroll_to(READING_TOP, speed=25)
    rec.snap(3500)

    # The market behind every page: the skew and the volatility surface.
    rec.scroll_to(0, speed=60)
    rec.click(page.locator("#market-button"))
    page.locator("#surface-graph canvas").first.wait_for(timeout=60_000)
    rec.settle(1500)
    rec.snap(1800)
    panel = page.locator(".offcanvas-body")
    end = panel.evaluate("(p) => p.scrollHeight - p.clientHeight")
    for i in range(1, 15):
        panel.evaluate("(p, y) => p.scrollTo({top: y, behavior: 'instant'})", end * ease(i / 14))
        rec.snap()
    rec.settle(800)
    rec.snap(4500)


def save_gif(frames: list[tuple[Image.Image, int]], path: Path) -> None:
    """Write the frames with one shared palette, identical neighbours merged."""
    merged: list[tuple[Image.Image, int]] = []
    for image, ms in frames:
        if merged and merged[-1][0].tobytes() == image.tobytes():
            merged[-1] = (merged[-1][0], merged[-1][1] + ms)
        else:
            merged.append((image, ms))

    height = round(HEIGHT * GIF_WIDTH / WIDTH)
    small = [i.resize((GIF_WIDTH, height), Image.Resampling.LANCZOS) for i, _ in merged]
    sample = small[:: max(1, len(small) // 24)]
    sheet = Image.new("RGB", (GIF_WIDTH, height * len(sample)))
    for k, image in enumerate(sample):
        sheet.paste(image, (0, k * height))
    palette = sheet.quantize(COLOURS, method=Image.Quantize.MEDIANCUT)
    indexed = [i.quantize(palette=palette, dither=Image.Dither.NONE) for i in small]

    indexed[0].save(
        path,
        save_all=True,
        append_images=indexed[1:],
        duration=[ms for _, ms in merged],
        loop=0,
        disposal=1,
    )
    seconds = sum(ms for _, ms in merged) / 1000
    size = path.stat().st_size / 1e6
    print(f"{path}: {len(merged)} frames, {seconds:.0f} s, {size:.1f} MB")


def main() -> None:
    """Record the walk and write the GIF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8050/")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--channel", default="msedge", help="installed browser to drive")
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.add_init_script(CURSOR)
        rec = Recorder(page)
        tour(rec, args.url)
        browser.close()
    save_gif(rec.frames, args.output)


if __name__ == "__main__":
    main()
