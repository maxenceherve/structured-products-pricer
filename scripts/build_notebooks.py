"""Write and execute the notebooks of ``notebooks/``, with their figures inline.

Run from the repository root:  uv run python scripts/build_notebooks.py
Every number quoted in the README comes from these notebooks or from the tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
KERNEL = {"name": "python3", "display_name": "Python 3", "language": "python"}

SETUP = """\
import math

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREEN, RED, ORANGE, GREY = "#1f4e79", "#1a9850", "#d73027", "#f46d43", "#7f7f7f"
"""

VALIDATION = [
    (
        "md",
        "# Pricing and validation\n\nThe implied smile, its density, the Dupire local "
        "volatility built from it, and the Monte Carlo engine checked against closed forms, every "
        "comparison in standard errors.",
    ),
    ("code", SETUP),
    (
        "code",
        """\
from spricer.bricks import DigitalOption, EuropeanOption
from spricer.market.defaults import default_market
from spricer.mc import local_vol, simulate

market = default_market()
surface = market.vol
strikes = np.linspace(50, 160, 221)
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
for t, color in zip((0.25, 1.0, 2.0), (RED, BLUE, GREEN), strict=True):
    left.plot(strikes, 100 * surface.vol(strikes, t), color=color, label=f"{t:g} y")
    right.plot(strikes, surface.density(strikes, t), color=color, label=f"{t:g} y")
left.set(title="Implied volatility (%)", xlabel="Strike (% of spot)")
right.set(title="Risk-neutral density of the spot", xlabel="Spot at maturity")
left.legend()
right.legend()
fig.tight_layout()
print("arbitrage-free:", surface.check_arbitrage().is_arbitrage_free)
""",
    ),
    (
        "md",
        "## Local volatility\n\nThe diffusion coefficient that reprices every European "
        "option of the surface (Dupire, in total variance). The short end is steep: the skew of "
        "short-dated options must come from a strongly spot-dependent volatility.",
    ),
    (
        "code",
        """\
fig, ax = plt.subplots(figsize=(6.5, 4))
spots = np.linspace(60, 140, 161)
for t, color in zip((0.1, 0.5, 1.0, 2.5), (RED, ORANGE, BLUE, GREEN), strict=True):
    ax.plot(spots, 100 * local_vol(market, spots, t), color=color, label=f"t = {t:g} y")
ax.set(title="Dupire local volatility (%)", xlabel="Spot (% of today)")
ax.legend()
fig.tight_layout()
""",
    ),
    (
        "md",
        "## The Monte Carlo reprices the smile\n\n200,000 paths (100,000 antithetic pairs), "
        "weekly steps graded from daily near today, Richardson extrapolation.",
    ),
    (
        "code",
        """\
times = [0.5, 2.0, 5.0]
paths = simulate(market, times, n_paths=200_000, seed=11)
worst = 0.0
print(f"{'maturity':>8} {'instrument':>22} {'exact':>9} {'MC':>9} {'± s.e.':>8} {'z':>6}")
for t in times:
    s, df = paths.spot_at(t), paths.discount(t)
    for k in (70.0, 100.0, 130.0):
        kind = "put" if k < 100 else "call"
        for brick in (EuropeanOption(kind, k, t), DigitalOption("call", k, t)):
            est = paths.estimate(brick.payoff(s) * df)
            exact = float(brick.value(market))
            z = (est.value - exact) / est.std_error
            worst = max(worst, abs(z))
            print(
                f"{t:>8g} {brick.label:>22} {exact:9.4f} {est.value:9.4f} "
                f"{est.std_error:8.4f} {z:+6.1f}"
            )
print(f"largest gap: {worst:.1f} standard errors")
""",
    ),
    (
        "md",
        "## Richardson extrapolation removes the Euler bias\n\nA six-month at-the-money "
        "digital is the most sensitive case: 400,000 paths, five seeds.",
    ),
    (
        "code",
        """\
digital = DigitalOption("call", 100.0, 0.5)
exact = float(digital.value(market))
rows = {}
for rich in (False, True):
    est = [simulate(market, [0.5], n_paths=400_000, seed=s, richardson=rich) for s in range(5)]
    values = [p.estimate(digital.payoff(p.spot_at(0.5)) * p.discount(0.5)) for p in est]
    mean = np.mean([v.value for v in values])
    error = math.sqrt(sum(v.std_error**2 for v in values)) / len(values)
    rows["Richardson" if rich else "Euler"] = (mean - exact, error)
for name, (bias, error) in rows.items():
    print(f"{name:>10}: bias {1e4 * bias:+.1f} bp of the digital ± {1e4 * error:.1f} bp "
          f"({bias / error:+.1f} standard errors)")
fig, ax = plt.subplots(figsize=(5, 3.5))
names = list(rows)
ax.bar(names, [1e4 * rows[n][0] for n in names], yerr=[2 * 1e4 * rows[n][1] for n in names],
       color=[RED, GREEN], capsize=6)
ax.axhline(0, color=GREY)
ax.set(title="ATM 6-month digital: bias (bp)", ylabel="MC - exact (bp), ± 2 s.e.")
fig.tight_layout()
""",
    ),
]

MODEL_RISK = [
    (
        "md",
        "# Model risk\n\nThe same product priced with one volatility per underlying and with "
        "the local volatility of the whole smile. The closed forms of path-dependent barriers use "
        "one volatility too.",
    ),
    ("code", SETUP),
    (
        "code",
        """\
from spricer.bricks import BarrierOption, DigitalOption, ZeroCoupon
from spricer.market import FlatVol, Market
from spricer.market.defaults import default_market
from spricer.mc import price_note
from spricer.monitoring import Monitoring
from spricer.products.capital_protection import find_note
from spricer.products.notes import Note, NoteLeg

market = default_market()
years = [1.0, 2.0, 3.0, 4.0, 5.0]
legs = [NoteLeg(f"Coupon {t:g}y", DigitalOption("call", 100.0, t), 5.0) for t in years]
legs += [NoteLeg("Redemption", ZeroCoupon(5.0), 100.0),
         NoteLeg("Protection", BarrierOption("put", "down-and-in", 100.0, 60.0, 5.0), -1.0)]
strip = Note("Coupon strip", 5.0, 100.0, tuple(legs), formula=lambda p: p)
exact = float(strip.value(market))
results = {"Smile (exact)": (exact, 0.0)}
for label, vol in (("One vol: barrier", market.vol.vol(60.0, 5.0)),
                   ("One vol: ATM", market.vol.vol(100.0, 5.0))):
    one = Market(100.0, market.rate, market.dividend_yield, FlatVol(float(vol)))
    mc = price_note(strip, one, n_paths=200_000, seed=6).total
    results[label] = (mc.value, mc.std_error)
local = price_note(strip, market, n_paths=200_000, seed=6).total
results["Local vol MC"] = (local.value, local.std_error)
for name, (v, e) in results.items():
    print(f"{name:>18}: {v:7.3f} ± {e:.3f}  (gap to exact {v - exact:+.2f})")
fig, ax = plt.subplots(figsize=(7, 3.8))
names = list(results)
ax.bar(names, [results[n][0] for n in names], yerr=[2 * results[n][1] for n in names],
       color=[BLUE, RED, ORANGE, GREEN], capsize=6)
ax.set_ylim(80, 92)
ax.set(title="5-year digital coupons + European PDI(100, 60)", ylabel="% of notional")
fig.tight_layout()
""",
    ),
    ("md", "## Continuous barriers: one volatility against the smile"),
    (
        "code",
        """\
spec = find_note("Shark note (up)")
for name in ("Shark note (up)", "Shark note (down)"):
    note = find_note(name).build({}, 3.0, 100.0, Monitoring.CONTINUOUS)
    closed = float(note.value(market))
    mc = price_note(note, market, n_paths=100_000, seed=3).total
    print(f"{name}: closed form (one vol) {closed:.2f}, local vol MC {mc.value:.2f} ± "
          f"{mc.std_error:.2f} -> gap {mc.value - closed:+.2f} points")
""",
    ),
]

HEDGING = [
    (
        "md",
        "# Greeks and hedging\n\nMonte Carlo Greeks with common random numbers, the Greeks of "
        "a vanilla against the spot, a delta hedge simulated through time, and the order of the "
        "Taylor P&L explain.",
    ),
    ("code", SETUP),
    (
        "code",
        """\
from spricer import strategies as strat
from spricer.hedging import derman_kamal_std, simulate_delta_hedge
from spricer.market import Basket, FlatVol, Market
from spricer.market.defaults import default_market
from spricer.mc.greeks import coupon_note_greeks
from spricer.products.coupon_notes import phoenix
from spricer.risk import residual_order

market = default_market()
note = phoenix(maturity=2.0, frequency=4, coupon_pa=8.0)
basket = Basket.single(market)
crn = coupon_note_greeks(note, basket, n_paths=50_000, seed=3)
independent = coupon_note_greeks(note, basket, n_paths=50_000, seed=3, common_random_numbers=False)
names = ["Delta", "Gamma", "Vega", "Rho"]


def pick(g):
    return [g.assets[0].delta, g.assets[0].gamma, g.assets[0].vega, g.rho]


ratios = [b.std_error / a.std_error for a, b in zip(pick(crn), pick(independent), strict=True)]
for n, a, r in zip(names, pick(crn), ratios, strict=True):
    print(f"{n:>6}: {a.value:+.4f} ± {a.std_error:.4f} with common random numbers; "
          f"noise {r:.1f}x larger without")
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(names, ratios, color=BLUE)
ax.set(title="Phoenix Greeks: noise without / with CRN", ylabel="ratio of standard errors")
fig.tight_layout()
""",
    ),
    (
        "md",
        "## How big a spot bump?\n\nStructured payoffs jump at their barriers: with common "
        "random numbers only the paths crossing a jump between the two bumps contribute, so a "
        "small bump is unbiased but noisy. A barrier reverse convertible has a closed form to "
        "measure both effects (flat volatility, 100,000 paths).",
    ),
    (
        "code",
        """from spricer.products.coupon_notes import barrier_reverse_convertible

brc = barrier_reverse_convertible(maturity=1.0, frequency=4, coupon_pa=6.0, barrier=70.0)
bricks = brc.replication(100.0)
flat_vol = Market(100.0, 0.03, 0.02, FlatVol(0.22))


def closed(h):
    v = [float(bricks.value(flat_vol.with_spot(100.0 * (1 + x)))) for x in (h, 0.0, -h)]
    return 0.01 * (v[0] - 2 * v[1] + v[2]) / h**2


exact = closed(1e-3)
for h in (0.01, 0.03, 0.05):
    g = coupon_note_greeks(
        brc, Basket.single(flat_vol), n_paths=100_000, seed=1, delta_bump=h, gamma_bump=h
    ).assets[0].gamma
    print(
        f"bump {100 * h:.0f} %: gamma {g.value:+.3f} ± {g.std_error:.3f} "
        f"(noise {100 * g.std_error / abs(exact):.0f} %, bias of the bump "
        f"{100 * (closed(h) - exact) / abs(exact):+.1f} %), exact {exact:+.3f}"
    )
""",
    ),
    ("md", "## Delta hedging: realised against implied volatility"),
    (
        "code",
        """\
flat = Market(100.0, 0.03, 0.01, FlatVol(0.20))


def call(t):
    return strat.call(100.0, t)


vega = 100 * float(call(1.0).greeks(flat).vega) / 100.0
fig, ax = plt.subplots(figsize=(7, 3.8))
for n, color in ((52, ORANGE), (252, BLUE)):
    sim = simulate_delta_hedge(call, flat, 1.0, realised_vol=0.20, rebalancings=n, n_paths=20_000)
    ax.hist(sim.pnl, bins=80, alpha=0.6, color=color, label=f"{n} rebalancings")
    # Standard error of the std on independent antithetic pairs, by the delta method.
    half = len(sim.pnl) // 2
    squares = (sim.pnl - sim.pnl.mean()) ** 2
    pairs = (squares[:half] + squares[half:]) / 2
    std_error = pairs.std(ddof=1) / math.sqrt(half) / (2 * sim.std)
    print(f"{n:>3} rebalancings: mean {sim.mean.value:+.3f} ± {sim.mean.std_error:.3f}, "
          f"std {sim.std:.3f} ± {std_error:.3f} "
          f"(Derman-Kamal {derman_kamal_std(vega, 0.20, n):.3f})")
ax.set(title="Delta-hedged long call, realised = implied = 20 %", xlabel="P&L (% of spot)")
ax.legend()
fig.tight_layout()
other = Market(100.0, 0.03, 0.01, FlatVol(0.25))
gap = float(call(1.0).value(other)) - float(call(1.0).value(flat))
sim = simulate_delta_hedge(call, flat, 1.0, realised_vol=0.25, rebalancings=252, n_paths=20_000)
print(f"realised 25 % vs implied 20 %: mean {sim.mean.value:.3f} ± {sim.mean.std_error:.3f}, "
      f"Black-Scholes price gap {gap:.3f}")
""",
    ),
    ("md", "## The Taylor residual is third order"),
    (
        "code",
        """\
def value(m, t):
    return float(call(t).value(m))


# the Greeks are central differences with 1 % bumps: a 1 % move is matched exactly
moves = np.array([16.0, 8.0, 4.0, 2.0])
res = np.abs(residual_order(value, flat, 1.0, moves))
slope = np.polyfit(np.log(moves), np.log(res), 1)[0]
print(f"log-log slope of the residual against the move: {slope:.2f}")
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.loglog(moves, res, "o-", color=BLUE, label=f"residual (slope {slope:.2f})")
ax.loglog(moves, res[-1] * (moves / moves[-1]) ** 3, "--", color=GREY, label="slope 3")
ax.set(title="Unexplained P&L of a call", xlabel="Spot move (%)", ylabel="|residual| (% of spot)")
ax.legend()
fig.tight_layout()
""",
    ),
]


def build(name: str, cells: list[tuple[str, str]]) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = KERNEL
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        for kind, src in cells
    ]
    NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(NOTEBOOKS)}}).execute()
    nbformat.write(nb, NOTEBOOKS / f"{name}.ipynb")
    print(f"wrote notebooks/{name}.ipynb")


if __name__ == "__main__":
    NOTEBOOKS.mkdir(exist_ok=True)
    build("01_pricing_and_validation", VALIDATION)
    build("02_model_risk", MODEL_RISK)
    build("03_greeks_and_hedging", HEDGING)
    # same style as the rest of the repository (outputs are kept)
    subprocess.run([sys.executable, "-m", "ruff", "format", str(NOTEBOOKS)], check=True)
