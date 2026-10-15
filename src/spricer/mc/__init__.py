"""Monte Carlo engine: local volatility paths, barriers, standard errors."""

from spricer.mc.correlation import check_correlation, nearest_correlation
from spricer.mc.local_vol import local_vol
from spricer.mc.paths import MCEstimate, PathSet, simulate
from spricer.mc.products import NoteMonteCarlo, brick_payoff, price_brick, price_note

__all__ = [
    "MCEstimate",
    "NoteMonteCarlo",
    "PathSet",
    "brick_payoff",
    "check_correlation",
    "local_vol",
    "nearest_correlation",
    "price_brick",
    "price_note",
    "simulate",
]
