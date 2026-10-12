"""Cross-check against QuantLib's AnalyticEuropeanEngine (independent implementation)."""

import pytest

from spricer.analytic import black_scholes as bs

ql = pytest.importorskip("QuantLib")

TODAY = ql.Date(2, ql.January, 2026)
DAY_COUNT = ql.Actual365Fixed()


def quantlib_option(payoff, s, days, r, q, v):
    ql.Settings.instance().evaluationDate = TODAY
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(s)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, q, DAY_COUNT)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, DAY_COUNT)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), v, DAY_COUNT)),
    )
    option = ql.EuropeanOption(payoff, ql.EuropeanExercise(TODAY + days))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option


CASES = [
    (s, k, days, r, q, v)
    for s, k in ((100.0, 100.0), (100.0, 80.0), (100.0, 125.0), (37.5, 40.0))
    for days in (30, 365, 1095)
    for r, q in ((0.03, 0.01), (-0.005, 0.0))
    for v in (0.12, 0.40)
]


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(("s", "k", "days", "r", "q", "v"), CASES)
def test_vanilla_matches_quantlib(kind, s, k, days, r, q, v):
    ql_type = ql.Option.Call if kind == "call" else ql.Option.Put
    option = quantlib_option(ql.PlainVanillaPayoff(ql_type, k), s, days, r, q, v)
    args = {
        "spot": s,
        "strike": k,
        "maturity": days / 365.0,
        "rate": r,
        "dividend_yield": q,
        "vol": v,
    }
    g = bs.vanilla_greeks(kind, **args)
    assert float(bs.vanilla_price(kind, **args)) == pytest.approx(
        option.NPV(), rel=1e-10, abs=1e-12
    )
    assert float(g.delta) == pytest.approx(option.delta(), rel=1e-9, abs=1e-12)
    assert float(g.gamma) == pytest.approx(option.gamma(), rel=1e-9, abs=1e-12)
    assert float(g.vega) == pytest.approx(option.vega(), rel=1e-9, abs=1e-12)
    assert float(g.theta) == pytest.approx(option.theta(), rel=1e-9, abs=1e-12)
    assert float(g.rho) == pytest.approx(option.rho(), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(("s", "k", "days", "r", "q", "v"), CASES)
def test_digital_matches_quantlib(kind, s, k, days, r, q, v):
    ql_type = ql.Option.Call if kind == "call" else ql.Option.Put
    option = quantlib_option(ql.CashOrNothingPayoff(ql_type, k, 1.0), s, days, r, q, v)
    args = {
        "spot": s,
        "strike": k,
        "maturity": days / 365.0,
        "rate": r,
        "dividend_yield": q,
        "vol": v,
    }
    g = bs.digital_greeks(kind, **args)
    assert float(bs.digital_price(kind, **args)) == pytest.approx(
        option.NPV(), rel=1e-10, abs=1e-12
    )
    assert float(g.delta) == pytest.approx(option.delta(), rel=1e-9, abs=1e-12)
    assert float(g.gamma) == pytest.approx(option.gamma(), rel=1e-9, abs=1e-12)
    assert float(g.vega) == pytest.approx(option.vega(), rel=1e-9, abs=1e-12)
    assert float(g.rho) == pytest.approx(option.rho(), rel=1e-9, abs=1e-12)
