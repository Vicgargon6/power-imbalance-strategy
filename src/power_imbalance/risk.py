"""Risk and performance metrics.

Task 2 asks for "a suitable metric" and then, separately, how to quantify the
risk. On this dataset those are the same question, because the naive answer
fails in an instructive way:

    always buy the day-ahead   median +29.75   mean +0.33   worst hour -4,713

A metric that rewards hit rate or total profit ranks that strategy highly. It
should not. So every summary here reports a central tendency, a dispersion, a
tail, and a concentration measure side by side, and the project's headline
metric is the Sharpe ratio of hourly P&L with CVaR quoted next to it — never
profit on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8760


def _as_series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x)


def sharpe(pnl: pd.Series, periods_per_year: int = HOURS_PER_YEAR) -> float:
    """Annualised Sharpe of hourly P&L. Risk-free rate ignored: the position is
    opened and closed within a day, so there is no funded capital to discount."""
    p = _as_series(pnl).dropna()
    if len(p) < 2 or p.std(ddof=1) == 0:
        return float("nan")
    return float(p.mean() / p.std(ddof=1) * np.sqrt(periods_per_year))


def sortino(pnl: pd.Series, periods_per_year: int = HOURS_PER_YEAR) -> float:
    """Sharpe's honest cousin for an asymmetric distribution: only downside
    deviation in the denominator."""
    p = _as_series(pnl).dropna()
    downside = p[p < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return float("nan")
    return float(p.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def var(pnl: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR: the loss exceeded in `alpha` of hours."""
    return float(_as_series(pnl).dropna().quantile(alpha))


def cvar(pnl: pd.Series, alpha: float = 0.05) -> float:
    """Expected loss in the worst `alpha` of hours.

    The metric that matters here. VaR tells you where the cliff starts; with a
    tail this heavy, what you need is the average depth of the fall.
    """
    p = _as_series(pnl).dropna()
    threshold = p.quantile(alpha)
    tail = p[p <= threshold]
    return float(tail.mean()) if len(tail) else float("nan")


def max_drawdown(pnl: pd.Series) -> float:
    """Deepest peak-to-trough of the cumulative P&L, in EUR/MWh."""
    c = _as_series(pnl).dropna().cumsum()
    return float((c - c.cummax()).min()) if len(c) else float("nan")


def tail_concentration(pnl: pd.Series, worst_pct: float = 0.01) -> float:
    """Share of the total absolute P&L produced by the worst hours.

    A strategy whose result is decided by five hours out of two thousand is not
    a strategy, it is a lottery ticket, and this number says so.
    """
    p = _as_series(pnl).dropna()
    if not len(p):
        return float("nan")
    k = max(1, int(len(p) * worst_pct))
    return float(p.nsmallest(k).abs().sum() / p.abs().sum())


def bootstrap_mean_ci(
    pnl: pd.Series, n: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Confidence interval for the mean, by block bootstrap over days.

    Blocks by day rather than by hour, because imbalance events cluster within
    a day and an hourly bootstrap would pretend the sample is far larger than
    it is. With 101 days the interval is wide, and it should be.
    """
    p = _as_series(pnl).dropna()
    if not isinstance(p.index, pd.DatetimeIndex):
        raise TypeError("bootstrap needs a datetime index to block by day")
    groups = [g.to_numpy() for _, g in p.groupby(p.index.normalize())]
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        means[i] = np.concatenate([groups[j] for j in pick]).mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summary(pnl: pd.Series, position: pd.Series | None = None) -> pd.Series:
    """One row per strategy. Ordered so the eye reads return, then risk, then
    whether the result is an artefact of a handful of hours."""
    p = _as_series(pnl).dropna()
    traded = p if position is None else p[position.reindex(p.index).fillna(0) != 0]
    lo, hi = bootstrap_mean_ci(p) if isinstance(p.index, pd.DatetimeIndex) else (np.nan, np.nan)
    out = {
        "hours": len(p),
        "hours_traded": len(traded),
        "share_traded": len(traded) / len(p) if len(p) else np.nan,
        "mean": p.mean(),
        "median": p.median(),
        "total": p.sum(),
        "mean_ci_low": lo,
        "mean_ci_high": hi,
        "hit_rate": float((traded > 0).mean()) if len(traded) else np.nan,
        "std": p.std(ddof=1),
        "sharpe_ann": sharpe(p),
        "sortino_ann": sortino(p),
        "var_5pct": var(p),
        "cvar_5pct": cvar(p),
        "max_drawdown": max_drawdown(p),
        "worst_hour": p.min(),
        "best_hour": p.max(),
        "tail_concentration_1pct": tail_concentration(p),
    }
    return pd.Series(out)


def compare(results: dict[str, tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    """`{name: (pnl, position)}` -> one row per strategy, sorted by Sharpe."""
    rows = {name: summary(pnl, pos) for name, (pnl, pos) in results.items()}
    df = pd.DataFrame(rows).T
    return df.sort_values("sharpe_ann", ascending=False)


def hours_to_detect(pnl: pd.Series, power: float = 0.8, alpha: float = 0.05) -> float:
    """How many trading hours are needed before the edge is distinguishable from zero.

    The question a risk committee should ask before allocating to a strategy with
    a thin edge and a fat tail: *if this works exactly as measured, how long until
    we can prove it?* And, symmetrically, how long before we could tell that it
    had stopped working.

    Standard two-sided test for a mean, with the sample's own dispersion:

        n = (z_{1-alpha/2} + z_{power})^2 * sigma^2 / mu^2

    On this dataset the answer is roughly 75 days at 80% power, against 52 days
    of out-of-sample history. The strategy cannot be validated by the data that
    produced it — which is a fact about the signal-to-noise ratio, not a defect
    of the backtest.
    """
    from scipy.stats import norm

    p = _as_series(pnl).dropna()
    mu, sd = p.mean(), p.std(ddof=1)
    if mu == 0:
        return float("inf")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(z**2 * sd**2 / mu**2)


def monthly_distribution(
    pnl: pd.Series, days: int = 21, n: int = 20000, seed: int = 0
) -> dict:
    """Distribution of a month of P&L, by bootstrap over whole trading days.

    Hourly CVaR is the right tail measure, but it is not the number a desk sizes
    against. What a desk needs is the distribution of a month, because that is
    the horizon of a stop-loss and of a conversation with a risk manager.
    """
    p = _as_series(pnl).dropna()
    daily = p.groupby(p.index.normalize()).sum().to_numpy()
    rng = np.random.default_rng(seed)
    sims = np.array([rng.choice(daily, days, replace=True).sum() for _ in range(n)])
    return {
        "days_per_month": days,
        "mean": float(sims.mean()),
        "p1": float(np.percentile(sims, 1)),
        "p5": float(np.percentile(sims, 5)),
        "p50": float(np.percentile(sims, 50)),
        "p95": float(np.percentile(sims, 95)),
        "prob_losing_month": float((sims < 0).mean()),
    }


def max_notional(monthly_loss: float, stop_loss_eur: float) -> float:
    """Notional in MWh per hour that keeps a bad month inside the stop-loss.

    Deliberately crude, and deliberately anchored on a percentile of the loss
    distribution rather than on the mean: sizing off the average outcome is how
    a desk discovers its limit by breaching it.

    If the percentile passed in is positive — as it is for the one-sided
    strategy at P5, where even a one-in-twenty month makes money — the stop is
    not the binding constraint at that confidence level. The caller should then
    size on a deeper percentile or on drawdown, and this returns NaN rather than
    infinity so that nobody reads it as "unlimited".
    """
    if monthly_loss >= 0:
        return float("nan")
    return float(stop_loss_eur / abs(monthly_loss))
