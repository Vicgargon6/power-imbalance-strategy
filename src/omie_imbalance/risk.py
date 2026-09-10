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
