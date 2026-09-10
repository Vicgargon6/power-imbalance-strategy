"""Turning a model score into a position size, and the trap on the way there.

The natural rule is "trade when the model is more than 80% confident". It is
also the rule that quietly breaks, because a gradient-boosted tree fitted on a
hundred days does not produce probabilities that mean what they say. On this
data the hours the model scores at 0.90 contain a short system 65% of the time.
A ladder built on those numbers is a ladder built on a label.

So the confidence here is defined by **realised frequency inside a score
bucket**, not by the number the model prints. The score orders the hours; the
data says what each rung is worth. That distinction is the whole module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .risk import cvar, sharpe


def calibration_table(score: pd.Series, is_short: pd.Series,
                      edges: tuple[float, ...] = (0, .3, .4, .5, .6, .7, .8, 1.01)) -> pd.DataFrame:
    """Claimed confidence against realised frequency. Read the error column.

    `score` is the model's confidence that the system will be short — the side
    the strategy actually takes. `is_short` is what happened.
    """
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (score >= lo) & (score < hi)
        if m.sum() < 10:
            continue
        rows.append({"bucket": f"{lo:.0%}-{min(hi, 1):.0%}", "hours": int(m.sum()),
                     "claimed": float(score[m].mean()), "realised": float(is_short[m].mean()),
                     "error": float(is_short[m].mean() - score[m].mean())})
    return pd.DataFrame(rows)


def ladder(score: pd.Series, spread: pd.Series, is_short: pd.Series,
           n_buckets: int = 10) -> pd.DataFrame:
    """The decision table: one row per score decile.

    Deciles rather than fixed probability cuts, because the cut points then come
    from the distribution of the score instead of from a number the model is not
    entitled to claim. Each row reports what the hours in it actually did.
    """
    dec = pd.qcut(score, n_buckets, labels=False, duplicates="drop") + 1
    out = pd.DataFrame({"score": score, "spread": spread, "short": is_short, "decile": dec})
    g = out.groupby("decile")
    tab = pd.DataFrame({
        "hours": g.size(),
        "mean_score": g["score"].mean(),
        "realised_p_short": g["short"].mean(),
        "mean_pnl": g["spread"].mean(),
        "median_pnl": g["spread"].median(),
        "sharpe": g["spread"].apply(sharpe),
        "cvar_5pct": g["spread"].apply(cvar),
        "worst": g["spread"].min(),
    })
    return tab


@dataclass(frozen=True)
class SizeLadder:
    """Position size per score decile, lowest decile first.

    The default is graded rather than binary: nothing below the median score,
    then a quarter, a half, three quarters and full size. Grading beats a single
    cut at the same average exposure — 6.08 against 5.19 EUR/MWh at an exposure
    of 0.35 — because the top deciles carry most of the edge and the ones just
    above the cut carry very little of it.
    """

    weights: tuple[float, ...] = (0, 0, 0, 0, 0, 0.25, 0.5, 0.75, 1.0, 1.0)

    def __post_init__(self):
        if not all(0 <= w <= 1 for w in self.weights):
            raise ValueError("weights must lie in [0, 1]")
        if list(self.weights) != sorted(self.weights):
            raise ValueError("weights must be non-decreasing in confidence")

    def size(self, score: pd.Series, reference: pd.Series | None = None) -> pd.Series:
        """Map scores to sizes through the decile boundaries.

        `reference` is the score history the boundaries are computed from. In a
        backtest it must be the training scores only — passing the test scores
        would leak the future into the position size, which is the subtle way
        this kind of rule cheats.
        """
        ref = reference if reference is not None else score
        n = len(self.weights)
        qs = np.quantile(ref, np.linspace(0, 1, n + 1)[1:-1])
        idx = np.searchsorted(qs, score.to_numpy(), side="right")
        return pd.Series(np.asarray(self.weights)[idx], index=score.index, name="size")
