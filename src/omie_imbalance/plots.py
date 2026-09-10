"""Figures. One idea per chart, same palette as the rest of the portfolio."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY, TEAL, STEEL = "#003D6A", "#6BC5CF", "#AABECD"
PALE, RED, GREEN = "#B5E2E7", "#A3320B", "#2E7D32"

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": STEEL,
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
        "axes.titlecolor": NAVY,
    }
)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def spread_distribution(df: pd.DataFrame, path: Path, clip: float = 200.0):
    """Why the mean is the wrong summary: the body is shifted positive, the tail
    is catastrophic, and they point in opposite directions."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})
    s = df["spread"]
    a1.hist(s.clip(-clip, clip), bins=80, color=TEAL, edgecolor="white", linewidth=0.3)
    a1.axvline(0, color=STEEL, lw=1)
    a1.axvline(s.median(), color=NAVY, lw=2, label=f"median {s.median():+.1f}")
    a1.axvline(s.mean(), color=RED, lw=2, ls="--", label=f"mean {s.mean():+.1f}")
    a1.set_xlabel(f"Spread, EUR/MWh (clipped at ±{clip:.0f} for display)")
    a1.set_ylabel("Hours")
    a1.legend(fontsize=7.6, framealpha=0.95)
    a1.set_title("The body says buy. The mean says nothing.", fontsize=10.5)

    ordered = s.sort_values().to_numpy()
    a2.plot(np.arange(len(ordered)) / len(ordered) * 100, ordered, color=NAVY, lw=1.4)
    a2.axhline(0, color=STEEL, lw=1)
    a2.set_yscale("symlog", linthresh=50)
    a2.set_xlabel("Percentile of hours")
    a2.set_ylabel("Spread, EUR/MWh (symlog)")
    a2.set_title(f"Worst hour {s.min():,.0f}, best {s.max():,.0f}", fontsize=10.5)
    return _save(fig, path)


def sign_relationship(df: pd.DataFrame, path: Path):
    """The finding the whole project rests on."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.2))
    for k, label, col in ((0, "System short", RED), (1, "System long", TEAL)):
        sub = df.loc[df.system_long == k, "spread"].clip(-200, 200)
        a1.hist(sub, bins=60, alpha=0.75, color=col, label=label, edgecolor="white", linewidth=0.3)
    a1.axvline(0, color=NAVY, lw=1.2)
    a1.set_xlabel("Spread, EUR/MWh (clipped for display)")
    a1.set_ylabel("Hours")
    a1.legend(fontsize=8, framealpha=0.95)
    a1.set_title("The two states barely overlap", fontsize=10.5)

    probs = [
        float((df.loc[df.system_long == 0, "spread"] > 0).mean()),
        float((df.loc[df.system_long == 1, "spread"] > 0).mean()),
    ]
    a2.bar(["System short", "System long"], probs, color=[RED, TEAL], width=0.55)
    for i, p in enumerate(probs):
        a2.text(i, p + 0.03, f"{p:.1%}", ha="center", fontsize=11, color=NAVY, weight="bold")
    a2.set_ylim(0, 1.15)
    a2.set_ylabel("P(spread > 0)")
    a2.set_title("Forecasting the spread sign is forecasting\nthe system sign. Nothing else.", fontsize=10.5)
    return _save(fig, path)


def hourly_shape(df: pd.DataFrame, path: Path):
    """The only structure that survives the information constraint."""
    g = df.groupby(df.index.hour)
    med = g["spread"].median()
    p_long = g["system_long"].mean()
    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    ax.bar(med.index, med.to_numpy(), color=np.where(med > 0, TEAL, RED), width=0.75)
    ax.axhline(0, color=NAVY, lw=1)
    ax.set_xlabel("Delivery hour")
    ax.set_ylabel("Median spread, EUR/MWh")
    ax2 = ax.twinx()
    ax2.plot(p_long.index, p_long.to_numpy(), "o-", color=NAVY, lw=1.8, ms=4)
    ax2.set_ylabel("P(system long)")
    ax2.set_ylim(0, 1)
    ax2.spines["right"].set_visible(True)
    ax.set_title("Midday the system is long and the spread is negative; the evening reverses it.\n"
                 "This is the solar profile, and it is knowable a day ahead.", fontsize=10.5)
    return _save(fig, path)


def equity_curves(results: dict[str, pd.Series], path: Path, exclude: tuple[str, ...] = ()):
    """Cumulative P&L per MWh. Oracle plotted separately when included, because
    it is a ceiling and squashes everything else."""
    fig, ax = plt.subplots(figsize=(9.4, 3.4))
    colors = [NAVY, TEAL, STEEL, GREEN, RED, "#7A5C99"]
    for i, (name, pnl) in enumerate(r for r in results.items() if r[0] not in exclude):
        ax.plot(pnl.index, pnl.cumsum(), lw=1.7, color=colors[i % len(colors)], label=name)
    ax.axhline(0, color=STEEL, lw=1)
    ax.set_ylabel("Cumulative P&L, EUR per MWh traded")
    ax.legend(fontsize=7.8, framealpha=0.95, loc="upper left")
    ax.set_title("Out-of-sample equity, walk-forward", fontsize=10.5)
    fig.autofmt_xdate()
    return _save(fig, path)


def risk_return(table: pd.DataFrame, path: Path, exclude: tuple[str, ...] = ()):
    """Mean against CVaR. The chart that answers 'a suitable metric'."""
    t = table.drop(index=[i for i in exclude if i in table.index])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.scatter(-t["cvar_5pct"], t["mean"], s=90, color=NAVY, zorder=3)
    for name, row in t.iterrows():
        ax.annotate(name, (-row["cvar_5pct"], row["mean"]), fontsize=8,
                    xytext=(6, 4), textcoords="offset points", color=NAVY)
    ax.axhline(0, color=RED, ls="--", lw=1.1)
    ax.set_xlabel("Tail risk: |CVaR 5%|, EUR/MWh")
    ax.set_ylabel("Mean P&L, EUR/MWh")
    ax.set_title("Return against the tail, not against the standard deviation", fontsize=10.5)
    return _save(fig, path)
