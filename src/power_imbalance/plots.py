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


def loss_is_one_sided(df: pd.DataFrame, path: Path):
    """Why restricting to one side works: a correct call cannot lose."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.3), gridspec_kw={"width_ratios": [1, 1.15]})
    short = df.loc[df.system_long == 0, "spread"]
    long_ = df.loc[df.system_long == 1, "spread"]
    for i, (lbl, x, col) in enumerate((("System\nshort", short, RED), ("System\nlong", long_, TEAL))):
        a1.bar(i, x.max(), 0.55, color=col, alpha=0.85)
        a1.bar(i, x.min(), 0.55, color=col, alpha=0.85)
        a1.text(i, x.max() + 400, f"{x.max():,.0f}", ha="center", fontsize=9, color=NAVY, weight="bold")
        a1.text(i, x.min() - 800, f"{x.min():,.0f}", ha="center", fontsize=9, color=NAVY, weight="bold")
    a1.axhline(0, color=NAVY, lw=1.4)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["System\nshort", "System\nlong"], fontsize=9)
    a1.set_ylabel("Spread range, EUR/MWh")
    a1.set_ylim(-6200, 9600)
    a1.set_title("A correct call cannot lose:\nthe spread never crosses zero within a state", fontsize=10.5)

    ext = df["spread"].abs().nlargest(15).index
    sub = df.loc[ext].sort_values("spread")
    cols = np.where(sub.system_long == 1, TEAL, RED)
    a2.barh(range(len(sub)), sub["spread"], color=cols)
    a2.axvline(0, color=NAVY, lw=1.2)
    a2.set_yticks([]); a2.set_xlabel("Spread, EUR/MWh")
    a2.set_title("The 15 most extreme hours: 12 with the system long,\nbut the largest of all is the seller's nightmare", fontsize=10.5)
    return _save(fig, path)


def one_sided_comparison(results: dict, stability: dict, path: Path):
    """Equity and split-half, both sides against buy only."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
    for (name, pnl), col in zip(results.items(), (NAVY, TEAL, STEEL)):
        a1.plot(pnl.index, pnl.cumsum(), lw=1.9, color=col, label=name)
    a1.axhline(0, color=STEEL, lw=1)
    a1.set_ylabel("Cumulative P&L, EUR per MWh traded")
    a1.legend(fontsize=7.8, framealpha=0.95, loc="upper left")
    a1.set_title("Out-of-sample equity", fontsize=10.5)
    fig.autofmt_xdate()

    names = list(stability)
    x = np.arange(len(names))
    first = [stability[k]["first_half"]["mean"] for k in names]
    second = [stability[k]["second_half"]["mean"] for k in names]
    a2.bar(x - 0.2, first, 0.4, color=STEEL, label="First half")
    a2.bar(x + 0.2, second, 0.4, color=NAVY, label="Second half")
    a2.axhline(0, color=RED, ls="--", lw=1.2)
    a2.set_xticks(x)
    a2.set_xticklabels([k.replace(", ", ",\n") for k in names], fontsize=7)
    a2.set_ylabel("Mean P&L, EUR/MWh")
    a2.legend(fontsize=7.6, framealpha=0.95)
    a2.set_title("Only the one-sided version\nsurvives the second half", fontsize=10.5)
    return _save(fig, path)
