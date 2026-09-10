"""End-to-end run: loads the data, backtests every strategy, writes the results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_imbalance import backtest as bt  # noqa: E402
from power_imbalance import data, information, risk  # noqa: E402
from power_imbalance.strategy import StrategyParams  # noqa: E402

OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

df, quality = data.load(ROOT / "data" / "raw" / "market_data.csv")
print(quality, "\n")
print("Does the system sign pin the sign of the spread?")
print(data.sign_agreement(df).to_string(), "\n")

info = information.InformationSet()
print(f"Minimum safe information lag: {info.min_safe_lag()} hours\n")

cfg_always = bt.BacktestConfig(params=StrategyParams())
cfg_filter = bt.BacktestConfig(params=StrategyParams(min_edge=20.0))

results = {
    "Always buy DA": bt.run_constant(df, +1, cfg_always),
    "Always sell DA": bt.run_constant(df, -1, cfg_always),
    "Two-stage model": bt.run_model(df, cfg_always),
    "Two-stage, calendar only": bt.run_model(df, bt.BacktestConfig(use_lags=False)),
    "Two-stage, LightGBM": bt.run_model(df, bt.BacktestConfig(classifier="lightgbm")),
    "Two-stage, vol-scaled": bt.run_model(
        df, bt.BacktestConfig(params=StrategyParams(sizing="kelly_like"))),
    "LightGBM, buy side only": bt.run_model(df, bt.BacktestConfig(
        classifier="lightgbm", params=StrategyParams(side="buy", min_confidence=0.2))),
    "LightGBM, sell side only": bt.run_model(df, bt.BacktestConfig(
        classifier="lightgbm", params=StrategyParams(side="sell", min_confidence=0.2))),
    "Seasonal median": bt.run_seasonal(df, cfg_always),
    "Seasonal median, |edge|>=20": bt.run_seasonal(df, cfg_filter, min_abs_median=20.0),
    "Oracle (system sign)": bt.run_oracle(df, cfg_always),
}

table = risk.compare({k: (v.pnl, v.position) for k, v in results.items()})
pd.set_option("display.width", 200, "display.max_columns", 30)
print(table.round(3).to_string(), "\n")
table.to_csv(OUT / "strategy_comparison.csv")

# ---------------------------------------------------------------- stability
# One number decides whether any of this is believable: does the edge survive
# into the second half of the out-of-sample period, or was it one good month?
stability = {}
for name in ("LightGBM, buy side only", "Two-stage, LightGBM", "Two-stage model", "Seasonal median, |edge|>=20"):
    p = results[name].pnl
    mid = p.index[len(p) // 2]
    halves = {}
    for label, sub in (("first_half", p[p.index < mid]), ("second_half", p[p.index >= mid])):
        lo, hi = risk.bootstrap_mean_ci(sub)
        halves[label] = {"from": str(sub.index.min().date()), "to": str(sub.index.max().date()),
                         "mean": float(sub.mean()), "ci_low": lo, "ci_high": hi,
                         "sharpe_ann": risk.sharpe(sub)}
    stability[name] = halves

print("STABILITY — the same strategy, split in two halves")
for name, h in stability.items():
    print(f"  {name}")
    for label, v in h.items():
        print(f"    {label:12} {v['from']}..{v['to']}  mean {v['mean']:+7.2f}"
              f"  [{v['ci_low']:+.2f}, {v['ci_high']:+.2f}]  Sharpe {v['sharpe_ann']:+.2f}")
print()

# ---------------------------------------------------------------- capacity
# What a desk would actually need to know before allocating to this.
best = results["LightGBM, buy side only"].pnl
capacity = {
    "hours_to_detect_80pct_power": risk.hours_to_detect(best, power=0.8),
    "hours_to_detect_90pct_power": risk.hours_to_detect(best, power=0.9),
    "hours_available_out_of_sample": int(len(best)),
    "monthly": risk.monthly_distribution(best),
}
# Size on the worst percentile that is actually a loss. For the one-sided
# strategy even the P5 month makes money, so the P1 is the binding one.
# Which constraint actually binds depends on the strategy. For the bidirectional
# version the monthly stop binds, because a one-in-twenty month loses money. For
# the one-sided version even the P1 month is roughly flat, so dividing by it
# explodes; there the binding constraint is the drawdown, and saying so is more
# useful than quoting a notional of thirty thousand megawatt hours.
m0 = capacity["monthly"]
dd = abs(risk.max_drawdown(best))
capacity["max_drawdown"] = -dd
capacity["binding_constraint"] = "monthly stop" if m0["p5"] < 0 else "drawdown"
capacity["max_notional_mwh_per_hour"] = {
    f"limit_{s // 1000}k": (risk.max_notional(m0["p5"], s) if m0["p5"] < 0 else s / dd)
    for s in (50_000, 100_000, 250_000)
}
m = capacity["monthly"]
print("CAPACITY — what a risk committee would ask")
print(f"  hours needed to prove the edge (80% power) : {capacity['hours_to_detect_80pct_power']:,.0f}"
      f"  ({capacity['hours_to_detect_80pct_power'] / 24:,.0f} days)")
print(f"  hours actually available out of sample     : {capacity['hours_available_out_of_sample']:,}"
      f"  ({capacity['hours_available_out_of_sample'] / 24:,.0f} days)")
print(f"  month of P&L, EUR per MWh of notional      : P1 {m['p1']:,.0f} · P5 {m['p5']:,.0f}"
      f" · median {m['p50']:,.0f} · P95 {m['p95']:,.0f}")
print(f"  probability of a losing month              : {m['prob_losing_month']:.1%}")
print(f"  binding constraint                         : {capacity['binding_constraint']}"
      f"  (P5 month {m['p5']:,.0f} · max drawdown {capacity['max_drawdown']:,.0f})")
for k, v in capacity["max_notional_mwh_per_hour"].items():
    print(f"  max notional with a {k.replace('limit_', '').replace('k', ' kEUR')} limit : {v:,.0f} MWh/h")
print()

summary = {
    "stability": stability,
    "capacity": capacity,
    "data": {
        "rows": quality.rows, "days": quality.days,
        "first": str(quality.first), "last": str(quality.last),
    },
    "sign_agreement": data.sign_agreement(df).to_dict(),
    "min_safe_lag_hours": info.min_safe_lag(),
    "strategies": json.loads(table.to_json(orient="index")),
}
(OUT / "results.json").write_text(json.dumps(summary, indent=2, default=float))
for name, res in results.items():
    key = name.lower().replace(" ", "_").replace(",", "").replace("|", "").replace(">=", "ge")
    res.pnl.to_frame("pnl").assign(position=res.position).to_csv(OUT / f"pnl_{key}.csv")
print(f"-> {OUT/'results.json'}, {OUT/'strategy_comparison.csv'}")

# ---------------------------------------------------------------- figures
from power_imbalance import plots  # noqa: E402

FIG = OUT / "figures"
plots.spread_distribution(df, FIG / "f1_spread_distribution.png")
plots.sign_relationship(df, FIG / "f2_sign_relationship.png")
plots.hourly_shape(df, FIG / "f3_hourly_shape.png")
plots.equity_curves({k: v.pnl for k, v in results.items()}, FIG / "f4_equity.png",
                    exclude=("Oracle (system sign)",))
plots.risk_return(table, FIG / "f5_risk_return.png", exclude=("Oracle (system sign)",))
print(f"-> {FIG}")

# ---------------------------------------------------------------- addendum figures
plots.loss_is_one_sided(df, FIG / "f6_one_sided_loss.png")
plots.one_sided_comparison(
    {k: results[k].pnl for k in ("LightGBM, buy side only", "Two-stage, LightGBM", "LightGBM, sell side only")},
    {k: stability[k] for k in ("LightGBM, buy side only", "Two-stage, LightGBM")},
    FIG / "f7_one_sided_comparison.png")
print(f"-> {FIG/'f6_one_sided_loss.png'}, {FIG/'f7_one_sided_comparison.png'}")
