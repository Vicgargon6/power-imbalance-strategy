"""End-to-end run: loads the data, backtests every strategy, writes the results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omie_imbalance import backtest as bt  # noqa: E402
from omie_imbalance import data, information, risk  # noqa: E402
from omie_imbalance.strategy import StrategyParams  # noqa: E402

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
for name in ("Two-stage, LightGBM", "Two-stage model", "Seasonal median, |edge|>=20"):
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

summary = {
    "stability": stability,
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
from omie_imbalance import plots  # noqa: E402

FIG = OUT / "figures"
plots.spread_distribution(df, FIG / "f1_spread_distribution.png")
plots.sign_relationship(df, FIG / "f2_sign_relationship.png")
plots.hourly_shape(df, FIG / "f3_hourly_shape.png")
plots.equity_curves({k: v.pnl for k, v in results.items()}, FIG / "f4_equity.png",
                    exclude=("Oracle (system sign)",))
plots.risk_return(table, FIG / "f5_risk_return.png", exclude=("Oracle (system sign)",))
print(f"-> {FIG}")
