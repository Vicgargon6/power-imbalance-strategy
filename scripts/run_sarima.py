"""SARIMA on the imbalance volume, as an alternative route to the sign.

The idea is reasonable: rather than classifying the state directly, forecast the
system's net imbalance volume with a seasonal time-series model and take the sign
of the forecast. It is a different inductive bias — explicit daily seasonality and
autoregressive structure instead of a tree over lagged features.

It is fitted at the same legal horizon as everything else: the forecast starts 36
hours after the last observation, so nothing inside the delivery day is used.

    python scripts/run_sarima.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_imbalance import data, risk  # noqa: E402
from power_imbalance.information import InformationSet  # noqa: E402

TRAIN_WINDOW_HOURS = 24 * 30
ORDER, SEASONAL = (1, 0, 1), (1, 0, 1, 24)

df, _ = data.load(ROOT / "data" / "raw" / "market_data.csv")
lag = InformationSet().min_safe_lag()
days = pd.Series(df.index.normalize().unique()).sort_values()

t0, preds = time.time(), []
for d in days[40:]:
    hist = df.imbalance_mwh[df.index.normalize() < d]
    today = df.imbalance_mwh[df.index.normalize() == d]
    if len(hist) < TRAIN_WINDOW_HOURS or today.empty:
        continue
    try:
        fit = SARIMAX(hist.iloc[-TRAIN_WINDOW_HOURS:], order=ORDER, seasonal_order=SEASONAL,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = fit.forecast(steps=lag + 24)[-24:]
        preds.append(pd.DataFrame({"pred": fc.values[:len(today)], "real": today.values,
                                   "spread": df.spread[today.index].values}, index=today.index))
    except Exception:
        continue

S = pd.concat(preds)
real_long = S.real > 0
acc = float(((S.pred > 0) == real_long).mean())
base = float(max(real_long.mean(), 1 - real_long.mean()))

out = {"model": f"SARIMA{ORDER}{SEASONAL}", "hours": int(len(S)),
       "train_window_hours": TRAIN_WINDOW_HOURS, "forecast_lag_hours": lag,
       "sign_accuracy": acc, "base_rate": base, "fit_seconds": round(time.time() - t0, 1),
       "strategies": {}}

print(f"{out['model']} on the imbalance volume · {len(S)} hours · {out['fit_seconds']}s")
print(f"  sign accuracy {acc:.1%}   base rate {base:.1%}")
for name, pos in (("both sides", np.where(S.pred < 0, 1, -1)),
                  ("buy side only", np.where(S.pred < 0, 1, 0)),
                  ("buy, |forecast| filter", np.where(S.pred < -S.pred.abs().quantile(0.3), 1, 0))):
    pnl = pd.Series(pos * S.spread.values, index=S.index)
    lo, hi = risk.bootstrap_mean_ci(pnl)
    out["strategies"][name] = {"mean": float(pnl.mean()), "ci_low": lo, "ci_high": hi,
                              "sharpe_ann": risk.sharpe(pnl), "cvar_5pct": risk.cvar(pnl),
                              "share_traded": float(np.mean(pos != 0))}
    print(f"  {name:24} mean {pnl.mean():+7.2f} [{lo:+6.2f},{hi:+6.2f}]  Sharpe {risk.sharpe(pnl):+6.2f}")

(ROOT / "reports" / "sarima.json").write_text(json.dumps(out, indent=2, default=float))
print(f"\n-> {ROOT/'reports'/'sarima.json'}")
