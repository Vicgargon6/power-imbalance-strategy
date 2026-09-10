"""Walk-forward backtest.

One rule, applied without exception: to trade day D, the model may only have
seen data up to and including day D-1, and the features for day D may only use
observations older than the information lag from `information.py`.

The model is refitted every day on an expanding window. With 101 days that is
cheap, and it is also the honest choice — a single train/test split would let
one lucky regime decide the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import align, build
from .information import InformationSet
from .models import SeasonalBaseline, TwoStageModel, oracle
from .strategy import StrategyParams, pnl, positions


@dataclass
class BacktestConfig:
    train_days: int = 40          # minimum history before the first trade
    refit_every: int = 1          # days between refits
    use_lags: bool = True
    classifier: str = "logistic"
    cost_per_mwh: float = 0.0
    params: StrategyParams = field(default_factory=StrategyParams)


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    position: pd.Series
    pnl: pd.Series
    config: BacktestConfig

    @property
    def daily(self) -> pd.Series:
        return self.pnl.groupby(self.pnl.index.normalize()).sum()

    @property
    def equity(self) -> pd.Series:
        return self.pnl.cumsum()


def _trading_days(index: pd.DatetimeIndex, train_days: int) -> list[pd.Timestamp]:
    days = pd.Series(index.normalize().unique()).sort_values()
    return list(days[train_days:])


def run_model(df: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    """Walk-forward the two-stage model."""
    config = config or BacktestConfig()
    info = InformationSet()
    X_all = build(df, info, use_lags=config.use_lags)
    X_all, y_long, y_spread = align(X_all, df["system_long"], df["spread"])
    dates = X_all.index.normalize()

    preds, model, last_fit = [], None, None
    for day in _trading_days(X_all.index, config.train_days):
        train = dates < day
        test = dates == day
        if train.sum() < 24 * 10 or test.sum() == 0:
            continue
        if model is None or last_fit is None or (day - last_fit).days >= config.refit_every:
            model = TwoStageModel(classifier=config.classifier).fit(
                X_all[train], y_long[train], y_spread[train]
            )
            last_fit = day
        preds.append(model.predict(X_all[test]))

    predictions = pd.concat(preds).sort_index()
    spread = df["spread"].reindex(predictions.index)
    pos = positions(predictions, config.params, realised_spread=df["spread"])
    return BacktestResult(predictions, pos, pnl(pos, spread, config.cost_per_mwh), config)


def run_seasonal(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
    min_abs_median: float = 0.0,
) -> BacktestResult:
    """Walk-forward the seasonal baseline: the hourly median spread, re-estimated
    each day on all history available up to the previous day."""
    config = config or BacktestConfig()
    preds = []
    for day in _trading_days(df.index, config.train_days):
        hist = df.loc[df.index.normalize() < day, "spread"]
        today = df.loc[df.index.normalize() == day]
        if len(hist) < 24 * 10 or today.empty:
            continue
        base = SeasonalBaseline(min_abs_median=min_abs_median).fit(hist)
        preds.append(base.predict(today.index))

    predictions = pd.concat(preds).sort_index()
    spread = df["spread"].reindex(predictions.index)
    pos = positions(predictions, config.params, realised_spread=df["spread"])
    return BacktestResult(predictions, pos, pnl(pos, spread, config.cost_per_mwh), config)


def run_constant(df: pd.DataFrame, side: int, config: BacktestConfig | None = None) -> BacktestResult:
    """Always buy (+1) or always sell (-1). The strategies a metric must reject."""
    config = config or BacktestConfig()
    days = _trading_days(df.index, config.train_days)
    idx = df.index[df.index.normalize().isin(days)]
    predictions = pd.DataFrame(
        {"p_long": 0.5, "confidence": 1.0, "spread_hat": float(side)}, index=idx
    )
    pos = pd.Series(float(side), index=idx, name="position")
    return BacktestResult(predictions, pos, pnl(pos, df["spread"].reindex(idx), config.cost_per_mwh), config)


def run_oracle(df: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    """Perfect foresight of the system sign. The ceiling, not a strategy."""
    config = config or BacktestConfig()
    days = _trading_days(df.index, config.train_days)
    sub = df[df.index.normalize().isin(days)]
    predictions = oracle(sub)
    pos = pd.Series(np.sign(sub["spread"].to_numpy()), index=sub.index, name="position")
    return BacktestResult(predictions, pos, pnl(pos, sub["spread"], config.cost_per_mwh), config)
