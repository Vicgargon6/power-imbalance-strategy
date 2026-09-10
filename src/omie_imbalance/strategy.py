"""From a forecast to a position.

The P&L of one MWh is exact and has no third case:

    position  +1  buy in the day-ahead, sell in the imbalance market  ->  +spread
    position  -1  sell in the day-ahead, buy in the imbalance market  ->  -spread
    position   0  do not trade                                        ->      0

So a strategy is a map from (probability, expected spread) to a number in
[-1, 1]. Everything interesting lives in two decisions: when to stand down, and
how large to go when you do trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyParams:
    """
    min_confidence   below this, do not trade. The single most valuable
                     parameter in the whole project: on this data a strategy
                     that always trades is a strategy that always pays the tail.
    min_edge         minimum expected spread, EUR/MWh, to bother.
    sizing           'binary'      full size whenever the filters pass
                     'confidence'  size proportional to |p - 0.5|
                     'kelly_like'  size proportional to edge / variance, capped
    max_position     cap in MWh per hour. Risk limit, not an optimisation knob.
    vol_lookback     hours used for the running spread volatility in kelly_like.
    """

    min_confidence: float = 0.0
    min_edge: float = 0.0
    sizing: str = "binary"
    max_position: float = 1.0
    vol_lookback: int = 168


def _direction(pred: pd.DataFrame) -> np.ndarray:
    """Buy the day-ahead when the spread is expected positive, sell otherwise.

    Taken from the predicted spread rather than from the probability, because
    the two agree by construction and the spread also carries the size.
    """
    return np.sign(pred["spread_hat"].to_numpy())


def positions(
    pred: pd.DataFrame,
    params: StrategyParams,
    realised_spread: pd.Series | None = None,
) -> pd.Series:
    """Turn forecasts into positions in MWh, one per delivery hour."""
    direction = _direction(pred)
    edge = np.abs(pred["spread_hat"].to_numpy())
    conf = pred["confidence"].to_numpy()

    trade = (conf >= params.min_confidence) & (edge >= params.min_edge)

    if params.sizing == "binary":
        size = np.ones(len(pred))
    elif params.sizing == "confidence":
        size = conf
    elif params.sizing == "kelly_like":
        if realised_spread is None:
            raise ValueError("kelly_like sizing needs the realised spread history")
        # Variance estimated only from the past, shifted so the current hour is
        # never used to size the position taken in it.
        vol = (
            realised_spread.shift(1)
            .rolling(params.vol_lookback, min_periods=24)
            .std()
            .reindex(pred.index)
            .bfill()
        )
        size = np.clip(edge / np.maximum(vol.to_numpy(), 1e-6), 0.0, 1.0)
    else:
        raise ValueError(f"unknown sizing rule: {params.sizing}")

    pos = direction * size * trade
    return pd.Series(np.clip(pos, -params.max_position, params.max_position),
                     index=pred.index, name="position")


def pnl(position: pd.Series, spread: pd.Series, cost_per_mwh: float = 0.0) -> pd.Series:
    """Realised P&L in EUR per MWh of notional.

    `cost_per_mwh` covers whatever a real desk pays to be in this trade at all
    — OMIE fees, the balance-responsible-party share, collateral. Small next to
    the spread, but it is the difference between a strategy that survives
    contact with an operations budget and one that does not.
    """
    p = position.reindex(spread.index).fillna(0.0)
    gross = p * spread
    cost = cost_per_mwh * p.abs()
    return (gross - cost).rename("pnl")
