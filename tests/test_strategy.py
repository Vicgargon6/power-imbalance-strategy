import numpy as np
import pandas as pd

from omie_imbalance.strategy import StrategyParams, pnl, positions


def _pred(spread_hat, p_long=None):
    """Direction comes from p_long, size from spread_hat. When p_long is not
    given it is derived so that the two stages agree, which is the normal case."""
    spread_hat = np.asarray(spread_hat, dtype=float)
    if p_long is None:
        p_long = np.where(spread_hat > 0, 0.0, np.where(spread_hat < 0, 1.0, 0.5))
    p_long = np.asarray(p_long, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(spread_hat), freq="h")
    return pd.DataFrame(
        {"spread_hat": spread_hat, "p_long": p_long, "confidence": np.abs(p_long - 0.5) * 2},
        index=idx,
    )


def test_pnl_identity_long_earns_the_spread():
    """The whole exercise reduces to this. If it breaks, nothing else matters."""
    pred = _pred([10.0, -10.0])
    pos = positions(pred, StrategyParams())
    spread = pd.Series([25.0, 25.0], index=pred.index)
    assert list(pnl(pos, spread)) == [25.0, -25.0]


def test_costs_are_charged_on_absolute_size():
    pred = _pred([10.0, -10.0])
    pos = positions(pred, StrategyParams())
    spread = pd.Series([0.0, 0.0], index=pred.index)
    assert list(pnl(pos, spread, cost_per_mwh=0.5)) == [-0.5, -0.5]


def test_min_edge_filter_stands_down():
    pred = _pred([5.0, 50.0])
    pos = positions(pred, StrategyParams(min_edge=20.0))
    assert list(pos) == [0.0, 1.0]


def test_confidence_sizing_is_bounded():
    pred = _pred([10.0] * 3, p_long=[0.5, 0.25, 0.0])
    pos = positions(pred, StrategyParams(sizing="confidence"))
    assert pos.min() >= 0 and pos.max() <= 1
    assert pos.iloc[0] == 0.0 and pos.iloc[2] == 1.0


def test_direction_comes_from_the_classifier_not_the_regression():
    """Regression test for a real bug: when the two stages disagree, the side is
    the classifier's. Taking it from the regression cost the whole signal."""
    pred = _pred([-40.0, -40.0], p_long=[0.1, 0.9])
    pos = positions(pred, StrategyParams())
    assert list(pos) == [1.0, -1.0]


def test_position_never_exceeds_the_limit():
    pred = _pred([1000.0, -1000.0])
    pos = positions(pred, StrategyParams(max_position=0.4))
    assert pos.abs().max() <= 0.4
