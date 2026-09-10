import numpy as np
import pandas as pd

from power_imbalance import risk


def _series(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="h"))


def test_cvar_is_at_least_as_bad_as_var():
    p = _series(np.random.default_rng(0).standard_t(3, 2000) * 50)
    assert risk.cvar(p) <= risk.var(p)


def test_max_drawdown_is_never_positive():
    assert risk.max_drawdown(_series([1.0, 2.0, -5.0, 1.0])) < 0


def test_tail_concentration_flags_a_single_bad_hour():
    """A strategy decided by one hour out of a thousand must score near 1."""
    values = np.full(1000, 1.0)
    values[0] = -100000.0
    assert risk.tail_concentration(_series(values)) > 0.9


def test_metric_rejects_the_naive_always_buy(df):
    """Regression test on the project's central claim: judged on total profit
    the naive strategy looks fine; judged on the tail it does not."""
    always_buy = df["spread"]
    assert always_buy.sum() > 0                      # profitable in total
    assert always_buy.median() > 20                  # and on the median hour
    assert risk.cvar(always_buy) < -100              # yet the tail is brutal
    assert risk.tail_concentration(always_buy) > 0.05


def test_bootstrap_interval_brackets_the_mean(df):
    lo, hi = risk.bootstrap_mean_ci(df["spread"], n=500)
    assert lo < df["spread"].mean() < hi
