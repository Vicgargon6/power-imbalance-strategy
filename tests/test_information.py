"""The tests that matter: they check the backtest cannot see the future."""
import pandas as pd
import pytest

from power_imbalance.features import build
from power_imbalance.information import InformationSet, assert_causal


def test_decision_time_is_noon_the_day_before():
    info = InformationSet()
    assert info.decision_time(pd.Timestamp("2024-02-05 07:00")) == pd.Timestamp("2024-02-04 12:00")


def test_lag_grows_across_the_delivery_day():
    """Every hour of D is decided at the same moment, so the last hour of the
    day faces the oldest information. If this ever stops holding, the gate
    closure model is wrong."""
    info = InformationSet()
    lags = [info.lag_hours(pd.Timestamp("2024-02-05") + pd.Timedelta(hours=h)) for h in range(24)]
    assert lags == sorted(lags)
    assert lags[0] == 13 and lags[-1] == 36


def test_min_safe_lag_covers_the_worst_hour():
    info = InformationSet()
    worst = max(info.lag_hours(pd.Timestamp("2024-02-05") + pd.Timedelta(hours=h)) for h in range(24))
    assert info.min_safe_lag() == worst


def test_features_contain_no_contemporaneous_outcome(df):
    X = build(df).dropna()
    assert_causal(X, df, min_lag=InformationSet().min_safe_lag())


def test_assert_causal_catches_a_leak(df):
    """Negative control: plant the leak and check the guard fires."""
    X = build(df).dropna()
    X = X.assign(leak=df["imbalance_mwh"].reindex(X.index))
    with pytest.raises(AssertionError):
        assert_causal(X, df, min_lag=36)


def test_no_lagged_feature_uses_the_current_day(df):
    """A 36-hour lag on hour 0 of day D reaches back to 12:00 of D-2, so no
    feature may correlate perfectly with anything inside day D itself."""
    X = build(df).dropna()
    lagged = [c for c in X.columns if "lag" in c or "same_hour" in c]
    for col in lagged:
        shifted = df["imbalance_mwh"].reindex(X.index)
        assert abs(X[col].corr(shifted)) < 0.9
