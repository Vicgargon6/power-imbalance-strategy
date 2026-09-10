"""Feature construction, restricted to what is knowable at gate closure.

Deliberately small. With 101 days of data and four raw columns, a large
feature set buys nothing except a better in-sample fit, and the whole point of
the exercise is to find out whether there is signal at all.

Two families:

  calendar   hour of day, day of week. Known with certainty and, as it turns
             out, the only family that carries usable signal — the daily shape
             of the Spanish system imbalance is driven by the solar profile.

  lagged     the system's own recent history, shifted by at least the minimum
             safe lag from `information.py`. Strongly autocorrelated at one
             hour and almost uninformative at the horizon we actually face.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .information import InformationSet


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour and weekday, encoded so a linear model can use them.

    The two harmonics matter: the imbalance shape has a midday trough and an
    evening peak, which a single sine cannot represent.
    """
    h = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    return pd.DataFrame(
        {
            "sin_h": np.sin(2 * np.pi * h / 24),
            "cos_h": np.cos(2 * np.pi * h / 24),
            "sin_2h": np.sin(4 * np.pi * h / 24),
            "cos_2h": np.cos(4 * np.pi * h / 24),
            "is_weekend": (dow >= 5).astype(float),
            "is_solar_hours": ((h >= 10) & (h <= 17)).astype(float),
        },
        index=index,
    )


def lagged_features(
    df: pd.DataFrame,
    info: InformationSet | None = None,
    extra_lags: tuple[int, ...] = (24, 48, 168),
) -> pd.DataFrame:
    """History of the system, shifted so every hour of the delivery day is legal.

    `min_lag` is the age of the freshest observation available for hour 23 of
    the delivery day, so applying it uniformly is conservative for every other
    hour. Cheap insurance against a subtle leak.
    """
    info = info or InformationSet()
    min_lag = info.min_safe_lag()

    out = pd.DataFrame(index=df.index)
    lags = tuple(sorted({min_lag, *(min_lag + L for L in extra_lags)}))
    for lag in lags:
        out[f"system_long_lag{lag}"] = df["system_long"].shift(lag)
        out[f"imbalance_mwh_lag{lag}"] = df["imbalance_mwh"].shift(lag)

    base = df["imbalance_mwh"].shift(min_lag)
    out["mwh_mean_24h"] = base.rolling(24).mean()
    out["mwh_std_24h"] = base.rolling(24).std()
    out["share_long_24h"] = df["system_long"].shift(min_lag).rolling(24).mean()
    out["share_long_168h"] = df["system_long"].shift(min_lag).rolling(168).mean()

    # Same clock hour on previous days: the daily shape is the strongest
    # structure in the data, so its own history is the natural lag to use.
    for d in (1, 2, 7):
        out[f"mwh_same_hour_d{d}"] = df["imbalance_mwh"].shift(24 * d + min_lag - 1)

    out.attrs["min_lag_hours"] = min_lag
    return out


def build(
    df: pd.DataFrame,
    info: InformationSet | None = None,
    use_lags: bool = True,
) -> pd.DataFrame:
    """Full design matrix. `use_lags=False` isolates the calendar-only model,
    which is the ablation that shows where the signal actually lives."""
    parts = [calendar_features(df.index)]
    if use_lags:
        parts.append(lagged_features(df, info))
    X = pd.concat(parts, axis=1)
    X.attrs["min_lag_hours"] = (info or InformationSet()).min_safe_lag()
    return X


def align(X: pd.DataFrame, y: pd.Series, *extra: pd.Series):
    """Drop the warm-up rows where lags are undefined, keeping everything aligned."""
    mask = X.notna().all(axis=1) & y.notna()
    for s in extra:
        mask &= s.notna()
    return (X[mask], y[mask], *[s[mask] for s in extra])
