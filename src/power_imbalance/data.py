"""Loading and validation of the day-ahead / imbalance dataset.

The raw file is hourly and has four fields: day-ahead price, imbalance price,
and the system's net imbalance volume. Everything else in this project is
derived from those.

Sign convention for the system volume, as given with the data:

    imbalance_mwh > 0  ->  the system is LONG  (surplus energy)
    imbalance_mwh < 0  ->  the system is SHORT (deficit energy)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["date", "hour", "da_price", "imbalance_price", "imbalance_mwh"]


@dataclass(frozen=True)
class DataQuality:
    """What the loader found. Printed in the notebook rather than asserted away."""

    rows: int
    days: int
    first: pd.Timestamp
    last: pd.Timestamp
    missing_hours: int
    duplicated_timestamps: int
    incomplete_days: dict[str, int]

    def __str__(self) -> str:
        return (
            f"{self.rows} hourly rows over {self.days} days, "
            f"{self.first:%Y-%m-%d} to {self.last:%Y-%m-%d}\n"
            f"  missing hours in the calendar : {self.missing_hours}\n"
            f"  duplicated timestamps         : {self.duplicated_timestamps}\n"
            f"  days without 24 hours         : {self.incomplete_days or 'none'}"
        )


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV extract and build a proper hourly index.

    The hour column is 1..24 (exchange convention), so hour 1 is the interval
    starting at 00:00. Timestamps are left naive local time: the dataset does
    not span a DST change (1 Jan - 10 Apr 2024), and inventing a timezone we
    cannot verify would be worse than not having one. If the series is ever
    extended, this is the first function that has to change.
    """
    df = pd.read_csv(path)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")

    df = df[RAW_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["hour"].astype(int)
    if not df["hour"].between(1, 25).all():
        raise ValueError("hour column outside the 1..25 range")

    df["ts"] = df["date"] + pd.to_timedelta(df["hour"] - 1, unit="h")
    df = df.dropna(subset=["da_price", "imbalance_price", "imbalance_mwh"])
    df = df.sort_values("ts").set_index("ts")
    return df[["da_price", "imbalance_price", "imbalance_mwh"]]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the two quantities the whole project turns on.

    spread    imbalance price minus day-ahead price. A long position taken in
              the day-ahead auction and closed in the imbalance market earns
              exactly this; a short position earns its negative. There is no
              third possibility, so the strategy problem is one-dimensional.

    system_long  1 when the system has surplus energy, 0 when it is short.
    """
    out = df.copy()
    out["spread"] = out["imbalance_price"] - out["da_price"]
    out["system_long"] = (out["imbalance_mwh"] > 0).astype(int)
    return out


def quality_report(df: pd.DataFrame) -> DataQuality:
    idx = df.index
    full = pd.date_range(idx.min(), idx.max(), freq="h")
    sizes = df.groupby(idx.normalize()).size()
    return DataQuality(
        rows=len(df),
        days=int(idx.normalize().nunique()),
        first=idx.min(),
        last=idx.max(),
        missing_hours=int(len(full) - len(idx.unique())),
        duplicated_timestamps=int(idx.duplicated().sum()),
        incomplete_days={str(d.date()): int(n) for d, n in sizes[sizes != 24].items()},
    )


def load(path: str | Path) -> tuple[pd.DataFrame, DataQuality]:
    """Convenience entry point: raw file in, model-ready frame and report out."""
    raw = load_raw(path)
    return add_derived(raw), quality_report(raw)


def sign_agreement(df: pd.DataFrame) -> pd.Series:
    """How tightly the system sign pins the sign of the spread.

    This is the single most important diagnostic in the project. If the
    agreement is near-total, forecasting the spread direction and forecasting
    the system direction are the same problem, and the second stage of the
    model is about size rather than about direction.
    """
    short = df.loc[df.system_long == 0, "spread"]
    long_ = df.loc[df.system_long == 1, "spread"]
    return pd.Series(
        {
            "P(spread>0 | system short)": float((short > 0).mean()),
            "P(spread>0 | system long)": float((long_ > 0).mean()),
            "P(spread>0) unconditional": float((df.spread > 0).mean()),
            "median spread | system short": float(short.median()),
            "median spread | system long": float(long_.median()),
            "n short": len(short),
            "n long": len(long_),
        }
    )


def tail_report(df: pd.DataFrame, column: str = "spread") -> pd.DataFrame:
    """The spread is not remotely Gaussian, and pretending otherwise is how a
    backtest becomes a fantasy. This table is meant to be read before any
    model is fitted."""
    s = df[column]
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    out = pd.DataFrame({"quantile": qs, "value": [float(s.quantile(q)) for q in qs]})
    out.attrs["mean"] = float(s.mean())
    out.attrs["median"] = float(s.median())
    out.attrs["std"] = float(s.std())
    out.attrs["worst"] = float(s.min())
    out.attrs["best"] = float(s.max())
    out.attrs["share_of_abs_sum_in_worst_1pct"] = float(
        s.nsmallest(max(1, len(s) // 100)).abs().sum() / s.abs().sum()
    )
    return out
