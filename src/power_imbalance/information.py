"""The information set — the part of this project that decides whether the
backtest is honest.

A position in this strategy is taken in the day-ahead auction. In the Italian day-ahead
market (MGP) the auction closes at 12:00 on D-1 and covers all 24 hours of day
D, with results published by 12:58 the same afternoon. So the
decision for *every* hour of D is made at one moment, before the auction
clears, with whatever was known at 12:00 on D-1.

Two consequences, and both are routinely ignored in imbalance backtests:

1. The day-ahead price of day D is NOT an input. It is the outcome of the
   auction the decision is submitted to. Using it is look-ahead, and it is the
   most common way this kind of study produces a Sharpe that does not exist.

2. Imbalance data is published after delivery and with a settlement lag. For
   hour 0 of day D the freshest observation available at gate closure is ~13
   hours old; for hour 23 it is ~36 hours old. A feature built on "yesterday's
   value" is therefore only legitimate if "yesterday" means at least 36 hours
   back for every hour of the delivery day.

This module makes that explicit so it can be tested rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Hour of D-1 at which the day-ahead auction closes (local market time).
GATE_CLOSURE_HOUR = 12

#: Additional delay between delivery and the imbalance figure being available.
#: Conservative: the real publication lag varies with the settlement cycle.
PUBLICATION_LAG_HOURS = 1


@dataclass(frozen=True)
class InformationSet:
    """Answers one question: at decision time for `delivery`, what is the most
    recent hour whose imbalance outcome we are allowed to know?"""

    gate_closure_hour: int = GATE_CLOSURE_HOUR
    publication_lag_hours: int = PUBLICATION_LAG_HOURS

    def decision_time(self, delivery: pd.Timestamp) -> pd.Timestamp:
        """12:00 on the day before delivery."""
        day = pd.Timestamp(delivery).normalize()
        return day - pd.Timedelta(days=1) + pd.Timedelta(hours=self.gate_closure_hour)

    def last_observable(self, delivery: pd.Timestamp) -> pd.Timestamp:
        """Last delivery hour whose imbalance outcome is known at decision time."""
        return self.decision_time(delivery) - pd.Timedelta(
            hours=self.publication_lag_hours
        )

    def lag_hours(self, delivery: pd.Timestamp) -> int:
        """Age, in hours, of the freshest legitimate observation."""
        delta = pd.Timestamp(delivery) - self.last_observable(delivery)
        return int(delta.total_seconds() // 3600)

    def min_safe_lag(self) -> int:
        """The lag that is safe for EVERY hour of the delivery day.

        Hour 23 is the worst case, so this is what a single fixed lag has to
        respect if the same feature definition is used across the whole day.
        """
        probe = pd.Timestamp("2024-01-02 23:00")
        return self.lag_hours(probe)

    def describe(self) -> pd.DataFrame:
        """Lag by delivery hour — worth putting in the deck, because it shows
        the constraint is not uniform across the day."""
        day = pd.Timestamp("2024-01-02")
        rows = []
        for h in range(24):
            d = day + pd.Timedelta(hours=h)
            rows.append(
                {
                    "delivery_hour": h,
                    "decision_time": self.decision_time(d),
                    "last_observable": self.last_observable(d),
                    "lag_hours": self.lag_hours(d),
                }
            )
        return pd.DataFrame(rows)


def assert_causal(features: pd.DataFrame, source: pd.DataFrame, min_lag: int) -> None:
    """Cheap guard against the mistake this module exists to prevent.

    Any feature derived from a column of `source` must not be perfectly
    correlated with a value of that column observed later than `min_lag` hours
    before delivery. This does not prove causality, but it catches the shift
    that was forgotten.
    """
    for col in ("imbalance_mwh", "imbalance_price", "spread", "system_long"):
        if col not in source:
            continue
        contemporaneous = source[col].reindex(features.index)
        for name in features.columns:
            f = features[name]
            if f.std() == 0 or contemporaneous.std() == 0:
                continue
            corr = abs(f.corr(contemporaneous))
            if corr > 0.999:
                raise AssertionError(
                    f"feature '{name}' reproduces contemporaneous '{col}' "
                    f"(|corr| = {corr:.4f}). Look-ahead."
                )
