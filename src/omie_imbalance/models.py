"""The two-stage model.

Stage 1 — classification. Predict P(system long) for each delivery hour. The
    sign of the system pins the sign of the spread almost deterministically in
    this dataset, so this is not a side quest: it is the whole directional
    decision, expressed as a probability rather than a label so the strategy
    can act on confidence instead of on a coin flip.

Stage 2 — regression. Predict the spread, with the stage-1 probability as an
    input. Direction is already handled; what this adds is size. A position
    worth taking when the expected move is 40 EUR/MWh is not worth taking when
    it is 4, and the strategy needs that distinction to know when to stand down.

Both stages are deliberately small models. 101 days of hourly data with four
raw columns does not support anything larger, and a gradient-boosted forest on
this sample would fit the January weather rather than the market.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    "logistic": lambda: Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=0.5)),
        ]
    ),
    "tree": lambda: DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=60, random_state=0
    ),
    "forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=30, random_state=0, n_jobs=-1
    ),
}


@dataclass
class TwoStageModel:
    """Stage 1 gives a probability; stage 2 turns it into an expected spread.

    The probability is passed to stage 2 in two forms: the raw value and its
    distance from 0.5. The second is what actually carries information about
    magnitude — a confident forecast in either direction tends to come with a
    larger move than an uncertain one.
    """

    classifier: str = "logistic"
    ridge_alpha: float = 10.0
    winsor: float | None = 500.0
    _clf: object = field(default=None, repr=False)
    _reg: object = field(default=None, repr=False)
    _cols: list[str] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    def _stage2_matrix(self, X: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
        Z = X.copy()
        Z["p_long"] = p
        Z["confidence"] = np.abs(p - 0.5) * 2.0
        return Z

    def _target(self, y_spread: pd.Series) -> pd.Series:
        """Winsorise the regression target.

        Not cosmetic. The raw spread reaches -4,713 and +8,171 EUR/MWh; a least
        squares fit on that is a fit to five observations. The cap applies to
        TRAINING only — the backtest always settles at the true, uncapped
        spread, so the tail is never hidden from the P&L, only from the fit.
        """
        if self.winsor is None:
            return y_spread
        return y_spread.clip(-self.winsor, self.winsor)

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y_long: pd.Series, y_spread: pd.Series):
        self._cols = list(X.columns)
        self._clf = CLASSIFIERS[self.classifier]()
        self._clf.fit(X, y_long)
        p = self._clf.predict_proba(X)[:, 1]

        self._reg = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=self.ridge_alpha))])
        self._reg.fit(self._stage2_matrix(X, p), self._target(y_spread))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._clf is None:
            raise RuntimeError("model is not fitted")
        X = X[self._cols]
        p = self._clf.predict_proba(X)[:, 1]
        spread_hat = self._reg.predict(self._stage2_matrix(X, p))
        return pd.DataFrame(
            {"p_long": p, "confidence": np.abs(p - 0.5) * 2.0, "spread_hat": spread_hat},
            index=X.index,
        )


@dataclass
class SeasonalBaseline:
    """The model to beat, and — spoiler — the one that wins.

    For each clock hour, the median spread observed so far. The median rather
    than the mean, because with this tail the mean of a training window is
    dominated by whichever extreme hour happened to fall inside it.

    It has no parameters to overfit and it encodes one real fact: the Spanish
    system is long in the middle of the day and short in the evening, because
    that is what solar does to it.
    """

    min_abs_median: float = 0.0

    def fit(self, y_spread: pd.Series):
        self.by_hour_ = y_spread.groupby(y_spread.index.hour).median()
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        med = self.by_hour_.reindex(index.hour).to_numpy()
        med = np.nan_to_num(med, nan=0.0)
        signal = np.where(np.abs(med) >= self.min_abs_median, med, 0.0)
        # Expressed as a probability so it plugs into the same strategy layer:
        # a negative median spread implies the system is expected to be long.
        p = np.where(signal > 0, 0.0, np.where(signal < 0, 1.0, 0.5))
        return pd.DataFrame(
            {"p_long": p, "confidence": np.abs(np.sign(signal)), "spread_hat": signal},
            index=index,
        )


def oracle(df: pd.DataFrame) -> pd.DataFrame:
    """Perfect knowledge of the system sign. Not a strategy — a ceiling.

    Worth computing because it prices information: the gap between the best
    implementable strategy and this number is what a genuine imbalance forecast
    would be worth, and therefore how much it is rational to spend on one.
    """
    p = 1.0 - df["system_long"].to_numpy()
    return pd.DataFrame(
        {
            "p_long": df["system_long"].to_numpy().astype(float),
            "confidence": np.ones(len(df)),
            "spread_hat": df["spread"].to_numpy(),
        },
        index=df.index,
    )
