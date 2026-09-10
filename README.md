# Day-ahead vs imbalance: a strategy study on the Spanish power market

A quantitative study of a single trade on the Spanish electricity market: take a
position in the OMIE day-ahead auction, close it in the REE imbalance market,
and keep the difference.

The interesting part is not the strategy. It is that the honest answer is
mostly negative, and that saying so precisely is worth more than a backtest
that isn't.

---

## The problem, stated exactly

You may buy or sell in the day-ahead auction and you must close the position in
the imbalance market. The P&L of one MWh has no third case:

```
buy  in day-ahead, sell in imbalance  ->  + (imbalance_price - day_ahead_price)
sell in day-ahead, buy  in imbalance  ->  - (imbalance_price - day_ahead_price)
```

So the strategy problem is one-dimensional: forecast the sign and size of the
spread `S = P_imbalance - P_DA`, hour by hour.

**Data.** 2,408 hourly observations, 1 January to 10 April 2024: day-ahead
price, imbalance price, and the system's net imbalance volume. Nothing else —
no weather, no demand, no cross-border. That constraint is part of the problem
and it shapes the conclusion.

---

## Three findings

### 1. The spread sign is the system sign

| | P(spread > 0) |
|---|---|
| System **short** (deficit) | **99.8%** |
| System **long** (surplus) | **0.0%** |

![Sign relationship](reports/figures/f2_sign_relationship.png)

Under single imbalance pricing this is close to mechanical: when the system is
short the imbalance price clears above the day-ahead price, and when it is long
it clears below. The consequence for modelling is decisive — **forecasting the
spread direction and forecasting the system direction are the same problem**.
A perfect forecast of the system sign and a perfect forecast of the spread sign
produce, on this data, exactly the same P&L.

That is why the model is built in two stages: a classifier for the direction,
and a regression for the size, with the classifier's probability as an input to
the regression. Direction is settled first because direction is everything.

### 2. That sign is not forecastable from this data at the moment of decision

The day-ahead auction closes at 12:00 on D-1 for all 24 hours of day D. So the
decision for hour 23 is made with information that is **36 hours old**, and the
day-ahead price itself is not available — it is the outcome of the auction the
order is submitted to. `information.py` encodes this, and the test suite fails
if a feature ever breaks it.

Under that constraint:

| Horizon | Accuracy on the system sign |
|---|---|
| Persistence at 1 hour | 79.4% (unusable — not known in time) |
| Persistence at 24 hours | 60.4% |
| Logistic regression, legal features, out of sample | **51.1%**, AUC **0.550** |
| Base rate (always predict the majority class) | 55.6% |

**The model loses to always guessing the majority class.** Not a tuning
failure: there is very little to find. The system's imbalance at a 13-to-36
hour horizon is driven by forecast error in wind, solar and demand — none of
which is in this dataset, and all of which is the reason a real desk buys them.

### 3. What survives is the daily shape

![Hourly shape](reports/figures/f3_hourly_shape.png)

P(system long) runs from 0.30 at hour 20 to 0.64 at hour 14, and the median
spread flips sign with it: −21 EUR/MWh at 16:00, +48 at 21:00. This is the
solar profile imprinted on the system imbalance, and unlike the stochastic part
it **is** known a day ahead, because the clock is known a day ahead.

---

## Results

Walk-forward, expanding window, refitted daily, 1,448 out-of-sample hours.
All strategies settle at the true uncapped spread.

| Strategy | Traded | Mean | 95% CI on the mean | Sharpe (ann.) | CVaR 5% | Worst hour |
|---|---|---|---|---|---|---|
| *Oracle — system sign (ceiling)* | *99%* | *56.00* | *[49.5, 64.8]* | *38.96* | *+7.94* | *0* |
| **Seasonal median, \|edge\| ≥ 20** | **70%** | **4.66** | **[−1.59, 10.60]** | **5.63** | **−131.6** | **−1,855** |
| Seasonal median | 97% | 5.93 | [−2.73, 16.99] | 3.82 | −146.5 | −1,855 |
| Always buy day-ahead | 100% | 3.33 | [−6.42, 12.14] | 2.14 | −191.0 | −4,551 |
| Two-stage model, calendar only | 100% | 1.86 | [−4.94, 7.80] | 1.20 | −168.1 | −4,551 |
| Two-stage model, full features | 100% | −0.57 | [−9.17, 6.54] | −0.37 | −174.1 | −4,551 |
| Always sell day-ahead | 100% | −3.33 | [−12.14, 6.42] | −2.14 | −116.0 | −1,026 |

![Equity curves](reports/figures/f4_equity.png)

**Read the confidence intervals before the Sharpe ratios.** Every implementable
strategy has a 95% interval on its mean that contains zero. With 101 days and a
tail this heavy, nothing here is statistically distinguishable from luck. The
ranking is informative; the levels are not.

Three things the table does say, and they are robust to that caveat:

- **The machine learning model is beaten by an hourly median.** A one-line
  seasonal baseline with no parameters outperforms the two-stage model, and
  removing the lagged features *improves* the model — they are noise. When the
  simplest thing wins, the finding is that there is no signal, not that the
  model needs another layer.
- **Standing down is worth more than any feature.** Filtering to hours where the
  expected edge exceeds 20 EUR/MWh cuts trading to 70% of hours, lowers the mean
  slightly, and improves Sharpe from 3.82 to 5.63 while cutting CVaR by 10%.
- **The gap to the oracle is the price of information.** 4.66 against 56.00. A
  genuine system-imbalance forecast is worth roughly twelve times the entire
  calendar edge, which is the argument for spending money on wind, solar and
  demand data rather than on a better classifier.

---

## Risk: the second question, and the one that decides the first

![Spread distribution](reports/figures/f1_spread_distribution.png)

The spread has a **median of +29.75 and a mean of +0.33**. Those two numbers
describe different worlds, and the difference is a tail: the worst hour in the
sample is −4,713 EUR/MWh, and the worst 1% of hours account for 12% of the
total absolute P&L of the naive strategy.

This is why "maximise profit" is a trap. A metric based on total profit, or on
hit rate, ranks *always buy the day-ahead* as a good strategy: it wins 52% of
hours and makes money overall. It also has a single hour that costs more than a
month of gains.

![Risk and return](reports/figures/f5_risk_return.png)

So the metric used here is the **Sharpe ratio of hourly P&L, never quoted
without CVaR beside it**, plus:

- **CVaR 5%** — the average loss in the worst 5% of hours. VaR says where the
  cliff is; with this distribution what matters is how far down it goes.
- **Maximum drawdown** on cumulative P&L.
- **Tail concentration** — the share of absolute P&L produced by the worst 1% of
  hours. A strategy whose result is decided by five hours out of two thousand
  should be labelled as such.
- **Block-bootstrap confidence interval on the mean**, resampling whole days
  rather than hours, because imbalance events cluster within a day.

For a desk actually running this, the risk controls that follow are position
limits per hour, a hard stand-down when the expected edge is small, and sizing
scaled to recent spread volatility rather than fixed.

---

## What this study does not claim

- Not a live strategy. 101 days, one winter-to-spring window, one price regime.
- No transaction costs beyond an optional flat `cost_per_mwh`; no market impact,
  no collateral cost, no assumption about being a price taker at scale.
- Single imbalance pricing is assumed throughout, as the data has one imbalance
  price. Under a dual-pricing regime the spread would be direction-dependent and
  the P&L identity above would need a second branch.
- The 2024 window sits in a period of high solar penetration and negative
  midday prices. The daily shape that carries the entire result is a feature of
  *that* regime and should be re-estimated, not assumed.

---

## Repository layout

```
src/omie_imbalance/
    data.py          loading, validation, derived quantities, tail diagnostics
    information.py   gate closure, information lag, the causality guard
    features.py      calendar and lagged features, built to respect that lag
    models.py        two-stage model, seasonal baseline, oracle ceiling
    strategy.py      forecasts -> positions -> P&L
    backtest.py      walk-forward engine, one refit per day
    risk.py          Sharpe, Sortino, VaR, CVaR, drawdown, tail concentration
    plots.py         the five figures in this README
scripts/run_backtest.py    end-to-end run, writes reports/
notebooks/01_analysis.ipynb   the same story, step by step
tests/                     19 tests, five of which exist to catch look-ahead
```

## Running it

```bash
pip install -r requirements.txt
python scripts/run_backtest.py     # writes reports/ and reports/figures/
pytest -q                          # 19 tests
```

## A note on the tests

Five of them do nothing except try to break the backtest:

- `test_lag_grows_across_the_delivery_day` — every hour of D is decided at the
  same moment, so hour 23 faces older information than hour 0.
- `test_features_contain_no_contemporaneous_outcome` — no feature reproduces an
  outcome from the delivery hour.
- `test_assert_causal_catches_a_leak` — a negative control: the leak is planted
  deliberately and the guard must fire.
- `test_pnl_identity_long_earns_the_spread` — the one identity the whole project
  rests on.
- `test_metric_rejects_the_naive_always_buy` — pins the central claim: profitable
  in total, brutal in the tail.

In a study like this, look-ahead is the failure mode that produces the most
impressive-looking wrong answer. It seemed worth testing for rather than
promising.

---

*Data: OMIE day-ahead and REE imbalance settlement, January–April 2024.*
