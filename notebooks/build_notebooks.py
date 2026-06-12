"""I build the three walkthrough notebooks programmatically with nbformat.

Run from the stat_arb root:  python3 notebooks/build_notebooks.py
Then execute them:           jupyter nbconvert --execute --to notebook \
                                 --inplace notebooks/0*.ipynb

I generate the notebooks from code rather than editing .ipynb JSON by
hand so that the narrative and the code cells stay under version control
in a readable form.
"""

import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).parent

SETUP = """\
# Setup: make src importable and load shared inputs
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks"
                       else Path.cwd()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src import (universe, cointegration, kalman_hedge, regime_detector,
                 signal_generator, backtest, report, config)

prices = universe.download_universe()
splits = universe.load_splits()
stocks = prices[universe.stock_columns(prices)]
vix = prices[config.VIX_TICKER]
train = universe.get_split(stocks, "train")
print("data:", prices.shape, "| splits:", splits)"""


def nb01():
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell("""\
# 01: Pair Selection via Cointegration

In this notebook I walk through the statistical pipeline that decides
*which pairs are tradeable*, step by step, using **training data only**
(years 1 to 3). Nothing here touches the validation or test years, and I
explain why that restriction matters at the end.

**The idea in plain English.** Two stocks in the same sector are often
driven by the same economics. Each stock's price on its own wanders like
a random walk: knowing where it is today tells me almost nothing about
where it settles long-term. But a particular *combination* of the two
(long one, short a fixed multiple of the other) can be stable. When that
combination strays far from its typical level, it tends to come back.
Such a combination is called a **cointegrated spread**, and "tends to
come back" is the property I trade.

Note that this is a stronger requirement than correlation. Two stocks can
be 95% correlated in daily returns while their *price levels* drift apart
forever. Cointegration is a statement about levels, not returns, which is
why I test it with unit-root machinery instead of a correlation matrix."""))
    c.append(nbf.v4.new_code_cell(SETUP))
    c.append(nbf.v4.new_markdown_cell("""\
## Step 1: Are the individual series non-stationary?

The **Augmented Dickey-Fuller (ADF) test** asks whether a series has a
*unit root*, meaning it wanders without an anchor. The null hypothesis is
"yes, it wanders". For cointegration to make sense, each price series
must individually wander. If a stock's price were already mean-reverting
on its own, I could trade it by itself, and the Engle-Granger machinery,
which looks for a stationary combination of two non-stationary series,
would not apply.

So at this step I want to **fail** to reject the null: p-value > 0.05
for both legs. This is one of the few places in statistics where a high
p-value is the desired outcome, and it is worth pausing on why: I am not
proving the series has a unit root (failing to reject is not proof), I am
screening out series that are clearly anchored already."""))
    c.append(nbf.v4.new_code_cell("""\
y_tkr, x_tkr = "MS", "BLK"   # my top-ranked pair, as the worked example
p_y = cointegration.adf_pvalue(train[y_tkr])
p_x = cointegration.adf_pvalue(train[x_tkr])
print(f"ADF p-value {y_tkr}: {p_y:.3f}  (> 0.05 means non-stationary, which is what I want)")
print(f"ADF p-value {x_tkr}: {p_x:.3f}  (> 0.05 means non-stationary, which is what I want)")"""))
    c.append(nbf.v4.new_markdown_cell("""\
## Step 2: Engle-Granger. Is some combination stationary?

I regress `y = alpha + beta*x` by OLS, then test whether the residual
`y - alpha - beta*x` is stationary. If the residual passes a
stationarity test, the pair is **cointegrated** and beta is the static
hedge ratio.

One statistical subtlety I handle deliberately: running a plain ADF test
on the *fitted* residuals uses the wrong critical values. The reason is
that OLS has already chosen alpha and beta to minimize the residual
variance, which makes the residuals look more stationary than a series I
observed directly would. The correct test is Engle-Granger with
**MacKinnon critical values**, available as
`statsmodels.tsa.stattools.coint`. I gate pairs on the `coint` p-value
and print the naive residual ADF p-value next to it; the naive value is
always smaller, in other words always flattering, and comparing the two
shows the size of the bias."""))
    c.append(nbf.v4.new_code_cell("""\
eg = cointegration.engle_granger(train[y_tkr], train[x_tkr])
print(f"OLS fit: {y_tkr} = {eg['alpha']:.2f} + {eg['beta']:.4f} * {x_tkr}")
print(f"Engle-Granger (MacKinnon) p-value: {eg['coint_pvalue']:.4f}  <- my gate")
print(f"naive residual ADF p-value:        {eg['resid_adf_pvalue']:.4f}  (flattering, shown for comparison)")

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(train[y_tkr], label=y_tkr); axes[0].plot(train[x_tkr], label=x_tkr)
axes[0].legend(); axes[0].set_title("Two wandering price series (train)")
axes[1].plot(eg["resid"], color="seagreen")
axes[1].axhline(0, color="k", lw=0.6)
axes[1].set_title("The combination is anchored: this is the cointegration residual")
plt.tight_layout(); plt.show()"""))
    c.append(nbf.v4.new_markdown_cell("""\
## Step 3: Half-life. How fast does it come back?

A spread can be statistically mean-reverting but still useless to trade,
either *too slow* (I would sit in a position for a year while the
relationship has time to break) or *too fast* (the reversion finishes
before my next-day fill executes). I quantify the speed by fitting an
AR(1) on residual changes:

`delta_resid_t = lambda * resid_{t-1} + noise`

If lambda is negative, the residual gets pulled back toward zero, and the
half-life of a dislocation is `-ln(2)/ln(1+lambda)`. This comes from
treating the residual as a discretized Ornstein-Uhlenbeck process: after
one half-life, the expected remaining dislocation is half the original.

I keep pairs whose half-life is between **5 and 60 trading days**: long
enough that the move survives a one-day execution delay, short enough
that capital recycles and the train-period relationship plausibly still
holds while I am in the trade."""))
    c.append(nbf.v4.new_code_cell("""\
hl = cointegration.half_life(eg["resid"])
print(f"half-life of mean reversion: {hl:.1f} trading days")"""))
    c.append(nbf.v4.new_markdown_cell("""\
## The full scan

I run every same-sector pair (both orderings, keeping the better fit)
through all three steps on training data. Survivors are ranked by
Engle-Granger p-value, then half-life. I freeze the result to
`results/selected_pairs.json` and **never recompute it** on later data.

Why never? If I re-selected pairs on the test year, I would keep exactly
the pairs that happened to remain cointegrated and drop the ones that
broke. That decision uses information a real trader would not have had,
and it is look-ahead bias at the portfolio level even though no single
price was peeked at."""))
    c.append(nbf.v4.new_code_cell("""\
pairs = cointegration.select_pairs(train)
pairs[["sector", "y", "x", "beta", "coint_pvalue", "resid_adf_pvalue",
       "half_life"]].round(4)"""))
    c.append(nbf.v4.new_markdown_cell("""\
### A non-cointegrated pair, for contrast

Same sector, both legs non-stationary, but no stable combination exists:
the "residual" itself wanders. This is exactly what Step 2 filters out.
If I traded this spread with z-scores anyway, every "dislocation" would
just be the residual drifting to a new level, and my mean-reversion bets
would have no statistical basis."""))
    c.append(nbf.v4.new_code_cell("""\
bad = cointegration.engle_granger(train["AAPL"], train["NVDA"])
print(f"AAPL~NVDA Engle-Granger p-value: {bad['coint_pvalue']:.3f} (not cointegrated)")
bad["resid"].plot(figsize=(11, 3.5), color="indianred",
                  title="AAPL vs NVDA 'residual': no anchor, nothing to trade")
plt.axhline(0, color="k", lw=0.6); plt.tight_layout(); plt.show()"""))
    return nb


def nb02():
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell("""\
# 02: Strategy Walkthrough on a Single Pair

Here I take **PEP/TGT**, my top-ranked Consumer pair (the classic KO/PEP
did not pass the cointegration gate on this training window), and follow
it through every layer of the system:

1. the static OLS hedge ratio vs the **Kalman filter's dynamic hedge ratio**
2. the spread and its **z-score**, built from trailing windows only
3. the **HMM regime gate**
4. the trades the backtester actually took in the test year"""))
    c.append(nbf.v4.new_code_cell(SETUP))
    c.append(nbf.v4.new_code_cell("""\
pairs = cointegration.select_pairs(train)
pair = pairs[(pairs.sector == "Consumer")].iloc[0]
params = json.loads(config.TUNED_PARAMS_JSON.read_text())
delta, center, window = (params["delta"], params["zscore_center"],
                         params["zscore_window"])
y_tkr, x_tkr = pair["y"], pair["x"]
print(f"pair: {y_tkr}/{x_tkr}, static beta={pair['beta']:.3f}, "
      f"half-life={pair['half_life']:.0f}d")
print(f"frozen params (tuned on validation): delta={delta:.0e}, "
      f"center='{center}', window={window}")"""))
    c.append(nbf.v4.new_markdown_cell("""\
## Dynamic vs static hedge ratio

The OLS beta is a single number estimated once over three years, so it
assumes the hedge relationship never changes. The Kalman filter instead
treats [beta, alpha] as a slowly drifting state and updates it each day
from that day's prices, using **filtered** estimates only. I never use
the Kalman smoother here: it conditions every day's estimate on the whole
sample including the future, which makes its smooth, attractive state
paths unusable for trading.

Two noise settings make or break this filter, and I calibrate both
instead of using textbook defaults:

- Observation noise **R**: I set it to the variance of the train OLS
  residual, which is how far this pair typically strays from its
  equilibrium. With the common default R = 1e-3 on dollar prices, the
  filter believes prices fit the model to within a few cents, absorbs
  every dislocation into the state, and leaves a "spread" that is white
  measurement noise. I hit exactly this failure while building the
  project; the tell-tale diagnostic was the spread half-life collapsing
  to roughly zero days while the OLS residual half-life was 25 days.
- Process noise **Q**: I scale the beta component by 1/scale(x)^2.
  Beta multiplies the price of x, so an unscaled random-walk step on beta
  moves the prediction by dollars per day when x trades at $800, and the
  state tracks dislocations through beta no matter how small delta is.
  After scaling, delta means the same thing for every pair: the fraction
  of observation variance the state may absorb per day."""))
    c.append(nbf.v4.new_code_cell("""\
y_full, x_full = stocks[y_tkr], stocks[x_tkr]
kf = kalman_hedge.kalman_filter_manual(y_full, x_full, delta,
                                       obs_var=pair["resid_var"])
# I implemented the filter twice; pykalman gives identical output (asserted):
print("manual vs pykalman max diff:",
      f"{kalman_hedge.self_test(train[y_tkr], train[x_tkr], delta, pair['resid_var']):.2e}")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(kf["beta"].iloc[60:], lw=1.2, label="Kalman beta (filtered)")
ax.axhline(pair["beta"], color="crimson", ls="--",
           label=f"static OLS beta = {pair['beta']:.3f}")
for name, (s, e) in splits.items():
    ax.axvline(pd.Timestamp(e), color="gray", lw=0.6, ls=":")
ax.legend(); ax.set_title(f"{y_tkr}/{x_tkr}: hedge ratio over time "
                          "(dotted lines mark split boundaries)")
plt.tight_layout(); plt.show()"""))
    c.append(nbf.v4.new_markdown_cell("""\
## The spread and its z-score

The tradeable spread is the residual under the *current filtered state*:
`spread_t = y_t - beta_t * x_t - alpha_t`. I can compute this at each
day's close using only data up to that close, so it carries no future
information.

For the z-score I use an **expanding window for the first 60
observations, then a rolling window**. The expanding start matters
because a rolling standard deviation estimated from the first three or
four observations is so noisy that it produces meaningless z-values of 5
or 10 on day three, which would either trigger nonsense trades or force
me to discard the early period entirely."""))
    c.append(nbf.v4.new_code_cell("""\
z = signal_generator.zscore(kf["spread"], center=center, window=window)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(kf["spread"].iloc[60:], lw=0.9, color="seagreen")
axes[0].axhline(0, color="k", lw=0.6)
axes[0].set_title(f"{y_tkr}/{x_tkr} Kalman spread")
axes[1].plot(z.iloc[60:], lw=0.9)
for lvl, c_ in [(2, "orange"), (-2, "orange"), (3.5, "red"), (-3.5, "red")]:
    axes[1].axhline(lvl, color=c_, lw=0.8, ls="--")
axes[1].set_title("z-score with my entry (+-2) and stop (+-3.5) bands")
plt.tight_layout(); plt.show()"""))
    c.append(nbf.v4.new_markdown_cell("""\
## The regime gate

Mean reversion is conditional: during stress periods correlations break
and spreads trend instead of reverting. I fit a 2-state Gaussian HMM on
**train-period features** (spread volatility, short and long return
correlations, VIX, spread momentum, all from trailing windows) and only
allow new entries when the model says the current day is in the
mean-reverting state.

Out of sample I classify each day with the *forward algorithm*: the state
probability at day t conditions only on observations up to t. I wrote
that recursion explicitly instead of calling `model.predict`, because
`predict` returns the Viterbi path, which chooses each day's state using
the entire series including the future. A regime gate built on Viterbi
states leaks information and makes the backtest look better than live
trading could ever be."""))
    c.append(nbf.v4.new_code_cell("""\
feats = regime_detector.build_features(y_full, x_full, kf["spread"], vix)
det = regime_detector.RegimeDetector().fit(feats.loc[:splits["train"][1]])
regimes = det.filtered_regimes(feats)
probs = det.filtered_probabilities(feats)
print(f"state labels: {det.state_labels}")
print(f"fraction of days mean-reverting: {regimes.mean():.1%}")

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(probs, lw=0.8)
ax.set_title("P(mean-reverting | data up to t): forward-filtered, no look-ahead")
ax.axhline(0.5, color="k", lw=0.6, ls="--")
plt.tight_layout(); plt.show()"""))
    c.append(nbf.v4.new_markdown_cell("""\
## Trades on this pair in the test year

The backtester makes decisions at one close and fills at the **next**
close, with 5 bps cost plus 2 bps slippage per side. Entries are
re-confirmed at the fill bar: if the dislocation vanished overnight, the
order is cancelled instead of filled. I added that rule after measuring
that the average |z| decayed from 2.55 at decision time to 0.77 at fill
time, which means that without re-confirmation I was systematically
buying one-day spikes after they had already snapped back.

Below is every trade the backtester took on this pair out of sample."""))
    c.append(nbf.v4.new_code_cell("""\
trades = pd.read_csv(config.RESULTS_DIR / "trades_test.csv",
                     parse_dates=["entry_date", "exit_date"])
mine = trades[trades.pair == f"{y_tkr}/{x_tkr}"]
display(mine.round(2))

start, end = splits["test"]
fig, ax = plt.subplots(figsize=(11, 4))
zt = z.loc[start:end]
ax.plot(zt, lw=0.9)
for _, t in mine.iterrows():
    col = "green" if t.pnl_dollars > 0 else "red"
    ax.axvspan(t.entry_date, t.exit_date, color=col, alpha=0.18)
    ax.scatter([t.entry_date], [t.entry_z], marker="v", color="k", zorder=5)
    ax.scatter([t.exit_date], [t.exit_z], marker="^", color=col, zorder=5)
for lvl in (2, -2): ax.axhline(lvl, color="orange", lw=0.8, ls="--")
ax.axhline(0, color="k", lw=0.6)
ax.set_title(f"{y_tkr}/{x_tkr} test-year z-score with trades "
             "(v = entry, ^ = exit; green span = winner)")
plt.tight_layout(); plt.show()"""))
    return nb


def nb03():
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell("""\
# 03: Performance Analysis

Everything below reads the frozen backtest artifacts. The discipline that
makes these numbers meaningful:

- I selected pairs on **train** only and fit the HMM on train only;
- I tuned the Kalman delta and the z-score settings on **validation**
  only, by running the actual backtest there;
- I evaluated the **test year exactly once**, with everything frozen.

A strategy that only looks good because its parameters were chosen on the
period being reported is not a strategy; it is a description of the
past. I structured this project so that mistake is hard to make."""))
    c.append(nbf.v4.new_code_cell(SETUP))
    c.append(nbf.v4.new_code_cell("""\
SPLITS = ["train", "validation", "test"]
metrics = {s: json.loads((config.RESULTS_DIR / f"metrics_{s}.json").read_text())
           for s in SPLITS}
table = pd.DataFrame(metrics).T
table.index.name = "split"
table.round(3)"""))
    c.append(nbf.v4.new_markdown_cell("""\
## My read of the results

Net of 14 bps round-trip costs, the strategy's edge is **approximately
zero**: modestly negative in-sample (the training window includes the
2022 bear market), positive on validation, and modestly negative out of
sample. I treat the validation number with suspicion on principle,
because I picked the hyperparameters there; selection bias applies to a
tuning set by construction, even when the tuning is done correctly.

I consider this the realistic outcome for daily-frequency cointegration
pairs trading on liquid US large caps. The literature documents this
anomaly being arbitraged down through the 2000s, with surviving edge
moving to higher frequencies and less liquid universes. The value of
this project is that the machinery measures the edge correctly. The
classic failure mode, a beautiful equity curve produced by look-ahead
bias, Viterbi-smoothed regimes, or test-set tuning, is exactly what this
codebase is built to prevent."""))
    c.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
for s in SPLITS:
    eq = pd.read_csv(config.RESULTS_DIR / f"equity_{s}.csv",
                     index_col=0, parse_dates=True)["equity"]
    axes[0].plot(eq, label=s, lw=1.2)
    dd = eq / eq.cummax() - 1
    axes[1].plot(dd, label=s, lw=1)
axes[0].legend(); axes[0].set_title("Equity curves (each split starts at $100k)")
axes[1].legend(); axes[1].set_title("Drawdowns")
axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
plt.tight_layout(); plt.show()"""))
    c.append(nbf.v4.new_markdown_cell("""\
## Degradation analysis: where does the test year lose?

Per-pair attribution tells me whether losses are broad (a cost or regime
problem affecting everything) or concentrated (specific cointegration
relationships breaking). A pair whose train-period relationship
physically broke in the test year, for example one company re-rating
after an earnings shift, is the main non-diversifiable risk of this
strategy, and no amount of signal engineering fixes it after the fact."""))
    c.append(nbf.v4.new_code_cell("""\
rows = []
for s in SPLITS:
    t = pd.read_csv(config.RESULTS_DIR / f"trades_{s}.csv")
    g = t.groupby("pair")["pnl_dollars"].sum()
    rows.append(g.rename(s))
attrib = pd.concat(rows, axis=1).fillna(0).round(0)
attrib["test_rank"] = attrib["test"].rank()
display(attrib.sort_values("test"))

t_test = pd.read_csv(config.RESULTS_DIR / "trades_test.csv")
print("\\ntest exit reasons:")
print(t_test.groupby("exit_reason")["pnl_dollars"]
      .agg(["count", "sum", "mean"]).round(0))"""))
    c.append(nbf.v4.new_markdown_cell("""\
## Sensitivity on the validation year (never on test)

How fragile was my hyperparameter choice? The grid below is the actual
tuning surface I used: validation Sharpe across every (delta, center,
window) combination. A trustworthy configuration sits on a plateau of
similar scores; an isolated peak means the "best" cell is mostly noise,
because with 18 cells and noisy Sharpe estimates, the maximum is biased
upward even if every configuration has zero true edge. This is the same
multiple-comparisons logic that applies to any grid search, and it is
why I report the whole surface instead of just the winner."""))
    c.append(nbf.v4.new_code_cell("""\
params = json.loads(config.TUNED_PARAMS_JSON.read_text())
surf = (pd.Series(params["signal_scores"]).rename("val_sharpe")
        .rename_axis("delta/center/window").to_frame().round(2))
display(surf)
print("chosen:", {k: params[k] for k in ("delta", "zscore_center", "zscore_window")})"""))
    c.append(nbf.v4.new_markdown_cell("""\
## Monthly returns and trade distribution

The full interactive versions of these charts are in
`results/report.html`."""))
    c.append(nbf.v4.new_code_cell("""\
eq_all = pd.concat([pd.read_csv(config.RESULTS_DIR / f"equity_{s}.csv",
                                index_col=0, parse_dates=True)["equity"]
                    for s in SPLITS])
eq_all = eq_all[~eq_all.index.duplicated()]
monthly = eq_all.resample("ME").last().pct_change().dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
monthly.plot(kind="bar", ax=axes[0], color=np.where(monthly > 0, "g", "r"))
axes[0].set_xticklabels([d.strftime("%Y-%m") for d in monthly.index],
                        rotation=90, fontsize=6)
axes[0].set_title("Monthly returns")
t_all = pd.concat([pd.read_csv(config.RESULTS_DIR / f"trades_{s}.csv")
                   for s in SPLITS])
axes[1].hist(t_all["pnl_dollars"], bins=40, color="steelblue")
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_title(f"Trade P&L distribution (n={len(t_all)})")
plt.tight_layout(); plt.show()"""))
    return nb


def nb04():
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell("""\
# 04: Walk-Forward Engine and the Breakdown Kill-Switch

v1 selected pairs once and traded them for years. This notebook covers
the two v2 changes that attack that directly.

**Walk-forward re-formation.** Every six months I re-run the entire
cointegration selection on the trailing two years (the formation window)
across a ~100-ticker universe, then trade the surviving pairs for the
next six months only. No pair is ever traded on evidence more than six
months stale. Each cycle's formation strictly precedes its trading
window, so the procedure is look-ahead-free by construction.

**The kill-switch.** Cointegration established on a formation window is
a hypothesis, not a property. I monitor every live pair two ways: a slow
rolling ADF re-test (blocks new entries when the spread stops looking
stationary) and a fast Bayesian online changepoint detector (BOCPD,
Adams and MacKay 2007) on standardized spread changes, which force-exits
when the posterior says the data-generating process changed recently. I
require the alarm to persist five consecutive days, because I measured
on synthetic data that a one-day outlier sustains the alarm about three
days while a genuine process change sustains it nine to ten."""))
    c.append(nbf.v4.new_code_cell(SETUP))
    c.append(nbf.v4.new_code_cell("""\
from src import walkforward, break_detector, ou_model

prices_v2 = universe.download_universe_v2()
splits_v2 = universe.load_splits_v2()
params_v2 = json.loads(config.TUNED_V2_JSON.read_text())
print("v2 universe:", prices_v2.shape)
print("frozen v2 params:", {k: params_v2[k] for k in ("delta","center","n_configs")})

cycles = walkforward.make_cycles(prices_v2.index, *splits_v2["development"])
print(f"{len(cycles)} development cycles; example:")
for cyc in cycles[:3]:
    print(f"  form {cyc['f_start'].date()} -> {cyc['f_end'].date()}, "
          f"trade {cyc['t_start'].date()} -> {cyc['t_end'].date()}")"""))
    c.append(nbf.v4.new_markdown_cell("""\
## Pair turnover: the reason walk-forward matters

If the same pairs survived every re-formation, walk-forward would be
pointless. The turnover chart shows the opposite: the cointegrated set
changes substantially every six months, which means a static selection
(v1) spends most of its life trading relationships that current data no
longer supports."""))
    c.append(nbf.v4.new_code_cell("""\
pairs_dev = pd.read_csv(config.RESULTS_DIR / "wf_pairs_dev.csv")
per_cycle = pairs_dev.groupby("cycle")["pair"].apply(set)
turnover = [np.nan] + [1 - len(per_cycle.iloc[i] & per_cycle.iloc[i-1])
                       / max(len(per_cycle.iloc[i]), 1)
                       for i in range(1, len(per_cycle))]
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(per_cycle.index, [t*100 if t==t else 0 for t in turnover], color="darkorange")
ax.set_ylabel("% pairs new vs prior cycle"); ax.set_xlabel("cycle")
ax.set_title("Pair turnover at each re-formation")
plt.tight_layout(); plt.show()
print(f"median turnover: {np.nanmedian(turnover):.0%}")"""))
    c.append(nbf.v4.new_markdown_cell("""\
## The OU s-score

For the score I fit the spread's AR(1)/Ornstein-Uhlenbeck dynamics on
the formation window and freeze the equilibrium m and the stationary
sigma_eq for the trading window. Unlike a short rolling std, sigma_eq
cannot be inflated by the dislocation being traded, so "s = 2" means the
same thing on every day of the trading window."""))
    c.append(nbf.v4.new_code_cell("""\
ci = walkforward.prepare_cycle(prices_v2, cycles[6], params_v2["delta"])
name, d = next(iter(ci["per_pair"].items()))
spread = d["kf"]["spread"]
ou = d["ou"]
print(f"pair {name}: OU m={ou['m']:.2f}, sigma_eq={ou['sigma_eq']:.2f}, "
      f"half-life={ou['half_life']:.1f}d")
s = ou_model.s_score(spread, ou)
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(s.iloc[60:], lw=0.9)
ax.axvline(ci["cycle"]["f_end"], color="k", ls=":", label="formation end")
for lvl in (2, -2): ax.axhline(lvl, color="orange", lw=0.8, ls="--")
ax.legend(); ax.set_title(f"{name}: OU s-score (params frozen at formation end)")
plt.tight_layout(); plt.show()"""))
    c.append(nbf.v4.new_markdown_cell("""\
## What the kill-switch buys

Same development run, same configuration, with and without the
kill-switch. The exit reason "break" marks force-exits where BOCPD
detected a process change under the persistence rule. The point of this
comparison is tail protection: the kill-switch should cut the worst
losses even if it costs some winners."""))
    c.append(nbf.v4.new_code_cell("""\
eq = pd.read_csv(config.RESULTS_DIR / "wf_equity_dev.csv", index_col=0,
                 parse_dates=True)["equity"]
eq_noks = pd.read_csv(config.RESULTS_DIR / "wf_equity_dev_noks.csv",
                      index_col=0, parse_dates=True)["equity"]
tr = pd.read_csv(config.RESULTS_DIR / "wf_trades_dev.csv")
tr_noks = pd.read_csv(config.RESULTS_DIR / "wf_trades_dev_noks.csv")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(eq, label="with kill-switch", lw=1.2)
ax.plot(eq_noks, label="without", lw=1.2, alpha=0.75)
ax.legend(); ax.set_title("Development equity: kill-switch ablation")
plt.tight_layout(); plt.show()

print("exit reasons (with kill-switch):")
print(tr["exit_reason"].value_counts().to_string())
print(f"\\nworst 5 trades with KS:    {sorted(tr['pnl_dollars'])[:5]}")
print(f"worst 5 trades without KS: {sorted(tr_noks['pnl_dollars'])[:5]}")"""))
    return nb


def nb05():
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell("""\
# 05: Meta-Labeling and Statistical Significance

Two final layers, one that tries to improve the strategy and one that
keeps me honest about whether anything improved at all.

**Meta-labeling** (Lopez de Prado). The walk-forward rule decides when a
trade triggers; a gradient-boosted classifier then estimates, from
conditions known at decision time, the probability that this particular
trigger wins. I train it on the development trade history with purged
cross-validation: chronological folds, training trades whose lifetimes
overlap a test fold (plus a 60-day embargo) are removed, because
overlapping trades share market state and ordinary K-fold would leak.

**Significance statistics** (Bailey and Lopez de Prado). I tuned over a
grid and kept the best cell, and the maximum of N noisy Sharpes is
biased upward even when nothing works. The deflated Sharpe ratio asks:
what is the probability the result beats what the luckiest of my N
trials would show by chance? I count every configuration I evaluated, so
N is the true trial count, not a guess."""))
    c.append(nbf.v4.new_code_cell(SETUP))
    c.append(nbf.v4.new_code_cell("""\
from src import meta_label, stats_validation

meta = json.loads((config.RESULTS_DIR / "meta_cv.json").read_text())
print(f"dev trades: {meta['n_trades']}, base win rate {meta['base_win_rate']:.1%}")
print(f"purged-CV fold AUCs: {[round(a,3) for a in meta['fold_aucs']]}")
print(f"mean AUC: {meta['mean_auc']:.3f}  (0.5 = no skill)")

imp = pd.Series(meta["importances"]).sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
imp.plot(kind="barh", ax=ax, color="seagreen")
ax.set_title("What makes a triggered entry win or lose (feature importances)")
plt.tight_layout(); plt.show()"""))
    c.append(nbf.v4.new_markdown_cell("""\
## Reading the AUC honestly

An AUC modestly above 0.5 under purged CV means the entry conditions
carry some information about trade outcomes; an AUC at or below 0.5
means the meta-model is noise and its test-year filtering should be
expected to do nothing. Either way the test-year ablation below shows
exactly what the filter did out of sample, because I ran the frozen test
year both with and without it."""))
    c.append(nbf.v4.new_code_cell("""\
rows = {}
for tag, eqf, trf in [
    ("test with meta", "wf_equity_test.csv", "wf_trades_test.csv"),
    ("test no meta", "wf_equity_test_nometa.csv", "wf_trades_test_nometa.csv")]:
    eq = pd.read_csv(config.RESULTS_DIR / eqf, index_col=0, parse_dates=True)["equity"]
    tr = pd.read_csv(config.RESULTS_DIR / trf)
    m = backtest.compute_metrics(eq, tr)
    rows[tag] = {k: m[k] for k in ("total_return","sharpe","max_drawdown",
                                   "n_trades","win_rate","profit_factor")}
    plt.plot(eq, label=tag, lw=1.2)
plt.legend(); plt.title("Reserved test year: meta-filter ablation")
plt.tight_layout(); plt.show()
display(pd.DataFrame(rows).T.round(3))"""))
    c.append(nbf.v4.new_markdown_cell("""\
## Signal or luck: PSR, DSR, bootstrap

The table below is, to me, the most important output of the entire
project. The deflated Sharpe benchmark ("E[max SR | luck]") is what the
best of my tuning trials would show if every configuration were pure
noise. Any Sharpe that does not clearly beat that number, with a
bootstrap interval that excludes zero, is not evidence of edge, no
matter how good the equity curve looks."""))
    c.append(nbf.v4.new_code_cell("""\
sig = json.loads((config.RESULTS_DIR / "significance.json").read_text())
df = pd.DataFrame(sig).T
df["ci_90"] = df["bootstrap_90ci"].apply(
    lambda d: f"({d['lo']:+.2f}, {d['hi']:+.2f})")
display(df[["sharpe_annual","psr","dsr","expected_max_sharpe_annual","ci_90"]]
        .astype({"sharpe_annual":float,"psr":float,"dsr":float,
                 "expected_max_sharpe_annual":float}).round(2))

params_v2 = json.loads(config.TUNED_V2_JSON.read_text())
print("\\nthe full tuning surface (what N trials actually looked like):")
print(pd.Series(params_v2["scores"]).round(2).to_string())"""))
    return nb


if __name__ == "__main__":
    for name, builder in [("01_pair_selection", nb01),
                          ("02_strategy_walkthrough", nb02),
                          ("03_performance_analysis", nb03),
                          ("04_walkforward_and_breaks", nb04),
                          ("05_metalabeling_and_significance", nb05)]:
        nb = builder()
        nb.metadata["kernelspec"] = {"name": "python3",
                                     "display_name": "Python 3",
                                     "language": "python"}
        nbf.write(nb, OUT / f"{name}.ipynb")
        print(f"built {name}.ipynb ({len(nb.cells)} cells)")
