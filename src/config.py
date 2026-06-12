"""Central configuration: universe, date splits, costs, and strategy tunables.

I keep every tunable in this one module so there is exactly one source of
truth. No module ever recomputes a train/val/test boundary or hardcodes a
threshold, which is how I prevent the subtle bug where two components
disagree about what "the training period" means.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

PRICES_PARQUET = DATA_DIR / "prices.parquet"
SPLITS_JSON = DATA_DIR / "splits.json"
SELECTED_PAIRS_JSON = RESULTS_DIR / "selected_pairs.json"
TUNED_PARAMS_JSON = RESULTS_DIR / "tuned_params.json"

# v2 (walk-forward) artifacts. I keep these separate from the v1 files so
# the v1 results stay exactly reproducible: v1 reads its own 5-year
# parquet and split file, v2 reads its own 10-year ones.
PRICES_V2_PARQUET = DATA_DIR / "prices_v2.parquet"
SPLITS_V2_JSON = DATA_DIR / "splits_v2.json"
TUNED_V2_JSON = RESULTS_DIR / "tuned_params_v2.json"
META_MODEL_PKL = RESULTS_DIR / "meta_model.pkl"

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
SECTORS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA",
                   "AMD", "INTC", "QCOM", "TXN", "AVGO"],
    "Financials": ["JPM", "BAC", "GS", "MS", "WFC",
                   "C", "BLK", "AXP", "V", "MA"],
    "Consumer":   ["KO", "PEP", "MCD", "SBUX", "WMT",
                   "TGT", "COST", "HD", "LOW", "NKE"],
    "Energy":     ["XOM", "CVX", "COP", "SLB", "EOG",
                   "MPC", "PSX", "VLO", "OXY", "HAL"],
}
ALL_TICKERS: list[str] = [t for sector in SECTORS.values() for t in sector]
VIX_TICKER = "^VIX"

# ---------------------------------------------------------------------------
# v2 universe: wider, with the sector ETF inside each sector group so that
# stock-vs-ETF pairs form naturally in the same-sector pair scan. I picked
# names that were public well before 2016 so a 10-year download has full
# history; anything that still ends up sparse (delistings, data gaps) is
# dropped by the >5%-missing cleaning rule rather than hand-curated away.
# A static list is survivorship-tilted (these are companies that survived
# to 2026); I accept and document that rather than pretending otherwise.
# ---------------------------------------------------------------------------
SECTORS_V2: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC",
                   "QCOM", "TXN", "AVGO", "ADBE", "CRM", "ORCL", "CSCO",
                   "IBM", "ACN", "INTU", "AMAT", "MU", "ADI", "LRCX",
                   "KLAC", "SNPS", "CDNS", "HPQ", "XLK"],
    "Financials": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP", "V",
                   "MA", "USB", "PNC", "SCHW", "BK", "STT", "COF", "MET",
                   "PRU", "AIG", "TRV", "ALL", "CB", "CME", "ICE", "SPGI",
                   "XLF"],
    "Consumer":   ["KO", "PEP", "MCD", "SBUX", "WMT", "TGT", "COST", "HD",
                   "LOW", "NKE", "PG", "CL", "KMB", "GIS", "K", "HSY",
                   "SYY", "KR", "DG", "DLTR", "YUM", "CMG", "ROST", "TJX",
                   "EL", "XLP", "XLY"],
    "Energy":     ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
                   "OXY", "HAL", "DVN", "APA", "FANG", "KMI", "OKE", "WMB",
                   "EQT", "TRGP", "XLE"],
}
ALL_TICKERS_V2: list[str] = [t for s in SECTORS_V2.values() for t in s]

LOOKBACK_YEARS_V2 = 10

# Walk-forward windows (trading days). I re-form pairs every half year on
# the trailing two years and trade them for the following half year, so
# every pair is at most six months past its formation evidence. The final
# year is reserved and evaluated exactly once with frozen settings.
FORMATION_DAYS = 504
TRADING_DAYS = 126
WF_STEP = 126
TEST_RESERVE_DAYS = 252
WF_MAX_PAIRS = 15           # pairs carried per cycle (top by coint p-value)

# Breakdown kill-switch (v2). I fixed these a priori instead of tuning
# them: every tuned knob inflates the selection-bias correction (deflated
# Sharpe N), and these have natural values. The ADF guard window equals
# the trading window; 0.20 is a deliberately loose "no longer looks
# cointegrated" level (the 0.05 selection gate would whipsaw). BOCPD
# hazard 1/100 encodes "regimes last ~5 months on average"; the trigger
# fires when most posterior mass says the data-generating process changed
# within the last two weeks.
BREAK_ADF_WINDOW = 126
BREAK_ADF_EVERY = 5         # re-test cadence in trading days
BREAK_ADF_P = 0.20          # block new entries above this p-value
BOCPD_HAZARD = 1.0 / 100.0
BOCPD_SHORT_RUN = 10        # "recent changepoint" = run length <= this
BOCPD_TRIGGER = 0.65        # P(run <= SHORT_RUN) above this counts as alarm
# A single outlier day pushes the run-length posterior down for 2-3 days;
# a genuine process change keeps it down for ~SHORT_RUN days because the
# new regime keeps surprising every long-run hypothesis. I verified this
# separation on synthetic data (6-sigma outlier: 3-day alarm run; true
# mean/vol change: 9-10 days), so I require the alarm to persist before
# acting on it.
BOCPD_PERSIST = 5           # consecutive alarm days required to force exit
BOCPD_MAX_RUN = 300         # run-length truncation (memory/speed)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
LOOKBACK_YEARS = 5          # download window
FFILL_LIMIT = 2             # forward-fill at most 2 consecutive missing days
MAX_MISSING_FRAC = 0.05     # drop tickers with >5% missing data

# Split fractions of the full date range: years 1-3 train, 4 val, 5 test.
TRAIN_FRAC = 3 / 5
VAL_FRAC = 1 / 5            # test gets the remainder

# ---------------------------------------------------------------------------
# Cointegration / pair selection (train data only)
# ---------------------------------------------------------------------------
ADF_NONSTATIONARY_P = 0.05  # individual series must FAIL to reject unit root
COINT_P_THRESHOLD = 0.05    # Engle-Granger p-value gate
HALF_LIFE_MIN = 5           # days
HALF_LIFE_MAX = 60          # days
MAX_SELECTED_PAIRS = 10     # candidates carried into the strategy

# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------
KALMAN_DELTA_GRID = [1e-5, 1e-4, 1e-3]  # tuned on validation set
# Observation noise R is calibrated PER PAIR to the variance of the train
# OLS residual (see cointegration.test_pair / kalman_hedge). This fallback
# is only used when no pair-specific value is supplied.
KALMAN_OBS_VAR_FALLBACK = 1.0
KALMAN_WARMUP = 60                       # days before spread is tradeable

# ---------------------------------------------------------------------------
# Regime detection (HMM)
# ---------------------------------------------------------------------------
HMM_N_STATES = 2
HMM_N_ITER = 200
HMM_RANDOM_STATE = 42
REGIME_VOL_WINDOW = 20      # rolling spread volatility
REGIME_CORR_SHORT = 5       # short rolling correlation window
REGIME_CORR_LONG = 20       # long rolling correlation window
REGIME_MOM_WINDOW = 20      # spread momentum window

# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
ZSCORE_WINDOW = 20          # default rolling window (spec baseline)
ZSCORE_WINDOW_GRID = [20, 40, 60]  # tuned on validation: a window short
                            # relative to the half-life (15-30d here) makes
                            # the rolling std spike-sensitive, so z reverts
                            # via the denominator instead of the spread
ZSCORE_EXPANDING_MIN = 20   # min obs for the early expanding window
ZSCORE_EXPANDING_UNTIL = 60 # use expanding window for the first 60 obs
# Z-score centering, chosen on the VALIDATION set:
#   "rolling": classic (spread - rolling_mean) / rolling_std
#   "zero":    spread / rolling_std. The Kalman alpha already absorbs the
#              equilibrium level, so the spread is mean-zero by
#              construction; subtracting a short rolling mean on top
#              double-centers and lets the mean chase dislocations.
ZSCORE_CENTER_MODES = ["rolling", "zero"]
ENTRY_Z = 2.0
EXIT_Z = 0.0
STOP_Z = 3.5

# ---------------------------------------------------------------------------
# Backtest / risk
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02       # 2% of current equity at risk per trade
COST_BPS_PER_SIDE = 5.0     # transaction cost, basis points per side
SLIPPAGE_BPS = 2.0          # additional slippage per side
MAX_CONCURRENT_POSITIONS = 3
MAX_HOLDING_DAYS = 60       # timeout exit
TRADING_DAYS_PER_YEAR = 252
