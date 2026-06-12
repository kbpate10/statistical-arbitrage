"""Stock universe: I download, clean, and split the price data here.

I compute the train/validation/test boundaries exactly once from the
downloaded date range and persist them to data/splits.json. Every
downstream module slices prices through `get_split()`, which reads that
one file. I designed it this way because the most common source of silent
data leakage in projects like this is two modules computing "the training
period" slightly differently; with a single persisted boundary file there
is exactly one definition of each split.

I use adjusted closes (yfinance auto_adjust=True) because raw closes jump
at dividends and splits, and those jumps would masquerade as spread
dislocations downstream.
"""

from __future__ import annotations

import json

import pandas as pd
import yfinance as yf

from . import config


def download_universe(force: bool = False) -> pd.DataFrame:
    """Download 5y of daily adjusted closes for the universe + VIX.

    Caches to data/prices.parquet; subsequent calls read the cache so the
    whole pipeline is reproducible and offline-friendly.
    """
    if config.PRICES_PARQUET.exists() and not force:
        return pd.read_parquet(config.PRICES_PARQUET)

    tickers = config.ALL_TICKERS + [config.VIX_TICKER]
    raw = yf.download(
        tickers,
        period=f"{config.LOOKBACK_YEARS}y",
        interval="1d",
        auto_adjust=True,   # adjusted close lands in the "Close" field
        progress=False,
    )
    prices = raw["Close"].copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()

    prices = clean_prices(prices)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(config.PRICES_PARQUET)
    _compute_and_save_splits(prices)
    return prices


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill gaps of at most FFILL_LIMIT days; drop sparse tickers.

    VIX is exempt from the drop rule (it is a feature, not a tradeable),
    but is forward-filled the same way.
    """
    prices = prices.ffill(limit=config.FFILL_LIMIT)

    missing_frac = prices.isna().mean()
    keep = missing_frac[missing_frac <= config.MAX_MISSING_FRAC].index.tolist()
    if config.VIX_TICKER not in keep and config.VIX_TICKER in prices.columns:
        keep.append(config.VIX_TICKER)
    dropped = sorted(set(prices.columns) - set(keep))
    if dropped:
        print(f"[universe] dropping sparse tickers: {dropped}")
    return prices[sorted(keep)]


def _compute_and_save_splits(prices: pd.DataFrame) -> dict:
    """Derive train/val/test date boundaries from the actual index."""
    idx = prices.index
    n = len(idx)
    train_end = idx[int(n * config.TRAIN_FRAC) - 1]
    val_end = idx[int(n * (config.TRAIN_FRAC + config.VAL_FRAC)) - 1]
    splits = {
        "train": [str(idx[0].date()), str(train_end.date())],
        "validation": [str((train_end + pd.Timedelta(days=1)).date()),
                       str(val_end.date())],
        "test": [str((val_end + pd.Timedelta(days=1)).date()),
                 str(idx[-1].date())],
    }
    config.SPLITS_JSON.write_text(json.dumps(splits, indent=2))
    return splits


def load_splits() -> dict:
    return json.loads(config.SPLITS_JSON.read_text())


def get_split(prices: pd.DataFrame, name: str) -> pd.DataFrame:
    """Slice prices to a named split ('train' | 'validation' | 'test')."""
    splits = load_splits()
    start, end = splits[name]
    return prices.loc[start:end]


def get_history_through(prices: pd.DataFrame, name: str) -> pd.DataFrame:
    """All data from the beginning of train through the END of `name`.

    I use this for walk-forward evaluation. Running the Kalman filter on
    the test year still requires the filter to have processed the
    preceding history (using only past data at every step), so I hand the
    filter the contiguous history ending at the split boundary and never
    anything after it.
    """
    splits = load_splits()
    _, end = splits[name]
    return prices.loc[:end]


def stock_columns(prices: pd.DataFrame) -> list[str]:
    return [c for c in prices.columns if c != config.VIX_TICKER]


# ---------------------------------------------------------------------------
# v2: 10-year wide universe for the walk-forward engine. Separate files so
# the v1 pipeline stays exactly reproducible.
# ---------------------------------------------------------------------------
def download_universe_v2(force: bool = False) -> pd.DataFrame:
    """Download 10y of daily adjusted closes for the v2 universe + VIX."""
    if config.PRICES_V2_PARQUET.exists() and not force:
        return pd.read_parquet(config.PRICES_V2_PARQUET)

    tickers = config.ALL_TICKERS_V2 + [config.VIX_TICKER]
    raw = yf.download(
        tickers,
        period=f"{config.LOOKBACK_YEARS_V2}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    prices = raw["Close"].copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = clean_prices(prices)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(config.PRICES_V2_PARQUET)
    _compute_and_save_splits_v2(prices)
    return prices


def _compute_and_save_splits_v2(prices: pd.DataFrame) -> dict:
    """v2 splits: everything except the last TEST_RESERVE_DAYS is the
    development span (walk-forward tuning happens there); the final year
    is the reserved test span, evaluated exactly once."""
    idx = prices.index
    dev_end = idx[-config.TEST_RESERVE_DAYS - 1]
    splits = {
        "development": [str(idx[0].date()), str(dev_end.date())],
        "test": [str(idx[-config.TEST_RESERVE_DAYS].date()),
                 str(idx[-1].date())],
    }
    config.SPLITS_V2_JSON.write_text(json.dumps(splits, indent=2))
    return splits


def load_splits_v2() -> dict:
    return json.loads(config.SPLITS_V2_JSON.read_text())


if __name__ == "__main__":
    px = download_universe()
    splits = load_splits()
    print(f"prices: {px.shape[0]} days x {px.shape[1]} tickers "
          f"({px.index[0].date()} -> {px.index[-1].date()})")
    for k, (s, e) in splits.items():
        sub = get_split(px, k)
        print(f"  {k:<11} {s} -> {e}  ({len(sub)} days)")
    n_missing = px[stock_columns(px)].isna().sum().sum()
    print(f"remaining missing cells (stocks): {n_missing}")
