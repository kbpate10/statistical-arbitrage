"""Meta-labeling: a second model that judges the first one's trades.

The idea (Lopez de Prado, Advances in Financial Machine Learning). The
primary strategy decides WHEN a trade is triggered; a secondary classifier
decides whether THIS PARTICULAR trigger is worth taking, using conditions
known at decision time. I am not asking a machine to find alpha in prices,
which invites overfitting; I am asking it the much easier question "given
that my rule fired, what is the probability this one wins?", for which I
have labeled examples: the walk-forward trade history.

Why purged cross-validation. Trades overlap in time and share market
conditions, so ordinary K-fold leaks: a training trade that was open
during a test trade's window carries information about the same market
state. I use chronological folds, purge any training trade whose exit
falls after the fold start minus an embargo equal to the maximum holding
period, and drop training trades entered before the fold ends plus the
same embargo on the far side.

Deliberately NOT tuned: the probability threshold is fixed at 0.5 and the
size multiplier at max(0, 2*(p-0.5)) a priori. Every tuned choice would
have to be counted in the deflated-Sharpe trial correction, and a
threshold sweep on the same data that trained the model is exactly the
kind of quiet snooping this project exists to avoid.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import config

FEATURES = ["abs_z", "half_life", "coint_p", "days_since_formation",
            "p_mr", "vix", "vol_ratio", "bocpd_p"]
F_COLS = [f"f_{f}" for f in FEATURES]
EMBARGO = pd.Timedelta(days=60)   # = max holding period, in calendar days
PROB_THRESHOLD = 0.5              # fixed a priori, not tuned


def build_dataset(trades: pd.DataFrame):
    """(X, y, entry_times, exit_times) from a walk-forward trade log."""
    t = trades.dropna(subset=F_COLS).copy()
    X = t[F_COLS].to_numpy(dtype=float)
    y = (t["pnl_dollars"] > 0).astype(int).to_numpy()
    entry = pd.to_datetime(t["entry_date"])
    exit_ = pd.to_datetime(t["exit_date"])
    return X, y, entry.reset_index(drop=True), exit_.reset_index(drop=True)


def _new_model() -> GradientBoostingClassifier:
    # Shallow trees and subsampling: with a few hundred rows the model
    # capacity has to stay small or it will memorize the dev period.
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=config.HMM_RANDOM_STATE)


def purged_cv(X, y, entry: pd.Series, exit_: pd.Series,
              n_splits: int = 5) -> dict:
    """Chronological purged K-fold. Returns per-fold AUCs and OOF preds."""
    order = np.argsort(entry.values)
    X, y = X[order], y[order]
    entry = entry.iloc[order].reset_index(drop=True)
    exit_ = exit_.iloc[order].reset_index(drop=True)

    bounds = np.linspace(0, len(y), n_splits + 1).astype(int)
    aucs, oof = [], np.full(len(y), np.nan)
    for k in range(n_splits):
        lo, hi = bounds[k], bounds[k + 1]
        test_idx = np.arange(lo, hi)
        t_start, t_end = entry.iloc[lo], entry.iloc[hi - 1]
        # Purge + embargo on both sides of the fold.
        train_mask = ((exit_ < t_start - EMBARGO)
                      | (entry > t_end + EMBARGO)).to_numpy()
        train_mask[test_idx] = False
        if train_mask.sum() < 30 or len(np.unique(y[test_idx])) < 2:
            continue
        model = _new_model().fit(X[train_mask], y[train_mask])
        p = model.predict_proba(X[test_idx])[:, 1]
        oof[test_idx] = p
        aucs.append(roc_auc_score(y[test_idx], p))
    return {"fold_aucs": aucs, "mean_auc": float(np.mean(aucs)) if aucs else np.nan,
            "oof": oof, "y_sorted": y, "entry_sorted": entry}


def train_final(trades: pd.DataFrame) -> GradientBoostingClassifier:
    """Fit on the full development trade history and persist."""
    X, y, _, _ = build_dataset(trades)
    model = _new_model().fit(X, y)
    config.META_MODEL_PKL.write_bytes(pickle.dumps(model))
    return model


def load_model() -> GradientBoostingClassifier:
    return pickle.loads(config.META_MODEL_PKL.read_bytes())


def make_entry_filter(model):
    """Size multiplier from the predicted win probability.

    p <= 0.5 skips the trade; above that, size scales linearly so a
    coin-flip trade gets nothing and a confident one gets full size.
    """
    def entry_filter(features: dict) -> float:
        vec = np.array([[features[f] for f in FEATURES]], dtype=float)
        p = float(model.predict_proba(vec)[0, 1])
        if p < PROB_THRESHOLD:
            return 0.0
        return min(1.0, 2.0 * (p - 0.5))
    return entry_filter


def feature_importances(model) -> pd.Series:
    return pd.Series(model.feature_importances_, index=FEATURES
                     ).sort_values(ascending=False)
