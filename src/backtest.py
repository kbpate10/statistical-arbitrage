"""Event-driven backtester. No look-ahead bias.

Execution model
---------------
The loop walks the calendar day by day. Signals observed at day t's close
become PENDING ORDERS that fill at day t+1's close (next-bar execution),
paying 5 bps cost + 2 bps slippage per side on gross notional. Each day:

    1. fill pending orders at today's close
    2. mark open positions to market
    3. evaluate today's signals -> queue orders for tomorrow

ENTRY RE-CONFIRMATION: an entry order only fills if the dislocation still
exists at the fill bar (same direction signal, regime still
mean-reverting, both evaluated on fill-day information). I added this
after measuring that the average |z| decayed from 2.55 at decision time
to 0.77 at fill time: without re-confirmation I was buying one-day spikes
AFTER they had already snapped back, which is pure adverse selection.
Exits always fill; I never cancel a risk reduction.

Walk-forward state
------------------
For each pair I compute the Kalman filter and HMM features on the
CONTIGUOUS history from the start of train through the end of the
requested split. Both are strictly free of forward-looking information
(filtered Kalman estimates, forward-algorithm HMM states,
trailing-window z-scores), so warming them up on prior history leaks
nothing. It just means the test year starts with a settled filter,
exactly as live trading would.

Frozen upstream of the backtest: selected pairs (train), HMM parameters
(train), Kalman delta and z-score settings (validation).
`run_backtest("test")` is the single out-of-sample evaluation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, cointegration, kalman_hedge, regime_detector
from . import signal_generator as sig
from . import universe
from .risk_model import (PortfolioState, Position, position_units,
                         transaction_cost)


# ---------------------------------------------------------------------------
# Data preparation (per pair, walk-forward safe)
# ---------------------------------------------------------------------------
def prepare_pair_base(prices: pd.DataFrame, pairs: pd.DataFrame,
                      delta: float, splits: dict,
                      through_split: str) -> dict[str, tuple]:
    """Kalman output + filtered regimes per pair (the expensive layer).

    History ends at `through_split`; the HMM is fit on train features only.
    """
    stocks = prices[universe.stock_columns(prices)]
    vix = prices[config.VIX_TICKER]
    train_end = splits["train"][1]
    _, end = splits[through_split]

    base = {}
    for _, p in pairs.iterrows():
        y = stocks[p["y"]].loc[:end]
        x = stocks[p["x"]].loc[:end]
        kf = kalman_hedge.kalman_filter_manual(y, x, delta,
                                               obs_var=p["resid_var"])
        feats = regime_detector.build_features(y, x, kf["spread"],
                                               vix.loc[:end])
        det = regime_detector.RegimeDetector().fit(feats.loc[:train_end])
        regimes = det.filtered_regimes(feats)
        base[f"{p['y']}/{p['x']}"] = (kf, regimes)
    return base


def frames_from_base(base: dict[str, tuple], center: str,
                     window: int) -> dict[str, pd.DataFrame]:
    """Cheap z-score layer on top of a prepared (Kalman, regimes) base."""
    return {pair: sig.build_signal_frame(kf, regimes, center=center,
                                         window=window)
            for pair, (kf, regimes) in base.items()}


def prepare_pair_frames(prices: pd.DataFrame, pairs: pd.DataFrame,
                        delta: float, splits: dict,
                        through_split: str,
                        center: str = "rolling",
                        window: int = config.ZSCORE_WINDOW,
                        ) -> dict[str, pd.DataFrame]:
    """Signal frames for every pair on history ending at `through_split`."""
    base = prepare_pair_base(prices, pairs, delta, splits, through_split)
    return frames_from_base(base, center, window)


def tune_signal_params(prices: pd.DataFrame, pairs: pd.DataFrame,
                       splits: dict,
                       force: bool = False) -> tuple[float, str, int]:
    """Choose (delta, z-score center, z-score window) jointly by VALIDATION
    backtest Sharpe, which is the actual trading objective rather than a
    proxy. Tuning the pieces separately against different objectives left
    them inconsistent with each other; tuning jointly against the real
    backtest removes that mismatch.

    Validation is the designated tuning set; the test year is untouched.
    The choice is persisted so the test backtest runs with frozen params.
    """
    params = (json.loads(config.TUNED_PARAMS_JSON.read_text())
              if config.TUNED_PARAMS_JSON.exists() else {})
    keys = ("delta", "zscore_center", "zscore_window")
    if all(k in params for k in keys) and not force:
        return params["delta"], params["zscore_center"], params["zscore_window"]

    start, end = splits["validation"]
    dates = prices.loc[start:end].index
    scores = {}
    for delta in config.KALMAN_DELTA_GRID:
        base = prepare_pair_base(prices, pairs, delta, splits, "validation")
        for center in config.ZSCORE_CENTER_MODES:
            for window in config.ZSCORE_WINDOW_GRID:
                frames = frames_from_base(base, center, window)
                equity, trades = run_event_loop(prices, frames, dates)
                m = compute_metrics(equity, trades)
                scores[(delta, center, window)] = m["sharpe"]
                print(f"[tune] delta={delta:.0e} center='{center}' "
                      f"window={window}: validation Sharpe "
                      f"{m['sharpe']:+.2f} ({m['n_trades']} trades)")

    best = max(scores, key=lambda k: scores[k])
    params["delta"], params["zscore_center"] = best[0], best[1]
    params["zscore_window"] = int(best[2])
    params["signal_scores"] = {f"{d:.0e}/{c}/{w}": s
                               for (d, c, w), s in scores.items()}
    config.TUNED_PARAMS_JSON.write_text(json.dumps(params, indent=2))
    return best


# ---------------------------------------------------------------------------
# Core event loop
# ---------------------------------------------------------------------------
def run_event_loop(prices: pd.DataFrame, frames: dict[str, pd.DataFrame],
                   dates: pd.DatetimeIndex,
                   initial_equity: float = config.INITIAL_CAPITAL,
                   kill_switch=None,
                   entry_filter=None,
                   feature_fn=None) -> tuple[pd.Series, pd.DataFrame]:
    """Daily loop over `dates`. Returns (equity curve, trade log).

    Hooks (all optional, all evaluated on information available at the
    time they are called):
      initial_equity  starting capital; the walk-forward engine chains
                      cycles by passing the previous cycle's ending equity
      kill_switch     fn(pair, date) -> "exit" | "block" | None. "exit"
                      force-closes an open position (reason "break") and
                      blocks new entries; "block" only blocks entries.
      feature_fn      fn(pair, date, row) -> dict. Decision-time features
                      attached to each entry and written into the trade
                      log (columns prefixed f_), forming the meta-labeling
                      dataset.
      entry_filter    fn(features) -> float size multiplier in [0, 1].
                      0 skips the entry. Applied at the fill bar, so the
                      meta-model can only veto or shrink, never peek.
    """
    state = PortfolioState(equity=initial_equity)
    cash = state.equity
    pending: list[dict] = []          # orders decided yesterday
    equity_curve = {}
    trades = []

    def close_position(pos: Position, date, py, px_, exit_z, reason):
        nonlocal cash
        pnl_gross = pos.market_value(py, px_)
        exit_cost = transaction_cost(pos.gross_notional(py, px_))
        pnl_net = pnl_gross - pos.entry_cost - exit_cost
        cash += pnl_net
        capital_base = pos.gross_notional(pos.entry_price_y, pos.entry_price_x)
        rec = {
            "pair": pos.pair, "direction": "long" if pos.direction == 1 else "short",
            "entry_date": pos.entry_date, "exit_date": date,
            "entry_z": pos.entry_z, "exit_z": exit_z,
            "pnl_dollars": pnl_net,
            "pnl_bps": 10_000 * pnl_net / capital_base if capital_base else np.nan,
            "holding_days": int((date - pos.entry_date).days),
            "exit_reason": reason,
        }
        if pos.features:
            rec.update({f"f_{k}": v for k, v in pos.features.items()})
        trades.append(rec)
        del state.positions[pos.pair]

    for date in dates:
        # ---- 1. fill yesterday's orders at today's close -------------------
        for order in pending:
            pair = order["pair"]
            y_tkr, x_tkr = pair.split("/")
            py, px_ = prices.at[date, y_tkr], prices.at[date, x_tkr]
            row = frames[pair].loc[date] if date in frames[pair].index else None

            if order["action"] == "exit" and pair in state.positions:
                z_now = row["z"] if row is not None else np.nan
                close_position(state.positions[pair], date, py, px_,
                               z_now, order["reason"])

            elif order["action"] == "entry" and state.can_open(pair):
                # Re-confirm on fill-day information: same-direction signal
                # and regime gate must still hold, else cancel the order.
                if row is None or sig.entry_signal(row) != order["direction"]:
                    continue
                size_mult = 1.0
                if entry_filter is not None and order.get("features"):
                    size_mult = float(entry_filter(order["features"]))
                    if size_mult <= 0:
                        continue
                units = size_mult * position_units(
                    state.equity, row["spread_std"], row["beta"], py, px_)
                if units <= 0:
                    continue
                pos = Position(
                    pair=pair, y=y_tkr, x=x_tkr,
                    direction=order["direction"], units=units,
                    beta=row["beta"], entry_date=date,
                    entry_z=row["z"], entry_price_y=py, entry_price_x=px_,
                    entry_cost=transaction_cost(
                        units * (py + abs(row["beta"]) * px_)),
                    features=order.get("features"),
                )
                state.positions[pair] = pos
        pending = []

        # ---- 2. mark to market ---------------------------------------------
        mtm = sum(
            pos.market_value(prices.at[date, pos.y], prices.at[date, pos.x])
            - pos.entry_cost
            for pos in state.positions.values()
        )
        state.equity = cash + mtm
        equity_curve[date] = state.equity

        # ---- 3. evaluate today's signals -> orders for tomorrow -------------
        entry_candidates = []
        for pair, frame in frames.items():
            if date not in frame.index:
                continue
            row = frame.loc[date]

            ks = kill_switch(pair, date) if kill_switch is not None else None
            if ks == "exit" and pair in state.positions:
                pending.append({"pair": pair, "action": "exit",
                                "reason": "break"})
                continue

            if pair in state.positions:
                pos = state.positions[pair]
                held = (date - pos.entry_date).days
                reason = sig.exit_signal(row["z"], pos.direction)
                if reason is None and held >= config.MAX_HOLDING_DAYS:
                    reason = "timeout"
                if reason is not None:
                    pending.append({"pair": pair, "action": "exit",
                                    "reason": reason})
            elif ks is None:    # entries blocked while any kill flag is up
                direction = sig.entry_signal(row)
                if direction != 0:
                    entry_candidates.append({
                        "pair": pair, "action": "entry",
                        "direction": direction, "z": row["z"],
                        "beta": row["beta"], "spread_std": row["spread_std"],
                        "features": (feature_fn(pair, date, row)
                                     if feature_fn is not None else None),
                    })

        # Strongest dislocations first if capacity is scarce.
        entry_candidates.sort(key=lambda o: -abs(o["z"]))
        slots = config.MAX_CONCURRENT_POSITIONS - len(state.positions)
        pending.extend(entry_candidates[:max(slots, 0)])

    # Force-close anything still open at the final bar.
    last = dates[-1]
    for pair in list(state.positions):
        pos = state.positions[pair]
        py, px_ = prices.at[last, pos.y], prices.at[last, pos.x]
        z_now = (frames[pair].at[last, "z"]
                 if last in frames[pair].index else np.nan)
        close_position(pos, last, py, px_, z_now, "end_of_period")
    state.equity = cash
    equity_curve[last] = state.equity

    equity = pd.Series(equity_curve).sort_index()
    trade_log = pd.DataFrame(trades)
    return equity, trade_log


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    rets = equity.pct_change().dropna()
    n_years = len(equity) / config.TRADING_DAYS_PER_YEAR
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    sharpe = (np.sqrt(config.TRADING_DAYS_PER_YEAR) * rets.mean() / rets.std()
              if rets.std() > 0 else np.nan)

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    m = {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar) if pd.notna(calmar) else None,
        "n_trades": int(len(trades)),
    }
    if len(trades):
        wins = trades[trades["pnl_dollars"] > 0]
        losses = trades[trades["pnl_dollars"] <= 0]
        gross_win = wins["pnl_dollars"].sum()
        gross_loss = abs(losses["pnl_dollars"].sum())
        m.update({
            "win_rate": float(len(wins) / len(trades)),
            "avg_win": float(wins["pnl_dollars"].mean()) if len(wins) else 0.0,
            "avg_loss": float(losses["pnl_dollars"].mean()) if len(losses) else 0.0,
            "profit_factor": (float(gross_win / gross_loss)
                              if gross_loss > 0 else np.inf),
            "avg_holding_days": float(trades["holding_days"].mean()),
        })
    return m


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_backtest(split_name: str, save: bool = True) -> dict:
    """Backtest one split with all upstream parameters frozen."""
    prices = universe.download_universe()
    splits = universe.load_splits()
    train = universe.get_split(prices[universe.stock_columns(prices)], "train")
    pairs = cointegration.select_pairs(train)
    delta, center, window = tune_signal_params(prices, pairs, splits)

    frames = prepare_pair_frames(prices, pairs, delta, splits, split_name,
                                 center=center, window=window)
    start, end = splits[split_name]
    dates = prices.loc[start:end].index

    equity, trades = run_event_loop(prices, frames, dates)
    metrics = compute_metrics(equity, trades)

    if save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        equity.to_csv(config.RESULTS_DIR / f"equity_{split_name}.csv",
                      header=["equity"])
        trades.to_csv(config.RESULTS_DIR / f"trades_{split_name}.csv",
                      index=False)
        (config.RESULTS_DIR / f"metrics_{split_name}.json").write_text(
            json.dumps(metrics, indent=2, default=str))
        # Regime series for the report overlay (fraction of pairs in MR).
        regime_frac = pd.concat(
            [f["regime_ok"].loc[start:end] for f in frames.values()],
            axis=1).mean(axis=1)
        regime_frac.to_csv(config.RESULTS_DIR / f"regimes_{split_name}.csv",
                           header=["frac_mean_reverting"])
    return {"equity": equity, "trades": trades, "metrics": metrics}


if __name__ == "__main__":
    for split in ["train", "validation", "test"]:
        out = run_backtest(split)
        m = out["metrics"]
        print(f"\n=== {split.upper()} ===")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k:<20} {v:,.4f}")
            else:
                print(f"  {k:<20} {v}")
