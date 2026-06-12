"""Walk-forward engine: re-form pairs every half year, trade them fresh.

Why walk-forward. v1 selected pairs once on 2021-2024 and traded them for
up to five more years. The pairs-trading literature (Do and Faff) and my
own v1 attribution agree on the failure mode: cointegration relationships
decay, and profits concentrate in the months right after formation. Here
I re-run the entire selection on a trailing FORMATION_DAYS window every
WF_STEP days and trade each cohort for only TRADING_DAYS days, so no pair
is ever traded on evidence more than six months stale.

Look-ahead discipline, cycle by cycle:
  * pair selection sees only the formation window;
  * the Kalman filter runs from the formation start, so it enters the
    trading window warm but having seen only past data;
  * the HMM and the OU parameters are fit on the formation window only;
  * the kill-switch series are forward-only by construction;
  * equity chains across cycles (cycle c starts with cycle c-1's ending
    capital), positions are force-closed at cycle end.

The development span is where I tune (delta, score mode); every
configuration evaluated is counted and stored, because the deflated
Sharpe correction in stats_validation needs the true number of trials.
The reserved test year is run exactly once with everything frozen.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import backtest, break_detector, config, cointegration
from . import kalman_hedge, ou_model, regime_detector
from . import signal_generator as sig
from . import universe

PAIRS_CACHE_JSON = config.RESULTS_DIR / "wf_pairs_cache.json"


# ---------------------------------------------------------------------------
# Cycle construction
# ---------------------------------------------------------------------------
def make_cycles(index: pd.DatetimeIndex, span_start, span_end) -> list[dict]:
    """Formation/trading windows whose TRADING window lies inside the span.

    Formations may reach back before span_start: that is correct
    walk-forward (a trader on the first test day has the trailing two
    years of history), not leakage, because formation always precedes
    trading.
    """
    cycles = []
    i = config.FORMATION_DAYS
    cid = 0
    while i < len(index):
        t_end_pos = min(i + config.TRADING_DAYS, len(index))
        f_start, f_end = index[i - config.FORMATION_DAYS], index[i - 1]
        t_start, t_end = index[i], index[t_end_pos - 1]
        if t_start >= pd.Timestamp(span_start) and t_start <= pd.Timestamp(span_end):
            cycles.append({
                "id": cid, "f_start": f_start, "f_end": f_end,
                "t_start": t_start, "t_end": min(t_end, pd.Timestamp(span_end)),
            })
            cid += 1
        i += config.WF_STEP
    return cycles


# ---------------------------------------------------------------------------
# Pair selection per cycle (delta-independent, cached on disk)
# ---------------------------------------------------------------------------
def _load_pairs_cache() -> dict:
    if PAIRS_CACHE_JSON.exists():
        return json.loads(PAIRS_CACHE_JSON.read_text())
    return {}


def select_cycle_pairs(stocks: pd.DataFrame, cyc: dict) -> pd.DataFrame:
    """Top pairs on one formation window, cached by formation end date."""
    cache = _load_pairs_cache()
    key = str(cyc["f_end"].date())
    if key not in cache:
        formation = stocks.loc[cyc["f_start"]:cyc["f_end"]]
        pairs = cointegration.find_pairs(formation, sectors=config.SECTORS_V2,
                                         with_naive_adf=False)
        pairs = pairs.head(config.WF_MAX_PAIRS)
        cache[key] = pairs.to_dict(orient="records")
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PAIRS_CACHE_JSON.write_text(json.dumps(cache, indent=1))
    return pd.DataFrame(cache[key])


# ---------------------------------------------------------------------------
# Per-cycle preparation (everything that depends on delta)
# ---------------------------------------------------------------------------
def prepare_cycle(prices: pd.DataFrame, cyc: dict, delta: float) -> dict:
    """Kalman, regimes, OU fit, and kill-switch for one cycle's pairs."""
    stocks = prices[universe.stock_columns(prices)]
    vix = prices[config.VIX_TICKER]
    pairs = select_cycle_pairs(stocks, cyc)
    trading_dates = stocks.loc[cyc["t_start"]:cyc["t_end"]].index

    per_pair = {}
    ks = break_detector.KillSwitch()
    for _, p in pairs.iterrows():
        name = f"{p['y']}/{p['x']}"
        y = stocks[p["y"]].loc[cyc["f_start"]:cyc["t_end"]]
        x = stocks[p["x"]].loc[cyc["f_start"]:cyc["t_end"]]
        kf = kalman_hedge.kalman_filter_manual(y, x, delta,
                                               obs_var=p["resid_var"])

        feats = regime_detector.build_features(
            y, x, kf["spread"], vix.loc[cyc["f_start"]:cyc["t_end"]])
        det = regime_detector.RegimeDetector().fit(feats.loc[:cyc["f_end"]])
        regimes = det.filtered_regimes(feats)
        p_mr = det.filtered_probabilities(feats)

        # OU on the formation spread, excluding the Kalman convergence
        # rows at the very start of the window.
        form_spread = kf["spread"].loc[:cyc["f_end"]].iloc[config.KALMAN_WARMUP:]
        ou = ou_model.fit_ou(form_spread)

        ks.add_pair(name, kf["spread"], cyc["f_end"], trading_dates)
        per_pair[name] = {"kf": kf, "regimes": regimes, "p_mr": p_mr,
                          "ou": ou, "row": p}

    return {"cycle": cyc, "pairs": pairs, "per_pair": per_pair,
            "killswitch": ks, "trading_dates": trading_dates}


def frames_for(ci: dict, center: str, window: int) -> dict[str, pd.DataFrame]:
    frames = {}
    for name, d in ci["per_pair"].items():
        if center == "ou" and d["ou"] is None:
            continue            # no stable OU fit: skip the pair this cycle
        frames[name] = sig.build_signal_frame(
            d["kf"], d["regimes"], center=center, window=window,
            ou_params=d["ou"], p_mr=d["p_mr"])
    return frames


def make_feature_fn(ci: dict, prices: pd.DataFrame):
    """Decision-time features for meta-labeling. Everything here is
    available at the close the entry decision is made on."""
    vix = prices[config.VIX_TICKER]
    ks = ci["killswitch"]
    f_end = ci["cycle"]["f_end"]

    def feature_fn(pair: str, date, row) -> dict:
        d = ci["per_pair"][pair]
        spread_hist = d["kf"]["spread"].loc[:date]
        sigma_ref = (d["ou"]["sigma_eq"] if d["ou"]
                     else spread_hist.loc[:f_end].std())
        srp = ks.short_run_prob.get(pair)
        return {
            "abs_z": float(abs(row["z"])),
            "half_life": float(d["row"]["half_life"]),
            "coint_p": float(d["row"]["coint_pvalue"]),
            "days_since_formation": int((date - f_end).days),
            "p_mr": float(row.get("p_mr", 0.5)),
            "vix": float(vix.loc[:date].iloc[-1]),
            "vol_ratio": float(spread_hist.tail(20).std() / sigma_ref),
            "bocpd_p": float(srp.get(date, 0.0)) if srp is not None else 0.0,
        }
    return feature_fn


# ---------------------------------------------------------------------------
# The walk-forward run
# ---------------------------------------------------------------------------
def run_walkforward(prices: pd.DataFrame, cycle_inputs: list[dict],
                    center: str, window: int = config.ZSCORE_WINDOW,
                    use_killswitch: bool = True,
                    entry_filter=None, with_features: bool = False,
                    initial_equity: float = config.INITIAL_CAPITAL,
                    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Chain the event loop across cycles. Returns (equity, trades, pairs log)."""
    equity_parts, trade_parts, pair_rows = [], [], []
    eq = initial_equity
    for ci in cycle_inputs:
        frames = frames_for(ci, center, window)
        feature_fn = make_feature_fn(ci, prices) if with_features else None
        equity, trades = backtest.run_event_loop(
            prices, frames, ci["trading_dates"], initial_equity=eq,
            kill_switch=(ci["killswitch"] if use_killswitch else None),
            entry_filter=entry_filter, feature_fn=feature_fn)
        eq = float(equity.iloc[-1])
        if len(trades):
            trades["cycle"] = ci["cycle"]["id"]
            trade_parts.append(trades)
        equity_parts.append(equity)
        for name in frames:
            pair_rows.append({"cycle": ci["cycle"]["id"],
                              "f_end": ci["cycle"]["f_end"],
                              "pair": name})
    equity_all = pd.concat(equity_parts)
    trades_all = (pd.concat(trade_parts, ignore_index=True)
                  if trade_parts else pd.DataFrame())
    return equity_all, trades_all, pd.DataFrame(pair_rows)


# ---------------------------------------------------------------------------
# Tuning on the development span
# ---------------------------------------------------------------------------
def tune_v2(prices: pd.DataFrame, force: bool = False) -> dict:
    """Grid over (delta, score mode) by development walk-forward Sharpe.

    The window stays fixed at the v1 value for "rolling"/"zero" and is
    irrelevant for "ou": every extra knob multiplies the trial count that
    the deflated Sharpe must correct for, so I keep the grid at 9 cells
    and record all of them.
    """
    if config.TUNED_V2_JSON.exists() and not force:
        return json.loads(config.TUNED_V2_JSON.read_text())

    splits = universe.load_splits_v2()
    dev_start, dev_end = splits["development"]
    cycles = make_cycles(prices.index, dev_start, dev_end)
    print(f"[wf-tune] {len(cycles)} development cycles")

    scores = {}
    for delta in config.KALMAN_DELTA_GRID:
        inputs = [prepare_cycle(prices, c, delta) for c in cycles]
        for center in ["rolling", "zero", "ou"]:
            equity, trades, _ = run_walkforward(prices, inputs, center)
            m = backtest.compute_metrics(equity, trades)
            scores[f"{delta:.0e}/{center}"] = m["sharpe"]
            print(f"[wf-tune] delta={delta:.0e} center='{center}': "
                  f"dev Sharpe {m['sharpe']:+.2f}, return "
                  f"{m['total_return']:+.1%}, {m['n_trades']} trades")

    best_key = max(scores, key=lambda k: scores[k])
    best_delta, best_center = best_key.split("/")
    params = {
        "delta": float(best_delta),
        "center": best_center,
        "window": config.ZSCORE_WINDOW,
        "n_configs": len(scores),
        "scores": scores,
    }
    config.TUNED_V2_JSON.write_text(json.dumps(params, indent=2))
    return params


if __name__ == "__main__":
    prices = universe.download_universe_v2()
    params = tune_v2(prices, force="--force" in __import__("sys").argv)
    print(f"\n[wf] chosen: delta={params['delta']:.0e} "
          f"center='{params['center']}' "
          f"({params['n_configs']} configurations evaluated)")
