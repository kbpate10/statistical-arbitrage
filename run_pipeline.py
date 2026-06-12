"""End-to-end pipelines: v1 (static splits) and v2 (walk-forward).

Usage:  python3 run_pipeline.py [--force]        v1 pipeline
        python3 run_pipeline.py v2 [--force]     v2 walk-forward pipeline
--force re-downloads data and re-runs selection/tuning from scratch.

I keep this as a thin orchestrator: every stage lives in src/ and can
also be run standalone or from a Jupyter notebook (the notebooks import
the same functions, so there is one implementation of everything).
"""

import json
import sys

from src import (backtest, cointegration, config, kalman_hedge, meta_label,
                 report, stats_validation, universe, walkforward)


def main(force: bool = False) -> None:
    if force:
        for f in [config.PRICES_PARQUET, config.SPLITS_JSON,
                  config.SELECTED_PAIRS_JSON, config.TUNED_PARAMS_JSON]:
            f.unlink(missing_ok=True)

    print("== 1/5 universe ==")
    prices = universe.download_universe()
    splits = universe.load_splits()
    print(f"   {prices.shape[0]} days x {prices.shape[1]} tickers, "
          f"splits: { {k: v for k, v in splits.items()} }")

    print("== 2/5 pair selection (train only) ==")
    stocks = prices[universe.stock_columns(prices)]
    pairs = cointegration.select_pairs(universe.get_split(stocks, "train"))
    print(f"   {len(pairs)} cointegrated pairs: "
          + ", ".join(f"{p['y']}/{p['x']}" for _, p in pairs.iterrows()))

    print("== 3/5 kalman self-test ==")
    top = pairs.iloc[0]
    train = universe.get_split(stocks, "train")
    diff = kalman_hedge.self_test(train[top["y"]], train[top["x"]],
                                  obs_var=top["resid_var"])
    print(f"   manual vs pykalman max diff {diff:.2e}")

    print("== 4/5 backtests (tuning on validation if not cached) ==")
    for split in ["train", "validation", "test"]:
        m = backtest.run_backtest(split)["metrics"]
        print(f"   {split:<11} return {m['total_return']:+7.2%}  "
              f"Sharpe {m['sharpe']:+5.2f}  maxDD {m['max_drawdown']:6.2%}  "
              f"trades {m['n_trades']}")

    print("== 5/5 report ==")
    print(f"   {report.generate_report()}")


def main_v2(force: bool = False) -> None:
    """v2: walk-forward + OU + kill-switch + meta-labeling + significance.

    Order matters: tuning and the meta-model only ever see the
    development span; the reserved test year is run at the very end with
    everything frozen, once with the meta filter (the headline, committed
    to in the plan before results were seen) and once without (ablation).
    """
    if force:
        for f in [config.PRICES_V2_PARQUET, config.SPLITS_V2_JSON,
                  config.TUNED_V2_JSON, config.META_MODEL_PKL,
                  walkforward.PAIRS_CACHE_JSON]:
            f.unlink(missing_ok=True)

    print("== 1/6 v2 universe (10y, wide) ==")
    prices = universe.download_universe_v2()
    splits = universe.load_splits_v2()
    print(f"   {prices.shape[0]} days x {prices.shape[1]} tickers; "
          f"dev {splits['development'][0]} -> {splits['development'][1]}, "
          f"test {splits['test'][0]} -> {splits['test'][1]}")

    print("== 2/6 tuning on development walk-forward ==")
    params = walkforward.tune_v2(prices)
    delta, center = params["delta"], params["center"]
    print(f"   chosen delta={delta:.0e} center='{center}' "
          f"({params['n_configs']} configurations tried)")

    print("== 3/6 development run (chosen config, with features) ==")
    dev_cycles = walkforward.make_cycles(prices.index,
                                         *splits["development"])
    dev_inputs = [walkforward.prepare_cycle(prices, c, delta)
                  for c in dev_cycles]
    eq_dev, tr_dev, pairs_dev = walkforward.run_walkforward(
        prices, dev_inputs, center, with_features=True)
    m_dev = backtest.compute_metrics(eq_dev, tr_dev)
    eq_dev.to_csv(config.RESULTS_DIR / "wf_equity_dev.csv",
                  header=["equity"])
    tr_dev.to_csv(config.RESULTS_DIR / "wf_trades_dev.csv", index=False)
    pairs_dev.to_csv(config.RESULTS_DIR / "wf_pairs_dev.csv", index=False)
    print(f"   dev: return {m_dev['total_return']:+.1%}, Sharpe "
          f"{m_dev['sharpe']:+.2f}, {m_dev['n_trades']} trades")

    # Ablation for the notebooks: same config without the kill-switch.
    eq_noks, tr_noks, _ = walkforward.run_walkforward(
        prices, dev_inputs, center, use_killswitch=False)
    eq_noks.to_csv(config.RESULTS_DIR / "wf_equity_dev_noks.csv",
                   header=["equity"])
    tr_noks.to_csv(config.RESULTS_DIR / "wf_trades_dev_noks.csv",
                   index=False)

    print("== 4/6 meta-labeling (purged CV on dev trades) ==")
    X, y, entry, exit_ = meta_label.build_dataset(tr_dev)
    cv = meta_label.purged_cv(X, y, entry, exit_)
    model = meta_label.train_final(tr_dev)
    imp = meta_label.feature_importances(model)
    (config.RESULTS_DIR / "meta_cv.json").write_text(json.dumps({
        "n_trades": int(len(y)), "base_win_rate": float(y.mean()),
        "fold_aucs": cv["fold_aucs"], "mean_auc": cv["mean_auc"],
        "importances": imp.round(4).to_dict(),
    }, indent=2))
    print(f"   {len(y)} dev trades, purged-CV AUC {cv['mean_auc']:.3f}, "
          f"top features: {', '.join(imp.head(3).index)}")

    print("== 5/6 frozen test year ==")
    test_cycles = walkforward.make_cycles(prices.index, *splits["test"])
    test_inputs = [walkforward.prepare_cycle(prices, c, delta)
                   for c in test_cycles]
    eq_test, tr_test, _ = walkforward.run_walkforward(
        prices, test_inputs, center, with_features=True,
        entry_filter=meta_label.make_entry_filter(model))
    m_test = backtest.compute_metrics(eq_test, tr_test)
    eq_test.to_csv(config.RESULTS_DIR / "wf_equity_test.csv",
                   header=["equity"])
    tr_test.to_csv(config.RESULTS_DIR / "wf_trades_test.csv", index=False)
    # Ablation: same frozen config without the meta filter.
    eq_tnm, tr_tnm, _ = walkforward.run_walkforward(
        prices, test_inputs, center)
    eq_tnm.to_csv(config.RESULTS_DIR / "wf_equity_test_nometa.csv",
                  header=["equity"])
    tr_tnm.to_csv(config.RESULTS_DIR / "wf_trades_test_nometa.csv",
                  index=False)
    m_tnm = backtest.compute_metrics(eq_tnm, tr_tnm)
    print(f"   test (meta):    return {m_test['total_return']:+.1%}, "
          f"Sharpe {m_test['sharpe']:+.2f}, {m_test['n_trades']} trades")
    print(f"   test (no meta): return {m_tnm['total_return']:+.1%}, "
          f"Sharpe {m_tnm['sharpe']:+.2f}, {m_tnm['n_trades']} trades")

    print("== 6/6 significance + report ==")
    trial_sharpes = list(params["scores"].values())
    sig = {
        "development": stats_validation.full_report(eq_dev, trial_sharpes),
        "test": stats_validation.full_report(eq_test, trial_sharpes),
        "test_nometa": stats_validation.full_report(eq_tnm, trial_sharpes),
    }
    (config.RESULTS_DIR / "significance.json").write_text(
        json.dumps(sig, indent=2, default=float))
    for k, v in sig.items():
        print(f"   {k}: SR {v['sharpe_annual']:+.2f}, PSR {v['psr']:.2f}, "
              f"DSR {v['dsr']:.2f}, 90% CI "
              f"({v['bootstrap_90ci']['lo']:+.2f}, "
              f"{v['bootstrap_90ci']['hi']:+.2f})")
    print(f"   {report.generate_report_v2()}")


if __name__ == "__main__":
    if "v2" in sys.argv:
        main_v2(force="--force" in sys.argv)
    else:
        main(force="--force" in sys.argv)
