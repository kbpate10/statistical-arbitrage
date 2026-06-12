"""Self-contained HTML performance report (plotly).

I read the artifacts saved by backtest.run_backtest() for all three
splits and render: equity curves, drawdowns, a monthly returns heatmap,
the trade P&L distribution, the regime overlay, and a cross-split metrics
table. I chose HTML over PDF because plotly charts stay interactive and
the file opens anywhere with no extra dependencies.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config

SPLITS = ["train", "validation", "test"]
SPLIT_COLORS = {"train": "#1f77b4", "validation": "#ff7f0e",
                "test": "#d62728"}


def _load(split: str) -> dict:
    equity = pd.read_csv(config.RESULTS_DIR / f"equity_{split}.csv",
                         index_col=0, parse_dates=True)["equity"]
    trades = pd.read_csv(config.RESULTS_DIR / f"trades_{split}.csv",
                         parse_dates=["entry_date", "exit_date"])
    metrics = json.loads(
        (config.RESULTS_DIR / f"metrics_{split}.json").read_text())
    regimes = pd.read_csv(config.RESULTS_DIR / f"regimes_{split}.csv",
                          index_col=0, parse_dates=True
                          )["frac_mean_reverting"]
    return {"equity": equity, "trades": trades, "metrics": metrics,
            "regimes": regimes}


def equity_figure(data: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.06,
                        subplot_titles=("Equity curve (each split starts at "
                                        "$100k)", "Drawdown"))
    for split in SPLITS:
        eq = data[split]["equity"]
        dd = eq / eq.cummax() - 1
        c = SPLIT_COLORS[split]
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=split,
                                 line=dict(color=c, width=1.4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=f"{split} DD",
                                 line=dict(color=c, width=1),
                                 fill="tozeroy", showlegend=False),
                      row=2, col=1)
    fig.update_yaxes(tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    fig.update_layout(height=560, template="plotly_white",
                      legend=dict(orientation="h", y=1.08))
    return fig


def monthly_heatmap(data: dict) -> go.Figure:
    eq = pd.concat([data[s]["equity"] for s in SPLITS])
    eq = eq[~eq.index.duplicated()]
    monthly = eq.resample("ME").last().pct_change().dropna()
    tbl = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    }).pivot(index="year", columns="month", values="ret")
    fig = go.Figure(go.Heatmap(
        z=tbl.values, x=[f"{m:02d}" for m in tbl.columns], y=tbl.index,
        colorscale="RdYlGn", zmid=0,
        text=np.where(np.isnan(tbl.values), "",
                      np.round(tbl.values * 100, 1)),
        texttemplate="%{text}", colorbar=dict(tickformat=".1%"),
    ))
    fig.update_layout(title="Monthly returns (%), splits concatenated",
                      height=380, template="plotly_white",
                      yaxis=dict(autorange="reversed", dtick=1))
    return fig


def pnl_distribution(data: dict) -> go.Figure:
    fig = go.Figure()
    for split in SPLITS:
        t = data[split]["trades"]
        if len(t):
            fig.add_trace(go.Histogram(x=t["pnl_dollars"], name=split,
                                       marker_color=SPLIT_COLORS[split],
                                       opacity=0.6, nbinsx=40))
    fig.update_layout(barmode="overlay", height=380,
                      title="Trade P&L distribution ($)",
                      template="plotly_white")
    return fig


def regime_overlay(data: dict) -> go.Figure:
    fig = go.Figure()
    for split in SPLITS:
        r = data[split]["regimes"]
        fig.add_trace(go.Scatter(x=r.index, y=r, name=split,
                                 line=dict(color=SPLIT_COLORS[split],
                                           width=1.2)))
    fig.update_layout(
        title=("Fraction of pairs in mean-reverting regime "
               "(new trades only allowed in this state)"),
        height=320, template="plotly_white", yaxis=dict(tickformat=".0%"))
    return fig


def metrics_table(data: dict) -> str:
    rows = [
        ("Total return", "total_return", "{:.2%}"),
        ("Annualized return", "annualized_return", "{:.2%}"),
        ("Sharpe ratio", "sharpe", "{:.2f}"),
        ("Max drawdown", "max_drawdown", "{:.2%}"),
        ("Calmar ratio", "calmar", "{:.2f}"),
        ("Trades", "n_trades", "{:d}"),
        ("Win rate", "win_rate", "{:.1%}"),
        ("Avg win", "avg_win", "${:,.0f}"),
        ("Avg loss", "avg_loss", "${:,.0f}"),
        ("Profit factor", "profit_factor", "{:.2f}"),
        ("Avg holding (days)", "avg_holding_days", "{:.1f}"),
    ]
    html = ["<table class='metrics'><tr><th>Metric</th>"]
    html += [f"<th>{s}</th>" for s in SPLITS]
    html.append("</tr>")
    for label, key, fmt in rows:
        html.append(f"<tr><td>{label}</td>")
        for s in SPLITS:
            v = data[s]["metrics"].get(key)
            html.append("<td>n/a</td>" if v is None
                        else f"<td>{fmt.format(v)}</td>")
        html.append("</tr>")
    html.append("</table>")
    return "".join(html)


def tuned_params_html() -> str:
    if not config.TUNED_PARAMS_JSON.exists():
        return ""
    p = json.loads(config.TUNED_PARAMS_JSON.read_text())
    return (f"<p class='params'>Frozen parameters (I tuned these on the "
            f"validation year only): Kalman &delta; = {p.get('delta'):.0e}, "
            f"z-score center = '{p.get('zscore_center')}', z-score window = "
            f"{p.get('zscore_window')}d. Pairs were selected on training "
            f"data only.</p>")


def generate_report(out_path=None) -> str:
    data = {s: _load(s) for s in SPLITS}
    figs = [equity_figure(data), monthly_heatmap(data),
            pnl_distribution(data), regime_overlay(data)]
    fig_html = [f.to_html(full_html=False,
                          include_plotlyjs=("cdn" if i == 0 else False))
                for i, f in enumerate(figs)]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Stat Arb Performance Report</title>
<style>
 body {{ font-family: -apple-system, Helvetica, Arial, sans-serif;
        max-width: 1100px; margin: 30px auto; color: #222; }}
 h1 {{ font-size: 26px; }} h2 {{ font-size: 19px; margin-top: 36px; }}
 table.metrics {{ border-collapse: collapse; width: 100%; }}
 table.metrics td, table.metrics th {{ border: 1px solid #ddd;
   padding: 7px 12px; text-align: right; }}
 table.metrics th {{ background: #f4f4f4; }}
 table.metrics td:first-child {{ text-align: left; }}
 .note {{ background: #fff7e0; border-left: 4px solid #e6a700;
   padding: 10px 14px; font-size: 14px; }}
 .params {{ color: #555; font-size: 14px; }}
</style></head><body>
<h1>Statistical Arbitrage Performance Report</h1>
<p>Pairs trading with Engle-Granger cointegration selection, Kalman-filter
dynamic hedge ratios, and HMM regime gating. I selected pairs on training
years 1 to 3, tuned all parameters on the validation year, and evaluated
the test year exactly once with everything frozen.</p>
{tuned_params_html()}
<div class="note"><b>How to read this report:</b> compare in-sample and
out-of-sample performance directly in the table below. I report the
degradation from train to test exactly as measured; no parameter was
chosen using test data.</div>
<h2>Metrics across splits</h2>
{metrics_table(data)}
<h2>Equity &amp; drawdown</h2>{fig_html[0]}
<h2>Monthly returns</h2>{fig_html[1]}
<h2>Trade P&amp;L distribution</h2>{fig_html[2]}
<h2>Regime overlay</h2>{fig_html[3]}
</body></html>"""

    if out_path is None:
        out_path = config.RESULTS_DIR / "report.html"
    out_path.write_text(html)
    return str(out_path)


# ---------------------------------------------------------------------------
# v2 report: walk-forward, kill-switch, meta-labeling, significance
# ---------------------------------------------------------------------------
def _v2_metrics_row(equity_csv: str, trades_csv: str) -> dict:
    from . import backtest
    eq = pd.read_csv(config.RESULTS_DIR / equity_csv, index_col=0,
                     parse_dates=True)["equity"]
    tr = pd.read_csv(config.RESULTS_DIR / trades_csv)
    return backtest.compute_metrics(eq, tr)


def generate_report_v2(out_path=None) -> str:
    """Self-contained HTML report for the v2 walk-forward system."""
    params = json.loads(config.TUNED_V2_JSON.read_text())
    sig = json.loads((config.RESULTS_DIR / "significance.json").read_text())
    meta = json.loads((config.RESULTS_DIR / "meta_cv.json").read_text())
    v1_test = json.loads(
        (config.RESULTS_DIR / "metrics_test.json").read_text())

    rows = {
        "v2 development (walk-forward)": _v2_metrics_row(
            "wf_equity_dev.csv", "wf_trades_dev.csv"),
        "v2 test, meta filter (headline)": _v2_metrics_row(
            "wf_equity_test.csv", "wf_trades_test.csv"),
        "v2 test, no meta (ablation)": _v2_metrics_row(
            "wf_equity_test_nometa.csv", "wf_trades_test_nometa.csv"),
        "v1 test (static pairs)": v1_test,
    }
    metric_defs = [
        ("Total return", "total_return", "{:.2%}"),
        ("Annualized return", "annualized_return", "{:.2%}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Max drawdown", "max_drawdown", "{:.2%}"),
        ("Trades", "n_trades", "{:d}"),
        ("Win rate", "win_rate", "{:.1%}"),
        ("Profit factor", "profit_factor", "{:.2f}"),
        ("Avg holding (days)", "avg_holding_days", "{:.1f}"),
    ]
    tbl = ["<table class='metrics'><tr><th>Metric</th>"]
    tbl += [f"<th>{k}</th>" for k in rows]
    tbl.append("</tr>")
    for label, key, fmt in metric_defs:
        tbl.append(f"<tr><td>{label}</td>")
        for m in rows.values():
            v = m.get(key)
            tbl.append("<td>n/a</td>" if v is None
                       else f"<td>{fmt.format(v)}</td>")
        tbl.append("</tr>")
    tbl.append("</table>")

    sig_tbl = ["<table class='metrics'><tr><th>Span</th><th>Sharpe</th>"
               "<th>PSR</th><th>DSR</th><th>E[max SR | luck]</th>"
               "<th>90% bootstrap CI</th></tr>"]
    for k, v in sig.items():
        ci_ = v["bootstrap_90ci"]
        sig_tbl.append(
            f"<tr><td>{k}</td><td>{v['sharpe_annual']:+.2f}</td>"
            f"<td>{v['psr']:.2f}</td><td>{v['dsr']:.2f}</td>"
            f"<td>{v['expected_max_sharpe_annual']:.2f}</td>"
            f"<td>({ci_['lo']:+.2f}, {ci_['hi']:+.2f})</td></tr>")
    sig_tbl.append("</table>")

    # Equity figure: dev + both test variants.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        row_heights=[0.6, 0.4], vertical_spacing=0.1,
                        subplot_titles=("Development walk-forward equity",
                                        "Reserved test year"))
    eq_dev = pd.read_csv(config.RESULTS_DIR / "wf_equity_dev.csv",
                         index_col=0, parse_dates=True)["equity"]
    fig.add_trace(go.Scatter(x=eq_dev.index, y=eq_dev, name="development",
                             line=dict(color="#1f77b4", width=1.2)),
                  row=1, col=1)
    for fname, label, color in [
            ("wf_equity_test.csv", "test with meta filter", "#d62728"),
            ("wf_equity_test_nometa.csv", "test without meta", "#999999")]:
        eq = pd.read_csv(config.RESULTS_DIR / fname, index_col=0,
                         parse_dates=True)["equity"]
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=label,
                                 line=dict(color=color, width=1.3)),
                      row=2, col=1)
    fig.update_yaxes(tickformat="$,.0f")
    fig.update_layout(height=620, template="plotly_white",
                      legend=dict(orientation="h", y=1.06))

    # Exit reasons and pair turnover.
    tr_dev = pd.read_csv(config.RESULTS_DIR / "wf_trades_dev.csv")
    reasons = tr_dev["exit_reason"].value_counts()
    fig2 = go.Figure(go.Bar(x=reasons.index, y=reasons.values,
                            marker_color="#1f77b4"))
    fig2.update_layout(title="Development exit reasons ('break' = "
                             "kill-switch fired)", height=320,
                       template="plotly_white")

    pairs_dev = pd.read_csv(config.RESULTS_DIR / "wf_pairs_dev.csv")
    per_cycle = pairs_dev.groupby("cycle")["pair"].apply(set)
    turnover = [np.nan] + [
        1 - len(per_cycle.iloc[i] & per_cycle.iloc[i - 1])
        / max(len(per_cycle.iloc[i]), 1)
        for i in range(1, len(per_cycle))]
    fig3 = go.Figure(go.Bar(x=per_cycle.index, y=turnover,
                            marker_color="#ff7f0e"))
    fig3.update_layout(title="Pair turnover per cycle (share of pairs not "
                             "carried over from the previous cycle)",
                       height=320, template="plotly_white",
                       yaxis=dict(tickformat=".0%"))

    imp = pd.Series(meta["importances"]).sort_values()
    fig4 = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                            marker_color="#2ca02c"))
    fig4.update_layout(title=(f"Meta-model feature importances "
                              f"(purged-CV AUC {meta['mean_auc']:.3f} on "
                              f"{meta['n_trades']} dev trades)"),
                       height=360, template="plotly_white")

    figs_html = [f.to_html(full_html=False,
                           include_plotlyjs=("cdn" if i == 0 else False))
                 for i, f in enumerate([fig, fig2, fig3, fig4])]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Stat Arb v2 Report</title>
<style>
 body {{ font-family: -apple-system, Helvetica, Arial, sans-serif;
        max-width: 1100px; margin: 30px auto; color: #222; }}
 h1 {{ font-size: 26px; }} h2 {{ font-size: 19px; margin-top: 36px; }}
 table.metrics {{ border-collapse: collapse; width: 100%; }}
 table.metrics td, table.metrics th {{ border: 1px solid #ddd;
   padding: 7px 12px; text-align: right; }}
 table.metrics th {{ background: #f4f4f4; }}
 table.metrics td:first-child {{ text-align: left; }}
 .note {{ background: #fff7e0; border-left: 4px solid #e6a700;
   padding: 10px 14px; font-size: 14px; }}
 .params {{ color: #555; font-size: 14px; }}
</style></head><body>
<h1>Statistical Arbitrage v2: Walk-Forward Report</h1>
<p>I re-form pairs every six months on a trailing two-year window across
a ~100-ticker universe (10 years of data), score dislocations with the
configuration tuned on the development span, monitor every live pair
with a Bayesian online changepoint kill-switch, and filter entries with
a meta-model trained on development trades under purged cross-validation.
The reserved final year was run once with everything frozen.</p>
<p class='params'>Frozen: Kalman delta = {params['delta']:.0e}, score
mode = '{params['center']}', {params['n_configs']} configurations tried
during tuning (this count feeds the deflated Sharpe).</p>
<div class="note"><b>Caveats I am not hiding:</b> the final year overlaps
v1's already-consumed test year, so it is out-of-sample with respect to
v2's parameters but not virgin data; the static ticker list carries
survivorship tilt; costs assume 14 bps round trip.</div>
<h2>Performance across spans (v1 comparison included)</h2>
{''.join(tbl)}
<h2>Is it signal or selection luck?</h2>
{''.join(sig_tbl)}
<p>PSR = probability the true Sharpe exceeds zero. DSR = probability it
exceeds what the best of {params['n_configs']} random configurations
would show. A DSR near 0.5 or below means the result is indistinguishable
from tuning luck.</p>
<h2>Equity curves</h2>{figs_html[0]}
<h2>Exit reasons</h2>{figs_html[1]}
<h2>Pair turnover</h2>{figs_html[2]}
<h2>Meta-model</h2>{figs_html[3]}
</body></html>"""

    if out_path is None:
        out_path = config.RESULTS_DIR / "report_v2.html"
    out_path.write_text(html)
    return str(out_path)


if __name__ == "__main__":
    print(f"[report] written: {generate_report()}")
