"""回测:计算信号发出后N个交易日的收益率,与基线对比"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HORIZONS = [1, 3, 5, 10]


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["Date"])
    return df.sort_values(["ticker", "Date"]).reset_index(drop=True)


def compute_forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """给每个 (ticker, Date) 附上未来N日收益率,以及相对QQQ基准的超额收益率"""
    out = []
    for ticker, g in prices.groupby("ticker"):
        g = g.sort_values("Date").reset_index(drop=True)
        for h in HORIZONS:
            g[f"fwd_ret_{h}d"] = g["Close"].shift(-h) / g["Close"] - 1
        out.append(g)
    result = pd.concat(out, ignore_index=True)

    # 用QQQ同期收益作为大盘基准,计算超额收益
    qqq = result[result["ticker"] == "QQQ"].set_index("Date")
    for h in HORIZONS:
        qqq_ret_map = qqq[f"fwd_ret_{h}d"]
        result[f"benchmark_ret_{h}d"] = result["Date"].map(qqq_ret_map)
        result[f"excess_ret_{h}d"] = result[f"fwd_ret_{h}d"] - result[f"benchmark_ret_{h}d"]

    return result


def align_signal_to_trading_day(signal_date, trading_days: np.ndarray):
    """把新闻日期对齐到当天或之后最近的交易日"""
    idx = np.searchsorted(trading_days, signal_date)
    if idx >= len(trading_days):
        return None
    return trading_days[idx]


def run_ticker_backtest(ticker: str, prices_fwd: pd.DataFrame) -> pd.DataFrame:
    sig = pd.read_csv(DATA_DIR / "signals" / f"{ticker}.csv", parse_dates=["date"])
    sig = sig.dropna(subset=["sentiment"])

    p = prices_fwd[prices_fwd["ticker"] == ticker].copy()
    trading_days = np.sort(p["Date"].unique())

    sig["trading_day"] = sig["date"].apply(lambda d: align_signal_to_trading_day(np.datetime64(d), trading_days))
    sig = sig.dropna(subset=["trading_day"])

    merged = sig.merge(p, left_on="trading_day", right_on="Date", how="left")
    return merged


def summarize(merged: pd.DataFrame, prices_fwd: pd.DataFrame, ticker: str):
    print(f"\n===== {ticker} (超额收益 = 个股收益 - 同期QQQ收益) =====")
    baseline = prices_fwd[prices_fwd["ticker"] == ticker]

    rows = []
    for sentiment in ["positive", "negative", "neutral"]:
        sub = merged[merged["sentiment"] == sentiment]
        if len(sub) == 0:
            continue
        for h in HORIZONS:
            col = f"excess_ret_{h}d"
            sig_rets = sub[col].dropna()
            base_rets = baseline[col].dropna()
            if len(sig_rets) < 3:
                continue
            t, p_val = stats.ttest_ind(sig_rets, base_rets, equal_var=False)
            rows.append({
                "sentiment": sentiment,
                "horizon": f"{h}d",
                "n": len(sig_rets),
                "signal_mean_excess_ret": sig_rets.mean(),
                "baseline_mean_excess_ret": base_rets.mean(),
                "signal_win_rate": (sig_rets > 0).mean(),
                "baseline_win_rate": (base_rets > 0).mean(),
                "p_value": p_val,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print(result.to_string(index=False))

    print("\n--- positive vs negative 直接对比(核心检验:情绪分类是否有方向区分度) ---")
    diff_rows = []
    for h in HORIZONS:
        col = f"excess_ret_{h}d"
        pos = merged[merged["sentiment"] == "positive"][col].dropna()
        neg = merged[merged["sentiment"] == "negative"][col].dropna()
        if len(pos) < 3 or len(neg) < 3:
            print(f"{h}d: 样本量不足 (positive n={len(pos)}, negative n={len(neg)})")
            continue
        t, p_val = stats.ttest_ind(pos, neg, equal_var=False)
        diff_rows.append({
            "horizon": f"{h}d", "positive_n": len(pos), "negative_n": len(neg),
            "positive_mean": pos.mean(), "negative_mean": neg.mean(),
            "diff": pos.mean() - neg.mean(), "p_value": p_val,
        })
    diff_df = pd.DataFrame(diff_rows)
    if not diff_df.empty:
        print(diff_df.to_string(index=False))

    return result, diff_df


def main(tickers):
    prices = load_prices()
    prices_fwd = compute_forward_returns(prices)

    all_results = []
    all_diffs = []
    for ticker in tickers:
        signals_path = DATA_DIR / "signals" / f"{ticker}.csv"
        if not signals_path.exists():
            print(f"[SKIP] {ticker}: 无信号文件")
            continue
        merged = run_ticker_backtest(ticker, prices_fwd)
        result, diff_df = summarize(merged, prices_fwd, ticker)
        result["ticker"] = ticker
        all_results.append(result)
        if not diff_df.empty:
            diff_df["ticker"] = ticker
            all_diffs.append(diff_df)

    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        out_path = DATA_DIR.parent / "outputs" / "backtest_summary.csv"
        out_path.parent.mkdir(exist_ok=True)
        final.to_csv(out_path, index=False)
        print(f"\n汇总结果已保存到 {out_path}")

    if all_diffs:
        diff_final = pd.concat(all_diffs, ignore_index=True)
        diff_final.to_csv(DATA_DIR.parent / "outputs" / "positive_vs_negative.csv", index=False)


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL"]
    main(tickers)
