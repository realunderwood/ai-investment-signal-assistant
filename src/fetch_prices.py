"""拉取标的的历史日线价格数据,存为 data/prices.csv"""
import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "QQQ"]  # QQQ作为大盘基准,用于计算超额收益
START = "2025-08-01"  # 比新闻区间早一个月,留出计算forward return的缓冲
END = "2026-09-16"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def main():
    frames = []
    for ticker in TICKERS:
        df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        if df.empty:
            print(f"[WARN] {ticker} 没有拉到数据")
            continue
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df["ticker"] = ticker
        frames.append(df[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]])
        print(f"{ticker}: {len(df)} 行, {df['Date'].min().date()} ~ {df['Date'].max().date()}")

    result = pd.concat(frames, ignore_index=True)
    out_path = DATA_DIR / "prices.csv"
    result.to_csv(out_path, index=False)
    print(f"\n已保存到 {out_path}, 共 {len(result)} 行")


if __name__ == "__main__":
    main()
