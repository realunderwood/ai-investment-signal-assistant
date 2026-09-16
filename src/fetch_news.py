"""按周分段拉取Finnhub公司新闻,存为 data/news_raw/{ticker}.csv
用法:
  python src/fetch_news.py --pilot          # 只跑AAPL近4周,验证pipeline
  python src/fetch_news.py --full           # 跑全部标的+全量区间
"""
import argparse
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["FINNHUB_API_KEY"]

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "news_raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def week_ranges(start: date, end: date):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=7), end)
        yield cur, nxt
        cur = nxt


def fetch_week(ticker: str, f: date, t: date, retries: int = 4):
    for attempt in range(retries):
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": ticker, "from": f.isoformat(), "to": t.isoformat(), "token": API_KEY},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()
            print(f"[WARN] {ticker} {f}~{t} status={r.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[RETRY {attempt+1}/{retries}] {ticker} {f}~{t}: {e}")
        time.sleep(2 * (attempt + 1))
    print(f"[FAIL] {ticker} {f}~{t}: 放弃,跳过这一周")
    return []


def fetch_ticker_news(ticker: str, start: date, end: date) -> pd.DataFrame:
    rows = []
    for f, t in week_ranges(start, end):
        items = fetch_week(ticker, f, t)
        for it in items:
            rows.append({
                "ticker": ticker,
                "news_id": it.get("id"),
                "datetime": it.get("datetime"),
                "headline": it.get("headline"),
                "summary": it.get("summary"),
                "source": it.get("source"),
                "url": it.get("url"),
            })
        time.sleep(0.3)  # 免费版限速 30次/秒,留足余量
    df = pd.DataFrame(rows).drop_duplicates(subset=["news_id"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--resume", action="store_true", help="跳过已经抓取过的标的")
    args = parser.parse_args()

    end = date(2026, 9, 15)
    if args.pilot:
        tickers = ["AAPL"]
        start = end - timedelta(weeks=4)
    else:
        tickers = TICKERS
        start = date(2025, 9, 16)  # 近12个月

    for ticker in tickers:
        out_path = DATA_DIR / f"{ticker}.csv"
        if args.resume and out_path.exists():
            print(f"{ticker}: 已存在,跳过")
            continue
        df = fetch_ticker_news(ticker, start, end)
        df.to_csv(out_path, index=False)
        print(f"{ticker}: {len(df)} 条新闻 -> {out_path}")


if __name__ == "__main__":
    main()
