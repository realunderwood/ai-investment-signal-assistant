"""过滤新闻:去掉泛市场噪音,保留真正跟公司相关的新闻,并按天限流"""
import re
from pathlib import Path

import pandas as pd

COMPANY_NAMES = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "NVDA": ["nvidia"],
    "AMZN": ["amazon"],
    "GOOGL": ["google", "alphabet"],
    "TSLA": ["tesla"],
}

# 常见的泛市场/聚合类标题模式,直接过滤掉
JUNK_PATTERNS = [
    r"stocks (are )?moving",
    r"techcheck",
    r"portfolio update",
    r"dow jones stocks",
    r"\bETF\b",
    r"tracking .* portfolio",
    r"here'?s what (else )?(he|she|they) (bought|purchased|sold)",
    r"morning briefing",
    r"evening edition",
    r"market wrap",
    r"stocks to watch",
]
JUNK_RE = re.compile("|".join(JUNK_PATTERNS), re.IGNORECASE)

MAX_PER_DAY = 5
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def filter_ticker(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "news_raw" / f"{ticker}.csv")
    df["date"] = pd.to_datetime(df["datetime"], unit="s").dt.date

    names = COMPANY_NAMES[ticker]
    name_mask = df["headline"].str.lower().apply(lambda h: any(n in str(h) for n in names))
    junk_mask = df["headline"].apply(lambda h: bool(JUNK_RE.search(str(h))))
    kept = df[name_mask & ~junk_mask].copy()

    # 每天限流:优先保留非Yahoo来源(聚合噪音更少),再按摘要长度排序取长的
    kept["is_yahoo"] = (kept["source"] == "Yahoo").astype(int)
    kept["summary_len"] = kept["summary"].fillna("").str.len()
    kept = kept.sort_values(["date", "is_yahoo", "summary_len"], ascending=[True, True, False])
    kept = kept.groupby("date", group_keys=False).head(MAX_PER_DAY)

    return kept.drop(columns=["is_yahoo", "summary_len"])


def main():
    total_before, total_after = 0, 0
    for ticker in COMPANY_NAMES:
        before = pd.read_csv(DATA_DIR / "news_raw" / f"{ticker}.csv")
        kept = filter_ticker(ticker)
        out_path = DATA_DIR / "news_filtered"
        out_path.mkdir(exist_ok=True)
        kept.to_csv(out_path / f"{ticker}.csv", index=False)
        print(f"{ticker}: {len(before)} -> {len(kept)} (过滤率 {1 - len(kept)/len(before):.0%})")
        total_before += len(before)
        total_after += len(kept)
    print(f"\n合计: {total_before} -> {total_after} (过滤率 {1 - total_after/total_before:.0%})")


if __name__ == "__main__":
    main()
