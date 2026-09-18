"""调用了3DeepSeek对每条新闻做情绪分类,输出结构化信号(并发+重试版)"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["DEEPSEEK_API_KEY"]
API_URL = "https://api.deepseek.com/chat/completions"

COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia",
    "AMZN": "Amazon", "GOOGL": "Google/Alphabet", "TSLA": "Tesla",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_WORKERS = 12
MAX_RETRIES = 3

SYSTEM_PROMPT = """你是一名股票研究助手。根据给定的新闻标题和摘要,判断这条新闻对该公司股价在短期(1-10个交易日)可能产生的影响方向。
只依据新闻内容本身的事实性质做判断(如财报超预期/不及预期、产品发布、监管处罚、高管变动、诉讼等),不要给出任何投资建议。
必须以JSON格式输出,格式如下:
{"sentiment": "positive|negative|neutral", "confidence": "high|medium|low", "reason": "一句话理由,不超过30字"}"""


def classify_news(company: str, headline: str, summary: str) -> dict:
    user_content = f"公司: {company}\n标题: {headline}\n摘要: {summary}"
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return {"sentiment": None, "confidence": None, "reason": f"ERROR after retries: {last_err}"}


def process_ticker(ticker: str):
    company = COMPANY_NAMES[ticker]
    df = pd.read_csv(DATA_DIR / "news_filtered" / f"{ticker}.csv").reset_index(drop=True)

    results = [None] * len(df)
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(classify_news, company, row["headline"], str(row.get("summary", ""))): i
            for i, row in df.iterrows()
        }
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  {ticker}: {done}/{len(df)}")

    out = pd.concat([df, pd.DataFrame(results)], axis=1)
    out_dir = DATA_DIR / "signals"
    out_dir.mkdir(exist_ok=True)
    out.to_csv(out_dir / f"{ticker}.csv", index=False)
    n_err = out["sentiment"].isna().sum()
    print(f"{ticker}: 完成 {len(out)} 条 (失败 {n_err}) -> {out_dir / f'{ticker}.csv'}")
    print(out["sentiment"].value_counts())


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL"]
    for t in tickers:
        process_ticker(t)
