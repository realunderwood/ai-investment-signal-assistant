"""生成 positive vs negative 差异检验的可视化图表(核心结论图)"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    df = pd.read_csv(OUT_DIR / "positive_vs_negative.csv")
    tickers = df["ticker"].unique()
    horizons = ["1d", "3d", "5d", "10d"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(horizons))
    width = 0.13

    for i, ticker in enumerate(tickers):
        sub = df[df["ticker"] == ticker].set_index("horizon").reindex(horizons)
        diffs = sub["diff"].values * 100
        sig = sub["p_value"].values < 0.05
        bars = ax.bar(x + i * width, diffs, width, label=ticker)
        for j, (bar, s) in enumerate(zip(bars, sig)):
            if s:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        "*", ha="center", va="bottom" if bar.get_height() >= 0 else "top",
                        fontsize=14, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(tickers) - 1) / 2)
    ax.set_xticklabels(horizons)
    ax.set_ylabel("正面信号 - 负面信号 超额收益差 (%)")
    ax.set_title("Positive vs Negative 信号超额收益差异(* = p<0.05)")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = OUT_DIR / "backtest_chart.png"
    plt.savefig(out_path, dpi=150)
    print(f"图表已保存到 {out_path}")


if __name__ == "__main__":
    main()
