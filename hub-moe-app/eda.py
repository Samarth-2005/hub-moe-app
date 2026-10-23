"""Quick dataset analysis:  python eda.py"""
import pandas as pd, numpy as np
m = pd.read_csv("data/models.csv")
print("rows", len(m), "| total monthly downloads", f"{m.downloads.sum():,}")
print("\nlicense values:", m.license.unique(), "-> license column carries NO information")
print("created_at range:", m.created_at.min(), "->", m.created_at.max(),
      "| rows on floor date 2022-03-02:", (m.created_at == "2022-03-02").sum())
print("\nTop-10 / top-100 download share:",
      round(m.downloads.nlargest(10).sum() / m.downloads.sum(), 3),
      round(m.downloads.nlargest(100).sum() / m.downloads.sum(), 3))
print("median downloads:", int(m.downloads.median()), "| min (selection cutoff):", m.downloads.min())
print("\nDownloads by task (M):\n",
      (m.groupby("pipeline_tag").downloads.sum().sort_values(ascending=False).head(10) / 1e6).round(1))
print("\nMedian downloads by library:\n",
      m.groupby("library").downloads.median().loc[m.library.value_counts().head(8).index].astype(int))
print("\nSpearman(downloads, likes):", round(m[["downloads", "likes"]].corr("spearman").iloc[0, 1], 3))
