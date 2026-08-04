"""
Independent pandas re-implementation of spark_pipeline.py's cleaning + feature
logic, used ONLY to sanity-check the row-count claims in README.md in an
environment where pyspark cannot be installed (no PyPI network access).
This does not replace spark_pipeline.py or prove Spark-specific runtime claims.
"""
import time
import pandas as pd

t0 = time.time()
df = pd.read_csv("data/raw_sales.csv", parse_dates=["date"])
raw_count = len(df)
print(f"Loaded {raw_count:,} raw rows")

clean = df[df["units_sold"].notna() & (df["units_sold"] >= 0)].copy()
clean_count = len(clean)
dropped = raw_count - clean_count
print(f"Dropped {dropped:,} invalid/null rows ({dropped/raw_count:.3%})")

store_daily = (
    clean.groupby(["store_id", "date"], as_index=False)
    .agg(total_units=("units_sold", "sum"), any_promo=("promo_flag", "max"))
)

store_daily = store_daily.sort_values(["store_id", "date"])
store_daily["dow"] = store_daily["date"].dt.dayofweek.map(lambda x: (x + 1) % 7 + 1)  # match Spark dayofweek (Sun=1)
store_daily["month"] = store_daily["date"].dt.month

g = store_daily.groupby("store_id")
store_daily["lag_1"] = g["total_units"].shift(1)
store_daily["lag_7"] = g["total_units"].shift(7)
store_daily["roll_mean_7"] = g["total_units"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
store_daily["roll_mean_28"] = g["total_units"].transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())

featured = store_daily.dropna()
n_stores = featured["store_id"].nunique()
n_rows = len(featured)
print(f"Built features for {n_stores} stores, {n_rows:,} store-day rows after lag/rolling windows")
print(f"Elapsed: {time.time()-t0:.2f}s (pandas, single-threaded, NOT a Spark timing comparison)")

featured.to_csv("data/features.csv", index=False)
