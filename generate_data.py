"""
generate_data.py
-----------------
Generates a SYNTHETIC multi-store, multi-SKU retail demand dataset.
This is simulated data (not a real company's data) designed to be large
enough and realistic enough (trend + weekly/annual seasonality + noise +
promo spikes + occasional missing/bad rows) to justify a Spark cleaning
and feature-engineering stage ahead of model training.

Output: data/raw_sales.csv  (~1M+ rows)
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_STORES = 60
N_SKUS = 25
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

dates = pd.date_range(START_DATE, END_DATE, freq="D")
n_days = len(dates)

print(f"Generating synthetic demand data: {N_STORES} stores x {N_SKUS} SKUs x {n_days} days "
      f"= {N_STORES * N_SKUS * n_days:,} rows")

rows = []
store_base = np.random.uniform(20, 200, N_STORES)      # store size/traffic factor
sku_base = np.random.uniform(0.5, 5.0, N_SKUS)          # product popularity factor

day_of_year = dates.dayofyear.values
day_of_week = dates.dayofweek.values
year_idx = (dates.year - dates.year.min()).values

annual_season = 1 + 0.35 * np.sin(2 * np.pi * (day_of_year / 365.25) + 1.2)
weekly_season = np.where(np.isin(day_of_week, [4, 5, 6]), 1.25, 0.95)  # weekend bump
trend = 1 + 0.05 * year_idx  # mild YoY growth

for s in range(N_STORES):
    for k in range(N_SKUS):
        base = store_base[s] * sku_base[k]
        demand = base * annual_season * weekly_season * trend
        noise = np.random.normal(1.0, 0.15, n_days)
        promo_mask = np.random.rand(n_days) < 0.03
        promo_lift = np.where(promo_mask, np.random.uniform(1.4, 2.2, n_days), 1.0)
        qty = np.clip(demand * noise * promo_lift, 0, None)
        qty = np.round(qty).astype(int)

        df = pd.DataFrame({
            "date": dates,
            "store_id": s,
            "sku_id": k,
            "units_sold": qty,
            "promo_flag": promo_mask.astype(int),
        })
        rows.append(df)

full = pd.concat(rows, ignore_index=True)

# inject some messiness for the Spark cleaning step to earn its keep
n_dirty = int(0.002 * len(full))
dirty_idx = np.random.choice(full.index, n_dirty, replace=False)
full.loc[dirty_idx[: n_dirty // 2], "units_sold"] = -1        # bad negative values
full.loc[dirty_idx[n_dirty // 2:], "units_sold"] = np.nan     # missing values

os.makedirs("data", exist_ok=True)
out_path = "data/raw_sales.csv"
full.to_csv(out_path, index=False)
print(f"Wrote {len(full):,} rows to {out_path}")
