"""
spark_pipeline.py
------------------
Distributed cleaning + feature engineering stage using PySpark.
Reads raw_sales.csv, drops/repairs bad rows, aggregates to daily
store-level totals, and builds lag + rolling-window features used
by the PyTorch forecasting model downstream.
"""
import time
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

t_start = time.time()

spark = (
    SparkSession.builder
    .appName("DemandForecastPipeline")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

t_session = time.time()

df = spark.read.csv("data/raw_sales.csv", header=True, inferSchema=True)
raw_count = df.count()
print(f"[spark] Loaded {raw_count:,} raw rows")

# --- Cleaning ---
# Drop nulls and impossible negative values (injected dirty rows)
clean = df.filter(F.col("units_sold").isNotNull() & (F.col("units_sold") >= 0))
clean_count = clean.count()
dropped = raw_count - clean_count
print(f"[spark] Dropped {dropped:,} invalid/null rows ({dropped/raw_count:.3%})")

# --- Aggregate to store-level daily total demand across all SKUs ---
store_daily = (
    clean.groupBy("store_id", "date")
    .agg(
        F.sum("units_sold").alias("total_units"),
        F.max("promo_flag").alias("any_promo"),
    )
)

# --- Feature engineering: lag features + rolling averages per store, via window functions ---
w = Window.partitionBy("store_id").orderBy("date")

featured = (
    store_daily
    .withColumn("dow", F.dayofweek("date"))
    .withColumn("month", F.month("date"))
    .withColumn("lag_1", F.lag("total_units", 1).over(w))
    .withColumn("lag_7", F.lag("total_units", 7).over(w))
    .withColumn("roll_mean_7", F.avg("total_units").over(w.rowsBetween(-7, -1)))
    .withColumn("roll_mean_28", F.avg("total_units").over(w.rowsBetween(-28, -1)))
)

featured = featured.dropna()

n_stores = featured.select("store_id").distinct().count()
n_feature_rows = featured.count()
print(f"[spark] Built features for {n_stores} stores, {n_feature_rows:,} store-day rows after lag/rolling windows")

# Write out for the PyTorch training step
featured.orderBy("store_id", "date").toPandas().to_csv("data/features.csv", index=False)

t_end = time.time()

print("\n--- Spark stage summary ---")
print(f"Raw rows processed:        {raw_count:,}")
print(f"Rows dropped in cleaning:  {dropped:,}")
print(f"Store-day feature rows:    {n_feature_rows:,}")
print(f"Spark session startup:     {t_session - t_start:.2f}s")
print(f"Total Spark job time:      {t_end - t_start:.2f}s")

spark.stop()
