from pyspark.sql import SparkSession
from pyspark.sql import functions as F 
import sys

target_date = sys.argv[1]
RAW_PATH = f"s3://rideflow/raw/{target_date}/data.parquet"
PROCESSED_PATH = f"s3://rideflow/processed/{target_date}/"

spark = SparkSession.builder.appName("ingest_job").getOrCreate()

df = spark.read.parquet(RAW_PATH)
print(f"Raw rows: {df.count():,}")

median_wait = df.approxQuantile("user_waiting_time_seconds", [0.5], 0.01)[0]

df_clean = (
    df
    .drop("Unnamed: 0")
    .dropDuplicates(["order_id"])
    .withColumn("date", F.to_date("date"))
    .withColumn(
        "user_waiting_time_seconds",
        F.when(F.col("user_waiting_time_seconds") <= 0, median_wait).otherwise(F.col("user_waiting_time_seconds"))
    )
    .fillna({"eta_std": 0.0, "eda_std": 0.0})
    .withColumn("ingested_at", F.current_timestamp())
)

print(f"Processed rows: {df_clean.count():,}")

df_clean.write.mode("overwrite").parquet(PROCESSED_PATH)

print(f"Written to: {PROCESSED_PATH}")
spark.stop()