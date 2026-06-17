from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, hour, lit, from_json
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    TimestampType,
)
import json
import sys

TYPE_MAPPING = {
    "string": StringType(),
    "integer": IntegerType(),
    "float": DoubleType(),
    "timestamp": TimestampType(),
}

CHECKPOINT = "s3://rideflow/checkpoints/ride_features_backfill/"
BUCKET = "rideflow"


def build_spark_schema(schema_config: dict) -> StructType:
    fields = []
    for name, spec in schema_config["fields"].items():
        fields.append(
            StructField(
                name,
                TYPE_MAPPING[spec["type"]],
                nullable=spec.get("nullable", True),
            )
        )
    return StructType(fields)


def add_realtime_features(df):
    df = df.withColumn(
        "supply_demand_ratio",
        col("num_drivers") / (col("num_orders") + lit(1)),
    )
    df = df.withColumn(
        "eta_confidence",
        col("eta_std") / (col("eta_avg") + lit(1)),
    )
    df = df.withColumn(
        "rush_hour",
        when(
            (hour(col("event_timestamp")).between(7, 9))
            | (hour(col("event_timestamp")).between(17, 19)),
            1,
        ).otherwise(0),
    )
    return df


def backfill(target_date: str):
    spark = SparkSession.builder.appName("streaming_backfill").getOrCreate()

    with open("data/raw/schemas/rides_raw_schema.json", "r") as f:
        schema_config = json.load(f)

    ride_schema = build_spark_schema(schema_config)

    input_path = f"s3://{BUCKET}/processed/{target_date}/data.parquet"
    df = spark.read.parquet(input_path)

    feature_df = add_realtime_features(df)
    output_path = f"s3://{BUCKET}/features_backfill/{target_date}/"
    feature_df.write.mode("overwrite").parquet(output_path)

    print(f"Backfilled {feature_df.count():,} rows → {output_path}")
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: spark-submit pipelines/spark/streaming_job.py <yyyy-mm-dd>")
        sys.exit(1)
    backfill(sys.argv[1])
