import sys
import logging
import pandas as pd
from pyspark.sql import SparkSession
from dotenv import load_dotenv

from data.feature.transformations import build_features
from data.feature.preprocessing import preprocess_for_training

load_dotenv()

BUCKET = "rideflow"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: spark-submit pipelines/spark/feature_job.py <yyyy-mm-dd>"
        )
    target_date = sys.argv[1]

    spark = SparkSession.builder.appName("feature_job").getOrCreate()

    processed_path = f"s3://{BUCKET}/processed/{target_date}/data.parquet"
    feature_path = f"s3://{BUCKET}/features/{target_date}/features.parquet"

    try:
        logging.info(f"Starting feature engineering job for {target_date}")
        logging.info(f"Reading data from {processed_path}")

        df = spark.read.parquet(processed_path)
        row_count = df.count()
        logging.info(f"Read {row_count:,} rows")

        # Convert to Pandas for feature engineering
        pdf = df.toPandas()
        pdf = preprocess_for_training(pdf)
        pdf = build_features(pdf, use_date=True, use_driver_agg=True)

        logging.info(f"Feature engineering complete: {pdf.shape[1]} features")

        # Write back via Spark for S3 compatibility
        spark.createDataFrame(pdf).write.mode("overwrite").parquet(feature_path)
        logging.info(f"Written features to {feature_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
