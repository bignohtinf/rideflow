import json
import pandas as pd
from confluent_kafka import Consumer, KafkaError
from feast import FeatureStore
from loguru import logger
from datetime import datetime

KAFKA_BROKER = "kafka:9092"
TOPIC = "rides-features"
BATCH_SIZE = 100
POLL_TIMEOUT = 1.0


def main():
    store = FeatureStore(repo_path="data/feature/")

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": "feast-online-writer",
        "auto.offset.reset": "latest",
    })
    consumer.subscribe([TOPIC])

    logger.info(f"Starting online store writer, consuming from {TOPIC}")

    buffer: list[dict] = []

    try:
        while True:
            msg = consumer.poll(POLL_TIMEOUT)

            if msg is None:
                if buffer:
                    _flush_to_store(store, buffer)
                    buffer = []
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Kafka error: {msg.error()}")
                continue

            record = json.loads(msg.value())
            if "event_timestamp" not in record:
                record["event_timestamp"] = datetime.utcnow().isoformat()
            buffer.append(record)

            if len(buffer) >= BATCH_SIZE:
                _flush_to_store(store, buffer)
                buffer = []

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if buffer:
            _flush_to_store(store, buffer)
    finally:
        consumer.close()


def _flush_to_store(store: FeatureStore, records: list[dict]):
    df = pd.DataFrame(records)

    if "event_timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])

    try:
        store.write_to_online_store(
            feature_view_name="order_raw_features",
            df=df,
        )
        logger.debug(f"Flushed {len(df)} records to online store")
    except Exception as e:
        logger.error(f"Failed to write to online store: {e}")


if __name__ == "__main__":
    main()
