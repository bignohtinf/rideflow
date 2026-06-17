import json
import math
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import (
    KafkaSource,
    KafkaOffsetsInitializer,
    KafkaSink,
    KafkaRecordSerializationSchema,
)
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common import WatermarkStrategy

KAFKA_BROKER = "kafka:9092"
SOURCE_TOPIC = "rides-raw"
SINK_TOPIC = "rides-features"


def add_realtime_features(record: dict) -> dict:
    num_drivers = record.get("num_drivers", 0)
    num_orders = record.get("num_orders", 0)
    record["supply_demand_ratio"] = num_drivers / (num_orders + 1)
    record["demand_supply_ratio"] = num_orders / (num_drivers + 1)

    # Confidence features
    eta_avg = record.get("eta_avg", 0)
    eta_std = record.get("eta_std", 0)
    eda_avg = record.get("eda_avg", 0)
    eda_std = record.get("eda_std", 0)
    record["eta_confidence"] = eta_std / (eta_avg + 1)
    record["eda_confidence"] = eda_std / (eda_avg + 0.01)

    # Trip value features
    distance = record.get("distance", 0)
    total_fee = record.get("total_fee", 0)
    record["fee_per_km"] = total_fee / (distance + 0.01)
    record["eta_per_km"] = eta_avg / (eda_avg + 0.01)
    record["eta_eda_ratio"] = eta_avg / (eda_avg + 0.01)
    record["pickup_to_trip_ratio"] = eda_avg / (distance + 0.01)

    # Binary flags
    record["is_short_trip"] = int(distance < 2)
    record["is_long_eta"] = int(eta_avg > 900)
    wait_seconds = record.get("user_waiting_time_seconds", 0)
    record["is_high_wait"] = int(wait_seconds > 120)
    record["is_negative_wait"] = int(wait_seconds < 0)
    record["is_single_driver"] = int(num_drivers == 1)

    # Time features
    hour = record.get("hour_of_day", 0)
    minute = record.get("minute_of_hour", 0)
    record["rush_hour"] = int(7 <= hour <= 9 or 17 <= hour <= 19)
    record["minutes_since_midnight"] = hour * 60 + minute
    record["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    record["hour_cos"] = math.cos(2 * math.pi * hour / 24)

    # Interaction features
    record["short_trip_rush"] = record["is_short_trip"] * record["rush_hour"]
    record["low_supply_flag"] = int(record["supply_demand_ratio"] < 0.2)
    record["low_supply_short_trip"] = record["low_supply_flag"] * record["is_short_trip"]
    record["high_eta_rush"] = record["is_long_eta"] * record["rush_hour"]

    return record


def main():
    env = StreamExecutionEnvironment.get_execution_environment()

    source = (
        KafkaSource.builder()
        .set_bootstrap_servers(KAFKA_BROKER)
        .set_topics(SOURCE_TOPIC)
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )

    stream = env.from_source(
        source,
        WatermarkStrategy.no_watermarks(),
        "rides-raw-source",
    )

    features = (
        stream
        .map(lambda x: json.loads(x))
        .map(add_realtime_features)
        .map(lambda x: json.dumps(x))
    )

    sink = (
        KafkaSink.builder()
        .set_bootstrap_servers(KAFKA_BROKER)
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(SINK_TOPIC)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        )
        .build()
    )

    features.sink_to(sink)
    env.execute("flink_consumer")


if __name__ == "__main__":
    main()
