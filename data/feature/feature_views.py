from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.redshift_source import RedshiftSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int64
import pandas as pd
import numpy as np


order_entity = Entity(
    name="order_id",
    description="Unique order/matching event identifier",
)

raw_features_source = RedshiftSource(
    table="order_raw_features",
    timestamp_field="event_timestamp",
)

derived_features_source = RedshiftSource(
    table="order_derived_features",
    timestamp_field="event_timestamp",
)

order_raw_features = FeatureView(
    name="order_raw_features",
    entities=[order_entity],
    ttl=timedelta(days=90),
    schema=[
        # Supply-demand
        Field(name="num_drivers", dtype=Int64),
        Field(name="num_orders", dtype=Int64),
        # ETA
        Field(name="eta_avg", dtype=Float64),
        Field(name="eta_std", dtype=Float64),
        Field(name="eta_min", dtype=Float64),
        # EDA
        Field(name="eda_avg", dtype=Float64),
        Field(name="eda_std", dtype=Float64),
        Field(name="eda_min", dtype=Float64),
        # Trip
        Field(name="distance", dtype=Float64),
        Field(name="total_fee", dtype=Float64),
        # Temporal
        Field(name="hour_of_day", dtype=Int64),
        Field(name="minute_of_hour", dtype=Int64),
        Field(name="rush_hour", dtype=Int64),
        # Behavioral
        Field(name="user_waiting_time_seconds", dtype=Float64),
    ],
    source=raw_features_source,
    online=True,
    tags={"team": "crp", "tier": "raw"},
)

order_derived_features = FeatureView(
    name="order_derived_features",
    entities=[order_entity],
    ttl=timedelta(days=90),
    schema=[
        # Supply-demand derived
        Field(name="supply_demand_ratio", dtype=Float64),
        Field(name="demand_supply_ratio", dtype=Float64),
        # Confidence
        Field(name="eta_confidence", dtype=Float64),
        Field(name="eda_confidence", dtype=Float64),
        # Trip value
        Field(name="fee_per_km", dtype=Float64),
        Field(name="eta_per_km", dtype=Float64),
        Field(name="eta_eda_ratio", dtype=Float64),
        Field(name="pickup_to_trip_ratio", dtype=Float64),
        # Binary flags
        Field(name="is_short_trip", dtype=Int64),
        Field(name="is_long_eta", dtype=Int64),
        Field(name="is_high_wait", dtype=Int64),
        Field(name="is_negative_wait", dtype=Int64),
        Field(name="is_single_driver", dtype=Int64),
        # Date features
        Field(name="day_of_week", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="is_friday", dtype=Int64),
        Field(name="rush_hour_weekday", dtype=Int64),
    ],
    source=derived_features_source,
    online=True,
    tags={"team": "crp", "tier": "derived"},
)

@on_demand_feature_view(
    sources=[
        order_raw_features,
        order_derived_features,
    ],
    schema=[
        Field(name="hour_sin", dtype=Float64),
        Field(name="hour_cos", dtype=Float64),
        Field(name="minutes_since_midnight", dtype=Int64),
        Field(name="short_trip_rush", dtype=Int64),
        Field(name="low_supply_flag", dtype=Int64),
        Field(name="low_supply_short_trip", dtype=Int64),
        Field(name="high_eta_rush", dtype=Int64),
    ],
)
def on_demand_interactions(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    # Cyclical time encoding
    df["hour_sin"] = np.sin(2 * np.pi * inputs["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * inputs["hour_of_day"] / 24)
    df["minutes_since_midnight"] = (inputs["hour_of_day"] * 60 + inputs["minute_of_hour"]).astype(int)
    # Interaction features
    df["short_trip_rush"] = (inputs["is_short_trip"] * inputs["rush_hour"]).astype(int)
    df["low_supply_flag"] = (inputs["supply_demand_ratio"] < 0.2).astype(int)
    df["low_supply_short_trip"] = (df["low_supply_flag"] * inputs["is_short_trip"]).astype(int)
    df["high_eta_rush"] = (inputs["is_long_eta"] * inputs["rush_hour"]).astype(int)
    return df
