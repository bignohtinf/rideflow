from feast import FeatureService
from data.feature.feature_views import (
    order_raw_features,
    order_derived_features,
    on_demand_interactions,
)

ride_completion_service = FeatureService(
    name="ride_completion_service",
    features=[
        order_raw_features,
        order_derived_features,
        on_demand_interactions,
    ],
    tags={"model": "ride_completion", "version": "1"},
)
