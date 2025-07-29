from __future__ import annotations 
import numpy as np
from typing import Literal

def aggregate_feature(
    sampler: "StreetSampler",
    feature_name: str,
    method: Literal["mean", "sum", "count", "min", "max", "median"] = "mean",
    new_attribute_name: str | None = None,
) -> None:
    if new_attribute_name is None:
        new_attribute_name = feature_name

    values_per_street = []

    for street in sampler.streets:
        values = street.get_point_attribute_values(feature_name)
        values = [v for v in values if v is not None]

        if not values:
            values_per_street.append(None)
            continue

        if method == "mean":
            result = float(np.mean(values))
        elif method == "sum":
            result = float(np.sum(values))
        elif method == "count":
            result = len(values)
        elif method == "min":
            result = float(np.min(values))
        elif method == "max":
            result = float(np.max(values))
        elif method == "median":
            result = float(np.median(values))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        values_per_street.append(result)

    sampler.tag_streets_raw(new_attribute_name, values_per_street)