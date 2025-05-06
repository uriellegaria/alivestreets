from __future__ import annotations


from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point

from ..sampling.street_sampler import StreetSampler  # relative import
from ..sampling.street import Street

# ────────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────────────────────────

def _to_plain(value: Any) -> Any:
    """Cast NumPy scalars/arrays to plain Python objects for DBF safety."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _point_on_segment(pt: Sequence[float], segment: LineString, tol: float) -> bool:
    """Return *True* if *pt* lies on *segment* within *tol* (units = CRS)."""
    return segment.distance(Point(pt[0], pt[1])) <= tol


def _aggregate(values: List[Any], method: str) -> Any:
    """Aggregate *values* by *method*; numeric preferred, categorical OK."""
    if len(values) == 0:
        return None
    try:
        if method == "mean":
            return float(np.mean(values))
        if method == "sum":
            return float(np.sum(values))
        if method == "count":
            return int(len(values))
        if method == "min":
            return float(np.min(values))
        if method == "max":
            return float(np.max(values))
        if method == "median":
            return float(np.median(values))
    except Exception:
        pass  # non‑numeric → fallback below
    try:
        return max(set(values), key=values.count)
    except Exception:
        return values[0]


# ────────────────────────────────────────────────────────────────────────────────
# Exporter class
# ────────────────────────────────────────────────────────────────────────────────

class StreetSamplerGeojsonExporter:
    """Exporter for :class:`StreetSampler`.

    Parameters
    ----------
    sampler
        The sampler instance.
    attribute_methods
        Mapping ``{attribute_name: aggregation_method}`` used **only** when
        ``aggregation='segment'``.
    """

    def __init__(
        self,
        sampler: StreetSampler,
        attribute_methods: Optional[Dict[str, Literal[
            "mean", "sum", "count", "min", "max", "median"
        ]]] = None,
    ) -> None:
        self._sampler: StreetSampler = sampler
        self._attribute_methods: Dict[str, str] = attribute_methods or {}

    # Public API ----------------------------------------------------------------

    def export(
        self,
        output_path: str,
        *,
        crs: str = "EPSG:4326",
        aggregation: Literal["street", "segment"] = "street",
        default_method: Literal[
            "mean", "sum", "count", "min", "max", "median"
        ] = "mean",
        distance_tol: float = 1e-9,
    ) -> None:

        rows: List[Dict[str, Any]] = []
        n_streets: int = len(self._sampler.streets)

        for i in range(0, n_streets):
            street: Street = self._sampler.streets[i]
            n_segments: int = len(street.street_segments)

            segment_lines: List[LineString] = []
            for j in range(0, n_segments):
                segment_lines.append(LineString(street.street_segments[j]))

            # Street‑level attributes prepared once
            street_attr_plain: Dict[str, Any] = {
                key: _to_plain(val) for key, val in street.attributes.items()
            }

            # Clean street name for export (empty if "unnamed_*")
            clean_name: str = "" if street.street_id.startswith("unnamed") else street.street_id

            # Iterate segments ---------------------------------------------------
            for j in range(0, n_segments):
                geom: LineString = segment_lines[j]
                row: Dict[str, Any] = {
                    "name": clean_name,
                    "length": street.get_complete_length(),
                    "geometry": geom,
                }

                if aggregation == "street":
                    row.update(street_attr_plain)
                else:  # segment‑level
                    for attr_name, point_values in street.point_attributes.items():
                        values_here: List[Any] = []
                        n_points: int = len(street.sampling_points)
                        for k in range(0, n_points):
                            pt = street.sampling_points[k]
                            if _point_on_segment(pt, geom, distance_tol):
                                if k < len(point_values):
                                    values_here.append(point_values[k])
                        if len(values_here) == 0:
                            row[attr_name] = None
                        else:
                            method = self._attribute_methods.get(attr_name, default_method)
                            row[attr_name] = _aggregate(values_here, method)

                rows.append(row)

        # Build & write GeoDataFrame --------------------------------------------
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
        gdf.to_file(output_path, driver="GeoJSON")
