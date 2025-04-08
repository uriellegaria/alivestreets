import os
from enum import Enum
from shapely.geometry import Polygon
import geopandas as gpd
import osmnx as ox
import logging

class PlaceType(Enum):
    PARKS = "parks"
    AMENITIES = "amenities"
    BUS_STOPS = "bus_stops"
    SCHOOLS = "schools"
    JOB_CENTERS = "job_centers"
    INSTITUTIONS = "institutions"
    HOMES = "homes"
    CENSUS_BLOCK = "census_block"
    SUBWAY = "subway"

def get_tags(place_type: PlaceType) -> dict:
    if place_type == PlaceType.PARKS:
        return {"leisure": "park"}

    elif place_type == PlaceType.SCHOOLS:
        return {"amenity": ["school", "university", "college"]}

    elif place_type == PlaceType.AMENITIES:
        return {"amenity": True}

    elif place_type == PlaceType.BUS_STOPS:
        return {"highway": "bus_stop"}

    elif place_type == PlaceType.JOB_CENTERS:
        return {"office": True, "amenity": "jobcentre"}

    elif place_type == PlaceType.INSTITUTIONS:
        return {
            "landuse": "institutional",
            "office": ["government", "ngo", "association"],
            "amenity": "social_facility"
        }

    elif place_type == PlaceType.HOMES:
        return {
            "building": [
                "house", "apartments", "detached", "terrace",
                "semidetached_house", "hut", "ger", "houseboat", "static_caravan"
            ]
        }

    elif place_type == PlaceType.SUBWAY:
        return {"railway": ["station", "halt", "platform"]}

    elif place_type == PlaceType.CENSUS_BLOCK:
        return {"boundary": "census"}

    return {}

def get_features_from_point(
    latitude: float,
    longitude: float,
    radius: float,
    place_type: PlaceType,
    crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame | None:
    tags = get_tags(place_type)
    logging.captureWarnings(True)
    try:
        gdf = ox.features_from_point((latitude, longitude), tags, dist=radius)
        gdf["geometry"] = gdf["geometry"].to_crs(crs).centroid
        return gdf
    except Exception as e:
        print(f"No features found for {place_type}: {e}")
        return None

def get_features_from_polygon(
    polygon: Polygon,
    place_type: PlaceType,
    crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame | None:
    logging.captureWarnings(True)
    tags = get_tags(place_type)
    try:
        gdf = ox.features_from_polygon(polygon, tags)
        gdf["geometry"] = gdf["geometry"].to_crs(crs).centroid
        return gdf
    except Exception as e:
        print(f"No features found for {place_type}: {e}")
        return None

def export_geojson(
    gdf: gpd.GeoDataFrame,
    path: str
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gdf = gdf.copy()
    gdf.to_file(path, driver="GeoJSON")

def export_all_from_polygon(
    output_dir: str,
    polygon: Polygon,
    verbose: bool = True
) -> None:
    for place_type in PlaceType:
        gdf = get_features_from_polygon(polygon, place_type)
        if gdf is not None and not gdf.empty:
            export_geojson(gdf[["geometry"]].copy(), os.path.join(output_dir, f"{place_type.value}.geojson"))
            if verbose:
                print(f"Exported {os.path.join(output_dir, f'{place_type.value}.geojson')}")

def export_all_from_point(
    output_dir: str,
    latitude: float,
    longitude: float,
    radius: float,
    verbose: bool = True
) -> None:
    for place_type in PlaceType:
        gdf = get_features_from_point(latitude, longitude, radius, place_type)
        if gdf is not None and not gdf.empty:
            export_geojson(gdf[["geometry"]].copy(), os.path.join(output_dir, f"{place_type.value}.geojson"))
            if verbose:
                print(f"Exported {os.path.join(output_dir, f'{place_type.value}.geojson')}")
