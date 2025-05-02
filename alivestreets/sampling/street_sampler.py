from typing import List, Optional, Any, Literal
from alivestreets.sampling.street import Street
from alivestreets.sampling.geometry import PointDistanceCalculator
import numpy as np
import unicodedata
import math
import pandas as pd
from shapely.geometry import LineString
import geopandas as gpd
import fiona
import matplotlib.pyplot as plt
from alivestreets.features.feature_aggregation import aggregate_feature
from tqdm import tqdm


class StreetSampler:
    def __init__(self, max_points: int, min_points_per_street: int = 1) -> None:
        """
        Initialize a StreetSampler.

        Parameters
        ----------
        max_points : int
            Maximum number of sampling points to distribute.
        min_points_per_street : int, optional
            Minimum number of sampling points per street.
        """
        self.max_points: int = max_points
        self.min_points_per_street: int = min_points_per_street
        self.streets: List[Street] = []
        self.unknown_counter: int = 0
    

    def get_street(self, street_id:str, contains:bool = False, integer_name: bool = False)-> Optional[Street]:
        if integer_name:
            return next((s for s in self.streets if s.street_id == street_id), None)

        street_id = unicodedata.normalize("NFC", street_id)

        for street in self.streets:
            sid = unicodedata.normalize("NFC", street.street_id)
            if not contains and sid == street_id:
                return street
            if contains and (street_id in sid or sid in street_id):
                return street

        return None
    

    def open_streets_from_gdf(self, gdf: gpd.GeoDataFrame) -> None:
        geom_elements = gdf["geometry"]
        street_names = list(gdf["name"])
        segment_lengths = list(gdf["length"])
        n_segments = len(geom_elements)

        for i in range(n_segments):
            segment = geom_elements[i]
            name = street_names[i]
            length = segment_lengths[i]

            # Handle missing or weird name values
            if name is None or (isinstance(name, float) and pd.isna(name)) or (isinstance(name, list) and len(name) == 0):
                name = f"unnamed_{self.unknown_counter}"
                self.unknown_counter += 1
            elif isinstance(name, list):
                name = name[0]

            if name not in [s.street_id for s in self.streets]:
                street = Street(name)
                self.streets.append(street)
            else:
                street = self.get_street(name)

            segment_points = [[pt[0], pt[1]] for pt in segment.coords]
            street.add_segment(segment_points)
    
    def sample_streets_no_intersections(self) -> None:
        """
        Sample points along the streets proportionally to their lengths,
        avoiding placing points at segment intersections.
        """
        weights = []
        for street in self.streets:
            weights.append(street.get_complete_length())

        weights = np.array(weights) / sum(weights)

        for i, street in enumerate(self.streets):
            n_points = int(weights[i] * self.max_points)
            if n_points < self.min_points_per_street:
                n_points = self.min_points_per_street

            sampled_points = street.sample_n_intermediate_points(n_points)
            street.set_sampling_points(sampled_points)
    

    def get_all_sampling_points(self) -> List[List[float]]:
        """
        Collect all sampling points from all streets.

        Returns
        -------
        List[List[float]]
            Flattened list of [lon, lat] sampling points.
        """
        points = []
        for street in self.streets:
            points.extend(street.sampling_points)
        return points
    

    def project_point(self, point: List[float]) -> tuple[List[float], Street]:
        """
        Project a given point to its closest location on any street.
        Parameters
        ----------
        point : List[float]
            [lon, lat] coordinate.

        Returns
        -------
        Tuple[List[float], Street]
            The projected point and the street it belongs to.
        """
        closest_street = None
        min_distance = float("inf")

        for street in self.streets:
            distance = street.get_distance_to_point(point)
            if distance < min_distance:
                min_distance = distance
                closest_street = street

        projected_point = closest_street.project_point_into_street(point)
        return projected_point, closest_street
    

    def sample_with_projected_locations(
    self,
    locations: List[List[float]],
    qualities: Optional[List[float]] = None,
    quality_threshold: float = 0.8) -> None:
        """
        Add sampling points by projecting external locations onto nearby streets.

        Parameters
        ----------
        locations : List[List[float]]
            Candidate [lon, lat] coordinates.
        qualities : Optional[List[float]], optional
            Quality score for each location (0–1). If given, will be thresholded.
        quality_threshold : float, optional
            Minimum quality required to keep a projected point.
        """
        use_quality = qualities is not None and len(qualities) == len(locations)

        for i, location in enumerate(tqdm(locations, desc="Projecting Locations")):
            projected_point, street = self.project_point(location)
            if use_quality:
                if qualities[i] > quality_threshold:
                    street.sampling_points.append(projected_point)
            else:
                street.sampling_points.append(projected_point)

    def restrict_distance_between_sampling_points(self, min_dist: float = 2.0) -> None:
        """
        Enforce a minimum distance between sampling points on each street.

        Parameters
        ----------
        min_dist : float
            Minimum allowed distance (in meters) between consecutive points.
        """
        calculator = PointDistanceCalculator()

        for street in self.streets:
            points = sorted(street.sampling_points, key=lambda x: (x[0], x[1]))
            if not points:
                continue

            filtered = [points[0]]
            for pt in points[1:]:
                last = filtered[-1]
                d = calculator.get_distance(pt[1], pt[0], last[1], last[0])
                if d >= min_dist:
                    filtered.append(pt)

            street.sampling_points = filtered
    
    def tag_streets(
    self,
    attribute_name: str,
    values: list[Any],
    method: Literal["mean", "sum", "count", "min", "max", "median"] = "mean"
) -> None:
        """
        Tag sampling point values to their streets and automatically aggregate them.

        Parameters
        ----------
        attribute_name : str
            The name of the feature (e.g., "accessibility").
        values : list
            One value per sampling point, in the same global order.
        method : str
            Aggregation method to produce a street-level value.
        """
        index = 0
        for street in self.streets:
            n = len(street.sampling_points)
            street_values = values[index:index + n]
            street.set_point_attribute_values(attribute_name, street_values)
            index += n

        if index != len(values):
            raise ValueError("Number of values does not match total sampling points.")

        aggregate_feature(self, attribute_name, method)
    
    

    def get_number_of_sampling_points(self) -> int:
        """
        Return total number of sampling points across all streets.

        Returns
        -------
        int
        """
        return len(self.get_all_sampling_points())
    
    def tag_streets_raw(self, attribute_name: str, values: list[Any]) -> None:
        if len(values) != len(self.streets):
            raise ValueError("Number of values must match number of streets.")
        for street, value in zip(self.streets, values):
            street.set_attribute_value(attribute_name, value)
    

    def get_street_of_nth_point(self, n: int) -> Optional[tuple[Street, int]]:
        """
        Given a global index n, find which street it belongs to and its index within that street.

        Parameters
        ----------
        n : int
            Index in the global sampling point list.

        Returns
        -------
        Tuple[Street, int] or None
        """
        counter = 0
        for street in self.streets:
            n_points = len(street.sampling_points)
            if counter + n_points > n:
                return street, n - counter
            counter += n_points
        return None
    

    def print_street_names(self) -> None:
        """
        Print the name of every street in the sampler.
        """
        for street in self.streets:
            print(street.street_id)
    

    def print_sampling(self) -> None:
        """
        Print the number of sampling points for each street.
        """
        for street in self.streets:
            print(f"{street.street_id}: {len(street.sampling_points)} points")
    
    def print_attributes(self, attribute_name: str) -> None:
        """
        Print the value of a given attribute for each street.
        Notice that if the attribute has not been registered it will be
        None for all streets.

        Parameters
        ----------
        attribute_name : str
            The attribute name to look up.
        """
        for street in self.streets:
            value = street.get_attribute_value(attribute_name)
            print(f"{street.street_id}: {value}")
    

    def euclidean_distance(self, point: List[float]) -> float:
        """
        Compute the Euclidean distance from (0, 0) to a given point.

        Parameters
        ----------
        point : List[float]
            [lon, lat] coordinate.

        Returns
        -------
        float
            Euclidean distance (not great-circle).
        """
        return np.sqrt(point[0]**2 + point[1]**2)


    def unknown_street_has_points(self, segment_points: List[List[float]]) -> Optional[Street]:
        """
        Check if any 'unnamed' street already contains the given segment.

        Parameters
        ----------
        segment_points : List[List[float]]
            List of [lon, lat] points in the segment.

        Returns
        -------
        Street or None
        """
        for street in self.streets:
            if "unnamed" in street.street_id:
                if segment_points in street.street_segments or segment_points[::-1] in street.street_segments:
                    return street
        return None
    

    def open_streets(self, path: str) -> None:
        """
        Load a street GeoJSON and populate the sampler.

        Parameters
        ----------
        path : str
            Path to the network GeoJSON file.
        """
        with fiona.open(path) as src:
            features = list(src)
            for f in features:
                props = f['properties']
                if 'name' in props:
                    name = props['name']
                    if isinstance(name, list):
                        props['name'] = name  # defer flattening
                else:
                    props['name'] = None
            geojson_dict = {
                "type": "FeatureCollection",
                "features": features
            }

        gdf = gpd.GeoDataFrame.from_features(geojson_dict)
        self.open_streets_from_gdf(gdf)
    

    def draw_sampling_scheme(
    self,
    width: int = 10,
    height: int = 10,
    point_color: str = "#FF4D9E",
    node_size: float = 1.0,
    edge_size: float = 1.0,
    edge_color: str = "#3A9AD9", 
    title: str = "Sampling Scheme"
) -> None:
        """
        Visualize the street network and sampling points.

        Parameters
        ----------
        width : int
            Width of the plot (in inches).
        height : int
            Height of the plot (in inches).
        point_color : str
            Color of sampling points.
        node_size : float
            Marker size for sampling points.
        edge_size : float
            Line width for street segments.
        edge_color : str
            Color of street lines.
        """
        plt.figure(figsize=(width, height))

        for street in self.streets:
            for segment in street.street_segments:
                for i in range(len(segment) - 1):
                    p1 = segment[i]
                    p2 = segment[i + 1]
                    plt.plot(
                        [p1[0], p2[0]], [p1[1], p2[1]],
                        color=edge_color,
                        linewidth=edge_size
                    )

            for point in street.sampling_points:
                plt.plot(
                    point[0], point[1],
                    marker="o",
                    color=point_color,
                    markersize=node_size
                )

        plt.axis("equal")
        plt.axis("off")
        plt.title(title)

    
    


    


    

    

    

    


    

    
    

    
    
    

    




