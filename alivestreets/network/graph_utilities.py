import networkx as nx
from shapely.geometry import LineString, Point
from alivestreets.sampling.street_sampler import StreetSampler
from alivestreets.sampling.geometry import PointDistanceCalculator
from typing import Tuple, List, Dict, Optional, Literal, Any
from collections import defaultdict
import numpy as np

def build_graph_from_sampler(
    sampler:StreetSampler, 
    precision: int = 7
)-> nx.MultiDiGraph:
    #Initialize the network
    G = nx.MultiDiGraph()
    calculator = PointDistanceCalculator()
    coord_to_node: Dict[Tuple[float, float], int] = {}
    node_counter = 0
    def get_node_id(coord:List[float])->int:
        nonlocal node_counter 
        rounded = tuple(np.round(coord, precision))
        #If the node is not still on the dictionary add it
        if(rounded not in coord_to_node):
            coord_to_node[rounded] = node_counter
            G.add_node(node_counter, x = rounded[0], y = rounded[1], geometry = Point(rounded))
            node_counter += 1
        
        return coord_to_node[rounded]
    
    for street in sampler.streets:
        for segment in street.street_segments:
            for i in range(len(segment)-1):
                p1 = segment[i]
                p2 = segment[i+1]
                u = get_node_id(p1)
                v = get_node_id(p2)
                geom = LineString([p1, p2])
                #Get the length of the connection
                length = calculator.get_distance(p1[1], p1[0], p2[1], p2[0])
                G.add_edge(u,v, key = 0, geometry = geom, length = length, street_id = street.street_id)

    return G


def compute_segment_attribute_map(
    sampler: StreetSampler,
    attribute_name: str,
    aggregation: str = "mean",
    fallback_k: int = 3,
    precision: int = 7
) -> Dict[Tuple[str, str], Optional[float]]:
    """
    Maps each segment (as WKT) to an aggregated attribute value using sampling points.

    Parameters
    ----------
    sampler : StreetSampler
        The street sampler containing segments and sampling point attributes.
    attribute_name : str
        Name of the sampling point attribute to use.
    aggregation : str
        Aggregation method: "mean", "median", "sum", etc.
    fallback_k : int
        Number of nearest sampling points to use if a segment has none on it.
    precision : int
        Coordinate rounding precision for consistent matching.

    Returns
    -------
    Dict[(str, str), float]
        Mapping from (street_id, segment_wkt) to aggregated attribute value.
    """
    calculator = PointDistanceCalculator()
    segment_to_value: Dict[Tuple[str, str], Optional[float]] = {}

    for street in sampler.streets:
        points = street.sampling_points
        values = street.point_attributes.get(attribute_name, [])
        if len(points) != len(values):
            continue  # Skip if misaligned

        sampling = list(zip(points, values))

        for segment in street.street_segments:
            line = LineString(segment)
            segment_wkt = line.wkt

            # Find sampling points close to the segment
            segment_values = []
            for pt, val in sampling:
                if val is None:
                    continue
                projected = line.project(Point(pt))
                projected_point = line.interpolate(projected)
                dist = Point(pt).distance(projected_point)
                if dist < 1e-6:  # Tight tolerance
                    segment_values.append(val)

            if segment_values:
                agg = np.mean(segment_values) if aggregation == "mean" else \
                      np.median(segment_values) if aggregation == "median" else \
                      np.sum(segment_values) if aggregation == "sum" else \
                      None
                segment_to_value[(street.street_id, segment_wkt)] = agg
            else:
                # Fallback to k nearest points in street
                if not sampling:
                    segment_to_value[(street.street_id, segment_wkt)] = None
                    continue

                segment_center = line.interpolate(0.5, normalized=True)
                distances = [
                    (calculator.get_distance(pt[1], pt[0], segment_center.y, segment_center.x), val)
                    for pt, val in sampling if val is not None
                ]
                distances.sort()
                k_nearest = [val for _, val in distances[:fallback_k]]

                if k_nearest:
                    agg = np.mean(k_nearest) if aggregation == "mean" else \
                          np.median(k_nearest) if aggregation == "median" else \
                          np.sum(k_nearest) if aggregation == "sum" else \
                          None
                    segment_to_value[(street.street_id, segment_wkt)] = agg
                else:
                    segment_to_value[(street.street_id, segment_wkt)] = None

    return segment_to_value



def attach_sampler_segment_attributes_to_graph(
    G: nx.MultiDiGraph,
    sampler: StreetSampler,
    attribute_name: str,
    aggregation: Literal["mean", "sum", "median"] = "mean",
    precision: int = 7,
    fallback_k: int = 3
) -> None:
    calculator = PointDistanceCalculator()
    segment_to_values = defaultdict(list)

    for street in sampler.streets:
        attr_values = street.point_attributes.get(attribute_name, [])
        points = street.sampling_points

        if len(attr_values) != len(points):
            continue  # skip if mismatched

        for pt, value in zip(points, attr_values):
            if value is None:
                continue
            min_dist = float("inf")
            best_segment = None

            for segment in street.street_segments:
                line = LineString(segment)
                dist = line.distance(Point(pt))
                if dist < min_dist:
                    min_dist = dist
                    best_segment = segment

            if best_segment:
                seg_key = (
                    tuple(np.round(best_segment[0], precision)),
                    tuple(np.round(best_segment[-1], precision))
                )
                segment_to_values[seg_key].append(value)

    # Now tag each edge in G with the aggregated segment value or fallback
    for u, v, k, data in G.edges(keys=True, data=True):
        coord_u = (round(G.nodes[u]["x"], precision), round(G.nodes[u]["y"], precision))
        coord_v = (round(G.nodes[v]["x"], precision), round(G.nodes[v]["y"], precision))
        key = (coord_u, coord_v)
        key_rev = (coord_v, coord_u)

        value = segment_to_values.get(key) or segment_to_values.get(key_rev)

        if value is not None:
            if aggregation == "mean":
                G.edges[u, v, k][attribute_name] = float(np.mean(value))
            elif aggregation == "sum":
                G.edges[u, v, k][attribute_name] = float(np.sum(value))
            elif aggregation == "median":
                G.edges[u, v, k][attribute_name] = float(np.median(value))
        else:
            # Fallback: search for closest k sampling points on the same street
            street_id = data.get("street_id")
            street = sampler.get_street(street_id, integer_name=True)
            if not street:
                continue

            midpoint = data["geometry"].interpolate(0.5, normalized=True)
            dists = []
            values = street.point_attributes.get(attribute_name, [])
            for pt, val in zip(street.sampling_points, values):
                if val is None:
                    continue
                d = calculator.get_distance(midpoint.y, midpoint.x, pt[1], pt[0])
                dists.append((d, val))

            if dists:
                dists.sort()
                fallback_vals = [v for _, v in dists[:fallback_k]]
                if aggregation == "mean":
                    G.edges[u, v, k][attribute_name] = float(np.mean(fallback_vals))
                elif aggregation == "sum":
                    G.edges[u, v, k][attribute_name] = float(np.sum(fallback_vals))
                elif aggregation == "median":
                    G.edges[u, v, k][attribute_name] = float(np.median(fallback_vals))


def attach_sampler_street_attributes_to_graph(
    G: nx.MultiDiGraph,
    sampler: StreetSampler,
    attribute_name: str,
    aggregation: Literal["mean", "sum", "median"] = "mean",
    none_value: Any = None,
) -> None:
    """
    Tag every edge in `G` with a single value per street (identified via `street_id`).

    Parameters
    ----------
    G : nx.MultiDiGraph
        Graph whose edges already store 'street_id'.
    sampler : StreetSampler
        The sampler holding per-street sampling-point attributes.
    attribute_name : str
        Name of the sampling-point attribute to aggregate.
    aggregation : {"mean","sum","median"}
        How to combine point-level values into the street value.
    none_value
        What to write when a street has no valid values (`None` keeps attribute absent).
    """

    # --- 1.  build one aggregated value per street ---------------------------------
    street_value: Dict[str, float] = {}
    for street in sampler.streets:
        values = [v for v in street.point_attributes.get(attribute_name, []) if v is not None]
        if not values:
            if none_value is not None:
                street_value[street.street_id] = none_value
            continue

        if aggregation == "mean":
            agg = float(np.mean(values))
        elif aggregation == "sum":
            agg = float(np.sum(values))
        elif aggregation == "median":
            agg = float(np.median(values))
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        street_value[street.street_id] = agg

    for _, _, data in G.edges(data=True):
        sid = data.get("street_id")
        if sid in street_value:
            val = street_value[sid]
            if val is not None:
                data[attribute_name] = val


