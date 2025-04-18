import networkx as nx
import numpy as np
import numpy.typing as npt
from shapely.geometry import Point, LineString
from typing import Optional, List
from alivestreets.sampling.geometry import PointDistanceCalculator
from tqdm import tqdm
from alivestreets.sampling.geometry import project_point_onto_graph_edges


def compute_geodesic_distance_to_amenities(
    point: list[float],
    amenities: list[list[float]],
) -> npt.NDArray[np.float64]:
    """
    Compute geodesic distances between a single point and multiple amenities.

    All coordinates are assumed to be in the form [longitude, latitude].

    Parameters
    ----------
    point : list of float
        The [lon, lat] coordinate of the reference point.
    amenities : list of [float, float]
        A list of [lon, lat] coordinates for amenity locations.

    Returns
    -------
    distances : np.ndarray
        Array of geodesic distances in meters.
    """
    calculator = PointDistanceCalculator()
    distances = np.array([
        calculator.get_distance(point[1], point[0], am[1], am[0])
        for am in amenities
    ], dtype=np.float64)

    return distances


def compute_network_distance_to_amenities(
    point: list[float],
    amenities: list[list[float]],
    G: nx.MultiDiGraph,
    weight: str = "length",
    precision: int = 7,
    undirected: bool = True
) -> list[float | None]:
        calculator = PointDistanceCalculator()
        distances: list[float | None] = []

        def cut_linestring_at_point(line: LineString, point: Point) -> tuple[LineString, LineString]:
            d = line.project(point)
            if d <= 0 or d >= line.length:
                return LineString([point, point]), line
            coords = list(line.coords)
            new_point = line.interpolate(d)
            split1 = []
            split2 = []
            total = 0
            for i in range(len(coords) - 1):
                seg = LineString([coords[i], coords[i + 1]])
                seg_len = seg.length
                if total + seg_len >= d:
                    split1.append(coords[i])
                    split1.append((new_point.x, new_point.y))
                    split2.append((new_point.x, new_point.y))
                    split2.extend(coords[i + 1:])
                    break
                else:
                    split1.append(coords[i])
                    total += seg_len
            return LineString(split1), LineString(split2)

        def add_virtual_node(G_: nx.MultiDiGraph, coord: list[float], prefix: str) -> Optional[str]:
            try:
                projected = project_point_onto_graph_edges(G_, coord)
            except ValueError:
                return None

            x, y = round(projected[0], precision), round(projected[1], precision)
            node_id = f"{prefix}_{x}_{y}"
            point_geom = Point(projected)

            if node_id in G_:
                return node_id

            G_.add_node(node_id, x=projected[0], y=projected[1], geometry=point_geom)

            for u, v, k, data in G_.edges(keys=True, data=True):
                geom = data.get("geometry", None)
                if geom is None:
                    continue
                if geom.distance(point_geom) < 1e-8:
                    G_.remove_edge(u, v, k)

                    geom1, geom2 = cut_linestring_at_point(geom, point_geom)
                    dist1 = calculator.get_distance(G_.nodes[u]["y"], G_.nodes[u]["x"], y, x)
                    dist2 = calculator.get_distance(G_.nodes[v]["y"], G_.nodes[v]["x"], y, x)

                    G_.add_edge(u, node_id, weight=dist1, geometry=geom1)
                    G_.add_edge(node_id, v, weight=dist2, geometry=geom2)

                    if undirected:
                        G_.add_edge(v, node_id, weight=dist2, geometry=LineString(list(geom2.coords)[::-1]))
                        G_.add_edge(node_id, u, weight=dist1, geometry=LineString(list(geom1.coords)[::-1]))
                    break

            return node_id

        for i, amenity in enumerate(amenities):
            G_aug = G.copy()
            if undirected:
                G_aug = G_aug.to_undirected()

            source_node = add_virtual_node(G_aug, point, prefix="source")
            target_node = add_virtual_node(G_aug, amenity, prefix=f"amenity{i}")

            if source_node is None or target_node is None:
                distances.append(None)
                continue

            try:
                d = nx.shortest_path_length(G_aug, source=source_node, target=target_node, weight=weight)
                distances.append(d)
            except nx.NetworkXNoPath:
                distances.append(None)

        return distances


def get_min_network_distances_to_amenities(
        origins: List[List[float]], 
        amenities: List[List[float]], 
        G: nx.MultiDiGraph,
        weight: str = "length",
        precision: int = 7, 
        undirected:bool = True
    )->List[float|None]:

    n_origins = len(origins)
    min_distances:List[float | None] = []

    for i in tqdm(range(0,n_origins)):
        point = origins[i]
        distances = compute_network_distance_to_amenities(point, 
        amenities, 
        G, 
        weight = weight, 
        precision = precision, 
        undirected = undirected)
    
        distances = [x for x in distances if not x is None]
        if(len(distances)>0):
            min_distances.append(np.min(distances))
        
        else:
            min_distances.append(None)
    
    return min_distances


def get_min_geodesic_distances_to_amenities(
    origins: List[List[float]], 
    amenities: List[List[float]]
)-> List[float|None]:

    min_distances:List[float|None] = []
    n_origins = len(origins)
    for i in tqdm(range(0,n_origins)):
        point = origins[i]
        distances = compute_geodesic_distance_to_amenities(point, amenities)
        distances = [x for x in distances if not x is None]

        if(len(distances)>0):
            min_distances.append(np.min(distances))
        else:
            min_distances.append(None)
    

    return min_distances







