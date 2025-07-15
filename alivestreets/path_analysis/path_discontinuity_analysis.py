import networkx as nx
from pyproj import Geod
from networkx.exception import NetworkXNoPath
from alivestreets.path_analysis.path_features import get_path_feature_metrics
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from alivestreets.path_analysis.path_features import get_path_average_absolute_difference
from tqdm import tqdm

geod = Geod(ellps="WGS84")  # Accurate earth model

from heapq import heappush, heappop

def _k_shortest_paths_multigraph(
    graph: nx.MultiDiGraph,
    source: int,
    target: int,
    k: int,
    weight: str = "length",
):
    """
    Yen’s k-shortest simple paths that works directly on a MultiDiGraph.
    Returns a list whose first element is the true shortest path.
    """
    def path_cost(path):
        return sum(
            min(data.get(weight, 1) for data in graph[u][v].values())
            for u, v in zip(path, path[1:])
        )

    try:
        initial = nx.shortest_path(graph, source, target, weight=weight)
    except NetworkXNoPath:
        return []

    accepted_paths = [initial]
    candidate_heap = []  # (cost, path)

    for _ in range(1, k):
        for i in range(len(accepted_paths[-1]) - 1):
            spur_node = accepted_paths[-1][i]
            root_path = accepted_paths[-1][: i + 1]

            removed_edges = []
            for p in accepted_paths:
                if len(p) > i and p[: i + 1] == root_path:
                    u, v = p[i], p[i + 1]
                    if not graph.has_edge(u, v):
                        continue
                    for key, data in list(graph[u][v].items()):
                        removed_edges.append((u, v, key, data))
                        graph.remove_edge(u, v, key)

            try:
                spur_path = nx.shortest_path(graph, spur_node, target, weight=weight)
                total_path = root_path[:-1] + spur_path
            except NetworkXNoPath:
                total_path = None

            # restore edges immediately, then compute cost
            for u, v, key, data in removed_edges:
                graph.add_edge(u, v, key, **data)

            if total_path is not None and total_path not in accepted_paths:
                heappush(candidate_heap, (path_cost(total_path), total_path))

        if not candidate_heap:
            break
        _, next_path = heappop(candidate_heap)
        accepted_paths.append(next_path)

    return accepted_paths

def get_trajectories_from_origins_to_endpoints(
    graph: nx.MultiDiGraph, 
    origin_points: List[Tuple[float, float]], 
    destination_points: List[Tuple[float, float]],
    weight: str = "length",
    min_length = 0,
    max_alternatives = 0,
    max_alternative_dist = 0
) -> List[Dict]:
    """
    Finds shortest path trajectories for all origin → destination combinations.

    Returns:
    - List of dicts with keys:
        - 'origin_coordinates'
        - 'destination_coordinates'
        - 'origin_node'
        - 'destination_node'
        - 'trajectory' (list of nodes)
    """
    results = []

    node_coords = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}

    for origin in tqdm(origin_points):
        ox, oy = origin
        origin_node = min(
            node_coords,
            key=lambda n: geod.line_length([ox, node_coords[n][0]], [oy, node_coords[n][1]])
        )

        for destination in destination_points:
            dx, dy = destination
            destination_node = min(
                node_coords,
                key=lambda n: geod.line_length([dx, node_coords[n][0]], [dy, node_coords[n][1]])
            )

            try:
                path   = nx.shortest_path(graph, origin_node, destination_node, weight=weight)
                length = nx.shortest_path_length(graph, origin_node, destination_node, weight=weight)
            except NetworkXNoPath:
                path, length = [], 0

            if length >= min_length:
                results.append({
                    "origin_coordinates": origin,
                    "destination_coordinates": destination,
                    "origin_node": origin_node,
                    "destination_node": destination_node,
                    "trajectory": path
                })

            # ── spaced-out alternatives (only if a shortest path exists) ─────
            if (
                path
                and max_alternatives > 0
                and max_alternative_dist > 0
            ):
                alt_paths = _k_shortest_paths_multigraph(
                    graph,
                    origin_node,
                    destination_node,
                    k=max_alternatives + 1,   # +1 because the first is the shortest
                    weight=weight,
                )[1:]                        # skip the true shortest

                shortest_len   = length
                shortest_edges = set(zip(path, path[1:]))

                alt_kept = 0
                for alt in alt_paths:
                    alt_len = sum(
                        min(d.get(weight, 1) for d in graph[u][v].values())
                        for u, v in zip(alt, alt[1:])
                    )
                    if alt_len < min_length:          # ← add this guard
                        continue

                    if alt_len - shortest_len > max_alternative_dist:
                        break   # further paths will only be longer (Yen order)

                    overlap = len(shortest_edges & set(zip(alt, alt[1:]))) / len(shortest_edges)
                    if overlap > 0.80:
                        continue  # too similar to shortest

                    results.append({
                        "origin_coordinates": origin,
                        "destination_coordinates": destination,
                        "origin_node": origin_node,
                        "destination_node": destination_node,
                        "trajectory": alt
                    })
                    alt_kept += 1
                    if alt_kept >= max_alternatives:
                        break

    return results

def assign_edge_discontinuity(
    graph: nx.MultiDiGraph,
    origin_points: List[Tuple[float, float]],
    destination_points: List[Tuple[float, float]],
    attribute_name: str,
    weight: str = "length",
    min_length = 0,
    max_alternatives = 0,
    max_alternative_dist = 0, 
    output_var_name = "discontinuity"
) -> None:
    """
    Assigns a 'discontinuity' metric to each edge in the graph based on the average absolute difference
    of the specified attribute across all trajectories that traverse the edge.

    Parameters:
    - graph: The street network as a MultiDiGraph.
    - origin_points: List of (lon, lat) tuples for origins.
    - destination_points: List of (lon, lat) tuples for destinations.
    - attribute_name: The name of the edge attribute to analyze.
    - weight: Edge attribute to use for pathfinding (default is 'length').
    """
    # Initialize a dictionary to collect discontinuity values for each edge
    edge_discontinuities = defaultdict(list)
    # Generate trajectories
    trajectories = get_trajectories_from_origins_to_endpoints(
        graph, 
        origin_points, 
        destination_points, 
        weight, 
        min_length = min_length,
        max_alternatives = max_alternatives, 
        max_alternative_dist = max_alternative_dist
    )

    for traj in trajectories:
        path = traj["trajectory"]
        if len(path) < 2:
            continue  # Skip if the path is too short

        # Compute the average absolute difference for the trajectory
        avg_abs_diff = get_path_average_absolute_difference(path, graph, attribute_name)
        if avg_abs_diff is None:
            continue  # Skip if the attribute data is missing

        # Assign the discontinuity value to each edge in the path
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            # Since it's a MultiDiGraph, there might be multiple edges between u and v
            if graph.has_edge(u, v):
                for key in graph[u][v]:
                    edge_discontinuities[(u, v, key)].append(avg_abs_diff)


    for (u, v, key), diffs in edge_discontinuities.items():
        if diffs:
            mean_discontinuity = sum(diffs) / len(diffs)
            graph[u][v][key][output_var_name] = mean_discontinuity
        else:
            graph[u][v][key][output_var_name] = None