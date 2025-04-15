import osmnx as ox
from typing import List
import numpy as np
import math
import networkx as nx
from shapely.geometry import Point, LineString


def project_point_onto_graph_edges(
    G: nx.MultiDiGraph,
    point: List[float]
) -> List[float]:
    """
    Projects a point onto the closest edge geometry in the graph.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Graph with 'geometry' attributes on edges.
    point : List[float]
        [lon, lat] coordinate.

    Returns
    -------
    projected_point : List[float]
        [lon, lat] of the closest point on the graph.
    """
    min_dist = float("inf")
    proj_coords = None
    p = Point(point)

    for _, _, _, data in G.edges(keys=True, data=True):
        geom = data.get("geometry", None)
        if geom is None:
            continue
        candidate = geom.interpolate(geom.project(p))
        dist = p.distance(candidate)
        if dist < min_dist:
            min_dist = dist
            proj_coords = [candidate.x, candidate.y]

    if proj_coords is None:
        raise ValueError("Could not project point onto any edge.")

    return proj_coords

def get_bearing(p1: List[float], p2: List[float]) -> float:
    """
    Calculate the compass bearing from point p1 to point p2.

    Parameters
    ----------
    p1 : List[float]
        [lon, lat] of the starting point
    p2 : List[float]
        [lon, lat] of the destination point

    Returns
    -------
    float
        Bearing in degrees (0–360)
    """
    lat1 = math.radians(p1[1])
    lat2 = math.radians(p2[1])
    diff_lon = math.radians(p2[0] - p1[0])

    x = math.sin(diff_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(diff_lon)

    initial_bearing = math.atan2(x, y)
    bearing = (math.degrees(initial_bearing) + 360) % 360

    return bearing

class PointDistanceCalculator:
    def get_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        return ox.distance.great_circle(lat1, lon1, lat2, lon2)


class Line:

    def __init__(self, point1:List[float], point2: List[float])->None:

        self.point1 = point1
        self.point2 = point2

    
    def __eq__(self, other:object)->bool:
        if not isinstance(other,Line):
            return NotImplemented
        

        return (
            self.point1[0] == other.point1[0] and
            self.point1[1] == other.point1[1] and
            self.point2[0] == other.point2[0] and
            self.point2[1] == other.point2[1]
        )
    

    def get_distance_to_point(self, 
                            point:List[float])-> float:

        a = np.array(self.point1)
        b = np.array(self.point2)
        p = np.array(point)

        ab = b - a
        ap = p - a

        denominator = np.dot(ab, ab)
        alpha = np.dot(ap, ap)/denominator if denominator != 0 else 0

        if(0<= alpha <=1):
            projection = a + alpha + ab
        
        else:
            projection = (a + b)/2
        
        return float(np.linalg.norm(p - projection))
    

    def get_point_projection(self, 
                            point:List[float]) -> np.ndarray:
        
        a = np.array(self.point1)
        b = np.array(self.point2)
        p = np.array(point)

        ab = b - a
        ap = p - a
        denominator = np.dot(ab, ab)
        alpha = np.dot(ab, ap)/denominator if denominator != 0 else 0

        projection = a + alpha*ab

        return projection
