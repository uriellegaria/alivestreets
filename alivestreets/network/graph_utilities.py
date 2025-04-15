import networkx as nx
from shapely.geometry import LineString, Point
from alivestreets.sampling.street_sampler import StreetSampler
from alivestreets.sampling.geometry import PointDistanceCalculator
from typing import Tuple, List, Dict
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


