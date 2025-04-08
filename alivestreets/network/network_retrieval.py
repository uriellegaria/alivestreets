import os
import osmnx as ox
import pandas as pd
import networkx as nx
from shapely.geometry import Polygon
import geopandas as gpd

def get_graph_from_point(
        lat:float, 
        lon:float, 
        radius:float, 
        network_type:str = "walk", 
        simplify:bool = True)->nx.MultiDiGraph:

    """
    Obtains the graph from a given (lat, lon) pair using OSMnx.
    """
    
    return ox.graph.graph_from_point((lat, lon), dist = radius, dist_type = "bbox", network_type = network_type, simplify = simplify)


def get_graph_from_polygon(
        polygon:Polygon,
        network_type: str = "walk", 
        simplify: bool = True
)-> nx.MultiDiGraph:
    """
    Obtains a graph from a given shapely polygon
    """
    return ox.graph.graph_from_polygon(polygon, simplify = simplify, network_type = network_type)


def clean_dataframe(
        gdf:gpd.GeoDataFrame
)->gpd.GeoDataFrame:
    
    """
    Turns the columns of a dataframe that are lists into a string
    """
    
    for column in gdf.columns:
        gdf[column] = gdf[column].apply(
            lambda x: str(x) if isinstance(x, list) else x
        )
    
    return gdf


def graph_to_dataframe(
        graph: nx.MultiDiGraph)->gpd.GeoDataFrame:
    
    _, edges = ox.graph_to_gdfs(graph)
    edges["geometry"] = edges.normalize()
    edges = edges.drop_duplicates("geometry")
    edges = clean_dataframe(edges)
    gdf = edges[['name', 'length', 'geometry']].copy()

    return gdf

def export_graph_geojson(
    graph: nx.MultiDiGraph,
    path: str,
    verbose: bool = True
) -> None:
    _, edges = ox.graph_to_gdfs(graph)
    edges['geometry'] = edges.normalize()
    edges = edges.drop_duplicates('geometry')

    if 'name' not in edges.columns or pd.isnull(edges['name']).all():
        unnamed_counter = 1

        def assign_name(row):
            nonlocal unnamed_counter
            row['name'] = f"Unnamed_{unnamed_counter}"
            unnamed_counter += 1
            return row

        edges = edges.apply(assign_name, axis=1)

    # 🟢 FIX: copy the slice before modifying
    gdf = edges[['name', 'length', 'geometry']].copy()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    gdf.to_file(path, driver='GeoJSON')

    if verbose:
        print(f"Exported: {path}")
    
    
