import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from typing import List, Optional, Any
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from contextily import add_basemap

def plot_trajectory_on_graph(
    G: nx.MultiDiGraph,
    trajectory: List[int],
    attribute_name: Optional[str] = None,
    ax: Optional[Any] = None,
    node_pos_key: str = "pos",
    default_color: str = "#C98F35",  # Burnt Gold for edges
    min_color: tuple = (58 / 255, 154 / 255, 217 / 255),
    max_color: tuple = (255 / 255, 77 / 255, 158 / 255),
    width: float = 3.0,
    node_size: float = 10, 
    edge_size: float = 3.0,
    alpha: float = 0.5, 
    node_color: str = "#3E7D3E",
    cmap=None, 
    zoom = False, 
    orientation = "horizontal",
    min_percentile: int = 0,      
    max_percentile: int = 100, 
    show_endpoints:bool = False, 
    color_startpoint:str = "#00b0b0", 
    color_endpoint:str = "#a1006b", 
    size_endpoints:int  = 20, 
    add_basemap: bool = False, 
    crs:str = "EPSG:4326",
    tile_url:str = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png",
    color_all_net: bool = False, 
    basemap_alpha = 0.8
) -> None:
    """
    Plot a trajectory over a graph using custom RGB color gradient based on an attribute.

    If `attribute_name` is None, will plot in solid default color.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True) if "x" in d and "y" in d}

    if add_basemap and crs != "EPSG:3857":
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
        pos = {n: transformer.transform(*xy) for n, xy in pos.items()}
        crs = "EPSG:3857" 

    # Draw background with fixed node/edge colors
    if color_all_net and attribute_name is not None:
        all_vals = [
            data.get(attribute_name)
            for _, _, data in G.edges(data=True)
            if data.get(attribute_name) is not None
        ]
        if all_vals:
            vmin, vmax = np.percentile(all_vals, [min_percentile, max_percentile])
            norm = Normalize(vmin=vmin, vmax=vmax)
            edge_colors = []
            for u, v, data in G.edges(data=True):
                val = data.get(attribute_name)
                if val is not None:
                    color = cmap(norm(val)) if cmap is not None else tuple(np.array(min_color) + norm(val) * (np.array(max_color) - np.array(min_color)))
                else:
                    color = default_color
                edge_colors.append(color)
            
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color=edge_colors,
                width=edge_size,
                alpha=alpha
            )
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_size=node_size,
                node_color=node_color,
                alpha=alpha
            )
        else:
            # fallback to fixed color if no valid values
            nx.draw(
                G,
                pos,
                ax=ax,
                node_size=node_size,
                node_color=node_color,
                edge_color=default_color,
                alpha=alpha,
                with_labels=False,
                width=edge_size
            )
    else:
        nx.draw(
            G,
            pos,
            ax=ax,
            node_size=node_size,
            node_color=node_color,
            edge_color=default_color,
            alpha=alpha,
            with_labels=False,
            width=edge_size
        )

    # Prepare edge data for trajectory
    edges = []
    values = []

    for i in range(len(trajectory) - 1):
        u = trajectory[i]
        v = trajectory[i + 1]
        if G.has_edge(u, v):
            for k in G[u][v]:
                attr = G[u][v][k].get(attribute_name) if attribute_name else None
                if attr is not None or attribute_name is None:
                    edges.append((u, v))
                    values.append(attr if attribute_name else None)
                    break  # Use first matching edge

    if not edges:
        return

    if attribute_name is None:
        # Solid color trajectory
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            edge_color=default_color,
            width=width,
            ax=ax
        )
    else:
        # Attribute-based color trajectory
        all_vals = [
        data.get(attribute_name) 
        for _, _, data in G.edges(data=True) 
        if data.get(attribute_name) is not None
        ]

        if not all_vals:
            return  # No valid attribute values found in the graph

        vmin, vmax = np.percentile(all_vals, [min_percentile, max_percentile])
        norm = Normalize(vmin=vmin, vmax=vmax)

        for (u, v), val in zip(edges, values):
            if val is None:
                color = default_color
            else:
                t = norm(val)
                if cmap is not None:
                    color = cmap(t)
                else:
                    color = tuple(np.array(min_color) + t * (np.array(max_color) - np.array(min_color)))
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                edge_color=[color],
                width=width,
                ax=ax
            )
    
    if show_endpoints and trajectory:
        x_start, y_start = pos[trajectory[0]]
        x_end, y_end = pos[trajectory[-1]]
        ax.scatter(x_start, y_start, c=color_startpoint, s=size_endpoints, zorder=10, edgecolors="white")
        ax.scatter(x_end, y_end, c=color_endpoint, s=size_endpoints, zorder=10, edgecolors="white")

    if zoom and len(edges) > 0:
        # Get all x, y coordinates of nodes used in the trajectory
        x_vals = []
        y_vals = []
        for node in trajectory:
            if node in pos:
                x, y = pos[node]
                x_vals.append(x)
                y_vals.append(y)

        if x_vals and y_vals:
            # Compute bounds with 5% padding
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)

            x_pad = (x_max - x_min) * 0.05
            y_pad = (y_max - y_min) * 0.05

            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    ax.set_title(f"Trajectory ({attribute_name})" if attribute_name else "Trajectory")
    ax.set_axis_off()
    ax.set_aspect('equal', adjustable='box')

    if attribute_name is not None and all_vals:

        cmap_obj = cmap if cmap is not None else LinearSegmentedColormap.from_list(
            "custom_gradient", [min_color, max_color]
        )
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])  # Required for colorbar

        cbar = plt.colorbar(sm, ax=ax, orientation=orientation, shrink=0.8, pad=0.01)
        cbar.set_label(attribute_name, fontsize=10)
    
    if add_basemap:
        import contextily as ctx
        # Try to use your custom tile_url as a provider, fallback to CartoDB Light if not supported
        try:
            import xyzservices
            provider = xyzservices.TileProvider(url=tile_url)
            ctx.add_basemap(ax, source=provider, crs=crs, alpha = basemap_alpha)
        except Exception:
            ctx.add_basemap(ax, source=tile_url, crs=crs, alpha = basemap_alpha)


def plot_attribute_time_series(
    trajectory: List[int],
    G: nx.MultiDiGraph,
    attribute_name: str,
    ax: Optional[Any] = None,
    color: str = "#FF4D9E",
    label: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot the time series of an edge attribute along a trajectory.

    Parameters
    ----------
    trajectory : List[int]
        Sequence of node IDs.
    G : nx.MultiDiGraph
        Graph containing the edge attribute.
    attribute_name : str
        Name of the edge attribute.
    ax : Optional[matplotlib.axes.Axes]
        Axis to draw on. If None, will create one.
    color : str
        Line color.
    label : Optional[str]
        Y-axis label.
    title : Optional[str]
        Plot title.
    """
    values = []
    for i in range(len(trajectory) - 1):
        u = trajectory[i]
        v = trajectory[i + 1]
        edge_value = None
        if G.has_edge(u, v):
            for k in G[u][v]:
                edge_value = G[u][v][k].get(attribute_name)
                if edge_value is not None:
                    break
        values.append(edge_value)

    x_vals = list(range(1, len(values) + 1))
    y_vals = values

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(x_vals, y_vals, marker="o", color=color, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel(label or attribute_name)
    ax.grid(True)
    if title:
        ax.set_title(title)