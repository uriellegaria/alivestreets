import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from typing import List, Optional, Any
from matplotlib.colors import Normalize

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
    node_size: int = 10, 
    edge_size: float = 3.0,
    alpha: float = 0.5, 
    node_color: str = "#3E7D3E"
) -> None:
    """
    Plot a trajectory over a graph using custom RGB color gradient based on an attribute.

    If `attribute_name` is None, will plot in solid default color.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Position lookup
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True) if "x" in d and "y" in d}

    # Draw background with fixed node/edge colors
    nx.draw(
        G,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,       # Deep Void Purple
        edge_color=default_color,   # Burnt Gold
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
        finite_vals = [v for v in values if v is not None]
        if not finite_vals:
            return
        vmin, vmax = min(finite_vals), max(finite_vals)
        norm = Normalize(vmin=vmin, vmax=vmax)

        for (u, v), val in zip(edges, values):
            if val is None:
                color = default_color
            else:
                t = norm(val)
                color = tuple(np.array(min_color) + t * (np.array(max_color) - np.array(min_color)))
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                edge_color=[color],
                width=width,
                ax=ax
            )

    ax.set_title(f"Trajectory ({attribute_name})" if attribute_name else "Trajectory")
    ax.set_axis_off()

    if attribute_name is None:
        # Solid color path
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=default_color, width=width, ax=ax)
    else:
        # Color path with custom RGB scale
        finite_vals = [v for v in values if v is not None]
        if not finite_vals:
            return
        vmin, vmax = min(finite_vals), max(finite_vals)
        norm = Normalize(vmin=vmin, vmax=vmax)

        for (u, v), val in zip(edges, values):
            if val is None:
                color = default_color
            else:
                t = norm(val)
                color = tuple(np.array(min_color) + t * (np.array(max_color) - np.array(min_color)))
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=[color], width=width, ax=ax)

    ax.set_title(f"Trajectory ({attribute_name})" if attribute_name else "Trajectory")
    ax.set_axis_off()

    if attribute_name is not None and finite_vals:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("custom_gradient", [min_color, max_color])
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # Required for colorbar

        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, pad=0.01)
        cbar.set_label(attribute_name, fontsize=10)


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