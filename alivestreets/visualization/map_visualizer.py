import matplotlib.pyplot as plt
from typing import Literal
import contextily as ctx
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pyproj import Transformer
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm

class MapVisualizer:
    def __init__(self) -> None:
        self.fig, self.ax = None, None
        self.global_min = float("inf")
        self.global_max = float("-inf")
        self.network_layers = []  # list of dicts: {sampler, attr, style}
        self.point_layers = []    # optional later
        self.polygon_layers = []

    def add_street_sampler(
        self,
        street_sampler,
        attribute_name: str,
        variable_type: Literal["continuous"],
        min_color: tuple = (58 / 255, 154 / 255, 217 / 255) ,
        max_color: tuple = (255 / 255, 77 / 255, 158 / 255),
        edge_width: float = 1,
        alpha: float = 1.0,
        none_color = "gray", 
        cmap = None

    ) -> None:
        """
        Queue a StreetSampler layer to be drawn later.

        This supports stacking multiple samplers into one visualization.
        """
        # Compute value range for normalization later
        for street in street_sampler.streets:
            v = street.attributes.get(attribute_name, None)
            if v is not None:
                self.global_min = min(self.global_min, v)
                self.global_max = max(self.global_max, v)

        self.network_layers.append({
            "sampler": street_sampler,
            "attribute": attribute_name,
            "type": variable_type,
            "min_color": min_color,
            "max_color": max_color,
            "cmap":cmap,
            "edge_width": edge_width,
            "alpha": alpha,
        })
    

    def draw_networks(
    self,
    transform_coords: bool = True,
    crs: str = "EPSG:4326",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    colorbar_orientation: Literal["horizontal", "vertical"] = "vertical", 
    apply_log: bool = False, 
    none_color = "gray",
    min_percentile: int = 0,
    max_percentile: int = 100
) -> None:
        if not self.network_layers:
            raise ValueError("No street networks added to visualize.")

        # Collect all valid attribute values across layers
        all_values = [
            street.attributes.get(layer["attribute"], None)
            for layer in self.network_layers
            for street in layer["sampler"].streets
            if street.attributes.get(layer["attribute"], None) is not None and (street.attributes.get(layer["attribute"], None) > 0 if apply_log else True)
        ]

        # Compute percentile-based normalization limits
        p_initial, p_final = np.percentile(all_values, [min_percentile, max_percentile])
        if not apply_log:
            norm = Normalize(vmin=p_initial, vmax=p_final)
        else:
            norm = LogNorm(vmin=max(p_initial, 1e-3), vmax=p_final)

        transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True) if transform_coords else None

        for layer in self.network_layers:
            sampler = layer["sampler"]
            attr = layer["attribute"]
            cmin = np.array(layer["min_color"])
            cmax = np.array(layer["max_color"])
            cmap = layer.get("cmap", None) 
            width = layer["edge_width"]
            alpha = layer["alpha"]

            for street in sampler.streets:
                value = street.attributes.get(attr, None)
                if value is None or (apply_log and value <= 0):
                    color = none_color
                else:
                    t = norm(value)
                    t = np.clip(float(t), 0, 1)
                    if cmap is not None:
                        cmap_obj = plt.get_cmap(cmap)
                        color = cmap_obj(t)
                    else:
                        color = tuple(cmin + t * (cmax - cmin))

                for segment in street.street_segments:
                    coords = segment
                    if transformer:
                        coords = [transformer.transform(x, y) for x, y in coords]

                    x, y = zip(*coords)
                    self.ax.plot(x, y, color=color, linewidth=width, alpha=alpha)

        if colorbar:
            from matplotlib.cm import ScalarMappable
            cmap_obj = plt.get_cmap(cmap) if cmap is not None else self._create_colormap(cmin, cmax)
            sm = ScalarMappable(norm=norm, cmap=cmap_obj)
            self.fig.colorbar(sm, ax=self.ax, orientation=colorbar_orientation, label=colorbar_label, pad=0.01)

    def initialize_map(self, figsize: tuple[int, int] = (10, 10)) -> None:
        """
        Create a new Matplotlib figure and axis for plotting.
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        

    def add_basemap(
        self,
        zoom: int | Literal["auto"] = "auto",
        alpha: float = 1.0,
        tile_url: str = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png"
    ) -> None:
        """
        Add a contextily basemap to the map using a specified CRS.
        """
        if self.ax is None:
            raise ValueError("Call initialize_map() before adding a basemap.")

        ctx.add_basemap(
            self.ax,
            crs="EPSG:3857",
            zoom=zoom,
            source=tile_url,
            attribution=False,
            interpolation="sinc",
            alpha=alpha
        )
    

    def finalize_map(self, title: str = "", save_path: str | None = None) -> None:
        """
        Show and optionally save the final map.
        """
        if self.ax is None:
            raise ValueError("Call initialize_map() before finalizing the map.")

        if title:
            self.ax.set_title(title)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        if save_path is not None:
            self.fig.savefig(save_path, dpi=600, bbox_inches="tight")

        plt.show()
    
    
    def _create_colormap(self, min_color: tuple, max_color: tuple) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list("custom_cmap", [min_color, max_color])
    

