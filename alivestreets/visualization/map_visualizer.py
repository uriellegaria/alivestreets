import matplotlib.pyplot as plt
from typing import Literal
import contextily as ctx
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pyproj import Transformer
from matplotlib.colors import LinearSegmentedColormap

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
        none_color = "gray"
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
    clipped_max_value = None, 
    clipped_min_value = None, 
    none_color = "gray"
) -> None:
        if not self.network_layers:
            raise ValueError("No street networks added to visualize.")

        transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True) if transform_coords else None
        norm = Normalize(vmin=self.global_min, vmax=self.global_max)

        for layer in self.network_layers:
            sampler = layer["sampler"]
            attr = layer["attribute"]
            cmin = np.array(layer["min_color"])
            cmax = np.array(layer["max_color"])
            width = layer["edge_width"]
            alpha = layer["alpha"]

            for street in sampler.streets:
                value = street.attributes.get(attr, None)
                if value is None:
                    color = none_color
                else:
                    t = (value - self.global_min) / (self.global_max - self.global_min) if self.global_max != self.global_min else 0
                    color = tuple(cmin + t * (cmax - cmin))

                for segment in street.street_segments:
                    coords = segment
                    if transformer:
                        coords = [transformer.transform(x, y) for x, y in coords]

                    x, y = zip(*coords)
                    self.ax.plot(x, y, color=color, linewidth=width, alpha=alpha)

        if colorbar:
            from matplotlib.cm import ScalarMappable
            cmap = self._create_colormap(cmin, cmax)
            sm = ScalarMappable(norm=norm, cmap=cmap)
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
    

