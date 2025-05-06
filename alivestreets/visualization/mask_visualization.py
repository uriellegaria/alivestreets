

from __future__ import annotations

from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgb


class TransparentMaskVisualizer:
    """Overlay segmentation masks on an image with adjustable transparency."""

    @staticmethod
    def _transparent_mask(
        mask: np.ndarray, color: Tuple[float, float, float], alpha: float
    ) -> np.ndarray:
        """Return an RGBA image where mask==1 pixels get `color` and alpha."""
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[mask == 1, :3] = color  # RGB (already 0-1 range)
        rgba[mask == 1, 3] = alpha   # A
        return rgba

    @staticmethod
    def _centroid(mask: np.ndarray) -> Tuple[float, float]:
        """Centroid (row, col) of a binary mask."""
        ys, xs = np.nonzero(mask)
        return float(ys.mean()), float(xs.mean())

    def visualize(
        self,
        image: np.ndarray,                    # H×W×3 (uint8 0-255 or float 0-1)
        masks: Sequence[np.ndarray],          # each H×W binary
        labels: Sequence[str],                # one per mask
        *,
        figsize: Tuple[int, int] = (7, 7),
        title: str = "",
        colors: Sequence[str] | None = None,  # hex strings or matplotlib-style
        alpha: float = 0.6,
        font_size: int = 12,
        font_color: str = "#ffffff",
        save_path: str | None = None,
        dpi: int = 600,
    ) -> None:
        """
        Plot `image` with semi-transparent `masks` and their `labels`.

        If `colors` is None, distinct hues are pulled from seaborn's tab20/20b/20c
        palettes, giving up to 60 visually separable colors.
        """
        if len(masks) != len(labels):
            raise ValueError("`masks` and `labels` must have the same length.")

        if colors is None:
            palette: List[Tuple[float, float, float]] = (
                sns.color_palette("tab20", 20)
                + sns.color_palette("tab20b", 20)
                + sns.color_palette("tab20c", 20)
            )
            colors_rgb: List[Tuple[float, float, float]] = palette[: len(masks)]
        else:
            # convert supplied colors to RGB 0-1 tuples
            colors_rgb = [to_rgb(c) for c in colors]

        plt.figure(figsize=figsize)
        plt.imshow(image / 255.0 if image.dtype != float or image.max() > 1 else image)

        for mask, label, rgb in zip(masks, labels, colors_rgb):
            overlay = self._transparent_mask(mask.astype(bool), rgb, alpha)
            plt.imshow(overlay)
            cy, cx = self._centroid(mask)
            plt.text(
                cx,
                cy,
                label,
                ha="center",
                va="center",
                fontsize=font_size,
                color=font_color,
            )

        if title:
            plt.title(title)
        plt.axis("off")

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        plt.show()