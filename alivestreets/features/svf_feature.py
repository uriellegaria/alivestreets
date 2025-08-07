from typing import Optional, List
import numpy as np
from ultralytics import YOLO
from alivestreets.features.street_view_feature import StreetViewFeatureExtractor
from PIL import Image
import cv2
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import requests


class Panoramic2FishEye:
    """
    Converts a panoramic image to a fisheye projection.
    """

    def __init__(self, img_width: int, img_height: int) -> None:
        """
        Parameters
        ----------
        img_width : int
            Width of the fisheye output.
        img_height : int
            Height of the fisheye output.
        """
        self.img_width: int = img_width
        self.img_height: int = img_height
        self.img: Optional[np.ndarray] = None
        self.radius: Optional[float] = None

    def set_panoramic(self, img: np.ndarray) -> None:
        """
        Set the panoramic image.

        Parameters
        ----------
        img : np.ndarray
            RGB panoramic image.
        """
        self.img = img

    def get_panoramic(self) -> Optional[np.ndarray]:
        """
        Returns the current panoramic image.

        Returns
        -------
        Optional[np.ndarray]
            The current panoramic image.
        """
        return self.img

    def convert_to_fisheye(self, radius_fraction: float = 1.0) -> np.ndarray:
        """
        Convert the panoramic image to a fisheye projection.

        Parameters
        ----------
        radius_fraction : float
            Fraction of the full image size to use as radius.

        Returns
        -------
        np.ndarray
            Fisheye-projected image.
        """
        if self.img is None:
            raise ValueError("Panoramic image not set.")

        self.radius = radius_fraction * min(self.img_width, self.img_height)
        mid_x: int = self.img_width // 2
        mid_y: int = self.img_height // 2

        panoramic_h, panoramic_w, _ = self.img.shape

        y_indices, x_indices = np.meshgrid(np.arange(self.img_height), np.arange(self.img_width), indexing="ij")
        distances = np.sqrt((x_indices - mid_x) ** 2 + (y_indices - mid_y) ** 2)

        valid_mask = distances < self.radius
        normalized_r = (distances / self.radius) * panoramic_h
        angles = np.arctan2(y_indices - mid_y, x_indices - mid_x)

        index1 = np.clip(normalized_r.astype(int), 0, panoramic_h - 1)
        index2 = np.clip(((angles + np.pi) * panoramic_w / (2 * np.pi)).astype(int), 0, panoramic_w - 1)

        fish_image = np.zeros((self.img_height, self.img_width, 3), dtype=self.img.dtype)
        fish_image[valid_mask] = self.img[index1[valid_mask], index2[valid_mask]]

        return fish_image
    

class SVFFeatureExtractor(StreetViewFeatureExtractor):
    def __init__(
        self,
        sky_class_id: int = None,
        confidence_threshold: float = 0,
        fisheye_width: int = 1000,
        fisheye_height: int = 1000,
        tile_size: int = 640,
        model_path: str = None,
    ) -> None:
        self.model_path: str = model_path
        self.sky_class_id: int = sky_class_id
        self.confidence_threshold: float = confidence_threshold
        self.fisheye_width: int = fisheye_width
        self.fisheye_height: int = fisheye_height
        self.tile_size: int = tile_size
        if(not model_path is None):
            try:
                self.model: YOLO = YOLO(model_path)
            except Exception as e:
                raise RuntimeError(f"Please input a valid model path or download with `download_model` method. Error: {e}")

    def get_masks(self, tile: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        results = self.model(tile, verbose=False, overlap_mask=True)
        masks: List[np.ndarray] = []
        confidences: List[float] = []

        if results[0].masks is not None:
            all_confidences = results[0].boxes.conf.numpy()
            class_ids = results[0].boxes.cls.numpy()
            for mask, class_id, conf in zip(results[0].masks.data, class_ids, all_confidences):
                if conf >= self.confidence_threshold and int(class_id) == self.sky_class_id:
                    masks.append(np.array(mask))
                    confidences.append(float(conf))

        return masks, confidences
    
    def download_model(self,
        model_path:str = "facades.pt") -> None:
        """
        downloads the model from hugging face if not provided.

        Parameters
        ----------
        model_path
            The path where the model will be saved. If not provided, it will be saved as "facades.pt".
        """

        URL = "https://huggingface.co/urilp4669/Facade_Segmentator/resolve/main/facades.pt"

        response = requests.get(URL, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded.")

            self.model_path = model_path
            self.model = YOLO(model_path)
            #For the default model the sky class id is 7
            self.sky_class_id = 7

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # No preprocessing needed, return as-is
        return image


    def compute(self, image: np.ndarray, debug:bool = False) -> dict[str, float]:
        """
        Computes Sky View Factor (SVF) from the input image.

        Parameters
        ----------
        image
            Input image as a numpy array.
        
        debug
            If true displays the masked image.
        
        Returns
        -------
        dict[str, float]
            A dictionary of the following form:
            {
                "SVF": Sky View Factor value,
                "sky_mask": Binary mask of the sky region,
                "confidences": List of confidence scores for detected sky regions
            }
        """
        original_h, original_w = image.shape[:2]
        num_tiles_x = (original_w + self.tile_size - 1) // self.tile_size
        num_tiles_y = (original_h + self.tile_size - 1) // self.tile_size

        sky_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        masked_image = np.zeros_like(image)
        all_confidences = []

        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                x_start = j * self.tile_size
                y_start = i * self.tile_size
                x_end = min(x_start + self.tile_size, original_w)
                y_end = min(y_start + self.tile_size, original_h)

                tile = image[y_start:y_end, x_start:x_end]

                # Pad tile if needed
                pad_x = self.tile_size - tile.shape[1]
                pad_y = self.tile_size - tile.shape[0]
                tile_padded = cv2.copyMakeBorder(tile, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                masks, confidences = self.get_masks(tile_padded)
                all_confidences.extend(confidences)

                if masks:
                    combined_tile_mask = np.maximum.reduce(masks)[:tile.shape[0], :tile.shape[1]]
                else:
                    combined_tile_mask = np.zeros((tile.shape[0], tile.shape[1]), dtype=np.uint8)

                extended_mask = combined_tile_mask[:, :, np.newaxis]
                masked_tile = tile * extended_mask

                sky_mask[y_start:y_end, x_start:x_end] = combined_tile_mask
                masked_image[y_start:y_end, x_start:x_end] = masked_tile
        
        kernel = np.ones((5, 5), np.uint8)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)

        masked_image = image.copy()
        masked_image[sky_mask == 0] = 0

        if(debug == True):
            plt.figure()
            plt.imshow(masked_image)    

        # Proceed to fisheye conversion
        fisheye_converter = Panoramic2FishEye(self.fisheye_width, self.fisheye_height)
        fisheye_converter.set_panoramic(masked_image)
        fish_image = fisheye_converter.convert_to_fisheye(radius_fraction=1)

        n, m = fish_image.shape[:2]
        mid_x, mid_y = m // 2, n // 2
        radius = fisheye_converter.radius

        y_indices, x_indices = np.ogrid[:n, :m]
        distances = np.sqrt((x_indices - mid_x) ** 2 + (y_indices - mid_y) ** 2)
        valid_mask = distances < radius

        total_pixels = np.count_nonzero(valid_mask)
        sky_pixels = np.count_nonzero(np.any(fish_image[valid_mask], axis=-1))

        svf = sky_pixels / total_pixels if total_pixels > 0 else 0.0


        return {"SVF": svf, "sky_mask":sky_mask, "confidences": all_confidences}
