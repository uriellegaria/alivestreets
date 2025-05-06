import os
import json
import numpy as np
from typing import List, Tuple, Optional
import cv2
from ultralytics.utils.downloads import attempt_download_asset  # or any GSV API lib you're using
import requests
import math
from typing import Tuple
from PIL import Image

def compute_bearing_between_locations(location1: Tuple[float, float], location2: Tuple[float, float]) -> float:
    """
    Computes the compass bearing from location1 to location2.

    Parameters
    ----------
    location1 : Tuple[float, float]
        The starting point as (latitude, longitude).
    location2 : Tuple[float, float]
        The destination point as (latitude, longitude).

    Returns
    -------
    float
        The compass bearing in degrees (0° = North, 90° = East).
    """
    lat1: float = math.radians(location1[0])
    lat2: float = math.radians(location2[0])
    diff_long: float = math.radians(location2[1] - location1[1])

    x: float = math.sin(diff_long) * math.cos(lat2)
    y: float = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff_long))

    initial_bearing: float = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing: float = (initial_bearing + 360) % 360

    return compass_bearing

class ImageStitcher:
    """
    A simple class to stitch multiple images into a panoramic image.
    """

    def stitch_images(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Stitches a list of images into a panorama.

        Parameters
        ----------
        images : List[np.ndarray]
            List of RGB images to stitch.

        Returns
        -------
        np.ndarray
            The stitched panoramic image.
        """
        cv_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
        stitcher = cv2.Stitcher_create()  
        stitcher.setRegistrationResol(-1)
        stitcher.setSeamEstimationResol(-1)
        stitcher.setCompositingResol(-1 )
        status, result = stitcher.stitch(cv_images)

        if status != cv2.Stitcher_OK:
            raise RuntimeError("Could not stitch images into a panorama.")

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result_rgb

class StreetViewImageCollector:

    def __init__(self, api_key: str, fov: int = 90, pitch: int = 0, size: Tuple[int, int] = (640, 640)) -> None:
        self.api_key: str = api_key
        self.fov: int = fov
        self.pitch: int = pitch
        self.size: Tuple[int, int] = size  # (width, height)
    

    def fetch_street_view_metadata(
    self,
    lat: float,
    lon: float
) -> dict:
        """
        Retrieve official Street View metadata from Google for a specific location.

        Parameters
        ----------
        lat : float
            Latitude of the location.
        lon : float
            Longitude of the location.

        Returns
        -------
        dict
            A dictionary containing the metadata returned by the Google Street View API.
            Includes keys like 'pano_id', 'date', 'status', 'location', and 'copyright'.
            If the location has no available imagery, the dictionary will include 'status' only.
        """
        url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lon}&key={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": f"HTTP_{response.status_code}"}

    def collect_panoramic_with_views(
    self,
    lat: float,
    lon: float,
    output_dir: str,
    num_views: int = 8,
    save_views: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Collect multiple views around a point and stitch into a panoramic image.

        Parameters
        ----------
        lat : float
            Latitude of the point.
        lon : float
            Longitude of the point.
        output_dir : str
            Directory where outputs will be saved.
        num_views : int
            Number of views to collect around the point.
        save_views : bool
            Whether to save individual views in a subfolder.

        Returns
        -------
        Tuple[np.ndarray, dict]
            The stitched panoramic image and metadata for the views.
        """
        os.makedirs(output_dir, exist_ok=True)
        views_dir = os.path.join(output_dir, "views")
        if save_views:
            os.makedirs(views_dir, exist_ok=True)

        delta_angle: float = 360.0 / num_views
        current_angle: float = 0.0
        images: List[np.ndarray] = []
        metadata: List[dict] = []

        overall_metadata = self.fetch_street_view_metadata(lat, lon)

        for i in range(num_views):
            url = f"https://maps.googleapis.com/maps/api/streetview?size={self.size[0]}x{self.size[1]}&location={lat},{lon}&heading={current_angle}&pitch={self.pitch}&fov={self.fov}&key={self.api_key}"
            response = requests.get(url, stream=True)
            img_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if image is None:
                raise RuntimeError(f"Failed to download or decode image at heading {current_angle}")

            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to RGB for stitching

            if save_views:
                view_path = os.path.join(views_dir, f"view_{i}.jpg")
                cv2.imwrite(view_path, image)

            metadata.append({
                "metadata":overall_metadata,
                "heading": current_angle,
                "pitch": self.pitch,
                "fov": self.fov,
                "size": self.size,
                "lat": lat,
                "lon": lon,
                "view_index": i
            })

            current_angle = (current_angle + delta_angle) % 360

        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Use ImageStitcher to stitch images into panorama
        stitcher = ImageStitcher()
        panoramic_image = stitcher.stitch_images(images)

        pano_path = os.path.join(output_dir, "panoramic.jpg")
        cv2.imwrite(pano_path, cv2.cvtColor(panoramic_image, cv2.COLOR_RGB2BGR))  

        return panoramic_image, {"metadata": metadata, "panoramic_path": pano_path}
    
    def collect_sidewalk_views(
    self,
    lat: float,
    lon: float,
    next_lat: float,
    next_lon: float,
    output_dir: str,
    mode: str = "both"
) -> Tuple[List[np.ndarray], List[dict]]:
        """
        Collect sidewalk-oriented views based on calculated bearing.

        Parameters
        ----------
        lat : float
            Latitude of the current point.
        lon : float
            Longitude of the current point.
        next_lat : float
            Latitude of the next point to compute bearing.
        next_lon : float
            Longitude of the next point.
        output_dir : str
            Directory where outputs will be saved.
        mode : str
            Which views to collect: "left", "right", or "both".

        Returns
        -------
        dict
            Metadata of the collected views.
        """
        os.makedirs(output_dir, exist_ok=True)
        views_dir = os.path.join(output_dir, "sidewalk_views")
        os.makedirs(views_dir, exist_ok=True)

        bearing: float = compute_bearing_between_locations((lat, lon), (next_lat, next_lon))
        metadata: List[dict] = []
        overall_metadata = self.fetch_street_view_metadata(lat, lon)

        if mode in ("left", "both"):
            left_heading: float = (bearing + 270) % 360
            url_left = f"https://maps.googleapis.com/maps/api/streetview?size={self.size[0]}x{self.size[1]}&location={lat},{lon}&heading={left_heading}&pitch={self.pitch}&fov={self.fov}&key={self.api_key}"
            response_left = requests.get(url_left, stream=True)
            img_array_left = np.asarray(bytearray(response_left.raw.read()), dtype=np.uint8)
            image_left = cv2.imdecode(img_array_left, cv2.IMREAD_COLOR)
            left_path = os.path.join(views_dir, "sidewalk_left.jpg")
            cv2.imwrite(left_path, image_left)
            metadata.append({
                "metadata":overall_metadata,
                "view": "left",
                "heading": left_heading,
                "lat": lat,
                "lon": lon,
                "path": left_path
            })

        if mode in ("right", "both"):
            right_heading: float = (bearing + 90) % 360
            url_right = f"https://maps.googleapis.com/maps/api/streetview?size={self.size[0]}x{self.size[1]}&location={lat},{lon}&heading={right_heading}&pitch={self.pitch}&fov={self.fov}&key={self.api_key}"
            response_right = requests.get(url_right, stream=True)
            img_array_right = np.asarray(bytearray(response_right.raw.read()), dtype=np.uint8)
            image_right = cv2.imdecode(img_array_right, cv2.IMREAD_COLOR)
            right_path = os.path.join(views_dir, "sidewalk_right.jpg")
            cv2.imwrite(right_path, image_right)
            metadata.append({
                "metadata":overall_metadata,
                "view": "right",
                "heading": right_heading,
                "lat": lat,
                "lon": lon,
                "path": right_path
            })

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        images = []
        image_right = None
        image_left = None
        if(os.path.exists(left_path) and os.path.exists(right_path)):
            img_right = np.array(Image.open(right_path))
            img_left = np.array(Image.open(left_path))
            images = [img_right, img_left]
        return images, metadata
    

    def collect_single_view(
    self,
    lat: float,
    lon: float,
    pitch:float,
    heading: float,
    output_dir: str,
    filename: str = "single_view.jpg"
) -> dict:
        """
        Collect a single view at a specific heading.

        Parameters
        ----------
        lat : float
            Latitude of the point.
        lon : float
            Longitude of the point.
        heading : float
            Heading direction for the view (0° = north).
        output_dir : str
            Directory where outputs will be saved.
        filename : str
            Name for the saved image file.

        Returns
        -------
        dict
            Metadata of the collected view.
        """
        os.makedirs(output_dir, exist_ok=True)

        url = f"https://maps.googleapis.com/maps/api/streetview?size={self.size[0]}x{self.size[1]}&location={lat},{lon}&heading={heading}&pitch={pitch}&fov={self.fov}&key={self.api_key}"
        response = requests.get(url, stream=True)
        img_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        image_path = os.path.join(output_dir, filename)
        cv2.imwrite(image_path, image)
        overall_metadata = self.fetch_street_view_metadata(lat, lon)

        metadata = {
            "metadata":overall_metadata,
            "view": "single",
            "heading": heading,
            "pitch": pitch,
            "fov": self.fov,
            "lat": lat,
            "lon": lon,
            "path": image_path
        }

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        if(os.path.exists(image_path)):
            img = np.array(Image.open(image_path))
        else:
            img = None
        
        return img, metadata
