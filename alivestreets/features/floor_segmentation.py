from alivestreets.features.street_view_feature import StreetViewFeatureExtractor
from typing import Dict, Optional, Callable, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import requests

class FloorFeatureExtractor(StreetViewFeatureExtractor):

    def __init__(
        self,
        method: str = "ml",
        model_path: Optional[str] = None,
        floor_features_dictionary: Optional[dict] = None,
        threshold: float = 0.3
        ) -> None:
        
        self.method: str = method
        self.model_path: Optional[str] = model_path
        self.floor_features_dictionary:Optional[Dict[str,int]] = floor_features_dictionary
        self.threshold: float = threshold

        self.model: Optional[YOLO] = None
        if(self.method == "ml" and model_path and floor_features_dictionary is not None):
            self.model = YOLO(model_path)

    def download_model(
        self,
        save_path:str = "floor_characteristics.pt") -> None:
        """
        Downloads a default model for floor features segmentation. The model was 
        trained on images from Mexicali, Mexico, which means that it might be better
        suited for desert-like environments.

        The classes included in the model are asphalt (0), built_structure (1), dust (2),
        floor_concrete (3), urban_void (4), vegetation (5).

        Parameters
        __________
        save_path
            Path where the .pt file will be downloaded.
        """
        URL = "https://huggingface.co/urilp4669/floor_features_mexicali/resolve/main/floor_characteristics.pt?download=true"
        response = requests.get(URL, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size = 8192):
                    f.write(chunk)
            print("Model downloaded")
            #The default model contains some classes that are not strictly speaking
            #floor features.
            floor_dictionary = {
                "asphalt":0,
                "built_structure":1,
                "dust":2,
                "floor_concrete":3, 
                "urban_void":4,
                "vegetation":5
            }
            self.floor_features_dictionary = floor_dictionary
            self.model = YOLO(save_path)
        else:
            raise Exception("Failed to download the model. Contemplate your life and internet connection.")


    def preprocess_image(self, image):
        return image

    def get_masks(
        self, 
        image: np.ndarray,
        class_name:str = "")->Tuple[List[np.ndarray], List[float]]:
        """
        Returns the masks and confidence scores for the requested floor feature class.

        Parameters
        ----------
        image
            The input image as a numpy array. As a recommendation use BGR format.
            for some reason the ultralytics library works better with BGR images.
        class_name
            The name of the floor feature to segment. Must be one of the keys in the
            floor_features_dictionary.
        
        Returns
        -------
        masks
            A list of boolean numpy arrays representing the masks for the requested class.
        """

        if(self.method == "ml"):
            if(self.floor_features_dictionary is None or self.floor_features_dictionary.get(class_name, None) is None):
                raise ValueError("Invalid class name or floor feature dictionary not set.")
            if(self.model is None):
                raise ValueError("Model has not been initialized. If you need a model use the download_model method.")

            results = self.model(image, verbose = False, overlap_mask = True)
            requested_id = self.floor_features_dictionary.get(class_name, None)
            masks: List[np.ndarray] = []
            confidences: List[float] = []
            if(results[0].masks is not None):
                detected_confidences = results[0].boxes.conf.numpy()
                class_ids = results[0].boxes.cls.numpy()
                for mask, class_id, conf in zip(results[0].masks.data, class_ids, detected_confidences):
                    if conf >= self.threshold and int(class_id) == requested_id:
                        mask_np = np.array(mask, dtype=bool)
                        masks.append(mask_np)
                        confidences.append(conf)
            return masks, confidences
        
        else:
            raise ValueError("Invalid method. Only ''ml'' is supported at this time.")
        

    def compute(
        self,
        input_feature_names: List[str],
        image: np.ndarray,
        confidence_threshold: float = 0.3,
        operation: Optional[Callable[..., Any]] = None
    )->Any:
        """
        Computes floor features for the given image. If `operation` is provided, it will be called with the
        `FloorFeatureExtractor` instance, the image, the input feature names, and the confidence threshold.
        If `operation` is None, it will return a dictionary with the area ratios and metadata of the requested
        floor features.

        Parameters
        ----------
        input_feature_names
            List of feature names used to compute the result. Each of them must correspond to a name in the
            floor_features_dictionary.
        image
            The input image as a numpy array.
        confidence_threshold
            Threshold that will be used to discriminate which detected masks correspond
            to the requested class instances. Defaults to 0.3.
        operation
            An custom operation that takes the `FloorFeatureExtractor` instance, the image,
            the input feature names, and the confidence threshold as arguments and returns any value.
        
        Returns
        -------
        Any
            The result of the custom operation if provided, otherwise a dictionary with area ratios
            and confidence of the requested floor features.
        """
        self.threshold = confidence_threshold
        if(operation is not None):
            return operation(self, image, input_feature_names, self.threshold)

        result = {}
        for name in input_feature_names:
            masks, confidences = self.get_masks(image, class_name=name)
            single_mask = np.logical_or.reduce(masks) if len(masks) > 0 else np.zeros(image.shape[:2], dtype=bool)
            area = np.sum(single_mask) / (single_mask.shape[0] * single_mask.shape[1])

            result[name] = {
                f"{name}_index": area,
                f"{name}_confidences": np.mean(confidences) if confidences else None
            }

        return result