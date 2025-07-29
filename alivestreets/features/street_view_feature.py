from abc import ABC, abstractmethod
import numpy as np

class StreetViewFeatureExtractor(ABC):
    
    @abstractmethod
    def download_model(self):
        """
        Download model weights or prepare the model.
        """
        pass

    @abstractmethod
    def get_masks(self, image: np.ndarray, confidence_threshold: float = 0.6):
        """
        Return class-wise masks from the image, usually from a segmentation model.
        Can return a dict of {class: [masks]}.
        """
        pass

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Optional hook to prepare the image before inference.
        """
        return image

    @abstractmethod
    def compute(self, image: np.ndarray) -> dict[str, float]:
        """
        Compute feature values (e.g., GVI, SVF, indices) from the image, 
        using the masks extracted in get_masks.
        Returns a dictionary like {"GVI": 0.31, "SVF": 0.52, ...}
        """
        pass