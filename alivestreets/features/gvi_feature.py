from alivestreets.features.street_view_feature import StreetViewFeatureExtractor
import numpy as np
from typing import Optional, List, Tuple, Any
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import requests


class GVIFeatureExtractor(StreetViewFeatureExtractor):


    def __init__(
        self,
        method: str = "ml",
        model_path: Optional[str] = None,
        vegetation_class_id: Optional[int] = None,
        threshold: float = 0.6
    ) -> None:
        self.method: str = method
        self.model_path: Optional[str] = model_path
        self.vegetation_class_id: Optional[int] = vegetation_class_id
        self.threshold: float = threshold

        self.model: Optional[YOLO] = None

        # Do not raise error yet — let user call `download_model()` later
        if self.method == "ml" and model_path and vegetation_class_id is not None:
            self.model = YOLO(model_path)
    

    def download_model(
        self, 
        save_path = "facades_model.pt"
        )->None:

        """
        Downloads the façade characteristics model, which includes the vegetation class. Note that the
        model works best with views directed towards building façades or sidewalk views. The results of the segmentation 
        will return also the confidence of the detections. 

        Parameters
        ----------
        save_path
            Path where the .pt file will be downloaded.
        """
        #Non-failure case
        URL = "https://tecmx-my.sharepoint.com/personal/uriel_legaria_tec_mx/_layouts/15/download.aspx?share=Ee9PBbcma2NGmLIUqYGRUKcBmOvlBh0NKnRPZkoTUFaFNg"

        response = requests.get(URL, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded.")

            self.model_path = save_path
            self.vegetation_class_id = 7
            self.model = YOLO(save_path)
        else:
            raise Exception("Failed to download the model.")



    def preprocess_image(
        self, 
        image:np.ndarray
        )->np.ndarray:

        return image
    
    def get_masks(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.6
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Returns a list of vegetation array masks. If segmentation is well conducted 
        different masks correspond to disconnected trees or vegetation indexes.

        Parameters
        ----------
        image
            input image
        
        confidence_threshold
            threshold that will be used to discriminate which detected masks correspond
            to vegetation instances. 

        Returns
        ------------
        masks
            list of masks, where each mask is a binary matrix. 
        """
  
        if self.method == "simple":
            return [self._get_simple_mask(image)], [1]

        elif self.method == "ml":
            if self.model is None:
                raise RuntimeError("Model is not loaded. Cannot extract masks.")
            results = self.model(image, verbose=False, overlap_mask=True)

            masks: List[np.ndarray] = []
            final_confidences: List[float] = []
            if results[0].masks is not None:
                confidences = results[0].boxes.conf.numpy()
                class_ids = results[0].boxes.cls.numpy()
                for mask, class_id, conf in zip(results[0].masks.data, class_ids, confidences):
                    if conf >= confidence_threshold and int(class_id) == self.vegetation_class_id:
                        masks.append(np.array(mask))
                        final_confidences.append(conf)

            return masks, final_confidences

        else:
            raise ValueError(f"Unsupported method '{self.method}'")
    

    def _get_simple_mask(
        self, 
        image: np.ndarray
        ) -> np.ndarray:
        
        """
        Gets a simple mask of vegetation based on color channels. The criteria for deciding that a pixel (R,G,B)
        has vegetation is simply R/G + B/G < 2*threshold.

        Returns
        -------
        mask
            A binary mask where 1 is written at vegetation locations.
        
        Parameters
        ----------
        image
            The image where vegetation will be detected.
        """
 
        n: int = np.size(image, 0)
        m: int = np.size(image, 1)
        mask: np.ndarray = np.zeros((n, m), dtype=np.uint8)

        for i in range(n):
            for j in range(m):
                r: int = image[i, j, 0]
                g: int = image[i, j, 1]
                b: int = image[i, j, 2]

                if g == 0:
                    continue

                ratio_r: float = r / g
                ratio_b: float = b / g

                if ratio_r < self.threshold and ratio_b < self.threshold:
                    mask[i, j] = 1

        return mask

    
    def compute(
        self, 
        image:np.ndarray, 
        debug:bool = False
        )->dict[str, Any]:
        
        """
        Computes the GVI of an image as the ratio of pixels occupied by pixels to the total number of 
        pixels in the image. 


        Returns
        -------
        results_dictonary
            Dictionary including
                "GVI" -> The computed Green View Index.
                "Confidences" -> A list of the detection confidence for each vegetation instance found in the image. 
                "mask"-> Binary mask marked as 1 on pixels corresponding to vegetation.
        """

        masks, confidences = self.get_masks(image, confidence_threshold=self.threshold)

        h: int = image.shape[0]
        w: int = image.shape[1]

        if not masks:
            return {
                "GVI": 0.0,
                "confidences": [],
                "mask": np.zeros((h, w), dtype=np.uint8)  # <- safe fallback mask
            }

        combined_mask: np.ndarray = np.logical_or.reduce(np.array(masks)).astype(np.uint8)

        if debug:
            segmented_image = (combined_mask[:, :, np.newaxis] * image).astype(np.uint8)
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(image)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Segmented Vegetation")
            plt.imshow(segmented_image)
            plt.axis("off")
            plt.show()

        gvi: float = float(np.sum(combined_mask)) / (h * w)
        if(gvi == 0):
            combined_mask == None
        
        results_dictionary = {
            "GVI": gvi,
            "confidences": confidences,
            "mask": combined_mask
        }
        return results_dictionary

    





