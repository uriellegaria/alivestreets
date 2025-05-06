
from alivestreets.sampling.street_sampler import StreetSampler
from typing import Tuple, List, Dict, Optional, Literal
from alivestreets.streetview_collection.streetview_collection import StreetViewImageCollector
import os
from tqdm import tqdm
import json

class StreetViewNetworkCollector:
    def __init__(
        self,
        mode: Literal["full_panoramics", "sidewalks", "directed"],
        api_key:str,
        fov:int = 90,
        pitch:int = 0,
        size:Tuple[int, int] = (640, 640)
        )->None:

        
        self.mode = mode
        self.api_key = api_key
        #Create the street view collector
        self.street_view_collector = StreetViewImageCollector(
            api_key,
            fov = fov,
            pitch = pitch,
            size = size

        )
    



    def collect_street_sampler_images(
        self,
        sampler:StreetSampler, 
        parent_output_location:str,
        num_views = 8,
        save_json = True, 
        headings = None
        )-> List[Dict]:

        """
        Collects images over the sampling points of a Street Sampler object.

        Returns
        -------
        point_dictionaries
            A list of dictionaries containing information for each sampling point. The fields of the dictionary are:
                "latitude" -> Latitude of the point.
                "longitude" -> Longitude of the point.
                "panoramic_view_path" -> Path to the panoramic view / None if the modality is sidewalks or directed.
                "view_paths" -> Paths to the views stored as part of the analysis.
                "view_directions" -> Heading angles at which the views were taken.
                "Pitch" -> pitch angle used for retrieval.
                "FOV" -> FOV used for retrieval.
                "metadata_path"-> metadata location path.

        Parameters
        __________

        sampler
            The street sampler. It must already contain sampling points for the images to be retrieved.
        parent_output_location
            Parent folder where the images will be stored.
        num_views
            Number of views used to construct panoramic (default 8 works well).
        save_json
            If true saves a json with the point dictionaries.
        headings
            The heading directions of the camera for the directed mode.
        """

        if(self.mode == "full_panoramics"):

            sampling_points = sampler.get_all_sampling_points()
            n_points = len(sampling_points)
            point_dictionaries = []

            for i in tqdm(range(0,n_points)):
                point = sampling_points[i]
                longitude = point[0]
                latitude = point[1]
                output_dir = os.path.join(parent_output_location, f"point_{i}")
                #Collect the panoramic
                panoramic, dictionary = self.street_view_collector.collect_panoramic_with_views(
                    latitude,
                    longitude,
                    output_dir,
                    num_views = num_views
                )
                view_paths = []
                view_directions = []

                for i in range(0,num_views):
                    potential_path = os.path.join(output_dir, "views", f"view_{i}.jpg")
                    if(os.path.exists(potential_path)):
                        view_paths.append(potential_path)
                        view_directions.append((360/n_views)*i)

                results_dictionary = {
                    "latitude":latitude,
                    "longitude":longitude,
                    "panoramic_view_path":dictionary["panoramic_path"],
                    "view_paths":view_paths,
                    "view_directions":view_directions,
                    "pitch":self.street_view_collector.pitch,
                    "fov":self.street_view_collector.fov,
                    "metadata_path":os.path.join(output_dir, "metadata.json")
                }

                point_dictionaries.append(results_dictionary)
            
            if(save_json):
                os.makedirs(parent_output_location, exist_ok = True)
                jsonl_path = os.path.join(parent_output_location, "point_dictionaries.jsonl")
                with open(jsonl_path, "w") as f:
                    for d in point_dictionaries:
                        json.dump(d, f)
                        f.write("\n")

            
            return point_dictionaries
    
    
        elif(self.mode == "sidewalks"):
            #Get the sampling points
            sampling_points = sampler.get_all_sampling_points()
            #Get reference points close to the sampling points

            #We are going to collect points in the same segment of the point of interest
            #To get the sidewalk views
            next_points = []
            n_points = len(sampling_points)

            for i in range(0,n_points):
                point = sampling_points[i]
                street, point_index = sampler.get_street_of_nth_point(i)
                reference_points = street.get_reference_points()
                reference_point = reference_points[point_index]
                next_points.append(reference_point)
            

            #Once you have collected the reference points we can begin to extract the required SV views
            point_dictionaries = []
            for i in tqdm(range(0,n_points)):
                point = sampling_points[i]
                reference_point = next_points[i]
                longitude = point[0]
                latitude = point[1]
                ref_longitude = reference_point[0]
                ref_latitude = reference_point[1]

                output_path = os.path.join(parent_output_location,f"point_{i}")

                images, metadata_dicts = self.street_view_collector.collect_sidewalk_views(
                    latitude, 
                    longitude, 
                    ref_latitude, 
                    ref_longitude,
                    output_path
                    )
                
                view_paths= [d.get("path", None) for d in metadata_dicts]
                if(len(view_paths) == 0):
                    view_paths = None
                
                view_directions = [d.get("heading", None) for d in metadata_dicts]

                if(len(view_directions) == 0):
                    view_directions = None
                
                

                
                results_dictionary = {
                    "latitude":latitude,
                    "longitude":longitude,
                    "panoramic_view_path":None,
                    "view_paths":view_paths,
                    "view_directions":view_directions,
                    "pitch":self.street_view_collector.pitch,
                    "fov":self.street_view_collector.fov,
                    "metadata_path":os.path.join(output_path, "metadata.json")
                }

                point_dictionaries.append(results_dictionary)
            

            if(save_json):
                os.makedirs(parent_output_location, exist_ok = True)
                jsonl_path = os.path.join(parent_output_location, "point_dictionaries.jsonl")
                with open(jsonl_path, "w") as f:
                    for d in point_dictionaries:
                        json.dump(d, f)
                        f.write("\n")

                
            return point_dictionaries


        elif(self.mode == "directed"):
            sampling_points = sampler.get_all_sampling_points()
            n_points = len(sampling_points)

            if(headings is None):
                raise Exception("In the directed mode you need to provide a list of camera headings.")
            
            if(len(headings) != len(sampling_points)):
                raise Exception("The number of headings provided to the function needs to match the number of sampling points")
            
            point_dictionaries = []
            for i in tqdm(range(0,n_points)):

                point = sampling_points[i]
                longitude = point[0]
                latitude = point[1]
                #Get the heading angle for the camera
                heading = headings[i]
                output_dir_path = os.path.join(parent_output_location,f"point_{i}")
                #Obtain the view and save it 
                img, metadata = self.street_view_collector.collect_single_view(
                    latitude,
                    longitude,
                    self.pitch,
                    heading,
                    output_dir_path
                )

            
                view_paths = [metadata["path"]] if metadata.get("path") else None
                view_directions = [heading]


                results_dictionary = {
                    "latitude":latitude,
                    "longitude":longitude,
                    "panoramic_view_path":None,
                    "view_paths":view_paths,
                    "view_directions":view_directions,
                    "pitch":self.street_view_collector.pitch,
                    "fov":self.street_view_collector.fov,
                    "metadata_path":os.path.join(output_dir_path, "metadata.json")
                }

                point_dictionaries.append(results_dictionary)
            
            if(save_json):
                os.makedirs(parent_output_location, exist_ok = True)
                jsonl_path = os.path.join(parent_output_location, "point_dictionaries.jsonl")
                with open(jsonl_path, "w") as f:
                    for d in point_dictionaries:
                        json.dump(d, f)
                        f.write("\n")

                
            return point_dictionaries





        

            

            








