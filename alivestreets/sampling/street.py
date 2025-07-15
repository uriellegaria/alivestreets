from typing import List, Dict, Any
from .geometry import PointDistanceCalculator
from .geometry import Line
from .geometry import get_bearing
import numpy as np


class Street:

    def __init__(self, street_id: str) -> None:
        #Name of the street
        self.street_id: str = street_id
        #The Street Segments that are part of the street
        #Each segment is a list of tuples (x0, y0)->(x1, y1)
        self.street_segments: List[List[List[float]]] = []
        self.segment_lengths: List[float] = []
        #Attributes of the street (e.g. GVI, Mean building height)
        self.attributes: Dict[str, Any] = {}
        #Sampling points where images are taken and attributes are computed. 
        self.sampling_points: List[List[float]] = []
        #Attribute values at point level. 
        self.point_attributes:Dict[str,List[Any]] = {}

    def __eq__(self, other: object) -> bool:
        #Two streets are regarded as equal if their id matches. 
        if not isinstance(other, Street):
            return NotImplemented
        return self.street_id == other.street_id

    def add_segment(self, segment: List[List[float]]) -> None:
        #Checkes tha the segment or the reverse version of it does not already belong to the street.
        if segment not in self.street_segments and segment[::-1] not in self.street_segments:
            self.street_segments.append(segment)
            calculator = PointDistanceCalculator()
            length = 0.0
            for i in range(len(segment) - 1):
                p1 = segment[i]
                p2 = segment[i + 1]
                length += calculator.get_distance(p1[1], p1[0], p2[1], p2[0])
            self.segment_lengths.append(length)

    def get_complete_length(self) -> float:
        return sum(self.segment_lengths)

    def get_points_list(self) -> List[List[float]]:
        points = []
        for segment in self.street_segments:
            for point in segment:
                if point not in points:
                    points.append(point)
        return points

    def get_number_of_points(self) -> int:
        return len(self.get_points_list())

    def get_distance_to_point(self, point: List[float]) -> float:
        lines = self.get_all_lines()
        min_distance = float("inf")
        for line in lines:
            distance = line.get_distance_to_point(point)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def get_closest_line(self, point: List[float]) -> Line:
        lines = self.get_all_lines()
        min_distance = float("inf")
        closest_line = None
        for line in lines:
            distance = line.get_distance_to_point(point)
            if distance < min_distance:
                min_distance = distance
                closest_line = line
        return closest_line

    def get_all_lines(self) -> List[Line]:
        lines: List[Line] = []
        for segment in self.street_segments:
            for i in range(len(segment) - 1):
                point1 = segment[i]
                point2 = segment[i + 1]
                line = Line(point1, point2)
                if line not in lines:
                    lines.append(line)
        return lines
    

    def project_point_into_street(self, point: List[float]) -> np.ndarray:
        lines = self.get_all_lines()
        min_distance = float("inf")
        closest_line = None
        for line in lines:
            distance = line.get_distance_to_point(point)
            if distance < min_distance:
                min_distance = distance
                closest_line = line
        return closest_line.get_point_projection(point)

    def get_google_formatted_sampling_points(self) -> List[tuple[float, float]]:
        return [(pt[1], pt[0]) for pt in self.sampling_points]

    def sampling_point_exists(self, point: List[float]) -> bool:
        for p in self.sampling_points:
            if p[0] == point[0] and p[1] == point[1]:
                return True
        return False

    def add_sampling_point(self, point: List[float]) -> None:
        if not self.sampling_point_exists(point):
            self.sampling_points.append(point)

    def get_number_of_sampling_points(self) -> int:
        return len(self.sampling_points)

    def sample_n_intermediate_points(self, n_points: int) -> List[List[float]]:
        if n_points <= 0:
            return []
        total_length = self.get_complete_length()
        interval = total_length / (n_points + 1)
        sampled = []
        distance = 0
        for _ in range(n_points):
            distance += interval
            point = self.get_point_at_distance(distance)
            if point is not None:
                sampled.append(point)
        return sampled

    def get_point_at_distance(self, dst: float) -> List[float]:
        cumulative_dst = 0
        segment = None
        for i in range(len(self.segment_lengths)):
            if dst >= cumulative_dst and dst <= cumulative_dst + self.segment_lengths[i]:
                segment = self.street_segments[i]
                break
            else:
                cumulative_dst += self.segment_lengths[i]

        excess_distance = dst - cumulative_dst
        sub_segment = None
        n_points = len(segment)
        calculator = PointDistanceCalculator()
        cumulative_dst = 0
        for i in range(n_points - 1):
            p1 = segment[i]
            p2 = segment[i + 1]
            dist = calculator.get_distance(p1[1], p1[0], p2[1], p2[0])
            if excess_distance >= cumulative_dst and excess_distance <= cumulative_dst + dist:
                sub_segment = [p1, p2]
                break
            else:
                cumulative_dst += dist

        alpha_values = np.linspace(0, 1, 100)
        vector1 = np.array(sub_segment[0])
        vector2 = np.array(sub_segment[1])

        for alpha in alpha_values:
            vector = vector1 + alpha * (vector2 - vector1)
            dist_to_1 = calculator.get_distance(vector[1], vector[0], vector1[1], vector1[0])
            if dist_to_1 - excess_distance > 0:
                return [vector[0], vector[1]]

        return [vector[0], vector[1]]

    def set_sampling_points(self, points: List[List[float]]) -> None:
        self.sampling_points = points

    def set_attribute_value(self, name: str, value: Any) -> None:
        self.attributes[name] = value

    def get_attribute_value(self, name: str) -> Any:
        return self.attributes.get(name, None)

    def set_point_attribute_values(self, name:str, values:List[Any])-> None:
        self.point_attributes[name] = values
    
    def get_point_attribute_values(self, name:str)->List[Any]:
        return self.point_attributes.get(name, None)


    def get_bearings(self) -> List[float]:
        delta_fraction = 0.1
        bearings = []
        for point in self.sampling_points:
            point_arr = np.array(point)
            line = self.get_closest_line(point)
            p1 = np.array(line.point1)
            p2 = np.array(line.point2)
            offset_vector = (p2 - p1) * delta_fraction
            ref_point = point_arr + offset_vector
            bearing = get_bearing(point_arr.tolist(), ref_point.tolist())
            bearings.append(bearing)
        return bearings
    

    def get_reference_points(self) -> List[float]:
        delta_fraction = 0.1
        reference_points = []
        for point in self.sampling_points:
            point_arr = np.array(point)
            line = self.get_closest_line(point)
            p1 = np.array(line.point1)
            p2 = np.array(line.point2)
            offset_vector = (p2 - p1) * delta_fraction
            ref_point = point_arr + offset_vector
            reference_points.append(ref_point)
        return reference_points
    











    

    


    

    
    

    

    