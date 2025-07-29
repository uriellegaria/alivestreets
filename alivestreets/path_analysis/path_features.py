from typing import List, Optional, Tuple
import networkx as nx
import numpy as np


def get_feature_time_series(
    trajectory: List[int],
    G: nx.MultiDiGraph, 
    attribute_name:str
)->List[Optional[float]]:

    values = []

    for i in range(0, len(trajectory) - 1):
        #Get start and end nodes
        u = trajectory[i]
        v = trajectory[i + 1]

        if G.has_edge(u,v):
            for k in G[u][v]:
                val = G[u][v][k].get(attribute_name, None)

                if(not val is None):
                    values.append(val)
                    break

            
        else:

            values.append(None)
    
    return values



def get_path_feature_mean(
    trajectory:List[int], 
    G:nx.MultiDiGraph, 
    attribute_name:str
) -> Optional[float]:

    #Get the time series of the feature over the trajectory
    values = get_feature_time_series(trajectory, G, attribute_name)

    #Filter out none values
    values = [x for x in values if not x is None]

    feature_mean = np.mean(values)

    if not values:
        return None
    
    return float(np.mean(values))


def get_path_feature_std(
    trajectory:List[int], 
    G:nx.MultiDiGraph, 
    attribute_name:str
)->Optional[float]:

    values = get_feature_time_series(trajectory, G, attribute_name)
    values = [x for x in values if not x is None]
    if(len(values) == 0):
        return None
    else:
        feature_std = np.std(values)
        return feature_std


def get_path_feature_range(
    trajectory: List[int], 
    G:nx.MultiDiGraph, 
    attribute_name:str
)-> Optional[float]:
    values = get_feature_time_series(trajectory, G, attribute_name)
    values = [x for x in values if not x is None]
    if(len(values) == 0):
        return None
    
    else:
        feature_range = np.max(values) - np.min(values)
        return feature_range


def get_path_average_absolute_difference(
    trajectory: List[int], 
    G:nx.MultiDiGraph,
    attribute_name:str
)-> Optional[float]:
    values = get_feature_time_series(trajectory, G, attribute_name)
    values = [x for x in values if not x is None]

    n_values = len(values)
    average_difference = 0
    if(n_values >= 2):
        for i in range (0, n_values - 1):

            average_difference += (1/(n_values -1))*np.abs(values[i+1] - values[i])

    return average_difference


def get_path_feature_distribution(
    trajectory:List[int], 
    G:nx.MultiDiGraph, 
    attribute_name:str,
    n_bins: int = -1
)-> Tuple[np.ndarray, np.ndarray]:

    feature_series = get_feature_time_series(trajectory, G, attribute_name)
    values = [x for x in feature_series if not x is None]
    n = len(values)

    #IF the time series is empty just return an empty distribution
    if(n == 0):
        return np.array([]), np.array([])

    #If no number of bins is specified the default will be sqrt(n)
    if(n_bins == -1):
        n_bins = int(np.sqrt(n))
    
    distribution = np.zeros(n_bins)
    max_value = np.max(values)
    min_value = np.min(values)

    #Case where there is only one value
    if(max_value == min_value):
        distribution[0] =1.0
        return np.linspace(min_value, max_value, n_bins), distribution
    
    x_values = np.linspace(min_value, max_value, n_bins)

    for value in values:
        #Assign a bin to each value and increase the count of that bin
        bin_idx = min(int(((value - min_value) / (max_value - min_value)) * n_bins), n_bins - 1)
        distribution[bin_idx] += 1
    
    distribution = distribution/np.sum(distribution)
    return x_values, distribution


def get_path_feature_entropy(
    trajectory:List[int], 
    G:nx.MultiDiGraph, 
    attribute_name:str, 
    n_bins: int = -1
)->Optional[float]:


    x_values, distribution = get_path_feature_distribution(trajectory, G, attribute_name, n_bins)
    entropy = 0
    if(len(distribution) == 0):
        return None
    for i in range(0, len(distribution)):
        P = distribution[i]
        if(P > 0):
            entropy = entropy + P*np.log(1/P)
    
    return entropy


def get_path_feature_metrics(
    trajectory: List[int],
    G: nx.MultiDiGraph,
    attribute_name: str,
    n_bins: int = -1
) -> dict:

    return {
        "mean": get_path_feature_mean(trajectory, G, attribute_name),
        "std": get_path_feature_std(trajectory, G, attribute_name),
        "range": get_path_feature_range(trajectory, G, attribute_name),
        "avg_abs_diff": get_path_average_absolute_difference(trajectory, G, attribute_name),
        "entropy": get_path_feature_entropy(trajectory, G, attribute_name, n_bins)
    }
    

    






