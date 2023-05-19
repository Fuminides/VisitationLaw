'''

This module contains utility functions.


'''
import pandas as pd
import numpy as np

def load_concat_csv(path: list[str]) -> pd.DataFrame:
    '''
    Returns the DataFrame containing the data of various csv files concatenated.
    
    :param path: the path of the csv file
    :return: the DataFrame containing the data of various csv files concatenated'''

    res = pd.DataFrame()
    for p in path:
        df = pd.read_csv(p)
        res = pd.concat([res, df])
    
    return res


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    from math import radians, cos, sin, asin, sqrt

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def incoming_pop_distance(state, distances: pd.DataFrame, edges: pd.DataFrame):
    # Sort states by distance
    sorted_states = distances[state].sort_values(ascending=False).index
    states = sorted_states
    result = pd.DataFrame(np.zeros((len(states), 3)), index=sorted_states, columns=['distance', 'pop_flow', 'pop_flow_distance'])
    
    for o_state in sorted_states:
        if o_state == state:
            continue
        
        try:
            pop_flow = edges.loc[o_state, state]
        except:
            pop_flow = 0

        distance = distances.loc[o_state, state]
        result.loc[o_state, 'distance'] = distance
        result.loc[o_state, 'pop_flow'] = pop_flow
        result.loc[o_state, 'pop_flow_distance'] = pop_flow / distance

    result.sort_values(by='distance', ascending=False, inplace=True)
    return result


def outgoing_pop_distance(state, distances: pd.DataFrame, edges:pd.DataFrame):
    # Sort states by distance
    sorted_states = distances.loc[state].sort_values(ascending=False).index
    states = sorted_states
    result = pd.DataFrame(np.zeros((len(states), 3)), index=sorted_states, columns=['distance', 'pop_flow', 'pop_flow_distance'])
    
    for o_state in sorted_states:
        if o_state == state:
            continue

        try:
            pop_flow = edges.loc[state, o_state]
        except:
            pop_flow = 0
            
        distance = distances.loc[o_state, state]
        result.loc[o_state, 'distance'] = distance
        result.loc[o_state, 'pop_flow'] = pop_flow
        result.loc[o_state, 'pop_flow_distance'] = pop_flow / distance

    result.sort_values(by='distance', ascending=False, inplace=True)
    return result