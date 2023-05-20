'''
This module contains functions to get the state of a given coordinates and vice versa
'''
import pandas as pd
import numpy as np

_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia', 'Puerto Rico']
cache_county = {}
county_coord = {}

def get_state_names() -> list[str]:
    '''
    Returns the list of state names
    '''
    return _states


def load_edges(studied_csv1: pd.DataFrame, states: list[str], state_coord_cache:dict, coord_state_cache:dict, field:str='visitor_flows') -> pd.DataFrame:
    '''
    Returns the edges of the connections between the states

    :param studied_csv1: the csv file containing the data
    :return: the edges of the connections between the states in a numpy array of states x states dimension.
    '''
    res = pd.DataFrame(np.zeros((len(states), len(states))), index=states, columns=states)

    # Iterate over the rows of the DataFrame
    for index, row in studied_csv1.iterrows():
        lat_o = round(row['lat_o'], 1)
        lng_o = round(row['lng_o'], 1)
        lat_d = round(row['lat_d'], 1)
        lng_d = round(row['lng_d'], 1)
        # Get the state of the origin and destination coordinates
        try:
            og_key = (lat_o, lng_o)
            state_o = state_coord_cache[og_key]
        except KeyError:
            state_o = approximate_state_from_coordinates(og_key[0], og_key[1])
            state_coord_cache[og_key] = state_o

            if state_o not in coord_state_cache.keys():
                coord_state_cache[state_o] = og_key
        
        try:
            ds_key = (lat_d, lng_d)
            state_d = state_coord_cache[ds_key]
        except KeyError:
            state_d = approximate_state_from_coordinates(ds_key[0], ds_key[1])
            state_coord_cache[ds_key] = state_d

            if state_d not in coord_state_cache.keys():
                coord_state_cache[state_d] = ds_key

        # Add the number of connections to the corresponding cell
        try:
            res[state_o][state_d] += row[field]
        except KeyError:
            pass
    
    return res


def get_state_from_coordinates(lat: float, lng: float, coord_state_cache:dict) -> str:
    '''
    Returns the state of the given coordinates

    :param lat: the latitude of the coordinates
    :param lng: the longitude of the coordinates
    :return: the state of the given coordinates
    '''
    from geopy.geocoders import Nominatim
    from utils import haversine
    geolocator = Nominatim(user_agent="myGeocoder")
    try:
        location = geolocator.reverse(str(lat) + ' ' + str(lng))

        try:
            address = location.raw['address']
        except AttributeError:
            return 'Hawaii'
        
        state = address.get('state')

    except:
        state = None
    
    if state is None:
        min_distance = 100000000
        for state in _states:
            coord = coord_state_cache[state]
            
            distance = haversine(float(coord[1]), float(coord[0]), lng, lat)
            if distance < min_distance:
                min_distance = distance
                closest_state = state
 
        state = closest_state

    return state


def get_coordinates_for_state(state: str) -> tuple[float, float]:
    '''
    Returns the coordinates of the given state

    :param state: the state of which we want the coordinates
    :return: the coordinates of the given state
    '''
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="myGeocoder")
    location = geolocator.geocode(state)

    return (location.latitude, location.longitude)


def fill_states_coords() -> dict:
    '''
    Returns a dictionary containing the coordinates of each state

    :return: a dictionary containing the coordinates of each state
    '''
    states_coords = {}
    for state in _states:
        states_coords[state] = get_coordinates_for_state(state)
    
    return states_coords


def init_coord_cache(init_cache_path: str) -> [dict, dict]:
    '''
    Returns the state-->coord and coord-->state dictionaries
    '''
    try:
        state_coord_cache = pd.read_csv(init_cache_path, index_col=0).to_dict()['state']
    except:
        print('Not previously computed coord-->state cache.')
        state_coord_cache = {}

    coord_state_cache = fill_states_coords()

    return state_coord_cache, coord_state_cache


def state_distance() -> pd.DataFrame:
    '''
    Returns the DataFrame containing the distances between each state
    '''
    from utils import haversine
    coord_state_cache = fill_states_coords()
    states = get_state_names()

    # Compute the distances between the states
    distances = pd.DataFrame(np.zeros((len(states), len(states))), index=states, columns=states)
    for i, state_og in enumerate(states):
        x, y = coord_state_cache[state_og]
        for j, state_dest in enumerate(states):
            x_d, y_d = coord_state_cache[state_dest]
            distances[state_og][state_dest] = haversine(x, y, x_d, y_d)

    return distances


def load_travel_data(path: str, verbose: bool=True) -> pd.DataFrame:
    '''
    Returns the DataFrame containing the travel data from all the csv files in the given path.
    '''
    import os
    files = os.listdir(path)   
    csv_files = [path + f for f in files if f.endswith('.csv')]
    states = get_state_names()

    # df = load_concat_csv(csv_files) # This wont scale in memory usage
    for i, f in enumerate(csv_files):
        if verbose:
            print(f'Loading file {i+1}/{len(csv_files)}')
        df = pd.read_csv(f)
        if i == 0:
            edges = load_edges(df, states, 'visitor_flows')
        else:
            edges += load_edges(df, states, 'visitor_flows')

    return edges


def approximate_state_from_coordinates(lat: float, lng: float) -> str:
    '''
    Returns the state of the given coordinates

    :param lat: the latitude of the coordinates
    :param lng: the longitude of the coordinates
    :return: the state of the given coordinates
    '''
    from utils import haversine

    states = get_state_names()
    min_distance = 100000000
    coord_state_cache = fill_states_coords()
    for state in states:
        coord = coord_state_cache[state]
        
        distance = haversine(float(coord[1]), float(coord[0]), lng, lat)
        if distance < min_distance:
            min_distance = distance
            closest_state = state

    return closest_state


def get_distances_between_coords(input: dict) -> pd.DataFrame:
    '''
    Receives a dictionary with each name and coords and returns the distance
    between each pair of coordinates
    '''
    from utils import haversine

    distances = pd.DataFrame(np.zeros((len(input), len(input))), index=input.keys(), columns=input.keys())
    for i, state_og in enumerate(input.keys()):
        x, y = input[state_og]
        for j, state_dest in enumerate(input.keys()):
            x_d, y_d = input[state_dest]
            distances[state_og][state_dest] = haversine(x, y, x_d, y_d)

    
    return distances


def get_county_transfer_dict(travel_file: pd.DataFrame) -> dict:
    travel_dict = {}
    # Iterate each row in the travel file
    for index, row in travel_file.iterrows():
        # Get the county name
        county_o = get_county_name(row['lat_o'], row['lng_o'])
        county_d = get_county_name(row['lat_d'], row['lng_d'])

        try:
            travel_dict[county_o]
        except:
            travel_dict[county_o] = {}
        
        try:
            travel_dict[county_o][county_d] += row['visitor_flows']
        except:
            travel_dict[county_o][county_d] = row['visitor_flows']
    
    return travel_dict


def dict_transfer_to_dataframe(travel_dict: dict):
    counties = set()
    for county_o in travel_dict.keys():
        counties.add(county_o)
        for county_d in travel_dict[county_o].keys():
            counties.add(county_d)

    counties = list(counties)
    travel_df = pd.DataFrame(np.zeros((len(counties), len(counties))), columns=counties, index=counties)

    for county_o in travel_dict.keys():
        for county_d in travel_dict[county_o].keys():
            travel_df[county_o][county_d] = travel_dict[county_o][county_d]

    
    return travel_df


def rawdata_to_edge_df(travel_file: pd.DataFrame) -> pd.DataFrame:
    travel_dict = get_county_transfer_dict(travel_file)
    return dict_transfer_to_dataframe(travel_dict)


def get_county_name(lat, lng):
    from uszipcode import SearchEngine
    global cache_county, county_coord
    try:
        return cache_county[(lat, lng)]
    except:
        search = SearchEngine()
        result = search.by_coordinates(lat, lng)
        
        try:
            aux = result[0].county
            cache_county[(lat, lng)] = aux
            county_coord[aux] = (lat, lng)
            return aux
        except:
            if len(result) > 0:
                aux = result.county
                cache_county[(lat, lng)] = aux
                county_coord[aux] = (lat, lng)
                return result.county
            else:
                cache_county[(lat, lng)] = None
                return None


def load_county_data_edges(data_folder_path):
    import os
    files = os.listdir(data_folder_path)   
    csv_files = [data_folder_path + f for f in files if f.endswith('.csv')]

    verbose=True
    for i, f in enumerate(csv_files):
        if verbose:
            print(f'Loading file {i+1}/{len(csv_files)}')
        df = pd.read_csv(f)
        if i == 0:
            edges = rawdata_to_edge_df(df)
        else:
            edges += rawdata_to_edge_df(df)
    edges.replace(np.nan, 0, inplace=True)
    
    return edges

def get_city_pop(county_name):
    import requests
    import json

    tmp = 'https://public.opendatasoft.com/api/records/1.0/search/?dataset=worldcitiespop&q=%s&sort=population&facet'
    url = tmp % (county_name + ',' + 'US')
    response = requests.get(url)
    data = json.loads(response.text)
    return data['records'][0]['fields']['population']