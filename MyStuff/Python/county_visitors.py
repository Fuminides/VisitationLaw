from uszipcode import SearchEngine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from geoutils import get_state_names, state_distance, load_travel_data, get_distances_between_coords
import geoutils
from utils import incoming_pop_distance, outgoing_pop_distance
cache_county = {}
cache_coord = {}

def get_county_name(lat, lng):
    global cache_county, cache_coord
    try:
        return cache_county[(lat, lng)]
    except:
        search = SearchEngine()
        result = search.by_coordinates(lat, lng)
        
        try:
            aux = result[0].county
            cache_county[(lat, lng)] = aux
            cache_coord[aux] = (lat, lng)
            return aux
        except:
            if len(result) > 0:
                aux = result.county
                cache_county[(lat, lng)] = aux
                cache_coord[aux] = (lat, lng)
                return result.county
            else:
                cache_county[(lat, lng)] = None
                return None


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

import sys

try:
    state = sys.argv[1]
    data_folder_path = sys.argv[2]
    vis_mode = sys.argv[3]
except:
    state = 'Los Angeles County'
    data_folder_path = './MyStuff/Data_small/'
    vis_mode = 'map'

edges = geoutils.load_county_data_edges(data_folder_path)
distances = get_distances_between_coords(cache_coord)
california_incoming = incoming_pop_distance(state, distances, edges)
california_outgoing = outgoing_pop_distance(state, distances, edges)
plt.plot(california_incoming['distance'], california_incoming['pop_flow'], 'r')
plt.plot(california_outgoing['distance'], california_outgoing['pop_flow'], 'b')
plt.legend(['Incoming', 'Outgoing'])
plt.title('Population flow vs distance for {}'.format(state))
plt.xlabel('Distance (km)')
plt.ylabel('Population flow')
plt.show()

print('Done')