import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import geopandas

state_coord_cache = {}
coord_state_cache = {}

def get_state_from_coordinates(lat: float, lng: float) -> str:
    '''
    Returns the state of the given coordinates

    :param lat: the latitude of the coordinates
    :param lng: the longitude of the coordinates
    :return: the state of the given coordinates
    '''
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="myGeocoder")
    location = geolocator.reverse(str(lat) + ' ' + str(lng))

    try:
        address = location.raw['address']
    except AttributeError:
        return 'Hawaii'
    state = address.get('state')
    
    if state is None:
        county = address['county']
        location = geolocator.geocode(county)
 
        state = location.raw['display_name'].split(',')[-2].strip()

    return state


def load_edges(studied_csv1: pd.DataFrame, states: list[str], field:str='visitor_flows') -> pd.DataFrame:
    '''
    Returns the edges of the connections between the states

    :param studied_csv1: the csv file containing the data
    :return: the edges of the connections between the states in a numpy array of states x states dimension.
    '''
    res = pd.DataFrame(np.zeros((len(states), len(states))), index=states, columns=states)

    # Iterate over the rows of the DataFrame
    for index, row in studied_csv1.iterrows():
        # Get the state of the origin and destination coordinates
        try:
            state_o = state_coord_cache[(row['lat_o'], row['lng_o'])]
        except KeyError:
            state_o = get_state_from_coordinates(row['lat_o'], row['lng_o'])
            state_coord_cache[(row['lat_o'], row['lng_o'])] = state_o
            coord_state_cache[state_o] = (row['lat_o'], row['lng_o'])
        
        try:
            state_d = state_coord_cache[(row['lat_d'], row['lng_d'])]
        except KeyError:
            state_d = get_state_from_coordinates(row['lat_d'], row['lng_d'])
            state_coord_cache[(row['lat_d'], row['lng_d'])] = state_d
            coord_state_cache[state_d] = (row['lat_d'], row['lng_d'])

        # Add the number of connections to the corresponding cell
        res[state_o][state_d] += row[field]
    
    return res

# Read data from CSV file
df = pd.read_csv('./MyStuff/weekly_state2state_03_02.csv')

# List with all the states in the USA
states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia', 'Puerto Rico', 'Guam', 'Virgin Islands', 'Northern Mariana Islands', 'American Samoa']

# Load the edges of the connections between the states
edges = load_edges(df, states, 'visitor_flows')
# Create a map centered on the US
map = Basemap(projection='merc', llcrnrlat=20, urcrnrlat=50,
              llcrnrlon=-130, urcrnrlon=-60, lat_ts=20, resolution='i')

# Draw coastlines and country boundaries
map.drawcoastlines()
map.drawcountries()

# Plot data on the map
map.scatter(df['lng_o'].values, df['lat_o'].values, marker='.', color='red')

# Draw the population exchanges between the states
for i in range(len(x_o)):
    for j in range(len(x_d)):
        if edges.iloc[i][j] > 0:
            map.drawgreatcircle(x_o[i], y_o[i], x_d[j], y_d[j], linewidth=int(edges.iloc[i][j]/1000000), color='green')
    

# Show the map
plt.show()


print('Done')