from uszipcode import SearchEngine
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia', 'Puerto Rico']

def approximate_state_from_coordinates(lat: float, lng: float) -> str:
    '''
    Returns the state of the given coordinates

    :param lat: the latitude of the coordinates
    :param lng: the longitude of the coordinates
    :return: the state of the given coordinates
    '''
    min_distance = 100000000
    for state in states:
        coord = coord_state_cache[state]
        
        distance = haversine(float(coord[1]), float(coord[0]), lng, lat)
        if distance < min_distance:
            min_distance = distance
            closest_state = state

    return closest_state

def get_state_from_coordinates(lat: float, lng: float) -> str:
    '''
    Returns the state of the given coordinates

    :param lat: the latitude of the coordinates
    :param lng: the longitude of the coordinates
    :return: the state of the given coordinates
    '''
    search = SearchEngine()
    result = search.by_coordinates(lat, lng)
    
    try:
        state = result[0].state
    except:
        if len(result) > 0:
            state = result.state
        else:
            state = None

    
    if state is None:
        min_distance = 100000000
        for state in states:
            coord = coord_state_cache[state]
            
            distance = haversine(float(coord[1]), float(coord[0]), lng, lat)
            if distance < min_distance:
                min_distance = distance
                closest_state = state
 
        state = closest_state

    return state

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

def load_edges(studied_csv1: pd.DataFrame, states: list[str], field:str='visitor_flows') -> pd.DataFrame:
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
    for state in states:
        states_coords[state] = get_coordinates_for_state(state)
    
    return states_coords

try:
    state_coord_cache = pd.read_csv('./MyStuff/state_coord_cache.csv', index_col=0).to_dict()['state']
    coord_state_cache = fill_states_coords()
    # coord_state_cache = pd.read_csv('./MyStuff/coord_state_cache.csv', index_col=0).to_dict()['coord']
except FileNotFoundError:
    state_coord_cache = {}
    coord_state_cache = fill_states_coords()


# List all CSV files from the folder
import os
path = './MyStuff/'
files = os.listdir(path)   
csv_files = [path + f for f in files if f.endswith('.csv')]


# df = load_concat_csv(csv_files) # This wont scale in memory usage
for i, f in enumerate(csv_files):
    print(f'Loading file {i+1}/{len(csv_files)}')
    df = pd.read_csv(f)
    if i == 0:
        edges = load_edges(df, states, 'visitor_flows')
    else:
        edges += load_edges(df, states, 'visitor_flows')

# Cache the distances
pd.DataFrame.from_dict(state_coord_cache, orient='index', columns=['state']).to_csv('./MyStuff/state_coord_cache.csv')
# pd.DataFrame.from_dict(coord_state_cache, orient='index', columns=['coord']).to_csv('./MyStuff/coord_state_cache.csv')

# Load the edges of the connections between the statesWhat does not kill 

# Compute and plot the total visitor flows for each state
total_visitor_flows = edges.sum(axis=1)
total_visitor_flows.sort_values(ascending=False, inplace=True)
'''
fig, ax = plt.subplots()
# Resize figure
fig.set_size_inches(50, 10.5)
plt.plot(range(len(total_visitor_flows)), total_visitor_flows.values)
# ax.set_xticklabels(total_visitor_flows.index)
plt.xticks(ticks=range(len(total_visitor_flows)), labels=total_visitor_flows.index, rotation = 60)
plt.show() # Power law distribution?
'''
######################################################
# Plot the edges of the connections between the states
######################################################
'''
# Create a map centered on the US
map = Basemap(projection='merc', llcrnrlat=20, urcrnrlat=50,
              llcrnrlon=-130, urcrnrlon=-60, lat_ts=20, resolution='i')

# Draw coastlines and country boundaries
map.drawcoastlines()
map.drawcountries()
map.drawstates()
# Plot data on the map
projected = map(df['lng_o'].values, df['lat_o'].values)
map.scatter(projected[0], projected[1], marker='.', color='red')


for i, state_og in enumerate(states):
    x, y = coord_state_cache[state_og]
    x_projected, y_projected = map(x, y)
    for j, state_dest in enumerate(states):
        # Draw the population exchanges between the states
        x_d, y_d = coord_state_cache[state_dest]
        x_d_projected, y_d_projected = map(x_d, y_d)

        # Draw the population exchanges between the states
        if edges[state_og][state_dest] > 0:
            map.plot([x,x_d], [y, y_d], color='blue', linewidth=edges[state_og][state_dest]/1000)
        

# Show the map
plt.show()

######################################################
# Plot a colomarp of the  total visitor flows for each state
######################################################

# Create the USA map
map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

# Load the shapefile, use the states resolution
map.readshapefile('MyStuff/st99_d00', name='states', drawbounds=True)

# Load the shapes
state_names = []
for shape_dict in map.states_info:
    state_names.append(shape_dict['NAME'])

ax = plt.gca() # get current axes instance

# get States and draw the filled polygon
from matplotlib.patches import Polygon
import matplotlib

cmap = matplotlib.cm.get_cmap('coolwarm')


for state in states:
    seg = map.states[state_names.index(state)]
    population_percentage = total_visitor_flows[state] / max(total_visitor_flows)

    poly = Polygon(seg, facecolor=cmap(population_percentage),edgecolor=cmap(population_percentage))
    ax.add_patch(poly)

plt.show()
'''
######################################################
# Distances between the states
######################################################
# Compute the distances between the states
distances = pd.DataFrame(np.zeros((len(states), len(states))), index=states, columns=states)
for i, state_og in enumerate(states):
    x, y = coord_state_cache[state_og]
    for j, state_dest in enumerate(states):
        # Draw the population exchanges between the states
        x_d, y_d = coord_state_cache[state_dest]
        distances[state_og][state_dest] = haversine(x, y, x_d, y_d)

# Check the distances vs population with respect to the studied state

def incoming_pop_distance(state):
    # Sort states by distance
    sorted_states = distances[state].sort_values(ascending=False).index
    result = pd.DataFrame(np.zeros((len(states), 3)), index=sorted_states, columns=['distance', 'pop_flow', 'pop_flow_distance'])
    
    for o_state in sorted_states:
        if o_state == state:
            continue

        pop_flow = edges.loc[o_state, state]
        distance = distances.loc[o_state, state]
        result.loc[o_state, 'distance'] = distance
        result.loc[o_state, 'pop_flow'] = pop_flow
        result.loc[o_state, 'pop_flow_distance'] = pop_flow / distance

    result.sort_values(by='distance', ascending=False, inplace=True)
    return result

def outgoing_pop_distance(state):
    # Sort states by distance
    sorted_states = distances.loc[state].sort_values(ascending=False).index
    result = pd.DataFrame(np.zeros((len(states), 3)), index=sorted_states, columns=['distance', 'pop_flow', 'pop_flow_distance'])
    
    for o_state in sorted_states:
        if o_state == state:
            continue

        pop_flow = edges.loc[state, o_state]
        distance = distances.loc[o_state, state]
        result.loc[o_state, 'distance'] = distance
        result.loc[o_state, 'pop_flow'] = pop_flow
        result.loc[o_state, 'pop_flow_distance'] = pop_flow / distance

    result.sort_values(by='distance', ascending=False, inplace=True)
    return result

state = 'California'
california_incoming = incoming_pop_distance(state)
california_outgoing = outgoing_pop_distance(state)
plt.plot(california_incoming['distance'], california_incoming['pop_flow'], 'r')
plt.plot(california_outgoing['distance'], california_outgoing['pop_flow'], 'b')
plt.legend(['Incoming', 'Outgoing'])
plt.title('Population flow vs distance for {}'.format(state))
plt.xlabel('Distance (km)')
plt.ylabel('Population flow')
plt.show()

print('Done')