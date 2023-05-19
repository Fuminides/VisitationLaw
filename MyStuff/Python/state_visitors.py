'''
This script will plot the total visitor flows for each state

'''
import utils
import geoutils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap

#####################################################
# Plot a colomarp of the  total visitor flows for each state
######################################################
import sys
try:
    data_folder_path = sys.argv[1]
except:
    state = 'California'
    data_folder_path = './MyStuff/Data/'

# Create the USA map
map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

# Load the shapefile, use the states resolution
map.readshapefile('MyStuff/st99_d00', name='states', drawbounds=True)

edges = geoutils.load_travel_data(data_folder_path, verbose=True)
total_visitor_flows = edges.sum(axis=1)
total_visitor_flows.sort_values(ascending=False, inplace=True)

# Load the shapes
state_names = []
for shape_dict in map.states_info:
    state_names.append(shape_dict['NAME'])

ax = plt.gca() # get current axes instance

# get States and draw the filled polygon
from matplotlib.patches import Polygon
import matplotlib

cmap = matplotlib.cm.get_cmap('coolwarm')

states = geoutils.get_state_names()



for state in states:
    seg = map.states[state_names.index(state)]
    population_percentage = total_visitor_flows[state] / max(total_visitor_flows)

    poly = Polygon(seg, facecolor=cmap(population_percentage),edgecolor=cmap(population_percentage))
    ax.add_patch(poly)

plt.show()