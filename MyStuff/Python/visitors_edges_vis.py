'''
 Plot the edges of the connections between the states
'''
import sys
import utils
import geoutils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

states = geoutils.get_state_names()
coord_state_cache = geoutils.get_state_coordinates()

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
            map.plot([x,x_d], [y, y_d], color='blue', linewidth=edges[state_og][state_dest]/1000) #WARNING: lines are plotting wrongly
        

# Show the map
plt.show()