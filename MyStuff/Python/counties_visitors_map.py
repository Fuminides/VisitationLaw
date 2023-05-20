'''
This script will plot the total visitor flows for each state

'''
import utils
import geoutils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#####################################################
# Plot a colomarp of the  total visitor flows for each state
######################################################
import sys
try:
    data_folder_path = sys.argv[1]
except:
    county = 'Los Angeles County'
    data_folder_path = './MyStuff/Data_small/'

# Import the geopandas and geoplot libraries
import geopandas as gpd
import geoplot as gplt
convert = lambda x: x.replace(',','')
# Load the json file with county coordinates
geoData = gpd.read_file('https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')
county_pops = pd.read_csv('./MyStuff/county_pops.csv', sep=';')
county_pops.columns = ['county', 'population']
county_pops['county'] = county_pops['county'].apply(lambda x: x.split('$')[0])
county_pops['population'] = county_pops['population'].apply(lambda x: int(convert(x)))
# Make sure the "id" column is an integer
geoData.id = geoData.id.astype(str).astype(int)

# Remove Alaska, Hawaii and Puerto Rico.
stateToRemove = ['02', '15', '72']
geoData = geoData[~geoData.STATE.isin(stateToRemove)]

# Basic plot with just county outlines
from shapely.geometry import MultiPolygon
import geopandas
geom = geoData.pop('geometry')
geom = geom.apply(lambda x: list(x.geoms) if isinstance(x, MultiPolygon) else x).explode()
geoData = geoData.join(geom, how='inner')
geoData = geopandas.GeoDataFrame(geoData, crs="EPSG:4326")
# gplt.polyplot(geoData, figsize=(12, 8));

# library
import pandas as pd
import seaborn as sns

# Read file
data = pd.read_csv('https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/unemployment-x.csv')

# Show the distribution of unemployment rate

fullData = geoData.merge(data, left_on=['id'], right_on=['id'])
fullData = fullData.merge(county_pops, left_on=['county'], right_on=['county'])


# Read file
# Initialize the figure
import matplotlib.pyplot as plt

# Load the edges data for the county
try:
    edges = pd.read_csv(data_folder_path + 'edges.csv', index_col=0)
except:
    edges = geoutils.load_county_data_edges(data_folder_path)
    #Cache the results
    edges.to_csv(data_folder_path + 'edges.csv', index=True)
incoming = edges[county]
outgoing = edges.loc[county]

# Add the missing counties from FullData to the incoming and outgoing data
for county_trial in fullData['county']:
    if county_trial not in incoming:
        incoming[county_trial] = 0
    if county_trial not in outgoing:
        outgoing[county_trial] = 0

fullData['incoming'] = fullData['county'].apply(lambda x: incoming[x])
fullData['outgoing'] = fullData['county'].apply(lambda x: outgoing[x])


# Set up the color sheme:
import mapclassify as mc
scheme_outgoing = mc.Quantiles(fullData['outgoing'], k=10)
scheme_incoming = mc.Quantiles(fullData['incoming'], k=10)
scheme_pop = mc.Quantiles(fullData['population'], k=10)

fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Map
gplt.choropleth(fullData, 
    hue="outgoing", 
    linewidth=.1,
    scheme=scheme_outgoing, cmap='inferno_r',
    legend=True,
    edgecolor='black',
    ax=ax
)

ax.set_title('Population transfer from: ' + str(county), fontsize=13);
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Map
gplt.choropleth(fullData, 
    hue="population", 
    linewidth=.1,
    scheme=scheme_pop, cmap='inferno_r',
    legend=True,
    edgecolor='black',
    ax=ax
)
ax.set_title('Population: ' + str(county), fontsize=13);
plt.show()
# edges = geoutils.load_county_data_edges(data_folder_path)

print('Done!')