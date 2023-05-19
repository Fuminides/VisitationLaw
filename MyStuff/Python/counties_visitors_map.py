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
    state = 'California'
    data_folder_path = './MyStuff/Data_small/'

# Import the geopandas and geoplot libraries
import geopandas as gpd
import geoplot as gplt

# Load the json file with county coordinates
geoData = gpd.read_file('https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')

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
sns.distplot( data["rate"], hist=True, kde=False, rug=False );

fullData = geoData.merge(data, left_on=['id'], right_on=['id'])
fullData.head(2)

# library
import pandas as pd
import seaborn as sns

# Read file
data = pd.read_csv('https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/unemployment-x.csv')

# Show the distribution of unemployment rate
sns.distplot( data["rate"], hist=True, kde=False, rug=False );

# Initialize the figure
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Set up the color sheme:
import mapclassify as mc
scheme = mc.Quantiles(fullData['rate'], k=10)
edges = geoutils.load_county_data_edges(data_folder_path)

# Map
gplt.choropleth(fullData, 
    hue="rate", 
    linewidth=.1,
    scheme=scheme, cmap='inferno_r',
    legend=True,
    edgecolor='black',
    ax=ax
);

ax.set_title('Unemployment rate in US counties', fontsize=13);
plt.show()
edges = geoutils.load_county_data_edges(data_folder_path)
