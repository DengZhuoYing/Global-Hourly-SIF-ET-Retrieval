import rasterio
import geopandas as gpd
import numpy as np
from numpy import nan, nanmean
from rasterio.features import geometry_mask
from shapely.geometry import box
from os.path import join
import pandas as pd
import time
import math
from numba import jit
from numpy import zeros

ftif = fr'F:\vpd\vpd_global_1982010100.tif'

with rasterio.open(ftif) as src:
    transform = src.transform
    raster_shape = src.shape
    raster_bounds = src.bounds

vector = gpd.read_file(fr'D:\bishe\grid\global01\global01.shp')

grid_to_pixels = []

with rasterio.open(ftif) as src:
    for idx, geom in enumerate(vector.geometry):
        geom_bounds = geom.bounds
        window = rasterio.windows.from_bounds(*geom_bounds, transform=transform)
        row_start, row_stop = window.row_off, window.row_off + window.height
        col_start, col_stop = window.col_off, window.col_off + window.width
        row_start = math.floor(row_start)
        col_start = math.floor(col_start)
        row_stop = math.ceil(row_stop)
        col_stop = math.ceil(col_stop)
        grid_to_pixels.append((row_start, row_stop, col_start, col_stop))

dfgrid = pd.DataFrame(grid_to_pixels)
print(dfgrid.describe())
dfgrid.to_csv(fr'grid_pixels_global_ERA5LAND.csv', index=False)

xy_grids = []
for idx, geom in enumerate(vector.geometry):
    x = geom.centroid.x
    y = geom.centroid.y
    xy_grids.append((x, y))

dfxy = pd.DataFrame(xy_grids)
dfxy.to_csv(fr'xy_grids_global_01.csv', index=False)

xy_grids = pd.read_csv(fr'xy_grids_global_01.csv')
hourly_sif24 = pd.DataFrame()
hourly_sif24['lon'] = xy_grids['0']
hourly_sif24['lat'] = xy_grids['1']
latitudes = np.linspace(90 - 0.05, -90 + 0.05, 1800)
longitudes = np.linspace(-180 + 0.05, 180 - 0.05, 3600)
lat_lon_idlist = []

for _, row in hourly_sif24.iterrows():
    lon = row['lon']
    lat = row['lat']
    lat_idx = np.argmin(abs(latitudes - lat))
    lon_idx = np.argmin(abs(longitudes - lon))
    lat_lon_idlist.append((lat_idx, lon_idx))

df = pd.DataFrame(lat_lon_idlist)
df.to_csv(fr'lat_lon_id_global_01.csv', index=False)

df1 = pd.read_csv(fr'lat_lon_id_global_01.csv')
df2 = pd.read_csv(fr'xy_grids_global_01.csv')
df3 = pd.read_csv(fr'grid_pixels_global_fpar.csv')
df4 = pd.read_csv(fr'grid_pixels_global_ERA5LAND.csv')
df5= pd.read_csv(fr'grid_pixels_global_CO2.csv')
df6= pd.read_csv(fr'grid_pixels_global_landcover.csv')
print(df1.describe())
print(df2.describe())
print(df3.describe())
print(df4.describe())
print(df5.describe())
print(df6.describe())