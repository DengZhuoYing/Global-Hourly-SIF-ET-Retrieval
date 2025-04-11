import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np


def create_grid(min_lon, max_lon, min_lat, max_lat, grid_size):
    lons = np.arange(min_lon, max_lon, grid_size)
    lats = np.arange(min_lat, max_lat, grid_size)

    polygons = []
    for lon in lons:
        for lat in lats:
            polygon = Polygon(
                [(lon, lat), (lon + grid_size, lat), (lon + grid_size, lat + grid_size), (lon, lat + grid_size)])
            polygons.append(polygon)

    grid = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    return grid


min_lon, max_lon = -180, 180
min_lat, max_lat = -90, 90

grid_size = 0.5

grid = create_grid(min_lon, max_lon, min_lat, max_lat, grid_size)

grid.to_file(r'full.shp')
