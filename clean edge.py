import geopandas as gpd
import time
from concurrent.futures import ThreadPoolExecutor


def process_grid_row(row, min_size):
    geometry = row['geometry']
    if geometry.area > min_size:
        return row


start = time.time()

grid_data = gpd.read_file(r'D:\bishe\grid\global05\weitichu.shp')
# min_size = 0.0499 * 0.0499
min_size = 0.499 * 0.499

fixed_grid = gpd.GeoDataFrame()

num_threads = 10

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_grid_row, row, min_size) for _, row in grid_data.iterrows()]
    results = [future.result() for future in futures]

fixed_grid = gpd.GeoDataFrame([result for result in results if result is not None], crs=grid_data.crs)

fixed_grid.to_file(r'D:\bishe\grid\global05\global05.shp')

end = time.time()
print(end - start)
