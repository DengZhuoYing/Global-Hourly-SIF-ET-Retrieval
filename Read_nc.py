import netCDF4 as nc
import rasterio
import numpy as np
from rasterio.transform import from_origin
from calendar import monthrange
import os

for year in range(1982, 2023):
    for month in range(1, 13):
        for day in range(1, monthrange(year, month)[1] + 1):
            outputET = fr'E:\ETeco-daily\ETeco-daily_global_{year}{month:02d}{day:02d}.tif'
            if os.path.exists(outputET):
                continue
            dataset = nc.Dataset(fr'D:\Global_SIF_Simulate\results_ET\HOUR-ETeco_{year}{month:02d}{day:02d}.nc4')

            ETeco = dataset.variables['ETeco'][:]
            ETeco = np.sum(ETeco, axis=0)

            with rasterio.open(
                    outputET,
                    'w',
                    driver='GTiff',
                    height=ETeco.shape[0], width=ETeco.shape[1]
                    , count=1, dtype=ETeco.dtype, crs='+proj=latlong',
                    transform=from_origin(-180, 90, 0.1, 0.1)) as dst:
                dst.write(ETeco, 1)

            print(f'{year}{month:02d}{day:02d}')
