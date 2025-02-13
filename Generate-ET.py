from os.path import exists
from calendar import monthrange
from numpy import zeros, pi, degrees, arcsin, cos, radians, sin, nanmean, full, linspace, int32, \
    where
from pandas import DataFrame, read_csv, date_range
from joblib import load
from multiprocessing import Pool
from numba import jit
import numba
import time
from xarray import DataArray, Dataset
from rasterio import open as rasterio_open
import warnings

warnings.filterwarnings("ignore")


def vector_to_nc(df_ET, latid, lonid, nc_file, year, month, day):
    latitudes = linspace(90 - 0.05, -90 + 0.05, 1800)
    longitudes = linspace(-180 + 0.05, 180 - 0.05, 3600)
    n_lat = 1800
    n_lon = 3600
    n = 24

    data_array_et = DataArray(
        dims=("time", "lat", "lon"),
        coords={
            "time": date_range(f"{year}-{month:02d}-{day:02d}", periods=n, freq='H'),
            "lat": latitudes,
            "lon": longitudes
        },
        data=full((n, n_lat, n_lon), 0)
    )

    for hour in range(n):
        matrix_ET = full((1800, 3600), 0, dtype=int32)
        etvalues = df_ET[hour].values.ravel()
        matrix_ET[latid, lonid] = etvalues
        data_array_et[hour, :, :] = matrix_ET

    ds = Dataset({"ETeco": data_array_et})

    ds.attrs["Long Name"] = f"Global Hourly Continuous ET on {year}-{month:02d}-{day:02d}"

    ds["ETeco"].attrs["Long Name"] = f"Hourly Continuous Evapotranspiration on {year}-{month:02d}-{day:02d}"
    ds["ETeco"].attrs["Units"] = "10\u207b\u00b3 W/m\u00b2"
    ds["ETeco"].attrs["Scale"] = "0.001"
    ds["ETeco"].attrs["Type"] = "int32"
    ds["ETeco"].attrs["Temporal Resolution"] = "One Hour"
    ds["ETeco"].attrs["Spatial Resolution"] = "0.1 Degree"
    ds["ETeco"].attrs["Fill Value"] = "0"
    ds["ETeco"].attrs["Projection"] = "Geographic"
    ds["ETeco"].attrs["Area"] = "Global Vegetation Area"
    ds["ETeco"].attrs["Dimension"] = "Rows:1800, Columns:3600"

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(nc_file, encoding=encoding)


@jit(nopython=True)
def myloop(grid_to_pixels, par_data, t2m_data, sm_data, vpd_data, xlist, ylist, year, month, day, hour, lclist,
           fparlist):
    average_sza = zeros(1486095)
    average_apar = zeros(1486095)
    average_t2m = zeros(1486095)
    average_sm = zeros(1486095)
    average_vpd = zeros(1486095)
    average_par = zeros(1486095)

    days_in_month = [31, 28 + (1 if year % 4 == 0 else 0), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for idx, (row_start, row_stop, col_start, col_stop) in enumerate(grid_to_pixels):
        centerx = xlist[idx]
        centery = ylist[idx]

        maskpar = par_data[row_start:row_stop, col_start:col_stop].ravel()
        average_par[idx] = nanmean(maskpar[maskpar >= 0])
        average_apar[idx] = average_par[idx] * fparlist[idx]

        if lclist[idx] != 0 and average_apar[idx] > 0:
            d = sum(days_in_month[:month - 1]) + day
            T = 2 * pi * (d - 1) / 365
            SD = (0.006918 - 0.399912 * cos(T) + 0.070257 * sin(T) - 0.006758 * cos(2 * T) + 0.000907 * sin(
                2 * T) - 0.002697 * cos(3 * T) + 0.00148 * sin(3 * T))
            ST = hour + 0.5 + (centerx - 0) / 15
            w = 15 * (ST - 12)
            sun_altitude = degrees(
                arcsin(sin(radians(centery)) * sin(SD) + cos(radians(centery)) * cos(SD) * cos(radians(w))))
            SZA = 90 - sun_altitude
            average_sza[idx] = cos(radians(SZA))

            maskt2m = t2m_data[row_start:row_stop, col_start:col_stop].ravel()
            average_t2m[idx] = nanmean(maskt2m[maskt2m >= 0])
            masksm = sm_data[row_start:row_stop, col_start:col_stop].ravel()
            average_sm[idx] = nanmean(masksm[masksm >= 0])
            maskvpd = vpd_data[row_start:row_stop, col_start:col_stop].ravel()
            average_vpd[idx] = nanmean(maskvpd[maskvpd >= 0])

    return average_sza, average_apar, average_t2m, average_sm, average_vpd, average_par


def generate(year):
    grid_to_pixels = read_csv(fr'grid_pixels_global_ERA5LAND.csv')
    grid_to_pixels = [tuple(x[1:]) for x in grid_to_pixels.itertuples()]
    grid_to_pixels = numba.typed.List(grid_to_pixels)
    xy_grids = read_csv(fr'xy_grids_global_01.csv')
    xlist = xy_grids['0'].values.ravel()
    ylist = xy_grids['1'].values.ravel()
    df_id = read_csv(fr'lat_lon_id_global_01.csv')
    latid = df_id['0'].values.ravel()
    lonid = df_id['1'].values.ravel()
    DEMlist = read_csv(fr'global_DEM.csv')['DEM'].values.ravel()
    ET_lgbm = load(r'ET_lgbm.pkl')

    if year < 2001:
        average_lc = read_csv(fr'D:\Global_SIF_Simulate\landcover\global_before_lc.csv')['landcover'].values.ravel()
    else:
        average_lc = read_csv(fr'D:\Global_SIF_Simulate\landcover\global_{year}_lc.csv')[f'{year}'].values.ravel()

    for month in range(1, 13):
        dffpar = read_csv(fr'D:\Global_SIF_Simulate\fpar\global_{year}{month:02d}_fpar.csv')

        for day in range(1, monthrange(year, month)[1] + 1):

            raster = fr'D:\Global_SIF_Simulate\results_ET\HC-ETeco_{year}{month:02d}{day:02d}.nc4'

            if exists(raster):
                continue
            else:
                try:
                    start = time.time()
                    hourly_et24 = DataFrame()

                    if day < 16:
                        average_fpar = dffpar[f'fpar{month:02d}01'].values.ravel()
                    else:
                        average_fpar = dffpar[f'fpar{month:02d}02'].values.ravel()

                    for hour in range(0, 24):
                        par_data = rasterio_open(fr'F:\par\par_global_{year}{month:02d}{day:02d}{hour:02d}.tif').read(1)
                        t2m_data = rasterio_open(fr'F:\t2m\t2m_global_{year}{month:02d}{day:02d}{hour:02d}.tif').read(1)
                        sm_data = rasterio_open(fr'F:\sm\sm_global_{year}{month:02d}{day:02d}{hour:02d}.tif').read(1)
                        vpd_data = rasterio_open(fr'F:\vpd\vpd_global_{year}{month:02d}{day:02d}{hour:02d}.tif').read(1)

                        average_sza, average_apar, average_t2m, average_sm, average_vpd, average_par = (
                            myloop(grid_to_pixels, par_data, t2m_data, sm_data, vpd_data, xlist, ylist, year, month,
                                   day, hour, average_lc, average_fpar))

                        x2 = DataFrame(
                            columns=['COSSZA', 'APAR', 'SM', 'VPD', 'VPD/T2M', 'DEM', 'PAR', 'FPAR', 'LANDCOVER'])
                        x2['COSSZA'] = average_sza
                        x2['APAR'] = average_apar
                        x2['SM'] = average_sm
                        x2['VPD'] = average_vpd
                        x2['VPD/T2M'] = where(average_t2m != 0, average_vpd / average_t2m, average_vpd / 0.01)
                        x2['DEM'] = DEMlist
                        x2['PAR'] = average_par
                        x2['FPAR'] = average_fpar
                        x2['LANDCOVER'] = average_lc
                        x2 = x2.fillna(0)

                        x_positive = x2[(x2['APAR'] > 0) & (x2['LANDCOVER'] != 0)].copy()
                        x_positive_without_landcover = x_positive.drop(columns=['LANDCOVER'])
                        x_positive['ET'] = (ET_lgbm.predict(x_positive_without_landcover, n_jobs=-1) * 1000).round()
                        x2['ET'] = 0
                        x2.loc[(x2['APAR'] > 0) & (x2['LANDCOVER'] != 0), 'ET'] = x_positive['ET']
                        hourly_et24[hour] = x2['ET']

                    vector_to_nc(hourly_et24, latid, lonid, raster, year, month, day)
                    print(f'{year}{month:02d}{day:02d}', time.time() - start)

                except Exception as e:
                    print(e)


if __name__ == '__main__':
    years = [year for year in range(1995, 2009)]
    pool = Pool(processes=14)
    pool.map(generate, years)
    pool.close()
    pool.join()
