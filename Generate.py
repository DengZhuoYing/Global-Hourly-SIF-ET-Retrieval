from os.path import exists
from calendar import monthrange
from numpy import zeros, pi, degrees, arcsin, cos, radians, sin, nanmean, full, linspace, int32, datetime64
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


def vector_to_nc(df_sif, df_sif_total_S, df_sif_total_D, df_ET, latid, lonid, nc_file, year, month, day):
    latitudes = linspace(90 - 0.05, -90 + 0.05, 1800)
    longitudes = linspace(-180 + 0.05, 180 - 0.05, 3600)
    n_lat = 1800
    n_lon = 3600
    n = 24

    data_array = DataArray(
        dims=("time", "lat", "lon"),
        coords={
            "time": date_range(f"{year}-{month:02d}-{day:02d}", periods=n, freq='H'),
            "lat": latitudes,
            "lon": longitudes
        },
        data=full((n, n_lat, n_lon), 0, dtype=int32)
    )

    data_array_total_S = DataArray(
        dims=("time", "lat", "lon"),
        coords={
            "time": date_range(f"{year}-{month:02d}-{day:02d}", periods=n, freq='H'),
            "lat": latitudes,
            "lon": longitudes
        },
        data=full((n, n_lat, n_lon), 0, dtype=int32)
    )

    data_array_total_D = DataArray(
        dims=("time", "lat", "lon"),
        coords={
            "time": date_range(f"{year}-{month:02d}-{day:02d}", periods=n, freq='H'),
            "lat": latitudes,
            "lon": longitudes
        },
        data=full((n, n_lat, n_lon), 0, dtype=int32)
    )

    data_array_et = DataArray(
        dims=("time", "lat", "lon"),
        coords={
            "time": date_range(f"{year}-{month:02d}-{day:02d}", periods=n, freq='H'),
            "lat": latitudes,
            "lon": longitudes
        },
        data=full((n, n_lat, n_lon), 0, dtype=int32)
    )

    for hour in range(n):
        matrix = full((1800, 3600), 0, dtype=int32)
        matrix_total_S = full((1800, 3600), 0, dtype=int32)
        matrix_total_D = full((1800, 3600), 0, dtype=int32)
        matrix_ET = full((1800, 3600), 0, dtype=int32)

        sifvalues = df_sif[hour].values
        sifvalues_total_S = df_sif_total_S[hour].values
        sifvalues_total_D = df_sif_total_D[hour].values
        etvalues = df_ET[hour].values

        matrix[latid, lonid] = sifvalues
        matrix_total_S[latid, lonid] = sifvalues_total_S
        matrix_total_D[latid, lonid] = sifvalues_total_D
        matrix_ET[latid, lonid] = etvalues

        data_array[hour, :, :] = matrix
        data_array_total_S[hour, :, :] = matrix_total_S
        data_array_total_D[hour, :, :] = matrix_total_D
        data_array_et[hour, :, :] = matrix_ET

    ds = Dataset({"SIFoco": data_array, "SIFtotal_S": data_array_total_S, "SIFtotal_D": data_array_total_D,
                  "ETeco": data_array_et})

    ds.attrs["Long Name"] = f"Global Hourly Continuous SIFoco and ETeco on {year}-{month:02d}-{day:02d}"

    ds["SIFoco"].attrs[
        "Long Name"] = f"Hourly Continuous Solar-induced Chlorophyll Fluorescence on {year}-{month:02d}-{day:02d}"
    ds["SIFoco"].attrs["Units"] = "10\u207b\u00b3 W/m\u00b2/µm/sr"
    ds["SIFoco"].attrs["Scale"] = "0.001"
    ds["SIFoco"].attrs["Type"] = "int32"
    ds["SIFoco"].attrs["Temporal Resolution"] = "One Hour"
    ds["SIFoco"].attrs["Spatial Resolution"] = "0.1 Degree"
    ds["SIFoco"].attrs["Fill Value"] = "0"
    ds["SIFoco"].attrs["Projection"] = "Geographic"
    ds["SIFoco"].attrs["Area"] = "Global Vegetation Area"
    ds["SIFoco"].attrs["Dimension"] = "Rows:1800, Columns:3600"

    ds["SIFtotal_S"].attrs[
        "Long Name"] = f"Hourly Continuous SIF Estimated to Total Canopy Emissions by SCOPE on {year}-{month:02d}-{day:02d}"
    ds["SIFtotal_S"].attrs["Units"] = "10\u207b\u00b3 W/m\u00b2/µm"
    ds["SIFtotal_S"].attrs["Scale"] = "0.001"
    ds["SIFtotal_S"].attrs["Type"] = "int32"
    ds["SIFtotal_S"].attrs["Temporal Resolution"] = "One Hour"
    ds["SIFtotal_S"].attrs["Spatial Resolution"] = "0.1 Degree"
    ds["SIFtotal_S"].attrs["Fill Value"] = "0"
    ds["SIFtotal_S"].attrs["Projection"] = "Geographic"
    ds["SIFtotal_S"].attrs["Area"] = "Global Vegetation Area"
    ds["SIFtotal_S"].attrs["Dimension"] = "Rows:1800, Columns:3600"

    ds["SIFtotal_D"].attrs[
        "Long Name"] = f"Hourly Continuous SIF Estimated to Total Canopy Emissions by Direct Calculation on {year}-{month:02d}-{day:02d}"
    ds["SIFtotal_D"].attrs["Units"] = "10\u207b\u00b3 W/m\u00b2/µm"
    ds["SIFtotal_D"].attrs["Scale"] = "0.001"
    ds["SIFtotal_D"].attrs["Type"] = "int32"
    ds["SIFtotal_D"].attrs["Temporal Resolution"] = "One Hour"
    ds["SIFtotal_D"].attrs["Spatial Resolution"] = "0.1 Degree"
    ds["SIFtotal_D"].attrs["Fill Value"] = "0"
    ds["SIFtotal_D"].attrs["Projection"] = "Geographic"
    ds["SIFtotal_D"].attrs["Area"] = "Global Vegetation Area"
    ds["SIFtotal_D"].attrs["Dimension"] = "Rows:1800, Columns:3600"

    ds["ETeco"].attrs["Long Name"] = f"Hourly Continuous Evapotranspiration on {year}-{month:02d}-{day:02d}"
    ds["ETeco"].attrs["Units"] = "10⁻⁹ kg H2O/m²/s"
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
    average_sza = zeros(1500579)
    average_apar = zeros(1500579)
    average_t2m = zeros(1500579)
    average_sm = zeros(1500579)
    average_vpd = zeros(1500579)
    average_par = zeros(1500579)

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
    grid_to_pixels = read_csv(fr'csv\grid_pixels_global_ERA5LAND.csv')
    grid_to_pixels = [tuple(x[1:]) for x in grid_to_pixels.itertuples()]
    grid_to_pixels = numba.typed.List(grid_to_pixels)
    xy_grids = read_csv(fr'csv\xy_grids_global_01.csv')
    xlist = xy_grids['0'].values
    ylist = xy_grids['1'].values
    df_id = read_csv(fr'csv\lat_lon_id_global_01.csv')
    latid = df_id['0'].values
    lonid = df_id['1'].values
    DEMlist = read_csv(fr'csv\global_DEM.csv')['DEM'].values
    OCO_lgbm = load(r'OCO_lgbm.pkl')
    OCO_lgbm_total_S = load(r'OCO_lgbm_total2.pkl')
    OCO_lgbm_total_D = load(r'OCO_lgbm_total1.pkl')
    ET_lgbm = load(r'ET_lgbm.pkl')

    if year < 2001:
        average_lc = read_csv(fr'D:\Global_SIF_Simulate\landcover\global_before_lc.csv')['landcover'].values
    else:
        average_lc = read_csv(fr'D:\Global_SIF_Simulate\landcover\global_{year}_lc.csv')[f'{year}'].values

    for month in range(1, 13):
        dffpar = read_csv(fr'D:\Global_SIF_Simulate\fpar\global_{year}{month:02d}_fpar.csv')

        for day in range(1, monthrange(year, month)[1] + 1):

            raster = fr'D:\Global_SIF_Simulate\results\HC-SIFoco-ETeco_{year}{month:02d}{day:02d}.nc4'

            if exists(raster):
                continue
            else:
                start = time.time()
                hourly_sif24 = DataFrame()
                hourly_sif24_total_S = DataFrame()
                hourly_sif24_total_D = DataFrame()
                hourly_et24 = DataFrame()

                if day < 16:
                    average_fpar = dffpar[f'fpar{month:02d}01'].values
                else:
                    average_fpar = dffpar[f'fpar{month:02d}02'].values

                doy = (datetime64(f'{year}-{month:02d}-{day:02d}') - datetime64(f'{year}-01-01')).astype(
                    'timedelta64[D]').astype(int) + 1
                doy = doy / 365 if year % 4 != 0 else doy / 366

                for hour in range(0, 24):
                    par_data = rasterio_open(fr'G:\par\par_global_{year}{month:02d}{day:02d}{hour:02d}.tif').read(1)
                    sm_data = rasterio_open(fr'G:\sm\sm_global_{year}{month:02d}{day:02d}{hour:02d}.tif').read(1)
                    vpd_data = rasterio_open(fr'F:\vpd\vpd_global_{year}{month:02d}{day:02d}{hour:02d}.tif').read(1)
                    t2m_data = rasterio_open(fr'F:\t2m\t2m_global_{year}{month:02d}{day:02d}{hour:02d}.tif').read(1)

                    average_sza, average_apar, average_t2m, average_sm, average_vpd, average_par = (
                        myloop(grid_to_pixels, par_data, t2m_data, sm_data, vpd_data, xlist, ylist, year, month,
                               day, hour, average_lc,
                               average_fpar))

                    x = DataFrame(
                        columns=['LANDCOVER', 'COSSZA', 'APAR', 'T2M', 'SM', 'VPD', 'DEM', 'PAR', 'FPAR', 'LON', 'LAT',
                                 'DOY'])
                    x['LANDCOVER'] = average_lc
                    x['COSSZA'] = average_sza
                    x['APAR'] = average_apar
                    x['T2M'] = average_t2m
                    x['SM'] = average_sm
                    x['VPD'] = average_vpd
                    x['DEM'] = DEMlist
                    x['PAR'] = average_par
                    x['FPAR'] = average_fpar
                    x['LON'] = xlist
                    x['LAT'] = ylist
                    x['DOY'] = doy

                    x = x.fillna(0)

                    x_positive = x[(x['APAR'] > 0) & (x['LANDCOVER'] != 0)].copy()
                    x_positive['SIFobs'] = (OCO_lgbm.predict(x_positive, n_jobs=-1) * 1000).round()
                    x['SIFobs'] = 0
                    x.loc[(x['APAR'] > 0) & (x['LANDCOVER'] != 0), 'SIFobs'] = x_positive['SIFobs']
                    hourly_sif24[hour] = x['SIFobs']

                    x = x.drop(columns=['SIFobs'])
                    x_positive = x[(x['APAR'] > 0) & (x['LANDCOVER'] != 0)].copy()
                    x_positive['SIFtotal_S'] = (OCO_lgbm_total_S.predict(x_positive, n_jobs=-1) * 1000).round()
                    x['SIFtotal_S'] = 0
                    x.loc[(x['APAR'] > 0) & (x['LANDCOVER'] != 0), 'SIFtotal_S'] = x_positive['SIFtotal_S']
                    hourly_sif24_total_S[hour] = x['SIFtotal_S']

                    x = x.drop(columns=['SIFtotal_S'])
                    x_positive = x[(x['APAR'] > 0) & (x['LANDCOVER'] != 0)].copy()
                    x_positive['SIFtotal_D'] = (OCO_lgbm_total_D.predict(x_positive, n_jobs=-1) * 1000).round()
                    x['SIFtotal_D'] = 0
                    x.loc[(x['APAR'] > 0) & (x['LANDCOVER'] != 0), 'SIFtotal_D'] = x_positive['SIFtotal_D']
                    hourly_sif24_total_D[hour] = x['SIFtotal_D']

                    x = x.drop(columns=['SIFtotal_D', 'LON', 'LAT', 'DOY', 'COSSZA'])
                    x_positive = x[(x['APAR'] > 0) & (x['LANDCOVER'] != 0)].copy()
                    x_positive['ET'] = ET_lgbm.predict(x_positive, n_jobs=-1) * 1000
                    x['ET'] = 0
                    x.loc[(x['APAR'] > 0) & (x['LANDCOVER'] != 0), 'ET'] = x_positive['ET']
                    x['NAMUDA'] = 2.501 - 0.00237 * x['T2M'] / 10
                    x['ET'] = (x['ET'] / x['NAMUDA']).round()  # 10** -9 kg H2O/m2/s
                    hourly_et24[hour] = x['ET']

                vector_to_nc(hourly_sif24, hourly_sif24_total_S, hourly_sif24_total_D, hourly_et24, latid, lonid,
                             raster, year, month, day)
                print(f'{year}{month:02d}{day:02d}', time.time() - start)


if __name__ == '__main__':
    years = [year for year in range(1993, 2015)]
    pool = Pool(processes=11)
    pool.map(generate, years)
    pool.close()
    pool.join()
