# This script downloads daily GEE and monthly GRACE data from Google Earth Engine for a given year and saves it to CSV files.
# It uses parallel processing to speed up the download process for multiple years.
# Authors: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu), Mr. Santanu Banerjee (bane7352@vandals.uidaho.edu)


import geopandas as gpd
import pandas as pd
import ee
import json
import os
import calendar
import time
import requests
from glob import glob
from joblib import Parallel, delayed
from http.client import RemoteDisconnected


def get_gee_data(
        year: int,
        daily_date_dict: dict,
        monthly_date_dict: dict,
        fc: ee.FeatureCollection,
        output_dir: str,
        gldas_ic: ee.ImageCollection,
        era5land_ic: ee.ImageCollection,
        grace_ic: ee.ImageCollection
) -> None:
    """
    Downloads daily GEE and monthly GRACE data from Google Earth Engine for a given year and saves it to CSV files.

    Args:
        year: The year for which to download data.
        daily_date_dict: Dictionary containing lists of daily dates for each year.
        monthly_date_dict: Dictionary containing lists of monthly dates for each year.
        fc: FeatureCollection containing geometries for which to download data.
        output_dir: Directory where the downloaded data will be saved.
        gldas_ic: GLDAS ImageCollection.
        era5land_ic: ERA5-Land ImageCollection.
        grace_ic: GRACE ImageCollection.

    Returns:
        None
    """
    data_band_names = [
        'rzsm_gldas_mm',
        'ssm_gldas_mm',
        'et_gldas_mm'
        'tws_gldas_mm',
        'ppt_era5_mm',
        'runoff_era5_mm',
        'tmax_era5_degC',
        'tmin_era5_degC'
    ]
    sum_data_band_names = [
        'et_gldas_mm',
        'tws_gldas_mm',
        'ppt_era5_mm',
        'runoff_era5_mm',
    ]
    ee.Initialize(
        project='grace-grb',
        opt_url='https://earthengine-highvolume.googleapis.com'
    )
    for gee_date in daily_date_dict[year]:
        year, month, day = gee_date.split('-')
        year = int(year)
        month = int(month)
        day = int(day)
        max_day = calendar.monthrange(year, month)[1]
        if day + 1 > max_day:
            gee_date_adv = f'{year + 1}-01-01' if month + 1 > 12 else f'{year}-{month + 1:02d}-01'
        else:
            gee_date_adv = f'{year}-{month:02d}-{day + 1:02d}'
        rzsm_gldas = gldas_ic.filterDate(gee_date, gee_date_adv) \
            .select('SoilMoist_RZ_tavg') \
            .mean() \
            .rename('rzsm_gldas_mm')
        ssm_gldas = gldas_ic.filterDate(gee_date, gee_date_adv) \
            .select('SoilMoist_S_tavg') \
            .mean() \
            .rename('ssm_gldas_mm')
        et_gldas = gldas_ic.filterDate(gee_date, gee_date_adv) \
            .select('Evap_tavg') \
            .mean() \
            .rename('et_gldas_mm')
        tws_gldas = gldas_ic.filterDate(gee_date, gee_date_adv) \
            .select('TWS_tavg') \
            .mean() \
            .rename('tws_gldas_mm')
        ppt_era5 = era5land_ic.filterDate(gee_date, gee_date_adv) \
            .select('total_precipitation_sum') \
            .mean() \
            .multiply(1000) \
            .rename('ppt_era5_mm')
        runoff_era5 = era5land_ic.filterDate(gee_date, gee_date_adv) \
            .select('surface_runoff_sum') \
            .mean() \
            .multiply(1000) \
            .rename('runoff_era5_mm')
        tmax_era5 = era5land_ic.filterDate(gee_date, gee_date_adv) \
            .select('temperature_2m_max') \
            .mean() \
            .add(-273.15) \
            .rename('tmax_era5_degC')
        tmin_era5 = era5land_ic.filterDate(gee_date, gee_date_adv) \
            .select('temperature_2m_min') \
            .mean() \
            .add(-273.15) \
            .rename('tmin_era5_degC')
        data_bands = [
            rzsm_gldas,
            ssm_gldas,
            et_gldas,
            tws_gldas,
            ppt_era5,
            runoff_era5,
            tmax_era5,
            tmin_era5
        ]
        sum_data_bands = [
            et_gldas,
            tws_gldas,
            ppt_era5,
            runoff_era5,
        ]
        data_img = rzsm_gldas
        for band in data_bands:
            data_img = data_img.addBands(band)
        data_img_sum = et_gldas
        for band in sum_data_bands:
            data_img_sum = data_img_sum.addBands(band)
        # Filter Over Feature Collection
        data_fc_mean = data_img.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=11000,
            tileScale=1
        )
        data_fc_sum = data_img_sum.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.sum(),
            scale=11000,
            tileScale=1
        )
        retry_download = True
        while retry_download:
            try:
                # Download table
                url_mean = data_fc_mean.getDownloadURL('csv', data_band_names)
                url_sum = data_fc_sum.getDownloadURL('csv', sum_data_band_names)
                # Download table
                df = pd.read_csv(url_mean)
                df.columns = [col.replace('mm', 'mean_mm') for col in df.columns]
                df_sum = pd.read_csv(url_sum)
                df_sum.columns = [col.replace('mm', 'sum_mm') for col in df_sum.columns]
                df = pd.concat([df, df_sum], axis=1)
                df['Date'] = pd.to_datetime(gee_date)
                # make Date column the first column
                cols = df.columns.tolist()
                cols = ['Date'] + [col for col in cols if col != 'Date']
                df = df[cols]
                df.to_csv(f'{output_dir}Daily_GEE_{gee_date}.csv', index=False)
                retry_download = False
            except (
                ee.EEException, requests.exceptions.RequestException,
                requests.exceptions.ConnectionError, RemoteDisconnected,
            ) as e:
                print('Error', e, 'during download!')
                print('Retrying download...')
                retry_download = True
            time.sleep(0.001)

    for month_yr in monthly_date_dict[year]:
        year, month = month_yr.split('-')
        year = int(year)
        month = int(month)
        month_year_adv = f'{year + 1}-01' if month + 1 > 12 else f'{year}-{month + 1:02d}'
        try:
            lwe_grace = grace_ic.filterDate(month_yr, month_year_adv) \
                .select('lwe_thickness') \
                .mean() \
                .multiply(100) \
                .rename('lwe_thickness_mm')

            data_fc_mean = lwe_grace.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=55660,
                tileScale=1
            )
            data_fc_sum = lwe_grace.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.sum(),
                scale=55660,
                tileScale=1
            )
            retry_download = True
            while retry_download:
                try:
                    # Download table
                    url_mean = data_fc_mean.getDownloadURL('csv', ['mean'])
                    url_sum = data_fc_sum.getDownloadURL('csv', ['sum'])
                    # Download table
                    df = pd.read_csv(url_mean)
                    df.columns = ['lwe_thickness_mean_mm']
                    df_sum = pd.read_csv(url_sum)
                    df_sum.columns = ['lwe_thickness_sum_mm']
                    df = pd.concat([df, df_sum], axis=1)
                    df['Date'] = pd.to_datetime(f'{month_yr}-01')
                    cols = df.columns.tolist()
                    cols = ['Date'] + [col for col in cols if col != 'Date']
                    df = df[cols]
                    df.to_csv(f'{output_dir}Monthly_GRACE_{month_yr}.csv', index=False)
                    retry_download = False
                except (
                        ee.EEException, requests.exceptions.RequestException,
                        requests.exceptions.ConnectionError, RemoteDisconnected,
                ) as e:
                    print('Error', e, 'during download!')
                    print('Retrying download...')
                    retry_download = True
                time.sleep(0.001)
        except ee.EEException:
            print(f"No GRACE data for {month_yr}")
            continue

def gee_data_download(
        input_vector_path: str,
        start_year: 2002,
        end_year: 2024,
        output_dir: str
) -> None:
    """
    Downloads data from Google Earth Engine based on the input GeoDataFrame.

    Args:
        input_vector_path: Path to the input vector shapefile or geojson.
        start_year: The starting year for the data download.
        end_year: The ending year for the data download.
        output_dir: Directory where the downloaded data will be saved.
    """
    try:
        input_gdf = gpd.read_file(input_vector_path)
        print("Shapefile loaded successfully.")
        ee.Initialize(
            project='grace-grb',
            opt_url='https://earthengine-highvolume.googleapis.com'
        )
        year_list = range(start_year, end_year + 1)
        gdf = input_gdf.to_crs("EPSG:4326")
        geo_json = gdf.to_json()
        fc = ee.FeatureCollection(json.loads(geo_json))
        temp_dir = f'{output_dir}temp/'
        os.makedirs(temp_dir, exist_ok=True)

        # download daily GEE data

        grace_ic = ee.ImageCollection('NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI')
        era5land_ic = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
        gldas_ic = ee.ImageCollection('NASA/GLDAS/V022/CLSM/G025/DA1D')

        daily_date_dict = {}
        monthly_date_dict = {}
        for year in year_list:
            date_list = []
            month_yr_list = []
            for month in range(1, 13):
                month_yr = f'{year}-{month:02d}'
                month_yr_list.append(month_yr)
                max_day = calendar.monthrange(year, month)[1]
                for day in range(1, max_day + 1):
                    date_str = f'{year}-{month:02d}-{day:02d}'
                    date_list.append(date_str)
            daily_date_dict[year] = date_list
            monthly_date_dict[year] = month_yr_list
        # Parallel processing for daily and monthly data
        num_years = len(year_list)
        print(f'Downloading {num_years} years of data parallely...')
        Parallel(n_jobs=num_years)(
            delayed(get_gee_data)(
                year,
                daily_date_dict,
                monthly_date_dict,
                fc,
                temp_dir,
                gldas_ic,
                era5land_ic,
                grace_ic
            ) for year in year_list
        )
        daily_gee_df = pd.concat(
            [pd.read_csv(daily_csv) for daily_csv in glob(f'{temp_dir}Daily_GEE_*.csv')]
        ).sort_values(by='Date')
        monthly_grace_df = pd.concat(
            [pd.read_csv(monthly_csv) for monthly_csv in glob(f'{temp_dir}Monthly_GRACE_*.csv')]
        ).sort_values(by='Date')

        daily_gee_df.to_csv(f'{output_dir}Daily_GEE.csv', index=False)
        monthly_grace_df.to_csv(f'{output_dir}Monthly_GRACE.csv', index=False)
    except Exception as e:
        print(f"Error {e}")


if __name__ == "__main__":
    input_dir = 'Data/'
    out_dir = f'{input_dir}Outputs/'
    os.makedirs(out_dir, exist_ok=True)
    shapefile_path = f"{input_dir}Ganga Basin Shapefile/Ganga_basin.shp"
    gee_data_download(
        input_vector_path=shapefile_path,
        start_year=2002,
        end_year=2024,
        output_dir=out_dir
    )