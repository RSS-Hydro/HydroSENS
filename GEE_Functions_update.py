import ee
import ee.mapclient
import geemap
import os
import pandas as pd

service_account = 'pycharm@ee-chloecampo20.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, r"C:\Users\ben\PycharmProjects\CN_Project\update\ee-chloecampo20-eb6cd5fb671d.json")
ee.Initialize(credentials)


def get_daily_weather(start_date, end_date, aoi):
    try:
        centroid = aoi.geometry().centroid(maxError=1)
    except Exception as e:
        centroid = aoi.geometry().bounds().centroid(maxError=1)
    era5_land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start_date, end_date)
    def extract_point_value(image):
        point_data = image.sample(region=centroid, scale=1).first()
        temp = point_data.get('temperature_2m')
        precip = point_data.get('total_precipitation_sum')
        return ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'temperature_2m': temp,
            'precipitation': precip
        })
    feature_collection = era5_land.map(extract_point_value)
    results = feature_collection.getInfo()['features']

    weather_data = {}
    for feature in results:
        props = feature['properties']
        temp_k = props.get('temperature_2m')
        precip_m = props.get('precipitation')
        if temp_k is not None and precip_m is not None:
            weather_data[props['date']] = {
                'temperature': temp_k - 273.15, 
                'precipitation': precip_m * 1000 
            }
    return weather_data


def get_sentinel2_dates(aoi, start_date, end_date, max_cloud_coverage=30):

    sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(aoi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', max_cloud_coverage))

    dates = sentinel2.aggregate_array('system:time_start').getInfo()

    date_list = [ee.Date(date).format('YYYY-MM-DD').getInfo() for date in dates]

    return date_list


def load_Sentinel2(aoi, StartDate, EndDate):
    """
    load_Sentinel2
        This function is used to obtain a collection of Sentinel 2 images that
        meet a specific criteria (e.g., cloud coverage <10%, date range). The images are sorted
        by cloud coverage in descending order, so that the first image selected in the mosaic function
        has the least cloud coverage.
    input:
        aoi: The area of interest
        StartDate: The start date of the satellite images
        EndDate: The end date of the satellite images;
    output:
        filtered_col: a collection of filtered images that meet the criteria

    """
    filtered_col1 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterDate(StartDate,EndDate)\
        .filterBounds(aoi) \
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than', 35)\
        .sort('CLOUDY_PIXEL_PERCENTAGE')\
        .select('B2', 'B3', 'B4', 'B7', 'B8', 'B8A', 'B11', 'B12')
    num_images = filtered_col1.size().getInfo()
    first_img = filtered_col1.first()


    return first_img, num_images

def load_Landsat(aoi, StartDate, EndDate):
    filtered_col1 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')\
        .filterDate(StartDate,EndDate)\
        .filterBounds(aoi) \
        .filterMetadata('CLOUD_COVER','less_than', 30)\
        .sort('CLOUD_COVER')\
        .select('SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7')
    num_images = filtered_col1.size().getInfo()
    first_img = filtered_col1.first()

    return first_img, num_images

def mosaic(filtered_col):
    """
    mosaic
        This function is used to mosaic images, should multiple encompass the aoi.
        The first image in the collection will be selected/mosaicked
    input:
        filtered_col: product of load_Sentinel2 function
    output:
        mosaic: first image in filtered_col, mosaicked if necessary

    """
    mosaic_col = filtered_col.mosaic()
    mosaic = mosaic_col.first()
    return mosaic


def resampling(image,shapefile_projection):
    """
    resampling
        This function is used to resample bands 8A, 11, and 12 to the 10m resolution of bands
        2, 3, and 4 using bilinear reasmpling.
    input:
        image: product of mosaic function
        shapefile_projection: projection of the shapefile
    output:
        resample: image with resampled bands

    """
    bands = image.select('B2', 'B3', 'B4', 'B7', 'B8', 'B8A', 'B11', 'B12')
    resample = bands.resample('bilinear').reproject(crs=shapefile_projection, scale=10)
    return resample


def getDEM(aoi):
    """
    getDEM
        This function is used to obtain the elevation data from FABDEM, the
        Forest And Buildings removed Copernicus 30m DEM in the area of interest.
    input:
            aoi: the area of interest
    output:
        DEM:  FABDEM image with the elevation data

    """
    DEM = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")\
        .filterBounds(aoi)\
        .mosaic()\
        .rename('elevation')
    return DEM


def Bandsexport(image,shapefile_projection,output,aoi):
    """
    Bandsexport
        This function is used to export the resampled Sentinel 2 images to a geotiff file.
        The output geotiff matches the aoi shapefile projection and bounds.
    input:
        image: image with bands 2,3,4,8A,11, and 12
        shapefile projection: projection of input shapefile
        output: output folder
        aoi: the area of interest
    output:
        Bands.tif in the user-designated output folder

    """
    output_file = os.path.join(output, "Bands.tif")

    band_names = ['B2', 'B3', 'B4', 'B7', 'B8', 'B8A', 'B11', 'B12']

    final_selected = image.select(band_names, band_names).float()

    geemap.ee_export_image(final_selected, output_file, scale=10, region=aoi.geometry(),
                           crs=shapefile_projection)
    return Bandsexport


def DEMexport(image,shapefile_projection,output,aoi):
    """
    DEMexport
        This function is used to export the elevation data of the FABDEM to a geotiff file.
        The output geotiff matches the aoi shapefile projection and bounds, along with a 10 m
        spatial resolution.
    input:
        image:image with FABDEM elevation data
         shapefile projection: projection of input shapefile
        output: output folder
        aoi: the area of interest
    output:
        DEM.tif in the user-designated output folder

    """
    output_file = os.path.join(output, "DEM.tif")

    band_names = ['elevation']

    final_selected = image.select(band_names, band_names)

    geemap.ee_export_image(final_selected, output_file, scale=10, region=aoi.geometry(),
                           crs=shapefile_projection)
    return DEMexport

def Bandsexport_Landsat(image,shapefile_projection,output,aoi):
    """
    Bandsexport
        This function is used to export the Landsat 8 images to a geotiff file.
        The output geotiff matches the aoi shapefile projection and bounds.
    input:
        image: image with bands 2,3,4,5, 6, and 7
        shapefile projection: projection of input shapefile
        output: output folder
        aoi: the area of interest
    output:
        Bands.tif in the user-designated output folder

    """
    output_file = os.path.join(output, "Bands.tif")

    band_names = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']

    final_selected = image.select(band_names, band_names).float()

    geemap.ee_export_image(final_selected, output_file, scale=30, region=aoi.geometry(),
                           crs=shapefile_projection)

def DEMexport_Landsat(image,shapefile_projection,output,aoi):
    """
    DEMexport
        This function is used to export the elevation data of the FABDEM to a geotiff file.
        The output geotiff matches the aoi shapefile projection and bounds, along with a 10 m
        spatial resolution.
    input:
        image:image with FABDEM elevation data
         shapefile projection: projection of input shapefile
        output: output folder
        aoi: the area of interest
    output:
        DEM.tif in the user-designated output folder

    """
    output_file = os.path.join(output, "DEM.tif")

    band_names = ['elevation']

    final_selected = image.select(band_names, band_names)

    geemap.ee_export_image(final_selected, output_file, scale=30, region=aoi.geometry(),
                           crs=shapefile_projection)
    return DEMexport
