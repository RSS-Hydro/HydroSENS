import osgeo
from osgeo import gdal, osr, ogr
osgeo.gdal.UseExceptions()
import rasterio
from rasterio.mask import mask
import numpy as np
import scipy
import geopandas as gpd
import pandas as pd
import csv
from numpy import ndarray
import timeit
import rioxarray
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import xarray as xr
from mesma.core import mesma, shade_normalisation
from scipy.interpolate import griddata
import netCDF4 as nc
import geemap
from datetime import datetime


def writeTCI(red_array, green_array, blue_array, reference, array_name, output):
    output_filename = os.path.join(output, array_name + ".tif")
    driver = gdal.GetDriverByName("GTiff")
    output_raster = driver.Create(output_filename,
                                  reference.RasterXSize,
                                  reference.RasterYSize,
                                  3,
                                  gdal.GDT_Float64)

    output_raster.SetProjection(reference.GetProjection())
    output_raster.SetGeoTransform(reference.GetGeoTransform())

    output_raster.GetRasterBand(1).WriteArray(red_array)
    output_raster.GetRasterBand(2).WriteArray(green_array)
    output_raster.GetRasterBand(3).WriteArray(blue_array)
    output_raster.FlushCache()


def CreateInt(array, reference, array_name, output):
    """
    CreateInt
        This function is used to create an integer geotiff from a numpy array.
    Parameters:
        array:  a numpy array of the image
        reference: another geotiff that will serve as a reference for the new image
        array_name: a string that will be used to name the output image
        output: path to output file
    Returns:
        None

    """
    output_filename = output + "\\" + array_name + ".tif"
    output_raster = gdal.GetDriverByName("GTiff").Create(output_filename, reference.RasterXSize,
                                                         reference.RasterYSize, 1, gdal.GDT_Int32)
    array_int = array.astype(np.int32)
    output_raster.SetProjection(reference.GetProjection())
    output_raster.SetGeoTransform(reference.GetGeoTransform())
    output_raster.GetRasterBand(1).WriteArray(array_int)




def CreateFloat(array, reference, array_name, output):
    """
    CreateFLoat
        This function is used to create a float geotiff from a numpy array.
    Parameters:
        array:  a numpy array of the image
        reference: another geotiff that will serve as a reference for the new image
        array_name: a string that will be used to name the output image
        output: path to output file
    Returns:
        None

    """
    output_filename = output + "\\" + array_name + ".tif"
    output_raster = gdal.GetDriverByName("GTiff").Create(output_filename, reference.RasterXSize,
                                                         reference.RasterYSize, 1, gdal.GDT_Float64)
    output_raster.SetProjection(reference.GetProjection())
    output_raster.SetGeoTransform(reference.GetGeoTransform())
    output_raster.GetRasterBand(1).WriteArray(array)



def Extract(raster_path, shapefile_path, output_path, nodata_value = -9999):
    """
    Extract
        Extracts a raster based on the extent of a shapefile.

    Parameters:
        raster_path: Path to the input raster file.
        shapefile_path: Path to the shapefile used as the extent mask.
        output_path: Path to the output extracted raster file.
        nodata_value : The value to be used for the nodata area

    Returns:
        None
    """
    with rasterio.open(raster_path) as src:
        shapefile = gpd.read_file(shapefile_path)
        geometry = shapefile.geometry.values[0]
        out_image, out_transform = mask(src, shapes=[geometry], crop=True)

        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        if nodata_value is not None:
            out_meta.update({"nodata": nodata_value})
            out_image[out_image == 0] = nodata_value

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

def Create_buffer(input_shapefile, output_shapefile):
    """"
    Create_buffer
        This function is used create a buffer around the original shapefile. The soil dataset raster is coarse and
        providing a buffered shapefile ensures that no information is lost as the raster is clipped to the study area.
    Parameters:
        input_shapefile: original input shapefile
        output_shapefile: path to new shapefile with additional 250 m buffer
    Returns:
        None

    """
    gdf = gpd.read_file(input_shapefile)
    buffered_geometries = gdf.geometry.buffer(250)
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered_geometries, crs=gdf.crs)
    buffered_gdf.to_file(output_shapefile)


def extract_raster(source, reference, output):
    """
    extract_raster
        This function is used extract a raster based on the dimensions of another raster.
    Parameters:
        source: raster to be clipped
        reference: raster that source will be clipped to
        output: output path (output + r"\desired_name.tif"
    Returns:
        None
    """
    with rasterio.open(source) as src:
        source_data = src.read()
        source_meta = src.meta.copy()
    with rasterio.open(reference) as ref:
        source_meta.update({
            'height': ref.height,
            'width': ref.width
        })
        with rasterio.open(output, 'w', **source_meta) as dst:
            dst.write(source_data)

def Fill(data):
    """
    Fill
        This function is used to fill nodata portions of the raster
        with information from the nearest neighbors. This is used with the global soil dataset, as the coarse nature
        of the raster causes areas of land surrounding water bodies are also being incorrectly labeled as nodata.
        This way, all of the possible land is accounted for.
    Parameters:
        data: geotiff read using matplotlib.pyplot
    Returns:
        Filled array

    """
    # create a boolean mask that identifies locations in array with valid data and creates mask
    mask = ~((data[:, :, 0] == 255) & (data[:, :, 1] == 255) & (data[:, :, 2] == 255))
    # create grid
    xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    xym =np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T
    # extracts data from array where mask is true
    data0 = np.ravel(data[:, :, 0][mask])
    # valid data points xym and corresponding data0 values are used to interpolate with nearest neighbor algorithm
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)
    # interpolated data is reshaped to original array
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
    return result0


def classification(CN_table,array1,array2):
    """
    classification
        This function is used to classify an array with CN values using the values of first array
         and the second array in a lookup table.
    Parameters:
        CN_table: .csv lookup table of CN values. The current table is based on the values found in Bera et al., 2022
        and USACE HEC-HMS TR-55 CN table
        array1: first array
        array2: second array
    Returns:
        Classified array
    """
    table = []
    with open(CN_table, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            row = [int(value) for value in row]
            table.append(row)
    classification_dict = {}

    # Creating a dictionary for classification
    for row in table[1:]:
        key = row[0]
        for i, value in enumerate(row[1:], start=1):
            if i == 0:
                continue
            dict_key = (key, i) # Create tuple key
            classification_dict[dict_key] = value # Store classification value

    # Populate the classified array based on classification dictionary
    classified_array: ndarray = np.empty_like(array1, dtype=np.float32)
    for i in range(array1.shape[0]):
        for j in range(array1.shape[1]):
            x = array1[i, j]
            y = array2[i, j]
            key = (x.item(), y.item())
            if key in classification_dict:
                classified_value = classification_dict[key]
                classified_array[i, j] = classified_value # Assign classification value
    return classified_array


def AMCIII(array):
    """
    AMCIII
        This function is used to convert the slope-corrected CN map to AMC III using the methodology of
        Mishra et al., 2008.
    Parameters:
        CNarr: the numpy array containing the CN values in AMC II
    Returns:
        AMC III CN array

    """
    np.seterr(invalid='ignore')
    CCN_arr = np.divide(array, (.43 + (array * 0.0057)))
    return CCN_arr


def AMCI(array):
    """
    AMCI
        This function is used to convert the slope-corrected CN map to AMC I using the methodology of
        Mishra et al., 2008.
    Paramters:
        CNarr: the numpy array containing the CN values in AMC II
    Returns:
        AMC I CN array

    """
    np.seterr(invalid='ignore')
    CCN_arr = np.divide(array, (2.2754-(0.012754*array)))
    return CCN_arr


def prepare_S2image(fpath, scale_factor=10000.0, min_val=0, max_val=10000, no_data_pixels=-9999):
    """
    prepare_image
        This function is used prepare an image for MESMA by scaling the reflectance from 0-1
    Parameters:
       fpath: filepath to an image
       scale_factor: scale factor of reflectance data (MESMA requires values to be in range 0-1)
       min_val: lower boundary of accepted reflectance values, all values <= min_val will be set to no_data_pixels
       max_val: upper boundary of accepted reflectance values, all values > max_val will be set to no_data_pixels
    Returns:
        3D xarray.DataArray with reflectance scaled to range 0-1

    """
    img = rioxarray.open_rasterio(fpath)
    return xr.where((img.min(dim='band')<=min_val) |
                    (img.max(dim='band')>max_val), no_data_pixels*scale_factor,
                    img).T / scale_factor


def prepare_L8image(fpath, scale_factor=100000.0, min_val=0, max_val=100000, no_data_pixels=-9999):
    """
    prepare_image
        This function is used prepare an image for MESMA by scaling the reflectance from 0-1
    Parameters:
       fpath: filepath to an image
       scale_factor: scale factor of reflectance data (MESMA requires values to be in range 0-1)
       min_val: lower boundary of accepted reflectance values, all values <= min_val will be set to no_data_pixels
       max_val: upper boundary of accepted reflectance values, all values > max_val will be set to no_data_pixels
    Returns:
        3D xarray.DataArray with reflectance scaled to range 0-1

    """
    img = rioxarray.open_rasterio(fpath)
    return xr.where((img.min(dim='band')<=min_val) |
                    (img.max(dim='band')>max_val), no_data_pixels*scale_factor,
                    img).T / scale_factor



def prepare_sli(fpath, num_bands):
    """
    prepare_sli
        This function is used prepare a spectral library for MESMA. The reflectance is scaled to range 0-1
    Parameters:
       fpath: filepath to an image
       num_bands: number of bands in the spectral library. Must match number of bands in input image
    Returns:
        (1) 1D array of strings, a class for each endmember in the library
        (2) 2D array of floats and shape (bands, endmembers) with reflectance scaled to range 0-1
    """
    sli = pd.read_csv(fpath)
    class_list = sli.MaterialClass.values
    em_spectra = sli[sli.columns[-num_bands:]].values.astype(float)
    em_spectra /= np.max(em_spectra)

    return class_list, em_spectra.T


def not_modelled_spots(arr1, arr2, arr3):
    """
     not_modelled_spots
         This function is used to fill areas with no appropriate MESMA models (where all fractions = 0) by interpolating
         nearby values. The final arrays are then normalized so that all fractions add up to 1.
     input:
         arr1, arr2, arr3: vegetation, impervious, and soil arrays obtained from the MESMA function
     output:
         normalized_array1, normalized_array2, normalized_array3: filled and normalized arrays for the final fraction maps

     """
    filled_array1 = np.copy(arr1)
    filled_array2 = np.copy(arr2)
    filled_array3 = np.copy(arr3)
    # Create the grid of non-zero pixel coordinates and interpolate values for 0 spots
    zero_indices = np.argwhere((arr1 == 0) & (arr2 == 0) & (arr3 == 0))
    non_zero_indices = np.argwhere((arr1 != 0) | (arr2 != 0) | (arr3 != 0))
    x = non_zero_indices[:, 1]
    y = non_zero_indices[:, 0]
    # Interpolate the values for zero spots
    filled_array1[zero_indices[:, 0], zero_indices[:, 1]] = griddata((x, y), arr1[non_zero_indices[:, 0], non_zero_indices[:, 1]], zero_indices, method='nearest')
    filled_array2[zero_indices[:, 0], zero_indices[:, 1]] = griddata((x, y), arr2[non_zero_indices[:, 0], non_zero_indices[:, 1]], zero_indices, method='nearest')
    filled_array3[zero_indices[:, 0], zero_indices[:, 1]] = griddata((x, y), arr3[non_zero_indices[:, 0], non_zero_indices[:, 1]], zero_indices, method='nearest')
    # Normalize each array
    stacked_array = np.stack((filled_array1, filled_array2, filled_array3))
    sums = np.sum(stacked_array, axis=0)
    normalized_array1 = filled_array1 / sums
    normalized_array2 = filled_array2 / sums
    normalized_array3 = filled_array3 / sums

    return normalized_array1, normalized_array2, normalized_array3


def trimmed_library(fpath, num_bands, row_numbers= None):
    """
     trimmed_library
        This function uses the output of the AMUSES function to extract the relevant spectra for MESMA
    Parameters:
        fpath: path to spectral library
        num_bands: number of bands in spectral library. Must match number of bands in input image
        row_numbers: row number in generic spectral library of each spectra selected by AMUSES
    Returns:
    .   (1) class_list array: The material classes for the trimmed library
        (2) em_spectra array: The  reflectance values for the trimmed library

     """
    df = pd.read_csv(fpath)
    if row_numbers is not None:
        df = df.iloc[row_numbers]

    class_list = df.MaterialClass.values
    em_spectra = df[df.columns[-num_bands:]].values.astype(float)

    # Normalize reflectance values to the range 0-1
    em_spectra /= np.max(em_spectra)
    return class_list, em_spectra.T


def doMESMA(class_list,img, trim_lib):
    """
     doMESMA
         This function carries out Multiple Endmember Spectral Mixture Analysis and subsequent shade normalization
     Parameters:
         class_list: Material classes (impervious, soil, vegetation) extracted from the spectral library
         img: Prepared input image
         trim_lib: Spectral library that has been pruned with the output of AMUSES
         output: path to output file
     Returns:
         3D array with vegetation, impervious surface, and soil fractions
     """

    # Setup MESMA model based on trimmed spectral library
    em_models = mesma.MesmaModels()
    em_models.setup(class_list)
    em_models = mesma.MesmaModels()
    em_models.setup(class_list)

    # Select 2-EM, 3-EM, 4-EM models for unmixing
    em_models.select_level(state=True, level=2)
    em_models.select_level(state=True, level=3)
    em_models.select_level(state=False, level=4)
    for i in np.arange(em_models.n_classes):
        em_models.select_class(state=True, index=i, level=2)
        em_models.select_class(state=True, index=i, level=3)
        em_models.select_class(state=False, index=i, level=4)



    out_fractions = np.zeros((len(np.unique(class_list)) + 1, img.shape[1], img.shape[2])) * np.nan
    total_start = timeit.default_timer()
    start_row = 0
    split = 10  # number of rows per to be unmixed at a time

    for chunk in range(start_row, img.shape[1], split):
        start = timeit.default_timer()
        MESMA = mesma.MesmaCore(n_cores=8)

        models, fractions, rmse, residuals = MESMA.execute(image=img[:, start_row:start_row + split, :].data,
                                                           library=trim_lib,
                                                           look_up_table=em_models.return_look_up_table(),
                                                           em_per_class=em_models.em_per_class,
                                                           constraints=(0, 1.0, -0.1, 0.8, 0.025, -9999, -9999),
                                                           no_data_pixels=np.where(
                                                               img[0, start_row:start_row + split,
                                                               :].data == -9999),
                                                           shade_spectrum=None,
                                                           fusion_value=0.007,
                                                           bands_selection_values=(0.99, 0.01)
                                                           )

        np.seterr(invalid='ignore')
        out_fractions[:, start_row:start_row + split, :] = fractions

        start_row = start_row + split

        stop = timeit.default_timer()

        # Obtain time for each chunk to be unmixed
        print('Chunk Time: ', stop - start)

    total_stop = timeit.default_timer()
    print(f"Total execution time: {total_stop - total_start:.2f} seconds")

    #Perform shade normalization
    out_shade = shade_normalisation.ShadeNormalisation.execute(out_fractions, shade_band=-1)
    return out_shade



