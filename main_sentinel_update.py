### Import required libraries ###
from GEE_Functions import *
from Functions import *
import matplotlib.pyplot
import glob
from spectral_libraries.core import amuses
from datetime import timedelta, datetime
import sys
import shutil
import time

# Start the timer
start_time = time.time()

def run_hydrosens (main_folder, start_date, end_date, output_master, amc, p):
    shapefiles = [f for f in os.listdir(main_folder) if f.endswith('.shp')]
    for shapefile in shapefiles:
        # Skip shapefiles that contain '_gcs' in their name
        if '_gcs' in shapefile:
            print(f"Skipping shapefile: {shapefile} (contains '_gcs')")
            continue
        shapefile_path = os.path.join(main_folder, shapefile)
        print(f"Processing shapefile: {shapefile_path}")
        aoi = geemap.shp_to_ee(shapefile_path)
        process_dates(start_date, end_date, aoi, output_master, amc, p, shapefile_path)

def process_dates(start_date, end_date, aoi, output_master, amc, p, shapefile_path):
     if isinstance(start_date, str):
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

    all_weather_data = get_daily_weather(start_date, end_date, aoi)

    dates_with_images = []
    vegetation_values = []
    impervious_values = []
    soil_values = []
    curve_number = []
    ndvi_values = []
    avg_temp = []
    avg_p = []

    date = start_date_dt
    while date <= end_date_dt:
        date_str = date.strftime('%Y-%m-%d')
        print(f"Checking for Sentinel-2 image on: {date_str}")

        StartDate = date
        EndDate = StartDate + timedelta(days=1, seconds=-1)
        filtered_col, num_images = load_Sentinel2(aoi, StartDate, EndDate)

        if num_images == 0:
            print(f"No S2 image found. Skipping to next date.")
            date += timedelta(days=1)
            continue

        print(f"Image found for {date_str}, beginning processing...")
        dates_with_images.append(date)
        output = create_output_folder(output_master, date)

        weather_day = all_weather_data.get(date_str)
        if weather_day:
            temperature = weather_day['temperature']
            precipitation = weather_day['precipitation']
            avg_temp.append(temperature)
            avg_p.append(precipitation)
        else:
            # Handle cases where weather data might be missing for a day
            avg_temp.append(np.nan)
            avg_p.append(np.nan)  


        dataframe = gpd.read_file(shapefile_path)
        shapefile_pro = dataframe.crs
        shapefile_projection = shapefile_pro.to_string()
        resample_img = resampling(filtered_col, shapefile_projection)
        DEM = getDEM(aoi)
        Bandsexport(resample_img, shapefile_projection, output, aoi)
        DEMexport(DEM, shapefile_projection, output, aoi)

        bands = gdal.Open(output + r"\Bands.tif")
        band_array = bands.ReadAsArray()
        arr3 = bands.GetRasterBand(2).ReadAsArray().astype('float64')
        arr4 = bands.GetRasterBand(3).ReadAsArray().astype('float64')
        arr8A = bands.GetRasterBand(6).ReadAsArray().astype('float64')
        arr11 = bands.GetRasterBand(7).ReadAsArray().astype('float64')

        # NDVI
        np.seterr(invalid='ignore')
        NDVI = (np.divide((arr8A - arr4), (arr8A + arr4), out=np.zeros_like(arr8A), where=(arr8A + arr4) != 0))
        ndvi_values.append(np.nanmean(NDVI))
        CreateFloat(NDVI, bands, "NDVI", output)

        # MNDWI
        np.seterr(invalid='ignore')
        MNDWI = (np.divide((arr3 - arr11), (arr3 + arr11), out=np.zeros_like(arr3), where=(arr3 + arr11) != 0))

        ### Water Mask ###
        # Default MNDWI threshold is 0

        reclassified_MNDWI = np.where(MNDWI > 0, 1, 0)
        CreateFloat(reclassified_MNDWI, bands, "null_MNDWI", output)
        del MNDWI

        # Sieve sparse, unconnected pixels in MNDWI to maintain contiguous water bodies
        null = gdal.Open(output + r"\null_MNDWI.tif", 1)
        Band = null.GetRasterBand(1)
        gdal.SieveFilter(srcBand=Band, maskBand=None, dstBand=Band, threshold=16, connectedness=8)
        del null, Band

        # Mask out water
        mask = gdal.Open(output + r"\null_MNDWI.tif")
        mask_array = mask.ReadAsArray()

        # Mask and save all bands of band_array
        driver = gdal.GetDriverByName('GTiff')
        output_raster_path = output + r"\bands_masked.tif"
        if os.path.exists(output_raster_path):
            os.remove(output_raster_path)
        output_raster = driver.Create(output + r"\bands_masked.tif", bands.RasterXSize, bands.RasterYSize,
                                      band_array.shape[0],
                                      gdal.GDT_Float64)
        output_raster.SetProjection(bands.GetProjection())
        output_raster.SetGeoTransform(bands.GetGeoTransform())

        for band_index in range(band_array.shape[0]):
            masked_band = np.where(mask_array == 0, band_array[band_index], 0)
            output_raster.GetRasterBand(band_index + 1).WriteArray(masked_band)

        output_raster.FlushCache()
        output_raster = None

        ### MESMA ###

        # Prepare image and spectral library for MESMA
        input_img = r"\Bands.tif"
        image = gdal.Open(output + input_img)
        image_array = image.ReadAsArray()
        image_array[np.isnan(image_array)] = -9999
        image_array[np.isinf(image_array)] = -9999
        image_array[image_array == 0] = -9999
        img = prepare_S2image(output + r"\bands_masked.tif")
        class_list_init_, initial_lib = prepare_sli(sli, num_bands=8)
        A = amuses.Amuses()
        em_spectra_dict = A.execute(image_array, initial_lib, 0.9, 0.95, 15, (0.0002, 0.02))
        em_spectra_list = list(em_spectra_dict.values())
        indices_array = em_spectra_dict['amuses_indices']
        class_list_init, em_spectra_trim = trimmed_library(sli,
                                                           num_bands=8, row_numbers=indices_array)

        output_file = output + r"\trimmed_library.csv"
        wavelengths = [490, 560, 665, 783, 842, 865, 1610, 2190]

        data = {
            "MaterialClass": class_list_init,
            **{str(wavelengths[i]): em_spectra_trim[i] for i in range(len(wavelengths))}
        }
        df = pd.DataFrame(data)
        material_order = ['vegetation', 'impervious', 'soil']
        df['MaterialClass'] = pd.Categorical(df['MaterialClass'], categories=material_order, ordered=True)
        df = df.sort_values('MaterialClass')
        output_csv = output + r"\trimmed_library.csv"
        df.to_csv(output_csv, index=False)
        class_list, trim_lib = prepare_sli(output + r"\trimmed_library.csv", num_bands=8)

        # Run MESMA algorithm using trimmed spectral library
        out_fractions = doMESMA(class_list, img, trim_lib)
        final = np.flip(out_fractions, axis=1)
        final = np.rot90(final, k=3, axes=(1, 2))
        soil = final[0]
        impervious = final[1]
        vegetation = final[2]
        vegetation_values.append(np.nanmean(vegetation))
        impervious_values.append(np.nanmean(impervious))
        soil_values.append(np.nanmean(soil))


        del img
        os.remove(output + r"\trimmed_library.csv")
        CreateFloat(soil, image, "soil", output)
        CreateFloat(impervious, image, "impervious", output)
        CreateFloat(vegetation, image, "vegetation", output)

        ### Global Soil Dataset Processing ###

        output_path = output + r"\new_shapefile.shp"
        Create_buffer(shapefile_path, output_path)

        # Matching global dataset projection to new shapefile in order to extract by mask
        data = gpd.read_file(output + r"\new_shapefile.shp")
        HSG250m_open = gdal.Open(HSG250m)
        setcrs = HSG250m_open.GetProjection()
        data = data.to_crs(setcrs)
        data.to_file(output + r"\new_shapefile.shp")
        del data

        # Extract study area from global dataset
        HSGraster = rasterio.open(HSG250m)
        newshape = gpd.read_file(output + r"\new_shapefile.shp")
        initialExtract = Extract(HSG250m, output + r"\new_shapefile.shp", output + r"\extracted.tif", nodata_value=255)

        # Reproject extracted raster to match MNDWI
        MNDWI = gdal.Open(output + r"\null_MNDWI.tif")
        setcrs = MNDWI.GetProjection()
        inputfile = output + r"\extracted.tif"
        output_raster = output + r"\HSG_match.tif"

        # Extract the resolution information from the MNDWI raster
        MNDWI_geotransform = MNDWI.GetGeoTransform()
        MNDWI_res = (MNDWI_geotransform[1], MNDWI_geotransform[5])
        warp = gdal.Warp(output_raster, inputfile, dstSRS=setcrs, xRes=MNDWI_res[0],
                         yRes=MNDWI_res[1], outputType=gdal.GDT_Int16)

        del inputfile, output_raster, MNDWI, warp
        os.remove(output + r"\extracted.tif")

        # Fill NoData holes in the extracted data
        reference = gdal.Open(output + r"\HSG_match.tif")
        data = matplotlib.pyplot.imread(output + r"\HSG_match.tif")
        filled = Fill(data)
        CreateInt(filled, reference, "filled", output)
        matplotlib.pyplot.close()
        reference = None
        del data, filled, reference

        os.remove(output + r"\new_shapefile.shp")
        os.remove(output + r"\new_shapefile.shx")
        os.remove(output + r"\new_shapefile.cpg")
        os.remove(output + r"\new_shapefile.dbf")
        os.remove(output + r"\new_shapefile.prj")

        # Reclassify to HSG value
        soilraster = gdal.Open(output + r"\filled.tif")
        reclass = soilraster.ReadAsArray()

        reclass[np.where((1 <= reclass) & (reclass <= 3))] = 4
        reclass[np.where((3 <= reclass) & (reclass <= 8))] = 3
        reclass[reclass == 10] = 3
        reclass[reclass == 11] = 2
        reclass[reclass == 9] = 2
        reclass[reclass == 12] = 1

        CreateInt(reclass, soilraster, "HSG_reclass", output)
        del soilraster

        extract_raster(output + r"\HSG_reclass.tif", output + r"\null_MNDWI.tif", output + r"\HSG_final.tif")

        os.remove(output + r"\HSG_reclass.tif")
        os.remove(output + r"\filled.tif")

        ### Initial CN classification for vegetation and soil ###

        # Reclassify NDVI
        NDVI = gdal.Open(output + r"\NDVI.tif")
        newNDVI = NDVI.ReadAsArray()
        newNDVI[newNDVI >= 0.62] = 10
        newNDVI[np.where((0.55 <= newNDVI) & (newNDVI < 0.62))] = 20
        newNDVI[(0.31 < newNDVI) & (newNDVI < 0.55)] = 30
        newNDVI[newNDVI <= 0.31] = 40

        # Reclassify Vegetation Fraction
        new_veg = vegetation.copy()
        new_veg[new_veg >= 0.75] = 3
        new_veg[(0.5 < new_veg) & (new_veg < 0.75)] = 2
        new_veg[new_veg <= 0.5] = 1

        # Combine
        array1 = new_veg + newNDVI
        array1[array1 == 42] = 41
        array1[array1 == 43] = 41
        array1[np.isnan(array1)] = 0
        array1[np.isinf(array1)] = 0

        CreateInt(array1, NDVI, "veghealth", output)
        finalshape = gpd.read_file(shapefile_path)
        masked = rasterio.open(output + r"\veghealth.tif")
        Extract(output + r"\veghealth.tif", shapefile_path, output + r"\Vegetation_Health.tif", nodata_value=255)
        masked = None
        os.remove(output + r"\veghealth.tif")

        # Get files
        file2 = gdal.Open(output + r"\HSG_final.tif")
        array2 = file2.ReadAsArray()
        CN_table = r"C:\Users\ben\PycharmProjects\CN_Project\CN_lookup.csv"

        # Vegetation CN Reclassification
        veg_reclass = classification(CN_table, array1, array2)

        # Soil CN Reclassification
        array3 = array1 * 0
        soil_reclass = classification(CN_table, array3, array2)

        file2 = None
        os.remove(output + r"\HSG_final.tif")

        # CCN calculation
        imp_CN = 98
        CCNarr = (soil_reclass * soil) + (veg_reclass * vegetation) + (imp_CN * impervious)

        ### Slope Correction ###

        # Create slope map isolating pixels >5%
        DEMfile = gdal.Open(output + r"\DEM.tif")
        DEM = DEMfile.ReadAsArray()
        cellsize = 10

        px, py = np.gradient(DEM, cellsize)
        slope_init = np.sqrt(px ** 2 + py ** 2)
        slope = np.degrees(np.arctan(slope_init))

        slope[slope < 5] = 0
        slope[slope == 90] = 0

        # Sharpley-Williams Method for slope correction

        AMC_III = AMCIII(CCNarr)
        CN_slope_SW = (1 / 3) * (AMC_III - CCNarr) * (1 - ((2 * 2.718281) ** (-13.86 * slope))) + CCNarr

        ### Conversion to different AMC if required ###

        while amc > 0:
            choice = amc
            if choice == 1:
                CCN_arr = AMCI(CN_slope_SW)
            elif choice == 3:
                CCN_arr = AMCIII(CN_slope_SW)
            else:
                CCN_arr = CN_slope_SW
            break

        # One last extraction to clean up edges of CCN map
        CCN_arr_final = np.where(mask_array == 0, CCN_arr, 0)

        CCN_arr_final[np.isnan(CCN_arr_final)] = 100
        CCN_arr_final[np.isinf(CCN_arr_final)] = 100
        CCN_arr_final[CCN_arr_final > 100] = 100
        CCN_arr_final[CCN_arr_final == 0] = 100
        curve_number.append(np.nanmean(CCN_arr_final))
        CreateInt(CCN_arr_final, DEMfile, "CCN_masked", output)
        finalshape = gpd.read_file(shapefile_path)
        masked = rasterio.open(output + r"\CCN_masked.tif")
        Extract(output + r"\CCN_masked.tif", shapefile_path, output + f"\CCN_final.tif", nodata_value=255)


        del masked, mask, DEMfile
        os.remove(output + r"\CCN_masked.tif")
        os.remove(output + r"\null_MNDWI.tif")
        os.remove(output + r"\HSG_match.tif")
        os.remove(output + r"\DEM.tif")
        os.remove(output+ r'\impervious.tif')

        ### Runoff Calculation ###

        """US Department of Agriculture (USDA) Natural Resources Conservation Service (NRCS) 
        CN method for determining the Runoff Coefficient
                Storage = 254 * (1-CN/100)
                Initial Abstraction = 0.2*S
                Runoff  = (P-Ia)^2/(P-Ia+S)
        """

        # Storage
        CCN = gdal.Open(output + r"\CCN_final.tif")
        CCN_array = CCN.ReadAsArray()

        storage = 254 * (1 - (CCN_array / 100.0))

        # Initial Abstraction
        Ia = 0.2 * storage

        # Runoff Coefficient
        # precipitation in mm

        runoff_c = (p - Ia) ** 2 / (p - Ia + storage)
        runoff_c[runoff_c < 0] = np.nan

        CreateFloat(runoff_c, CCN, "Runoff", output)
        date += timedelta(days=1)

    data = {
        'date': [d.strftime('%Y-%m-%d') for d in dates_with_images],
        'veg_mean': vegetation_values,
        'soil_mean': soil_values,
        'curve_number': curve_number,
        'ndvi': ndvi_values,
        'temperature': avg_temp,
        'precipitation': avg_p
            }
    print("extracted data", data)
    df = pd.DataFrame(data)
    df = df[
    (df[['veg_mean', 'soil_mean', 'curve_number', 'ndvi', 'temperature']] != 0).all(axis=1)
]


    shapefile_name = os.path.splitext(shapefile_path)[0]
    if dates_with_images:
        s2_data = {
            'date': [d.strftime('%Y-%m-%d') for d in dates_with_images],
            'veg_mean': vegetation_values,
            'impervious_mean': impervious_values,
            'soil_mean': soil_values,
            'curve_number': curve_number,
            'ndvi': ndvi_values,
        }
        df_s2 = pd.DataFrame(s2_data).dropna()
        output_s2_csv = os.path.join(output_master, f"{shapefile_name}_s2.csv")
        df_s2.to_csv(output_s2_csv, index=False)

    if all_weather_data:
        df_climate = pd.DataFrame.from_dict(all_weather_data, orient='index')
        df_climate.index.name = 'date'
        df_climate.reset_index(inplace=True)
        output_climate_csv = os.path.join(output_master, f"{shapefile_name}_climate.csv")
        df_climate.to_csv(output_climate_csv, index=False)


    # # Delete extra files
    suffix = "_gcs"
    pattern = f"*{suffix}.*"
    files = glob.glob(os.path.join(output_master, pattern))
    for file in files:
        os.remove(file)

    for date in dates_with_images:
        date_folder = os.path.join(output_master, date.strftime('%Y-%m-%d'))

        if os.path.isdir(date_folder):
            try:
                os.chmod(date_folder, 0o777)  # Ensure permissions allow deletion
                shutil.rmtree(date_folder)
                print(f"Deleted folder: {date_folder}")
            except Exception as e:
                print(f"Failed to delete {date_folder}: {e}")


    return dates_with_images


def create_output_folder(base_output, date):
    """Create a subfolder for the specific date."""
    # Convert date to string format YYYY-MM-DD
    date_str = date.strftime('%Y-%m-%d')
    # Create folder path
    folder_path = os.path.join(base_output, date_str)
    # Check if folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


## Load necessary inputs and specify output folder ###
output_master = r"Z:\Main\RSS-Hydro\Projects\01_Commercial\ADA\02_Work\02 Models\HydroSENS\Togo"

HSG250m = r"Z:\Main\RSS-Hydro\Projects\04_Product line\HydroSENS\02_Deliverables\02_Data\sol_texture.class_usda.tt_m_250m_b0..0cm_1950..2017_v0.2.tif"
sli = r"Z:\Main\RSS-Hydro\Projects\04_Product line\HydroSENS\02_Deliverables\02_Data\VIS_speclib_sentinel.csv"

StartDate = '2024-04-12'
EndDate = '2024-04-25'

amc = 2  # AMC I (1), AMC II (2), AMC III (3)
p = 100  # Precipitation in mm

run_hydrosens(output_master, StartDate, EndDate, output_master, amc, p)

end_time = time.time()

# Calculate and print the duration
print(f"Script took {end_time - start_time:.2f} seconds to run.")



