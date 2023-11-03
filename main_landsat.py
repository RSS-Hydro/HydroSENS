### Import required libraries ###
from GEE_Functions import *
from Functions import *
import matplotlib.pyplot
import glob
from spectral_libraries.core import amuses
import datetime
import sys


### Load necessary inputs and specify output folder ###
output = r"path to output folder"
shapefile = r"path to shapefile"

HSG250m = r"path to soil texture file"
sli = r"path to spectral library"

amc = 2 # 1, 2, or 3
p = 10 # Precipitation in mm
StartDate = '2022-08-29' #YYYY-MM-DD
EndDate = '2022-09-02'#YYYY-MM-DD


### Preprocessing ###

aoi = geemap.shp_to_ee(shapefile)
dataframe = gpd.read_file(shapefile)
shapefile_pro = dataframe.crs
shapefile_projection = shapefile_pro.to_string()
#
# #Load and select image
filtered_col, num_images = load_Landsat(aoi, StartDate, EndDate)
if num_images == 0:
        print("No images found for desired time period")
        sys.exit(0)
# Process the selected image
mosaic = mosaic(filtered_col)
time = mosaic.get('system:time_start').getInfo()
date = datetime.datetime.utcfromtimestamp(time / 1000).strftime('%Y-%m-%d')

date = str(date)
print(date)


DEM = getDEM(aoi)
Bandsexport_Landsat(mosaic,shapefile_projection,output,aoi)

# Obtain shapefile geometry and projection
DEMexport_Landsat(DEM,shapefile_projection,output,aoi)

# Delete extra files created by GEE
suffix = "_gcs"
pattern = f"*{suffix}.*"
files = glob.glob(os.path.join(output, pattern))
for file in files:
    os.remove(file)

### Indices Calculations ###

bands = gdal.Open(output + r"\Bands.tif")
band_array = bands.ReadAsArray()
arr3 = bands.GetRasterBand(2).ReadAsArray().astype('float64')
arr4 = bands.GetRasterBand(3).ReadAsArray().astype('float64')
arr5 = bands.GetRasterBand(4).ReadAsArray().astype('float64')
arr6 = bands.GetRasterBand(5).ReadAsArray().astype('float64')


# NDVI
np.seterr(invalid='ignore')
NDVI = (np.divide((arr5-arr4), (arr5+arr4), out=np.zeros_like(arr5), where=(arr5 + arr4) != 0))
CreateFloat(NDVI, bands, "NDVI", output)

# MNDWI
np.seterr(invalid='ignore')
MNDWI = (np.divide((arr3-arr6), (arr3+arr6), out=np.zeros_like(arr3), where=(arr3 + arr6) != 0))



### Water Mask ###

reclassified_MNDWI = np.where(MNDWI > -0.05, 1,0)
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
output_raster = driver.Create(output + r"\bands_masked.tif", bands.RasterXSize, bands.RasterYSize, band_array.shape[0], gdal.GDT_Float64)
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
img = prepare_L8image(output + r"\bands_masked.tif")
class_list_init_, initial_lib = prepare_sli(sli, num_bands = 6)
A = amuses.Amuses()
em_spectra_dict =A.execute(image_array, initial_lib,  0.9, 0.95, 15, (0.0002, 0.02))
em_spectra_list = list(em_spectra_dict.values())
indices_array = em_spectra_dict['amuses_indices']
class_list_init, em_spectra_trim = trimmed_library(sli,
                              num_bands=6, row_numbers=indices_array)

output_file = output + r"\trimmed_library.csv"
wavelengths = [482, 561, 655, 865, 1609, 2201]

data = {
    "MaterialClass": class_list_init,
    **{str(wavelengths[i]): em_spectra_trim[i] for i in range(len(wavelengths))}
}
df = pd.DataFrame(data)
material_order = ['vegetation', 'impervious','soil']
df['MaterialClass'] = pd.Categorical(df['MaterialClass'], categories=material_order, ordered=True)
df = df.sort_values('MaterialClass')
output_csv = output + r"\trimmed_library.csv"
df.to_csv(output_csv, index=False)
class_list, trim_lib = prepare_sli(output + r"\trimmed_library.csv", num_bands = 6)


# Run MESMA algorithm using trimmed spectral library
out_fractions = doMESMA(class_list, img, trim_lib)
final = np.flip(out_fractions, axis =1)
final = np.rot90(final, k=3, axes=(1, 2))
soil = final[0]
impervious = final[1]
vegetation = final[2]
# soil, impervious,vegetation  = not_modelled_spots(array1, array2, array3)

del img
os.remove(output+r"\trimmed_library.csv")
CreateFloat(soil, image, "soil", output)
CreateFloat(impervious, image, "impervious", output)
CreateFloat(vegetation, image, "vegetation", output)


### Global Soil Dataset Processing ###

output_path = output + r"\new_shapefile.shp"
Create_buffer(shapefile, output_path)

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
CreateInt(filled,reference,"filled", output)
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

CreateInt(reclass,soilraster,"HSG_reclass", output)
del soilraster

extract_raster(output + r"\HSG_reclass.tif", output + r"\null_MNDWI.tif", output + r"\HSG_final.tif")

os.remove(output + r"\HSG_reclass.tif")
os.remove(output + r"\filled.tif")


### Initial CN classification for vegetation and soil ###

# Reclassify NDVI
NDVI = gdal.Open(output + r"\NDVI.tif")
newNDVI = NDVI.ReadAsArray()
newNDVI[newNDVI >= 0.62]=10
newNDVI[np.where((0.55 <= newNDVI) & (newNDVI < 0.62))] = 20
newNDVI[(0.31 < newNDVI) & (newNDVI < 0.55)] = 30
newNDVI[newNDVI <= 0.31] = 40

# Reclassify Vegetation Fraction
new_veg = vegetation.copy()
new_veg[new_veg >= 0.75]= 3
new_veg[(0.5 < new_veg) & (new_veg < 0.75)] = 2
new_veg[new_veg <= 0.5]= 1

# Combine
array1 = new_veg + newNDVI
array1[array1 == 33] = 32
array1[array1 == 42] = 41
array1[array1 == 43] = 41
array1[np.isnan(array1)] = 0
array1[np.isinf(array1)] = 0

CreateInt(array1, NDVI, "veghealth", output)
finalshape = gpd.read_file(shapefile)
masked = rasterio.open(output + r"\veghealth.tif")
Extract(output + r"\veghealth.tif", shapefile, output + r"\Vegetation_Health.tif", nodata_value=255)
masked = None
os.remove(output + r"\veghealth.tif")

# Get files
file2 = gdal.Open(output + r"\HSG_final.tif")
array2 = file2.ReadAsArray()
CN_table = 'CN_lookup.csv'

# Vegetation CN Reclassification
veg_reclass = classification(CN_table,array1, array2)


# Soil CN Reclassification
array3 = array1*0
soil_reclass = classification(CN_table,array3,array2)

file2 = None
os.remove(output + r"\HSG_final.tif")

# CCN calculation
imp_CN = 98
CCNarr = (soil_reclass * soil) + (veg_reclass * vegetation) + (imp_CN * impervious)


### Slope Correction ###

# Create slope map isolating pixels >5%
DEMfile = gdal.Open(output + r"\DEM.tif")
DEM = DEMfile.ReadAsArray()
cellsize = 30

px, py = np.gradient(DEM, cellsize)
slope_init = np.sqrt(px ** 2 + py ** 2)
slope = np.degrees(np.arctan(slope_init))

slope[slope < 5] = 0
slope[slope == 90] = 0

# Sharpley-Williams Method for slope correction

AMC_III = AMCIII(CCNarr)
CN_slope_SW = (1/3)*(AMC_III-CCNarr)*(1-((2*2.718281)**(-13.86*slope))) + CCNarr



### Conversion to different AMC if requested ###

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
CCN_arr_final= np.where(mask_array == 0, CCN_arr, 0)

CCN_arr_final[np.isnan(CCN_arr_final)] = 100
CCN_arr_final[np.isinf(CCN_arr_final)] = 100
CCN_arr_final[CCN_arr_final > 100] = 100
CCN_arr_final[CCN_arr_final == 0] = 100

CreateInt(CCN_arr_final, DEMfile, "CCN_masked", output)
finalshape = gpd.read_file(shapefile)
masked = rasterio.open(output + r"\CCN_masked.tif")
Extract(output + r"\CCN_masked.tif", shapefile, output + r"\CCN_final.tif", nodata_value=255)

del  masked,mask, DEMfile
os.remove(output + r"\CCN_masked.tif")
os.remove(output + r"\null_MNDWI.tif")
os.remove(output+ r"\HSG_match.tif")
os.remove(output + r"\DEM.tif")


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
# precipitation in mm - user input

runoff_c = (p - Ia) ** 2 / (p - Ia + storage)
runoff_c[runoff_c < 0] = np.nan

CreateFloat(runoff_c, CCN, "Runoff", output)


### Put CCN file in vector format for use in software, such as HEC-RAS ###

in_path = output + r"\CCN_final.tif"
out_path = output + r"\CCN_final.shp"

src_ds = gdal.Open(in_path)
src_srs = osr.SpatialReference()
src_srs.ImportFromWkt(src_ds.GetProjection())
drv = ogr.GetDriverByName("ESRI Shapefile")
dst_ds = drv.CreateDataSource(out_path)
dst_layername = 'CCN_final'
dst_layer = dst_ds.CreateLayer(dst_layername, srs=src_srs)
fld = ogr.FieldDefn("CN", ogr.OFTInteger)
dst_layer.CreateField(fld)
dst_field = dst_layer.GetLayerDefn().GetFieldIndex("CN")
gdal.Polygonize(src_ds.GetRasterBand(1), None, dst_layer, dst_field, [], callback=None)
del src_ds
del dst_ds
os.remove(output + r"\bands_masked.tif")
