import geemap
import geopandas as gpd
import pycrs
import ee
import ee.mapclient
import os
import rasterio
import numpy as np
from rasterio.merge import merge


def getDEM(aoi):
    DEM = ee.ImageCollection("COPERNICUS/DEM/GLO30")\
        .filterBounds(aoi)\
        .mosaic()\
        .rename('DEM')
    return DEM


def DEMexport(image, shapefile_projection, output, aoi):
  output_file = os.path.join(output, "DEM.tif")

  band_names = ['DEM']

  final_selected = image.select(band_names, band_names)

  geemap.ee_export_image(final_selected, output_file, scale=5, region=aoi.geometry(),
                          crs=shapefile_projection)
  return DEMexport

def count_images_in_roi(collection, roi):
    filtered_collection = collection.filterBounds(roi)
    image_count = filtered_collection.size().getInfo()
    return image_count

def split_and_export_images(roi_shapefile, output_folder, num_rows=100, num_cols=100):
    gdf = gpd.read_file(roi_shapefile)
    if len(gdf) != 1:
        raise ValueError("Shapefile must contain exactly one feature (ROI).")
    roi_geom = gdf.geometry.iloc[0]
    roi_geom_json = roi_geom.__geo_interface__

    if roi_geom_json['type'] == 'Polygon':
        roi = ee.Geometry.Polygon(roi_geom_json['coordinates'])
    elif roi_geom_json['type'] == 'MultiPolygon':
        roi = ee.Geometry.MultiPolygon(roi_geom_json['coordinates']).convexHull()
    else:
        raise ValueError("Geometry must be a Polygon or MultiPolygon.")
    bounds = roi.bounds().getInfo()['coordinates'][0]

    roi_parts = []
    margin = 0.01

    for i in range(num_rows):
        for j in range(num_cols):
            left = bounds[0][0] + (bounds[2][0] - bounds[0][0]) * (j / num_cols) - margin
            right = bounds[0][0] + (bounds[2][0] - bounds[0][0]) * ((j + 1) / num_cols) + margin
            bottom = bounds[0][1] + (bounds[2][1] - bounds[0][1]) * (i / num_rows) - margin
            top = bounds[0][1] + (bounds[2][1] - bounds[0][1]) * ((i + 1) / num_rows) + margin

            part_geometry = ee.Geometry.Rectangle([left, bottom, right, top])
            roi_parts.append(part_geometry)

    for i, part_geometry in enumerate(roi_parts):
        collection = ee.ImageCollection("COPERNICUS/DEM/GLO30")

        num_images_in_roi = count_images_in_roi(collection, part_geometry)

        print(f"Number of images in ROI part {i+1}: {num_images_in_roi}")

        if num_images_in_roi == 1:

            image = collection.filterBounds(part_geometry).first()

            DEM = image.select("DEM") #b1


            export_params = {
                'ee_object': DEM,
                'filename': os.path.join(output_folder, f'image_part_{i}.tif'),
                'scale': 30.922,
                'region': part_geometry,
            }
            geemap.ee_export_image(**export_params)

        else:

            filtered_collection = collection.filterBounds(part_geometry)
            composite_image = filtered_collection.mosaic()

            intersection = composite_image.clip(part_geometry)
            if intersection.geometry().area().getInfo() > 0:

                DEM = composite_image.select("DEM")
                export_params = {
                    'ee_object': DEM,
                    'filename': os.path.join(output_folder, f'image_part_{i}.tif'),
                    'scale': 30.922,
                    'region': intersection.geometry(),
                }
                geemap.ee_export_image(**export_params)


def merge_tiff_files(output_folder, merged_output_path):
    tif_files = [f for f in os.listdir(output_folder) if f.endswith(".tif")]

    src_files_to_mosaic = [rasterio.open(os.path.join(output_folder, tif_file)) for tif_file in tif_files]

    mosaic, out_trans = merge(src_files_to_mosaic)

    with rasterio.open(merged_output_path, "w", driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
                       count=mosaic.shape[0], dtype=mosaic.dtype, crs=src_files_to_mosaic[0].crs,
                       transform=out_trans) as dest:
        dest.write(mosaic)

    for src_file in src_files_to_mosaic:
        src_file.close()

def download_dem(roi_shapefile, out_dir, out_name="DEM.tif"):
    output_folder = os.path.join(out_dir, "split_images")
    os.makedirs(output_folder, exist_ok=True)
    split_and_export_images(roi_shapefile, output_folder)

    merged_output_path = os.path.join(out_dir, out_name)
    merge_tiff_files(output_folder, merged_output_path)
    merged_output_path_ = os.path.splitext(merged_output_path)[0] + "_.tif"
    with rasterio.open(merged_output_path) as dem_src:
        dem_array = dem_src.read(1)

        for i in [0, -1]:
            dem_array[i, :] = np.nanmean(dem_array[i - 1:i + 2, :], axis=0)

        for j in [0, -1]:
            dem_array[:, j] = np.nanmean(dem_array[:, j - 1:j + 2], axis=1)

        with rasterio.open(merged_output_path_, 'w', **dem_src.profile) as dem_dst:
            dem_dst.write(dem_array, 1)


    for file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file)
        os.remove(file_path)
    os.rmdir(output_folder)

    return merged_output_path_