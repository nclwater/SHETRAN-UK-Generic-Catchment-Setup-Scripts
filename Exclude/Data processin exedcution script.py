"""
Data Processing Execution Script

This script is for running code that takes too long to run at scale using notebooks locally (this script can be run easily on the Blades).
It is a dynamic script intended to be edited frequently. No code should be left here that is not saved elsewhere.
"""
import shutil
import os
import os
import zipfile

# import rasterio
from rasterio.merge import merge
import numpy as np

# from scipy.ndimage import generic_filter
# import geopandas as gpd
import pandas as pd

import rasterio.features
# from shapely.geometry import box
# import math
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds

import geopandas as gpd

root = 'I:/SHETRAN_GB_2021/02_Input_Data/01 - National Data Inputs for SHETRAN UK/'
resolution_output = 100
print(resolution_output)

def write_ascii(
        array: np,
        ascii_ouput_path: str,
        xllcorner: float,
        yllcorner: float,
        cellsize: float,
        ncols: int = None,
        nrows: int = None,
        NODATA_value: int = -9999,
        data_format: str = '%1.1f'):

        if len(array.shape) > 0:
            nrows, ncols = array.shape

        file_head = "\n".join(
            ["ncols         " + str(ncols),
             "nrows         " + str(nrows),
             "xllcorner     " + str(xllcorner),
             "yllcorner     " + str(yllcorner),
             "cellsize      " + str(cellsize),
             "NODATA_value  " + str(NODATA_value)])

        with open(ascii_ouput_path, 'wb') as output_filepath:
            np.savetxt(fname=output_filepath, X=array,
                       delimiter=' ', newline='\n', fmt=data_format, comments="",
                       header=file_head
                       )


def read_ascii_raster(file_path, data_type=int, return_metadata=True, replace_NA=False):
    """
    Read ascii raster into numpy array, optionally returning headers.
    """
    headers = []
    dc = {}
    with open(file_path, 'r') as fh:
        for i in range(6):
            asc_line = fh.readline()
            headers.append(asc_line.rstrip())
            key, val = asc_line.rstrip().split()
            dc[key] = val
    ncols = int(dc['ncols'])
    nrows = int(dc['nrows'])
    xll = float(dc['xllcorner'])
    yll = float(dc['yllcorner'])
    cellsize = float(dc['cellsize'])
    nodata = float(dc['NODATA_value'])

    arr = np.loadtxt(file_path, dtype=data_type, skiprows=6)
    if replace_NA:
       arr[arr==nodata] = np.nan

    headers = '\n'.join(headers)
    headers = headers.rstrip()

    if return_metadata:
        return arr, ncols, nrows, xll, yll, cellsize, nodata, headers, dc
    else:
        return arr


# Define a function to calculate the mean of valid neighbors:
def fill_holes(values):
    # This will fill all holes with a value in a neighboring cell.

    center = values[4]  # Center pixel in the 3x3 window
    if np.isnan(center):  # If the center is a hole
        neighbors = values[np.arange(len(values)) != 4]  # Exclude the center
        valid_neighbors = neighbors[~np.isnan(neighbors)]  # Keep valid neighbors
        if len(valid_neighbors) > 0:  # Fill only if there are valid neighbors
            return valid_neighbors.mean()
    return center  # Return the original value if not a hole


# Function for aggregating numpy cells:
def cell_reduce(array, block_size, func=np.nanmean):
    """
    Resample a NumPy array by reducing its resolution using block aggregation.
    Parameters:
    - array: Input NumPy array.
    - block_size: Factor by which to reduce the resolution.
    - func: Aggregation function (e.g., np.nanmean, np.nanmin, np.nanmax).
            Recomended to use nanmean etc. else you will lose coverage
    """
    shape = (array.shape[0] // block_size, block_size, array.shape[1] // block_size, block_size,)

    return func(array.reshape(shape), axis=(1, 3), )



# 1. Load the hydrogeology Data and dissolve to group entries with the same information:
print('Reading')
hydro_shape = gpd.read_file("I:/SHETRAN_GB_2021/07_GIS/hydrogeology625k/HydrogeologyUK_IoM_v5.shp")

# 1b. Format the Low Productivity Names as there are two that do not match: Check with list(set(hydro_shape['CHARACTER']))
print('Processing')
hydro_shape['CHARACTER'].replace('Low productive aquifer', 'Low productivity aquifer', inplace=True)
hydro_shape['FLOW_MECHA'].fillna('No flow mechanism', inplace=True)

# 2. Give each rock unit an ID:
hydro_shape_dissolved = hydro_shape.dissolve(['ROCK_UNIT', 'CLASS', 'CHARACTER', 'FLOW_MECHA', 'SUMMARY'])
# hydro_shape_dissolved.reset_index(inplace=True)
# hydro_shape_dissolved.index.names = ['ID']
hydro_shape_dissolved['ID'] = np.arange(1, hydro_shape_dissolved.shape[0]+1)

# 3. Convert IDs into a raster of desired resolution and correct extents:
# 3a. Define parameters
bounds = (0, 0, 661000, 1241000)  # (x_min, y_min, x_max, y_max)
no_data_value = -9999

# 3b. Calculate raster dimensions
width = int((bounds[2] - bounds[0]) / resolution_output)  # Columns
height = int((bounds[3] - bounds[1]) / resolution_output)  # Rows
transform = rasterio.transform.from_bounds(*bounds, width, height)

# 3b. Rasterize the shapefile
shapes = ((geom, value) for geom, value in zip(hydro_shape_dissolved.geometry, hydro_shape_dissolved['ID']))
raster_data = rasterio.features.rasterize(shapes, out_shape=(height, width), transform=transform, fill=no_data_value)

# Step 5: Save the raster to an ASCII file
print('Writting')
with rasterio.open(
        f'{root}/Processed Data/APM Raster {resolution_output}m.asc', "w", driver="AAIGrid", height=height,
        width=width, count=1, dtype="float32", crs=hydro_shape_dissolved.crs,  # Use CRS from shapefile
        transform=transform, nodata=no_data_value) as dst:
    dst.write(raster_data, 1)

