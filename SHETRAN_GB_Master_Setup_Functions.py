# -------------------------------------------------------------
# SHETRAN Generic Catchment Simulation Functions
# -------------------------------------------------------------
# Ben Smith, adapted from previous codes.
# 27/07/2022
# -------------------------------------------------------------
# This code holds the functions required for SHETRAN Generic
# Catchment Simulation Creator. Function updates should always
# be backwards compatible and as simple as possible to aid future
# users.
#
# Notes:
# CHESS rainfall reads in the Y coordinates backwards, if you
# change the meteorological inputs then check the coordinates.
# -------------------------------------------------------------

# --- Load in Packages ----------------------------------------
import os
# import itertools
import xarray as xr
import pandas as pd
import geopandas as gpd
import copy
import datetime
import multiprocessing as mp
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_fill_holes
import shutil

# import hydroeval as he  # Slightly tricky to install needed for calculating objective functions
# https://pypi.org/project/hydroeval/ - open conda prompt: pip install hydroeval


# --- Create Functions ----------------------------------------
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

    # If required, swap out the no data values for np.nan:
    if replace_NA:
       arr[arr==nodata] = np.nan

    headers = '\n'.join(headers)
    headers = headers.rstrip()

    if return_metadata:
        return arr, ncols, nrows, xll, yll, cellsize, nodata, headers, dc
    else:
        return arr


# Create a function that can write ascii style data:
def write_ascii(
        array: np, ascii_ouput_path: str, xllcorner: float, yllcorner: float,
        cellsize: float, ncols: int = None, nrows: int = None, NODATA_value: int = -9999, data_format: str = '%1.1f'):

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


# Define a function to calculate the mean of valid neighbors (used for processing input rasters):
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


def get_date_components(date_string, fmt='%Y-%m-%d'):
    date = datetime.datetime.strptime(date_string, fmt)
    return date.year, date.month, date.day


def get_veg_string(vegetation_array_for_library, static_input_dataset):
    """
    Get string containing vegetation details for the library file.
    """

    veg_vals = [int(v) for v in np.unique(vegetation_array_for_library[vegetation_array_for_library != -9999])]
    # strickler_dict = {1: 0.6, 2: 3, 3: 0.5, 4: 1, 5: 0.25, 6: 2, 7: 5}
    # strickler_dict

    # Extract the vegetation properties from the metadata
    veg_props = static_input_dataset.land_cover_lccs.attrs["land_cover_key"].loc[
        static_input_dataset.land_cover_lccs.attrs["land_cover_key"]["Veg Type #"].isin(veg_vals)].copy()
    # veg_props["strickler"] = [strickler_dict[item] for item in veg_props["Veg Type #"]]

    # Write the subset of properties out to a string
    veg_string = veg_props.to_csv(header=False, index=False)
    # veg_string = "<VegetationDetail>" + veg_string[:-1].replace("\n", "</VegetationDetail>\n<VegetationDetail>") +
    # "</VegetationDetail>\n"
    tmp = []
    for veg_line in veg_string[:-1].split('\n'):
        tmp.append('<VegetationDetail>' + veg_line.rstrip() + '</VegetationDetail>')
    veg_string = '\n'.join(tmp)
    return veg_string


def get_soil_strings(orig_soil_types, new_soil_types, static_input_dataset):
    """
    Get the unique soil columns out for the library file.
    """

    orig_soil_types = [int(v) for v in orig_soil_types]
    new_soil_types = [int(v) for v in new_soil_types]

    # Find the attributes of those columns
    soil_props = static_input_dataset.soil_type_APM.attrs["soil_key"].loc[
        static_input_dataset.soil_type_APM.attrs["soil_key"]["Soil Category"].isin(
            orig_soil_types)].copy()  # Change soil_type_APM to soil_type to use old soils!

    for orig_type, new_type in zip(orig_soil_types, new_soil_types):
        soil_props.loc[soil_props['Soil Category'] == orig_type, 'tmp0'] = new_type
    soil_props['Soil Category'] = soil_props['tmp0'].values
    soil_props['Soil Category'] = soil_props['Soil Category'].astype(int)

    # Rename the soil types for the new format of shetran
    soil_props["New_Soil_Type"] = soil_props["Soil Type"].copy()
    aquifer_types = ["NoGroundwater", "LowProductivityAquifer", "ModeratelyProductiveAquifer",
                     "HighlyProductiveAquifer"]

    soil_props['tmp1'] = np.where(
        (~soil_props['Soil Type'].isin(aquifer_types)),
        'Top_' + soil_props['Soil Type'],
        soil_props['Soil Type']
    )
    soil_props['tmp2'] = np.where(
        (~soil_props['Soil Type'].isin(aquifer_types)),
        'Sub_' + soil_props['Soil Type'],
        soil_props['Soil Type']
    )
    soil_props['New_Soil_Type'] = np.where(
        soil_props['Soil Layer'] == 1, soil_props['tmp1'], soil_props['tmp2']
    )

    # Assign a new soil code to the unique soil types
    soil_codes = soil_props.New_Soil_Type.unique()
    soil_codes_dict = dict(zip(soil_codes, [i + 1 for i in range(len(soil_codes))]))
    soil_props["Soil_Type_Code"] = [soil_codes_dict[item] for item in soil_props.New_Soil_Type]

    # Select the relevant information for the library file
    soil_types = soil_props.loc[:, ["Soil_Type_Code", "New_Soil_Type", "Saturated Water Content",
                                    "Residual Water Content", "Saturated Conductivity (m/day)",
                                    "vanGenuchten- alpha (cm-1)", "vanGenuchten-n"]]
    soil_types.drop_duplicates(inplace=True)

    soil_cols = soil_props.loc[:, ["Soil Category", "Soil Layer", "Soil_Type_Code", "Depth at base of layer (m)"]]

    # Write the subset of properties out to a string
    soil_types_string = soil_types.to_csv(header=False, index=False)
    soil_cols_string = soil_cols.to_csv(header=False, index=False)

    # soil_types_string = "<SoilProperty>" + soil_types_string[:-1].replace("\n", "</SoilProperty>\n<SoilProperty>")
    # + "</SoilProperty>\n" soil_cols_string = "<SoilDetail>" + soil_cols_string[:-1].replace("\n",
    # "</SoilDetail>\n<SoilDetail>") + "</SoilDetail>\n"

    tmp = []
    for line in soil_types_string[:-1].split('\n'):
        tmp.append('<SoilProperty>' + line.rstrip() + '</SoilProperty>')
    soil_types_string = '\n'.join(tmp)

    tmp = []
    for line in soil_cols_string[:-1].split('\n'):
        tmp.append('<SoilDetail>' + line.rstrip() + '</SoilDetail>')
    soil_cols_string = '\n'.join(tmp)

    return soil_types_string, soil_cols_string

def calculate_library_channel_parameters(grid_resolution):
    # Set the number of upstream cells required to generate a channel - should be exponential, but linear seems to work.
    grid_accumulation = str(int(2000/grid_resolution))  # 2000 seems to work well as a simple value.
    # Set the minimum channel drop between cells. Default is 0.5m per 1km. Linear relationship with resolution.
    channel_drop = str(grid_resolution/2000)
    return grid_accumulation, channel_drop


def create_library_file(
        sim_output_folder, catch, veg_string, soil_types_string, soil_cols_string,
        sim_startime, sim_endtime, grid_resolution, prcp_timestep=24, pet_timestep=24):
    """Create library file."""

    start_year, start_month, start_day = get_date_components(sim_startime)
    end_year, end_month, end_day = get_date_components(sim_endtime)

    # Calculate channel parameters for library file:
    grid_accumulation_value, channel_drop_value = calculate_library_channel_parameters(grid_resolution)

    output_list = [
        '<?xml version=1.0?><ShetranInput>',
        '<ProjectFile>{}_ProjectFile</ProjectFile>'.format(catch),
        '<CatchmentName>{}</CatchmentName>'.format(catch),
        '<DEMMeanFileName>{}_DEM.asc</DEMMeanFileName>'.format(catch),
        '<DEMminFileName>{}_MinDEM.asc</DEMMinFileName>'.format(catch),
        '<MaskFileName>{}_Mask.asc</MaskFileName>'.format(catch),
        '<VegMap>{}_LandCover.asc</VegMap>'.format(catch),
        '<SoilMap>{}_Soil.asc</SoilMap>'.format(catch),
        '<LakeMap>{}_Lake.asc</LakeMap>'.format(catch),
        '<PrecipMap>{}_Cells.asc</PrecipMap>'.format(catch),
        '<PeMap>{}_Cells.asc</PeMap>'.format(catch),
        '<VegetationDetails>',
        '<VegetationDetail>Veg Type #, Vegetation Type, Canopy storage capacity (mm), Leaf area index, '
        'Maximum rooting depth(m), AE/PE at field capacity,Strickler overland flow coefficient</VegetationDetail>',
        veg_string,
        '</VegetationDetails>',
        '<SoilProperties>',
        '<SoilProperty>Soil Number,Soil Type, Saturated Water Content, Residual Water Content, Saturated Conductivity '
        '(m/day), vanGenuchten- alpha (cm-1), vanGenuchten-n</SoilProperty> Avoid spaces in the Soil type names',
        soil_types_string,
        '</SoilProperties>',
        '<SoilDetails>',
        '<SoilDetail>Soil Category, Soil Layer, Soil Type, Depth at base of layer (m)</SoilDetail>',
        soil_cols_string,
        '</SoilDetails>',
        '<InitialConditions>0</InitialConditions>',
        '<PrecipitationTimeSeriesData>{}_Precip.csv</PrecipitationTimeSeriesData>'.format(catch),
        '<PrecipitationTimeStep>{}</PrecipitationTimeStep>'.format(prcp_timestep),
        '<EvaporationTimeSeriesData>{}_PET.csv</EvaporationTimeSeriesData>'.format(catch),
        '<EvaporationTimeStep>{}</EvaporationTimeStep>'.format(pet_timestep),
        '<MaxTempTimeSeriesData>{}_Temp.csv</MaxTempTimeSeriesData>'.format(catch),
        '<MinTempTimeSeriesData>{}_Temp.csv</MinTempTimeSeriesData>'.format(catch),
        '<StartDay>{}</StartDay>'.format(start_day, '02'),
        '<StartMonth>{}</StartMonth>'.format(start_month, '02'),
        '<StartYear>{}</StartYear>'.format(start_year),
        '<EndDay>{}</EndDay>'.format(end_day, '02'),
        '<EndMonth>{}</EndMonth>'.format(end_month, '02'),
        '<EndYear>{}</EndYear>'.format(end_year),
        f'<RiverGridSquaresAccumulated>{grid_accumulation_value}</RiverGridSquaresAccumulated> Number of upstream '
        'grid squares needed to produce a river channel. A larger number will have fewer river channels.',
        '<DropFromGridToChannelDepth>2</DropFromGridToChannelDepth> Minimum value is 2 (standard for 1km).If there are '
        'numerical problems with error 1060 this can be increased.',
        f'<MinimumDropBetweenChannels>{channel_drop_value}</MinimumDropBetweenChannels> Depends on the grid size and '
        'catchment steepness. 1m/1km is a sensible starting point but more gently sloping catchments it can be reduced.',
        '<RegularTimestep>1.0</RegularTimestep> This is the standard Shetran timestep it is automatically reduced in '
        'rain. The standard value is 1 hour. The maximum allowed value is 2 hours.',
        '<IncreasingTimestep>0.05</IncreasingTimestep> speed of increase in timestep after rainfall back to the '
        'standard timestep. The standard value is 0.05. If if there are numerical problems with error 1060 it can be '
        'reduced to 0.01 but the simulation will take longer.',
        '<SimulatedDischargeTimestep>24.0</SimulatedDischargeTimestep> This should be the same as the measured '
        'discharge.',
        '<SnowmeltDegreeDayFactor>0.0002</SnowmeltDegreeDayFactor> Units  = mm s-1 C-1',
        '</ShetranInput>',
    ]
    output_string = '\n'.join(output_list)

    with open(sim_output_folder + catch + "_LibraryFile.xml", "w") as f:
        f.write(output_string)


def create_static_maps(static_input_dataset, xll, yll, ncols, nrows, cellsize,
                       static_output_folder, headers, catch, mask, nodata=-9999):
    """
    Write ascii files for DEM, minimum DEM, lake map, vegetation type map and soil map.
    """

    # Helper dictionary of details of static fields:
    # #- keys are names used for output and values are lists of variable name in
    # master static dataset alongside output number format
    static_field_details = {
        'DEM': ['surface_altitude', '%.2f'],
        'MinDEM': ['surface_altitude_min', '%.2f'],
        'Lake': ['lake_presence', '%d'],
        'LandCover': ['land_cover_lccs', '%d'],
        'Soil': ['soil_type_APM', '%d'],
    }

    xur = xll + (ncols * cellsize) - 1
    yur = yll + (nrows * cellsize) - 1
    catch_data = static_input_dataset.sel(y=slice(yur, yll), x=slice(xll, xur))

    # If we have additional data loaded in that we want to cut out, add these to the fields above:
    static_field_details_names = [v[0] for v in static_field_details.values()]
    static_input_data_names = list(catch_data.keys())
    names_to_add = [n for n in static_input_data_names if n not in static_field_details_names]
    for name in names_to_add:
        static_field_details[name] = [name, '%d']

    # Save each variable to ascii raster
    for array_name, array_details in static_field_details.items():
        array = copy.deepcopy(catch_data[array_details[0]].values)

        # Renumber soil types so consecutive from one
        if array_name == 'Soil':

            array_new = np.zeros(shape=array.shape)
            array_new[array_new == 0] = -9999

            orig_soil_types = np.unique(array[mask != nodata]).tolist()
            new_soil_types = range(1, len(orig_soil_types) + 1)
            for orig_type, new_type in zip(orig_soil_types, new_soil_types):
                array_new[array == orig_type] = new_type
            array = array_new

        array[mask == nodata] = nodata
        # if array_name in ['DEM', 'MinDEM']:
        #     array[mask <= 0] = 0.01

        # Remove any values <0 in the DEMs as these will crash the prepare.exe.
        if array_name in ['DEM', 'MinDEM']:
            array[(array > -9999) & (array <= 0)] = 0.01

        # Write the data out:
        map_output_path = static_output_folder + catch + '_' + array_name + '.asc'
        np.savetxt(
            map_output_path, array, fmt=array_details[1], header=headers, comments=''
        )

        # Get vegetation and soil type arrays for library file construction
        if array_name == 'LandCover':
            vegetation_arr = array
        if array_name == 'Soil':
            soil_arr = array

    # Also save mask:
    np.savetxt(
        static_output_folder + catch + '_Mask.asc', mask, fmt='%d', header=headers,
        comments=''
    )

    return vegetation_arr, soil_arr, orig_soil_types, new_soil_types


def find_rainfall_files(year_from, year_to):
    x = [str(y) + '.nc' for y in range(year_from, year_to + 1)]
    return x


def find_CHESS_temperature_or_PET_files(folder_path, year_from, year_to):
    files = sorted(os.listdir(folder_path))
    x = []
    for fn in files:
        if fn[-3:] == ".nc":
            if int(fn.split('_')[-1][:4]) in range(year_from, year_to + 1):
                x.append(fn)
    return x


def load_and_crop_xarray(filepath, mask):
    # Load the dataset
    ds = xr.open_dataset(filepath)

    # Get the bounding box from the mask
    mask_bounds = {  # We are adding a buffer just to be safe, this seems to help!
        "x_min": mask.x.min().item()-1000,
        "x_max": mask.x.max().item()+1000,
        "y_min": mask.y.min().item()-1000,
        "y_max": mask.y.max().item()+1000,
    }

    # Crop the dataset to the mask's bounding box. Do this according to which way y axis is ordered (differs for different datasets).
    y_coords = ds.y.values
    if (y_coords[1] - y_coords[0]) > 0:
        cropped_ds = ds.sel(
            # Cropping with .sel() may load the full dataset if it is too large. If this is a concern, consider using rioxarray for better I/O efficiency.
            x=slice(mask_bounds["x_min"], mask_bounds["x_max"]),
            y=slice(mask_bounds["y_min"], mask_bounds["y_max"]),
        )
        # print("y-coordinates are increasing.")
    else:
        cropped_ds = ds.sel(
            x=slice(mask_bounds["x_min"], mask_bounds["x_max"]),
            y=slice(mask_bounds["y_max"], mask_bounds["y_min"]),
        )
        # print("y-coordinates are decreasing.")

    return cropped_ds

def build_climate_data_with_xarray(cells_map_filepath, netCDF_filepath_list, h5_variable_name,
                                   start_date, end_date, climate_csv_output_filepath):
    """

    :param cells_map_filepath: filepath to a Cells ascii file.
    :param netCDF_filepath_list: Must be a list of full filepaths.
    :param h5_variable_name: 'pet' (CHESS), 'tas' (CHESS), rainfall_amount (GEAR).
    :param climate_csv_output_filepath:
    :return:
    """

    # Load the ASCII raster of cell numbers
    with rasterio.open(cells_map_filepath) as src:
        mask_data = src.read(1)  # Read the first (and likely only) band
        mask_transform = src.transform

    # Create coordinate arrays:
    mask_height, mask_width = mask_data.shape
    mask_x = np.arange(mask_width) * mask_transform[0] + mask_transform[2]
    mask_y = np.arange(mask_height) * mask_transform[4] + mask_transform[5]

    # The mask works best if it uses centroids, but this may change depending on the data source!
    # TODO: This is for further consideration / checks, but works well so far.
    mask_x = mask_x + mask_transform[0] / 2
    mask_y = mask_y - mask_transform[0] / 2

    # Build the mask into an xarray:
    mask = xr.DataArray(
        mask_data,
        coords={"y": mask_y, "x": mask_x},
        dims=["y", "x"],
        name="mask",
    )

    # Load the climate data:
    print('------------ Reading datasets...')
    datasets = [load_and_crop_xarray(fp, mask) for fp in netCDF_filepath_list]
    print('------------ Formatting arrays...')

    # Stack these into a single xarray ordered by time variable:
    stacked_ds = xr.concat(datasets, dim="time")

    # Crop the data to the desired time period:
    ds_cropped = stacked_ds.sel(time=slice(start_date, end_date))

    # Interpolate the stacked dataset to the mask's grid resolution.
    # This will take the nearest cell if the mask cell crosses multiple climate cells.
    # TODO: You may wish to change this to linear interpolation (or similar). This tends to prioritise the lower left value.
    stacked_resampled = ds_cropped.interp(
        x=mask.x,
        y=mask.y,
        method="nearest"  # Or 'linear' for smoother interpolation
    )

    # -- Now begin to process the data into csv format.

    # Flatten the mask and rainfall data:
    mask_flat = mask.values.flatten()  # Convert mask to a 1D array
    climate_flattened = stacked_resampled[h5_variable_name].values.reshape(stacked_resampled.sizes["time"],
                                                                           -1)  # Reshape rainfall to [time, all grid cells]

    # Get indices of active cells:
    active_indices = mask_flat > 0

    # Filter active cells from rainfall data:
    climate_active_cells = climate_flattened[:, active_indices]

    # Create column names based on the mask indices:
    column_names = [int(idx) for idx in mask_flat[active_indices]]

    # Create the DataFrame that will be the csv:
    climate_df = pd.DataFrame(climate_active_cells, columns=column_names)

    # If using temperature, change from Kelvin to degrees:
    if h5_variable_name == 'tas':
        climate_df -= 273.15

    # Round the data to 1 decimal place
    climate_df = climate_df.round(1)

    # Add the time column:
    climate_df.insert(len(climate_df.columns), "Time", stacked_resampled["time"].values)

    # Save to CSV:
    climate_df.to_csv(climate_csv_output_filepath, index=False)


def run_build_climate_data_with_xarray(mask_filepath, climate_output_folder, catchment, climate_startime, climate_endtime,
                                       prcp_folder, tas_folder, pet_folder):
    """
    This will run the function for building climate csv's and the cell map.
    The tas and pet files are located, the rainfall files are built from scratch as their names are easy. This doesn't havet o be the case and could be changed.
    :param climate_output_folder:
    :param catchment:
    :param climate_startime:
    :param climate_endtime:
    :param prcp_folder:
    :param tas_folder:
    :param pet_folder:
    :return:
    """

    # Create the cell map:
    map_output_path = climate_output_folder + catchment + '_Cells.asc'
    make_cell_map(mask_filepath, map_output_path)

    # Get the start years (used for making rainfall filenames and selecting climate files):
    start_year, _, _ = get_date_components(climate_startime)
    end_year, _, _ = get_date_components(climate_endtime)

    print("-------- Processing rainfall data.")
    prcp_input_file_names = find_rainfall_files(start_year, end_year)
    prcp_input_files = [os.path.join(prcp_folder, file) for file in prcp_input_file_names]
    series_output_path = climate_output_folder + catchment + '_Precip.csv'

    # if not os.path.exists(series_output_path):
    build_climate_data_with_xarray(
        cells_map_filepath=map_output_path,
        netCDF_filepath_list=prcp_input_files,
        h5_variable_name='rainfall_amount',
        start_date=climate_startime, end_date=climate_endtime,
        climate_csv_output_filepath=series_output_path)

    # Make temperature time series
    print("-------- Processing temperature data.")
    tas_input_filenames = find_CHESS_temperature_or_PET_files(tas_folder, start_year, end_year)
    tas_input_files = [os.path.join(tas_folder, file) for file in tas_input_filenames]
    series_output_path = climate_output_folder + catchment + '_Temp.csv'

    # if not os.path.exists(series_output_path):
    build_climate_data_with_xarray(
        cells_map_filepath=map_output_path,
        netCDF_filepath_list=tas_input_files,
        h5_variable_name='tas',
        start_date=climate_startime, end_date=climate_endtime,
        climate_csv_output_filepath=series_output_path)

    # --- PET
    print("-------- Processing evapotranspiration data.")
    # Make PET time series
    pet_input_filenames = find_CHESS_temperature_or_PET_files(pet_folder, start_year, end_year)
    pet_input_files = [os.path.join(pet_folder, file) for file in pet_input_filenames]
    series_output_path = climate_output_folder + catchment + '_PET.csv'
    # if not os.path.exists(series_output_path):
    build_climate_data_with_xarray(
        cells_map_filepath=map_output_path,
        netCDF_filepath_list=pet_input_files,
        h5_variable_name='pet',
        start_date=climate_startime, end_date=climate_endtime,
        climate_csv_output_filepath=series_output_path)


def process_catchment(
        catch, mask_path, simulation_startime, simulation_endtime, output_subfolder, static_inputs, resolution,
        produce_climate=True, prcp_data_folder=None, tas_data_folder=None, pet_data_folder=None  # ,q=None
):
    """
    Create all files needed to run shetran-prepare.
    produce_climate is true or false option. If False, climate files will not be created.
    """

    if not os.path.isdir(output_subfolder):
        os.mkdir(output_subfolder)

    try:
        # Read mask
        print(catch, ": reading mask...")
        mask, ncols, nrows, xll, yll, cellsize, _, headers, _ = read_ascii_raster(
            mask_path, data_type=int, return_metadata=True)

        # Create static maps and return vegetation_array (land cover) and soil arrays/info
        print(catch, ": creating static maps...")
        vegetation_array, soil_array, orig_soil_types, new_soil_types = create_static_maps(
            static_inputs, xll, yll, ncols, nrows, cellsize, output_subfolder, headers, catch, mask)

        # Get strings of vegetation and soil properties/details for library file
        # print(catch, ": creating vegetation (land use) and soil strings...")
        veg_string = get_veg_string(vegetation_array, static_inputs)
        soil_types_string, soil_cols_string = get_soil_strings(orig_soil_types, new_soil_types, static_inputs)

        # Create library file
        # print(catch, ": creating library file...")
        create_library_file(output_subfolder, catch, veg_string, soil_types_string, soil_cols_string,
                            simulation_startime, simulation_endtime, grid_resolution=resolution)

        # Create climate time series files (and cell ID map)
        if produce_climate:
            print(catch, ": Creating climate files...")
            # create_climate_files(simulation_startime, simulation_endtime, mask_path, catch, output_subfolder,
            # prcp_data_folder, tas_data_folder, pet_data_folder)  # ^^ This method does not work well for catchments
            # with resolutions != 1000m. run_build_climate_data_with_xarray is preferred.

            run_build_climate_data_with_xarray(
                mask_filepath=mask_path,
                climate_output_folder=output_subfolder,
                catchment=catch,
                climate_startime=simulation_startime,
                climate_endtime=simulation_endtime,
                prcp_folder=prcp_data_folder,
                tas_folder=tas_data_folder,
                pet_folder=pet_data_folder)

        # sys.exit()

    except Exception as E:
        print(E)
        pass


def process_mp(mp_catchments, mp_mask_folders, mp_output_folders, mp_simulation_startime,
               mp_simulation_endtime, mp_static_inputs, mp_resolution, mp_prcp_data_folder, mp_tas_data_folder,
               mp_pet_data_folder, mp_produce_climate=False, num_processes=10):
    manager = mp.Manager()
    # q = manager.Queue()
    pool = mp.Pool(num_processes)

    jobs = []
    for catch in np.arange(0, len(mp_catchments)):
        job = pool.apply_async(process_catchment,
                               (mp_catchments[catch], mp_mask_folders[catch], mp_simulation_startime,
                                mp_simulation_endtime, mp_output_folders[catch], mp_static_inputs, mp_resolution,
                                mp_produce_climate, mp_prcp_data_folder, mp_tas_data_folder, mp_pet_data_folder))

        jobs.append(job)

    for job in jobs:
        job.get()

    # q.put('kill')
    pool.close()
    pool.join()


def read_static_asc_csv(static_input_folder,
                        UDM_2017=False,
                        UDM_SSP2_2050=False, UDM_SSP2_2080=False,
                        UDM_SSP4_2050=False, UDM_SSP4_2080=False,
                        NFM_max=False, NFM_bal=False):
    """
    This functions will load in the raw data for the UK, i.e. asc and csv files, and convert these to the dictionary
    object used in the setups. There should be 7 files with the following names, all in the same folder (argument):
        - SHETRAN_UK_DEM.asc
        - SHETRAN_UK_minDEM.asc
        - SHETRAN_UK_lake_presence.asc
        - SHETRAN_UK_LandCover.asc
        - Vegetation_Details.csc
        - SHETRAN_UK_SoilGrid_APM.asc
        - SHETRAN_UK_SoilDetails.csc

        - UDM_GB_LandCover_2017.asc
        - UDM_GB_LandCover_2050.asc
        - UDM_GB_LandCover_2080.asc

        - NFMmax_GB_Woodland.asc
        - NFMmax_GB_Storage.asc
        - NFMbalanced_GB_Woodland.asc
        - NFMmax_GB_Storage.asc

    All .asc files should have the same extents and cell sizes.

    :param static_input_folder:
    :param UDM_2017: True or False depending on whether you want to use the default CEH 2007 or the UDM baseline map.
    :param UDM_2050: True or False depending on whether you want to use the default CEH 2007 or the UDM 2050 map.
    :param UDM_2080: True or False depending on whether you want to use the default CEH 2007 or the UDM 2080 map.
    :param NFM_bal:
    :param NFM_max:
    :return:
    """

    # Raise an error if there are multiple land covers selected:
    if (UDM_2017 + UDM_SSP2_2050 + UDM_SSP2_2080 + UDM_SSP4_2050 + UDM_SSP4_2080) > 1:
        raise ValueError("Multiple UDM land cover maps are 'True' in setup script; only a single map can be used.")

    # Raise an error if there are multiple NFM maps selected:
    if (NFM_max + NFM_bal) > 1:
        raise ValueError("Multiple NFM maps are 'True' in setup script; only a single map can be used.")

    # Load in the coordinate data (assumes all data has same coordinates:
    _, ncols, nrows, xll, yll, cellsize, _, _, _ = read_ascii_raster(static_input_folder + "SHETRAN_UK_DEM.asc",
                                                                  return_metadata=True)

    # Create eastings and northings. Note, the northings are reversed to match the maps
    eastings = np.arange(xll, ncols * cellsize + yll, cellsize)
    northings = np.arange(yll, nrows * cellsize + yll, cellsize)[::-1]
    eastings_array, northings_array = np.meshgrid(eastings, northings)

    # Set the desired land cover:
    if UDM_2017:
        LandCoverMap = "UDM_GB_LandCover_2017.asc"
    elif UDM_SSP2_2050:
        LandCoverMap = "UDM_GB_LandCover_SSP2_2050.asc"
    elif UDM_SSP2_2080:
        LandCoverMap = "UDM_GB_LandCover_SSP2_2080.asc"
    elif UDM_SSP4_2050:
        LandCoverMap = "UDM_GB_LandCover_SSP4_2050.asc"
    elif UDM_SSP4_2080:
        LandCoverMap = "UDM_GB_LandCover_SSP4_2080.asc"
    else:
        LandCoverMap = "SHETRAN_UK_LandCover.asc"

    # Create xarray database to load/store the static input data:
    ds = xr.Dataset({
        "surface_altitude": (["y", "x"],
                             np.loadtxt(static_input_folder + "SHETRAN_UK_DEM.asc", skiprows=6),
                             {"units": "m"}),
        "surface_altitude_min": (["y", "x", ],
                                 np.loadtxt(static_input_folder + "SHETRAN_UK_minDEM.asc", skiprows=6),
                                 {"units": "m"}),
        "lake_presence": (["y", "x"],
                          np.loadtxt(static_input_folder + "SHETRAN_UK_lake_presence.asc", skiprows=6)),
        "land_cover_lccs": (["y", "x"],
                            np.loadtxt(static_input_folder + LandCoverMap, skiprows=6),
                            {"land_cover_key": pd.read_csv(static_input_folder + "Vegetation_Details.csv")}),
        "soil_type_APM": (["y", "x"],
                          np.loadtxt(static_input_folder + "SHETRAN_UK_SoilGrid_APM.asc", skiprows=6),
                          {"soil_key": pd.read_csv(static_input_folder + "SHETRAN_UK_SoilDetails.csv")})
    },
        coords={"easting": (["y", "x"], eastings_array, {"projection": "BNG"}),
                "northing": (["y", "x"], northings_array, {"projection": "BNG"}),
                "x": (["x"], eastings, {"projection": "BNG"}),
                "y": (["y"], northings, {"projection": "BNG"})})

    # Load in the GB NFM Max map from Sayers and Partners:
    if NFM_max:
        ds["NFM_max_storage"] = (["y", "x"],
                                 np.loadtxt(static_input_folder + "NFMmax_GB_Storage.asc", skiprows=6))
        ds["NFM_max_woodland"] = (["y", "x"],
                                  np.loadtxt(static_input_folder + "NFMmax_GB_Woodland.asc", skiprows=6))
    if NFM_bal:
        ds["NFM_balanced_storage"] = (["y", "x"],
                                      np.loadtxt(static_input_folder + "NFMbalanced_GB_Storage.asc", skiprows=6))
        ds["NFM_balanced_woodland"] = (["y", "x"],
                                       np.loadtxt(static_input_folder + "NFMbalanced_GB_Woodland.asc", skiprows=6))

    return ds


def resolution_string(res):
    if res not in [1000, 500, 100]:
        print(f'Resolution is set to {str(res)}. This is incorrect and should instead be a numeric value of either '
              f'1000, 500, or 100 .')
    else:
        return f'{str(res)}/'

def make_cell_map(mask_filepath, output_filepath=None, write=True):
    """
    This will build the Cells.txt file from a mask.
    :param mask_filepath:
    :param output_filepath:
    :param write: True or False
    :return:
    """
    m, _, _, x, y, cs, _, _, d = read_ascii_raster(mask_filepath, data_type=int, return_metadata=True)
    m[m == 0] = np.arange(1, len(m[m == 0]) + 1)
    if write:
        write_ascii(m, output_filepath, x, y, cs, data_format='%1.0f')
    return m, d

def create_catchment_mask_from_shapefile(shapefile_path, output_ascii_path, resolution, fix_holes=True):
    """
    Converts a shapefile (proj:BNG) into a raster mask with specified resolution.
    Optionally fills any internal holes.
    - shapefile_path: Path to the input shapefile.
    - output_ascii_path: Path to save the output ASCII (.asc) raster file.
    - resolution: The resolution (cell size) of the raster in map units.
    - fix_holes: Boolean flag, if True, fills internal holes in the mask.
    """

    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Get the bounding box of the shapefile
    minx, miny, maxx, maxy = gdf.total_bounds

    # Round the bounding box to the nearest resolution
    minx = np.floor(minx / resolution) * resolution
    miny = np.floor(miny / resolution) * resolution
    maxx = np.ceil(maxx / resolution) * resolution
    maxy = np.ceil(maxy / resolution) * resolution

    # Compute raster dimensions
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    # Define transform
    transform = rasterio.transform.from_origin(minx, maxy, resolution, resolution)

    # Create an empty raster with no_data_value
    raster_data = np.full((height, width), -9999, dtype=np.float32)

    # Rasterize the shapefile (set inside-mask pixels to 0)
    shapes = [(geom, 0) for geom in gdf.geometry]
    rasterized = rasterize(shapes, out_shape=raster_data.shape, transform=transform, fill=-9999,
                           dtype=np.float32)

    # If fix_holes is True, fill any holes in the mask
    if fix_holes:
        binary_mask = (rasterized != -9999).astype(int)  # Convert to binary
        filled_mask = binary_fill_holes(binary_mask).astype(int)  # Fill holes
        rasterized[filled_mask == 1] = 0  # Apply filled mask to raster

    # Save to ASCII file
    with rasterio.open(output_ascii_path, "w", driver="AAIGrid", height=height, width=width, count=1,
                       dtype=np.float32, crs=gdf.crs, transform=transform, nodata=-9999) as dst:
        dst.write(rasterized, 1)

    print(f"Raster mask saved as: {output_ascii_path}")
    if fix_holes:
        print("Holes were detected and filled before saving.")


# --- Calculate Objective Functions for Flows -----------------
def shetran_obj_functions(regular_simulation_discharge_path: str, recorded_discharge_path: str,
                          start_date: str, period: list = None, recorded_date_discharge_columns: list = None,
                          return_flows=False, return_period=False):
    """
    Notes:
    - Assumes daily flow data, can be altered within function.
    - Assumes that recorded flows have dates and are regularly spaced, with no gaps.
    - NAs will be skipped from the analysis. NA count will be returned.

    TODO - consider whether you can add code that allows you to take other columns
            from the record so that they can be visualised at the end.

    regular_simulation_discharge_path:  Path to the txt file
    recorded_discharge_path:            Path to the csv file
    start_date:                         The start date of the simulated flows: "DD-MM-YYYY"
    period:                             The period to use (i.e. calibration/validation) as a list of dates:
                                        ["YYY-MM-DD", "YYY-MM-DD"].
                                        Leave blank if you want to use the whole thing.
                                        Leave as single item in list if you want to use until the end of the data.
    recorded_date_discharge_columns:    The columns (as a list) that contain the date and then flow data.
    RETURNS:                            The NSE value as an array.
    """

    # --- Read in the flows for Sim and Rec:
    if recorded_date_discharge_columns is None:
        recorded_date_discharge_columns = ["date", "discharge_vol"]

    flow_rec = pd.read_csv(recorded_discharge_path,
                           usecols=recorded_date_discharge_columns,
                           parse_dates=[recorded_date_discharge_columns[0]])

    # Set the columns to the following so that they are always correctly referenced:
    # (Do not use recorded_date_discharge_columns!)
    flow_rec.columns = ["date", "discharge_vol"]
    flow_rec = flow_rec.set_index('date')

    # Read in the simulated flows:
    flow_sim = pd.read_csv(regular_simulation_discharge_path)
    flow_sim.columns = ["flow"]

    # --- Give the simulation dates:
    flow_sim['date'] = pd.date_range(start=start_date, periods=len(flow_sim), freq='D')
    flow_sim = flow_sim.set_index('date').shift(-1)
    # ^^ The -1 removes the 1st flow, which is the flow before the simulation.

    # --- Resize them to match
    flows = flow_sim.merge(flow_rec, on="date")
    # ^^ Merge removes the dates that don't coincide. Beware missing record data!

    # Select the period for analysis (if given):
    if period is not None:
        if len(period) == 1:
            flows = flows[flows.index >= period[0]]
        if len(period) == 2:
            flows = flows[(flows.index >= period[0]) & (flows.index <= period[1])]

    # --- Do the comparison
    flow_NAs = np.isnan(flows["discharge_vol"])  # The NAs are actually automatically removed

    # Calculate the objective function:
    obj_funs = {"NSE": np.round(he.evaluator(he.nse, flows["flow"], flows["discharge_vol"]), 2),
                "KGE": np.round(he.evaluator(he.kge, flows["flow"], flows["discharge_vol"]), 2),
                "RMSE": np.round(he.evaluator(he.rmse, flows["flow"], flows["discharge_vol"]), 2),
                "PBias": np.round(he.evaluator(he.pbias, flows["flow"], flows["discharge_vol"]), 2)}

    # Print out the % of data that are NA:
    print(str(round(len(np.arange(len(flow_NAs))[flow_NAs]) / len(flows) * 100, 3)) + "% of comparison data are NA")

    if (period is not None) & (return_period):
        obj_funs["period"] = period

    if return_flows:
        obj_funs["flows"] = flows

    return obj_funs


# --- Sweep Files from Blades to Folder -----------------------
def folder_copy(source_folder, destination_folder, overwrite=False, outputs_only=False, complete_only=False):
    """
    I:/SHETRAN_GB_2021/scripts/Blade_Sweeper.py" will execute this function for the Blades and CONVEX.

    :param source_folder: E.g. "C:/BenSmith/Blade_SHETRANGB_OpenCLIM_UKCP18rcm_220708_APM/Temp_simulations/"
    :param destination_folder: E.g. "I:/SHETRAN_GB_2021/UKCP18rcm_220708_APM_GB/"
    :param overwrite: For if you want to overwrite the destination folder (False/True)
    :param outputs_only: For if you only want to copy "outputs_..." files (False/True)
    :param complete_only: For if you only want to copy completed files, based on PRI file (False/True)
    :return: A list of copied files
    """

    # Check whether the destination folder exists (make it if not):
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)

    # Get a list of the folders to copy:
    files_2_copy = os.listdir(source_folder)

    # Set conditions to skip incomplete simulations if desired:
    if complete_only:
        pri_file = [i for i in files_2_copy if i.endswith("pri.txt")]

        # If there isn't a PRI file, skip the copy:
        if len(pri_file) == 0:
            return source_folder + " was not copied as it is incomplete."

        # If there is, then check completeness:
        with open(source_folder + pri_file[0], 'r') as f:
            lines = f.read().split("\n")
            comp_line = lines[-24]
            if not comp_line.startswith("Normal completion of SHETRAN run:"):
                # If incomplete, skip the copy, else continue:
                return source_folder + " was not copied as it is incomplete."

    # If NOT overwriting files, remove duplicates from source list:
    if not overwrite:
        destination_files = os.listdir(destination_folder)
        files_2_copy = [i for i in files_2_copy if i not in destination_files]

    # If you only want to copy outputs, only include these in the copy list:
    if outputs_only:
        files_2_copy = [i for i in files_2_copy if "output" in i]

    # Copy each of the remaining files across:
    if len(files_2_copy) > 0:
        for file in files_2_copy:
            shutil.copy2(source_folder + file, destination_folder + file)
        return files_2_copy
    else:
        return "No files to copy..."


# --- Get Date Components from a data string ------------------
def get_date_components(date_string, fmt='%Y-%m-%d'):
    # "1980/01/01"
    date = datetime.datetime.strptime(date_string, fmt)
    return date.year, date.month, date.day

