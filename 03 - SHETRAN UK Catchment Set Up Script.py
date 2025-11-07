"""
-------------------------------------------------------------
SHETRAN GB Master Simulation Creator
-------------------------------------------------------------
Ben Smith, adapted from previous codes.
27/07/2022
-------------------------------------------------------------
This script is the standard script for generating SHETRAN files/models.
Changes to the setup should be generic and user friendly / comprehensible (unless that makes it super complex).
Be aware that this script often uses 'Try' statements, which means that sometimes errors are hidden.


--- USER INSTRUCTIONS:
The GitHub directory contains a .yml file that should enable users to set up the required environment.
First, download / sync the GitHub directory onto your machine. Then open an Anaconda 3 Prompt and run the
following code to setup the environment:
> conda env create -f SHETRAN_UK_env.yml
Or, to create it in a specific directory:
> conda env create -p path\to\desired\env\directory\env_SHETRAN_UK --file path\to\yml\file\SHETRAN_UK_env.yml
You can then activate this environment or configure your python idle to use it.
The YML file was built using:
> conda env export > path\to\yml\file\SHETRAN_UK_env.yml
The built environment can be updated with a newer yml file using:
> conda env update --file SHETRAN_UK_env.yml --prune

The user should only have to edit:
 --- Set File Paths
 --- Set Processing Methods
Make sure that you update the correct parts of the dictionaries.
'process_single_catchment["single"]' controls 'multiprocessing["process_multiple_catchments"]'

--- NOTES:
This code did not run well on the blades last time I tried it  (for the UDM setup) and created easting
and northing files. I wonder whether this is due to the wrong version of xarray, but that is just a guess.
This script uses...

This file will build catchments - these require a catchment mask and all input datasets to be present.
Datasets can be created by downloading the required data and formatting it using 'Create SHETRAN Raster Data.ipynb'.

Processing the climate data is very slow due to its size. Improvements to the code / method for doing this are welcome.
The code loads all climate data, cuts out the bit required for the catchment, and then drops the data. Very laborious.
An attempt was made to speed this up by loading in all climate data first and then processing multiple catchments
however this struggled with memory issues and was dropped.

--- PARAMETERS:
AE/PE at field capacity: for trees this is artificially high (1, but could be higher). This is because the evaporation
    at the top of the canopy is high compared to grass at surface level, which is what the measurement of PET will be.

--- MASKS:
Masks should be .asc  format, they can have .txt as a file name extension.
Masks MUST align with the nearest unit of resolution! Else your climate data will be blank. For example:
    - 1000m masks should have extents rounded to the nearest 1000m;
    - 500m masks should have extents ending in 000 or 500 etc.
Mask paths used in multiprocessing will be:
  mask_folder_prefix + sim_mask_path + sim_name + Mask.txt
  Leave sim_mask_path in the csv blank if there are no additional bits to add in here.
  mask_folder_prefix can be changed to "" and the prefixed folder specified in the CSV
   instead if more specific naming convention is needed.
Masks MUST NOT have isolated cells / diagonally connected cells. This will stop the
SHETRAN Prepare.exe script from setting up. Remove these manually (and check any
corresponding cell maps). Later version of SHETRAN Prepare may correct for this issue.

An efficient way of creating masks from a shapefile is to open them in QGIS, then use:
1. Raster > Rasterize (Vector to Raster).
    Set raster size units to geo-referenced units. width/height to desired resolution. CRS to BNG.
    Extents as per constraints above. Burn in value = 0 and No data value = -9999.
2. Raster > Convert Format. Set parameters as above and write as .asc file

--- Resolution
The models can currently be set up in four resolutions: 1000m, 500m, 200m and 100m, but this is easily adaptable.

This is controlled simply by the mask, which must be of the desired resolution and align with
the 1000m, 500m, 200m or 100m grid. The input data should also match that resolution, this is done
automatically by selecting the appropriate data folder.

If the mask resolution does not match the static data resolution you will get an ERROR similar to:
    >> boolean index did not match indexed array along dimension 0;
    >> dimension is 12 but corresponding boolean dimension is 6
This is probably because one of the two have incorrect file paths.


--- Climate Data:
The script has been set up to run using both HadUK and CHESS/GEAR meteorological datasets. Set the folder path
below - the script will work out the format of the data and process it accordingly.
On our servers these can be found here:
- 'I:/HADUK/HADUK_precip_daily/'
- 'I:/CEH-GEAR_rainfall_daily/'


--- Northern Ireland:
NI datasets are not fully included into the setup and so are processed differently. They
do not have gridded climate data, instead, this is available in processed form for a
selection of catchments (this was provided by Helen He at UEA). These only run until the
end of 2010. The Northern Ireland runs should run from 01/01/1980 - 01/01/2011. This was
incorrect in initial versions and the library files were corrected manually.


--- Troubleshooting
1. The following error may be due to reading in a mask file (or equivalent) that has decimal places in its header.
You can probably just delete the decimal places or improve the read ascii function.
    YourScriptName.py:50: DeprecationWarning: loadtxt(): Parsing an integer via a float is deprecated.  To avoid this warning, you can:
        * make sure the original data is stored as integers.
        * use the `converters=` keyword argument.  If you only use
          NumPy 1.23 or later, `converters=float` will normally work.
        * Use `np.loadtxt(...).astype(np.int64)` parsing the file as
          floating point and then convert it.  (On all NumPy versions.)
      (Deprecated NumPy 1.23)
      arr = np.loadtxt(file_path, dtype=data_type, skiprows=6)


---
TODO:
 - Update the climate input data to most current period.
 - Update the scripts so that they can take the UKCP18 data.
 - Update the scripts to include other functions (e.g. UKCP18).
 - Test the multiprocessing - this wasn't used in anger with the historical runs, so check against UKCP18.
    Check q works.
 - Add a mask checker to ensure that extents are rounded to the nearest resolution unit.
 - Consider that creating climate simulations with this will create incorrect model durations,
    as SHETRAN runs on calendar years, but climate years are only 360 days.
 - Load climate data in before running through catchments. As in 1a Quickload of the UKCP setup. One experiment with
    this managed to load all data and then cut it out, however the volume of data that was read in prior to the cutting
     was waaaaaay too large, causing python to crash.
 - Consider changing Vegetation_Details.csv to LandCover_details.csv
 - Update YML file on github.
 - Use cProfile to explore bottlenecks in the code.
 - Higher resolution grids may need a higher (or lower) Strickler coefficient as there is more topographic variation.
 - The Temp and PET data loads whole years, rather than just the months needed. Change this to speed it up.
 - Recent PET data (>=2016) has lat/long (or doesn't) and this differs from earlier data.
    Fix code so that it can load 01/12/2015 to 01/02/2016.

-------------------------------------------------------------
"""

# --- Load in Packages ----------------------------------------
import SHETRAN_GB_Master_Setup_Functions as SF
import pandas as pd
import numpy as np
import time
import os

# -------------------------------------------------------------
# --- USER INPUTS ---------------------------------------------
# -------------------------------------------------------------

# --- Set File Paths -------------------------------------------

# Climate input folders:
create_climate_data = True  # [True/False]
rainfall_input_folder = 'I:/HADUK/HADUK_precip_daily/'  # 'I:/CEH-GEAR/CEH-GEAR_rainfall_daily/'  #
temperature_input_folder = 'I:/CHESS/CHESS_temperature_daily/'
PET_input_folder = 'I:/CHESS/CHESS_PET_daily/'

# Set Model periods (model will include given days): 'yyyy-mm-dd'
start_time = '1980-01-01'
end_time = '2019-12-31'

# Model Resolution: This controls channel parameters in the Library file
resolution = 1000  # [Cell size in meters - options are: 1000, 500, 200, 100. Integer]

# Static Input Data Folder:
raw_input_folder = "I:/SHETRAN_GB_2021/02_Input_Data/00 - Raw ASCII inputs for SHETRAN UK/1000m_v2/"  # (Set the number to the resolution)

# --- Set Processing Methods -----------------------------------
# catchments = ['39065', '39101', '39037', '42026',  '25003', '23006', '24003', '23007']
catchment = '23007'
process_single_catchment = dict(
    single=True,
    simulation_name=f'{catchment}',
    mask_path=f"I:/SHETRAN_GB_2021/02_Input_Data/1kmBngMasks_Processed/{catchment}_Mask.txt",
    output_folder=f"S:/02 - Python Optimiser/02_Simulations/Chalk Catchments/{catchment}/")  # end with '/'

# Choose Single / Multiprocessing:
multiprocessing = dict(
    process_multiple_catchments=not process_single_catchment["single"],
    simulation_list_csv='I:/SHETRAN_GB_2021/01_Scripts/Other/OFFLINE Generic Catchment Setup Script/Simulation_Setup_List.csv',
    mask_folder_prefix='I:/SHETRAN_GB_2021/02_Input_Data/1kmBngMasks_Processed/',  # 1kmBngMasks_Processed/', # I:\SHETRAN_GB_2021\02_Input_Data\superseded\1kmBngMasks
    output_folder_prefix="I:/SHETRAN_GB_2021/04_Historical_Simulations/SHETRAN_UK_APM_Historical_HADUK/",  # 'I:/SHETRAN_GB_2021/02_Input_Data/NFM Catchment Maps/NFM_Maximum/',
    use_multiprocessing=True,  # May only work on Blades?
    n_processes=2,  # For use on the blades
    use_groups=True,  # [True, False][1]
    group="1")  # String. Not used when use_groups == False.


# # --- Set Non-Default Land Cover Types --- 1km Only ------------------------
# This is no longer in use, it was used for building the OpenCLIM simulations but has since been dropped and
# is not backwards compatible. It can probably be added back in if needed...
# # Urban development: if you wish to use land cover from the Newcastle University Urban Development Model (SELECT ONE).
# Use_UrbanDevelopmentModel_2017 = False  # True / False (Default)
# Use_UrbanDevelopmentModel_SSP2_2050 = False
# Use_UrbanDevelopmentModel_SSP2_2080 = False
# Use_UrbanDevelopmentModel_SSP4_2050 = False
# Use_UrbanDevelopmentModel_SSP4_2080 = False
#
# # Natural Flood Management: if you wish to use additional forest and storage from Sayers and Partners (SELECT ONE).
# Use_NFM_Max_Woodland_Storage_Addition = False  # True /False (Default)
# Use_NFM_Bal_Woodland_Storage_Addition = False


# -------------------------------------------------------------
# --- CALL FUNCTIONS FOR SETUP --------------------------------
# -------------------------------------------------------------

if __name__ == "__main__":

    if process_single_catchment["single"] == multiprocessing["process_multiple_catchments"]:
        print("WARNING - SINGLE AND MULTIPLE PROCESSES SELECTED. Check process_single_catchment & multiprocessing.")
        pass

    # --- Import the Static Dataset -------------------------------
    # static_data = SF.read_static_asc_csv(
    #     static_input_folder=raw_input_folder,
    #     UDM_2017=Use_UrbanDevelopmentModel_2017,
    #     UDM_SSP2_2050=Use_UrbanDevelopmentModel_SSP2_2050,
    #     UDM_SSP2_2080=Use_UrbanDevelopmentModel_SSP2_2080,
    #     UDM_SSP4_2050=Use_UrbanDevelopmentModel_SSP4_2050,
    #     UDM_SSP4_2080=Use_UrbanDevelopmentModel_SSP4_2080,
    #     NFM_max=Use_NFM_Max_Woodland_Storage_Addition,
    #     NFM_bal=Use_NFM_Bal_Woodland_Storage_Addition)

    static_data = SF.read_static_asc_csv(
        DEM_path=os.path.join(raw_input_folder, 'SHETRAN_UK_DEM.asc'),
        DEMminimum_path=os.path.join(raw_input_folder, 'SHETRAN_UK_minDEM.asc'),
        Lake_map_path=os.path.join(raw_input_folder, 'SHETRAN_UK_lake_presence.asc'),
        Land_cover_map_path=os.path.join(raw_input_folder, 'SHETRAN_UK_LandCover_CEH2007.asc'),
        Land_cover_table_path=os.path.join(raw_input_folder, 'SHETRAN_UK_LandCover_CEH2007.csv'),
        Subsurface_map_path=os.path.join(raw_input_folder, 'SHETRAN_UK_Subsurface_ESD_BGSsuper_BGShydrogeol.asc'),
        Subsurface_table_path=os.path.join(raw_input_folder, 'SHETRAN_UK_Subsurface_ESD_BGSsuper_BGShydrogeol.csv'),
        NFM_max=False, NFM_bal=False,
        NFM_storage_path=None, NFM_forest_path=None)

    # --- Import the Climate Datasets -----------------------------
    # if create_climate_data:
    #
    #     # Find Climate Files to load for Each Variable:
    #     start_year = int(start_time[0:4])
    #     end_year = int(end_time[0:4])
    #
    #     print("  Reading rainfall...")
    #     prcp_input_files = SF.find_rainfall_files(start_year, end_year)
    #     rainfall_dataset = SF.read_climate_data(root_folder=rainfall_input_folder, filenames=prcp_input_files)
    #
    #     print("  Reading temperature...")
    #     tas_input_files = SF.find_temperature_or_PET_files(temperature_input_folder, start_year, end_year)
    #     temperature_dataset = SF.read_climate_data(root_folder=temperature_input_folder, filenames=tas_input_files)
    #
    #     print("  Reading PET...")
    #     pet_input_files = SF.find_temperature_or_PET_files(PET_input_folder, start_year, end_year)
    #     pet_dataset = SF.read_climate_data(root_folder=PET_input_folder, filenames=pet_input_files)
    #
    # else:
    #     rainfall_dataset = temperature_dataset = pet_dataset = None

    # --- Call Functions to Process a Single Catchment ------------
    if process_single_catchment["single"]:
        print("Processing single catchment...", process_single_catchment["simulation_name"])
        SF.process_catchment(
            catch=process_single_catchment["simulation_name"],
            mask_path=process_single_catchment["mask_path"],
            simulation_startime=start_time, simulation_endtime=end_time, resolution=resolution,
            output_subfolder=process_single_catchment["output_folder"], static_inputs=static_data,
            produce_climate=create_climate_data, prcp_data_folder=rainfall_input_folder,
            tas_data_folder=temperature_input_folder, pet_data_folder=PET_input_folder)

    # --- Call Functions if Setting Up Multiple Catchments --------

    if multiprocessing["process_multiple_catchments"]:
        print("Processing multiple catchments...")

        # Read a list of simulation names to process:
        catchments_csv = pd.read_csv(multiprocessing["simulation_list_csv"], keep_default_na=False)

        # Check whether the simulation is in the group we're processing:
        if multiprocessing["use_groups"]:
            catchments_csv["Group"] = catchments_csv["Group"].astype('str')
            catchments_csv = catchments_csv[catchments_csv["Group"] == multiprocessing["group"]]

        # Get a list of the simulation/catchment names:
        simulation_names = list([str(c) for c in catchments_csv["Simulation_Name"]])

        # Create a list of file paths to the catchment masks:
        simulation_masks = [multiprocessing["mask_folder_prefix"] +
                            str(catchments_csv["Additional_Mask_Path"][x]) +
                            str(catchments_csv["Simulation_Name"][x]) + "_Mask.txt"
                            for x in catchments_csv["Simulation_Name"].index]

        # Create a list of output paths for the processed catchments:
        output_folders = [multiprocessing["output_folder_prefix"] +
                          str(catchments_csv["Additional_Output_Path"][x]) +
                          str(catchments_csv["Simulation_Name"][x]) + "/"
                          for x in catchments_csv.index]

        if multiprocessing["use_multiprocessing"]:
            print("Using multi-processing...")

            # Run the multiprocessing catchment setup:
            SF.process_mp(mp_catchments=simulation_names, 
                          mp_mask_folders=simulation_masks,
                          mp_output_folders=output_folders, 
                          mp_simulation_startime=start_time,
                          mp_simulation_endtime=end_time, 
                          mp_static_inputs=static_data,
                          mp_resolution=resolution,
                          mp_produce_climate=create_climate_data,
                          mp_prcp_data_folder=rainfall_input_folder,
                          mp_tas_data_folder=temperature_input_folder,
                          mp_pet_data_folder=PET_input_folder,
                          num_processes=multiprocessing["n_processes"])

        else:
            print("Using single processor...")
            for c in np.arange(0, len(simulation_names)):
                # Run the single processor catchment setup (for multiple catchments):
                SF.process_catchment(catch=simulation_names[c],
                                     mask_path=simulation_masks[c],
                                     simulation_startime=start_time,
                                     simulation_endtime=end_time,
                                     output_subfolder=output_folders[c],
                                     static_inputs=static_data,
                                     produce_climate=create_climate_data,
                                     prcp_data_folder=rainfall_input_folder,
                                     tas_data_folder=temperature_input_folder,
                                     pet_data_folder=PET_input_folder)
                time.sleep(1)
    print("Finished Processing Catchments")

