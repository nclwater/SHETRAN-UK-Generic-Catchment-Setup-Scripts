# -------------------------------------------------------------
# SHETRAN Post Simulation Functions
# -------------------------------------------------------------
# Ben Smith
# 08/08/2022
# -------------------------------------------------------------
# This code is written to provide users with a range of
# functions that are generally useful for manipulating and
# analysing SHETRAN simulations. These include data extraction,
# analysis and visualisation.
#
# You can add these functions to other projects / scripts using the following code:
#   import sys
#   sys.path.append("I:/SHETRAN_GB_2021/01_Scripts/Other/OFFLINE Generic Catchment Setup Script/")
#   import SHETRAN_Post_Simulation_Functions as sf
# -------------------------------------------------------------


# --- Load in Packages ----------------------------------------
import os
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
import hydroeval as he  # https://pypi.org/project/hydroeval/ - open conda prompt: pip install hydroeval
import plotly.graph_objects as go


def SwapDateFormat(DateString, year_first=True, sep_character='/'):
    """
    Function to swap date format between "dd-mm-yyyy" and "yyyy-mm-dd" (or with '/' instead of '-')

    param DateString: Date with format "dd-mm-yyyy" or "yyyy-mm-dd"
    param year_first: Boolean - True if the year is the first part of the date, else False
    param sep_character: The character to use to separate the date parts in the output string

    Note that the input date can use either '-' or '/' as a separator, but the output will use sep_character.
    """

    # Normalize input separator
    normalized_date = DateString.replace('/', '-')

    # Try to infer format from content
    for fmt in ('%d-%m-%Y', '%Y-%m-%d'):
        try:
            date_obj = datetime.strptime(normalized_date, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError("Date format not recognized. Expected dd-mm-yyyy or yyyy-mm-dd with '-' or '/' separators.")

    # Output format
    if year_first:
        output_format = f'%Y{sep_character}%m{sep_character}%d'
    else:
        output_format = f'%d{sep_character}%m{sep_character}%Y'

    return date_obj.strftime(output_format)


def to_date(date):
    """
    Converts a date string into a datetime object.
    :param date: A string of a date - can me year first or last. Seperator can be '-' or '-'.
    :return: datetime object
    """
    formatted_date = SwapDateFormat(DateString=date, year_first=True, sep_character='/')
    date_date = datetime.strptime(formatted_date, "%Y/%m/%d")
    return date_date


# --- Calculate Objective Functions for Flows -----------------
def obj_functions(recorded_timeseries,
                  simulated_timeseries,
                  period: list = None,
                  return_flows=False):
    """
    Notes:
    - Assumes daily flow data, can be altered within function.
    - Assumes that recorded flows have dates and are regularly spaced, with no gaps.
    - NAs will be skipped from the analysis. NA count will be returned.
    :param recorded_timeseries: Pandas DataFrame of input timeseries with date index.
    :param simulated_timeseries: Pandas DataFrame of input timeseries with date index.
    :param period: The period to use (i.e. calibration/validation) as a list of dates:
                    ["YYY-MM-DD", "YYY-MM-DD"].
                    Leave blank if you want to use the whole thing.
                    Leave as single item in list if you want to use until the end of the data.
                    Dates will be formatted and so it should not matter whether the seperators are '-' or '/' or
                    whether year is first or second (month must be middle).
    :param return_flows: Set to True/False according to whether you want to return a pd DataFrame of the timeseries used in calculations
    :return: NSE and other objective function values as an array.
    """

    # --- Resize them to match
    merged_df = recorded_timeseries.merge(simulated_timeseries, left_index=True, right_index=True)
    # ^^ Merge removes the dates that don't coincide. Beware missing record data!

    # Select the period for analysis (if given):
    if period is not None:
        period = [to_date(p) for p in period]
        if len(period) == 1:
            merged_df = merged_df[merged_df.index >= period[0]]
        if len(period) == 2:
            merged_df = merged_df[(merged_df.index >= period[0]) & (merged_df.index <= period[1])]

    # Calculate the objective function:
    # rec, sim = merged_df.columns
    merged_df.columns = ['rec', 'sim']
    performance_df = {
        "NSE": np.round(he.evaluator(obj_fn=he.nse, simulations=merged_df['sim'], evaluation=merged_df['rec']), 2),
        "RMSE": np.round(he.evaluator(obj_fn=he.rmse, simulations=merged_df['sim'], evaluation=merged_df['rec']), 2),
        "KGE": np.round(he.evaluator(obj_fn=he.kge, simulations=merged_df['sim'], evaluation=merged_df['rec'])[0], 2),
        "KGE_r": np.round(he.evaluator(obj_fn=he.kge, simulations=merged_df['sim'], evaluation=merged_df['rec'])[1], 2),
        "KGE_a": np.round(he.evaluator(obj_fn=he.kge, simulations=merged_df['sim'], evaluation=merged_df['rec'])[2], 2),
        "KGE_B": np.round(he.evaluator(obj_fn=he.kge, simulations=merged_df['sim'], evaluation=merged_df['rec'])[3], 2),
        "PBias": np.round(he.evaluator(obj_fn=he.pbias, simulations=merged_df['sim'], evaluation=merged_df['rec']), 2),
        # 'Pearsonsr': pearsonr(merged_df['rec'], merged_df['sim'])  # PSC uses different function so columns may be ok reversed.
        # "merged_df": merged_df
    }

    if return_flows:
        performance_df["Flows"] = round(merged_df, 2)

    return performance_df


'''
def shetran_obj_functions(regular_simulation_discharge_path: str,
                          recorded_timeseries_path: str,
                          start_date: str,
                          period: list = None,
                          recorded_date_discharge_columns: list = None,
                          return_flows=False, return_period=False):
    """
    Notes:
    - Assumes daily flow data, can be altered within function.
    - Assumes that recorded flows have dates and are regularly spaced, with no gaps.
    - NAs will be skipped from the analysis. NA count will be returned.
    - A more updated version of this function has been copied above.

    regular_simulation_discharge_path:  Path to the txt file
    recorded_timeseries_path:            Path to the csv file
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

    flow_rec = pd.read_csv(recorded_timeseries_path,
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
'''


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


def get_lib_from_flow(discharge_filepath: str):
    """
    Function for extracting the model name from the discharge file path. Used in load_shetran_LibraryDate.
    :param discharge_filepath:  String of the file path to the SHETRAN output discharge_sim_regulartimestep.
    :return: The filepath to the model library file (assuming typical naming convention.
    """
    # Split the filepath into the directory and the file name:
    f_path = os.path.dirname(discharge_filepath)
    f_name = os.path.basename(discharge_filepath)

    # Remove the standard text from the string to get the simulation name:
    name = f_name.split('_discharge_sim_regulartimestep')[0].replace('output_', '')

    # If there is an additional part to the name then store it:
    name_sufix = f_name.split('_discharge_sim_regulartimestep')[1].replace('.txt', '')

    # Reconstruct the library file path:
    lib_path = os.path.join(f_path, f'{name}_LibraryFile{name_sufix}.xml')

    return lib_path


def get_library_date(filepath, start=True, sepchar='/'):
    """
    :param filepath: String of the file path to the SHETRAN simulation library file.
    :param start: True or False depending on whether you want to edit the start or the end date
    :return: A string of the date in format dd/mm/yyyy.
    """
    prefix = "Start" if start else "End"
    # Read file in read mode 'r'
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith(f"<{prefix}Day>"):
                sd = line.split('>')[1].split('<')[0]
            if line.startswith(f"<{prefix}Month>"):
                sm = line.split('>')[1].split('<')[0]
            if line.startswith(f"<{prefix}Year>"):
                sy = line.split('>')[1].split('<')[0]
        return f'{sd}{sepchar}{sm}{sepchar}{sy}'


# --- Load SHETRAN Regular Discharge File --------------------
def load_shetran_discharge(flow_path: str, simulation_start_date=None, discharge_column: int = 0):
    """
    This will take the simulation dates from the library file and the output timestep from the regular discharge file
    and load in the SHETRAN discharge at the outlet (or specified points).
    :param discharge_column: integer specifying which column to read from the discharge file. This is only relevant if
            multiple discharge points are being used (as specified using frame 47 inthe rundata file).
    :param flow_path: String of the file path to the SHETRAN output discharge_sim_regulartimestep.
    :param simulation_start_date: string 'dd/mm/yyyy'
    :return: Pandas dataframe of flows.
    """

    # Check whether the discharge file contains a single outlet discharge, or discharges from multiple points:
    checker = pd.read_csv(flow_path, skiprows=1, nrows=1, header=None, sep='"" | ', engine='python')
    if checker[0].values[0] == 'Outlet':
        row_skip = 2
    else:
        row_skip = 1

    # Load in the flow:
    flow = pd.read_csv(flow_path, skiprows=row_skip, header=None,
                       usecols=[discharge_column], sep='"" | ', engine='python')

    # If a simulation start date was not given, extract one from the library file or use a default.
    if simulation_start_date is None:
        try:
            # See whether there is a library file with a date in it and use that:
            lib_name = get_lib_from_flow(flow_path)
            # print(lib_name)
            simulation_start_date = get_library_date(lib_name)
        except:
            # If not Library file, use a default date. December 1980 for long UKCP18 scenarios, else Jan 1980.
            simulation_start_date = "12/1/1980" if len(flow) >= 30000 else "1/1/1980"
            print(f'Warning >load_shetran_LibraryDate< No date given or library file found.'
                  f'Using default simulation start date: {simulation_start_date}')

    # Get the timestep of the data from the discharge file:
    timestep = pd.read_csv(flow_path, nrows=1, header=None)
    timestep = timestep[0][0].replace(' ', '')
    timestep = timestep.split('timestep')[-1][:-5]

    # This could be less than daily, so convert it to minutes as freq must have
    # integers not floats (24.00h or 8.5h won't work but 24h or 1440min will):
    timestep_min = str(float(timestep) * 60)

    # Build the flow dataframe
    flow.index = pd.date_range(start=pd.to_datetime(simulation_start_date, dayfirst=True),
                               periods=len(flow), freq=f'{timestep_min}min')  # f'{timestep}h')
    flow.index.name = "Date"
    flow.columns = ["Flow"]
    return flow


# --- Load and prepare non-SHETRAN flow data ------------------
def load_discharge(flow_path: str):
    """
    Load in non-SHETRAN flow data. Assumes two columns: Date and Flow. Column names not important. Date format is.
    :param flow_path: String of path to the csv file.
    :return:
    """
    # Check how many columns are in the file: (the formatted files that we use have two columns but the CAMELS dataset have 11, of which Dicharge_Vol is the 6th column)
    column_check = pd.read_csv(flow_path, sep='\t|,', parse_dates=[0], dayfirst=True, engine='python', skiprows=1,
                               nrows=1)

    # If there are 11 columns then assume it is CAMELS:
    if column_check.shape[1] == 11:
        flow = pd.read_csv(flow_path, sep='\t|,', parse_dates=[0], dayfirst=True, engine='python', usecols=[0, 5])
        if 'discharge_vol' not in flow.columns:
            raise ValueError(
                "The dataset was expected to be a CAMELS dataset with 11 columns. " \
                "The CAMELS column 'discharge_vol' was not found and so the datasource is uncertain. " \
                "Check the data source matches what you expect (ideally 2 columns: Date and Flow).")

    # If not then assume it is a standard two-column file:
    else:
        flow = pd.read_csv(flow_path, sep='\t|,', parse_dates=[0], dayfirst=True, engine='python', skiprows=1)

    # Rename the columns if needed (assuming the first column contains dates and the second column contains values)
    flow.columns = ['Date', 'Flow']
    # Set the 'Date' column as the index
    flow.set_index('Date', inplace=True)
    # flow['Flow'][flow['Flow'] < 0] = np.nan  # Old, use below instead.
    flow.loc[flow['Flow'] < 0, 'Flow'] = np.nan
    return flow


# --- Load and prepare SHETRAN and recorded flow data ---------
def load_discharge_files(discharge_path, st=None):
    """
    Load in discharge data from SHETRAN or other sources
    :param discharge_path:
    :param st: start date (dd/mm/yyyy) used for loading the SHETRAN regular discharge file, which does not have date.
    :return:
    """
    if '_discharge_sim_regulartimestep' in discharge_path:
        print('Loading SHETRAN regular timestep flow output')
        flow = load_shetran_discharge(discharge_path, simulation_start_date=st)
    else:
        print('Loading non-SHETRAN data')
        flow = load_discharge(discharge_path)
    # else:
    #     print('Unsure which format to use...')

    flow = round(flow, 2)
    return flow


# --- Plot Flow Data Interactively -----------------------------
def plot_flow_datasets(flow_path_list: dict, sim_start_date=None, Figure_input=None):
    """
    This will plot an interactive line plot of flow data from different (correctly formatted) sources.
    Example:
        paths = {
        'NRFA 69035': 'myfolder/NRFA Gauged Daily Flow 69035.csv',
        'SHETRAN 69035': 'myfolder/output_69035_discharge_sim_regulartimestep.txt
        }
        plot_flow_datasets(flow_path_list=paths)
    :param flow_path_list: Dictionary of labels and paths to SHETRAN _discharge_sim_regulartimestep files
            or other flow datasets.
    :param Figure_input: a plotly Figure to be updated. If starting a new figure, leave as None / blank.
    :return:

    TODO: Make an update to this where you can add new traces (so if you wanted to add data with different start dates.)
    """
    # Set up figure either as fresh, or from the inputted figure:
    fig = go.Figure() if Figure_input is None else Figure_input

    # Run through each dictionary item.
    for sim_name, sim_data in flow_path_list.items():
        # Load the flow.
        flow = load_discharge_files(sim_data, sim_start_date)
        # Build the plot.
        fig.add_trace(go.Scatter(x=flow.index, y=flow.Flow, name=sim_name, opacity=0.8))
    # Update layout properties:
    fig.layout['xaxis'] = dict(title=dict(text='Date'))
    fig.layout['yaxis'] = dict(title=dict(text='Flow (cumecs)'))
    fig.update_layout(title_text="Catchment Discharge")
    # Show the plot:

    # fig.show()  # This has been changed as this does not work in Dash Apps, which need the plotly object, not just the shown figure. You will need to add '.show()' to code that is using the shown output not the updated plotly object. 
    return fig


# --- Load SHETRAN Regular Groundwater Level Output ----------
def load_shetran_regular_groundater_level(level_path, st=None):
    """
    Function to read in daily groundwater outputs from SHETRAN's automated Groundwater
        level output (water table elements). This is only available in later (March 2024)
        versions.
    st: start date (dd/mm/yyyy) used for loading the SHETRAN regular level file, which does not have date.
    """
    level = pd.read_csv(level_path, skiprows=1, header=None, sep='\t|,', engine='python')
    level.columns = ["Date", "Level"]
    if st is None:
        # TODO - check why you have made this strange 12/1 date (UKCP18?).
        st = "12/1/1980" if len(level) >= 30000 else "1/1/1980"
    level.Date = pd.date_range(start=st, periods=len(level), freq='D')
    level.set_index('Date', inplace=True)
    return level


# --- Load recorded Groundwater Level ------------------------
def load_recorded_groundwater_level(level_path):
    """
    This will read the first two columns of a csv containing groundwater levels.
    If the timeseries data is in the 3rd column then this will not be returned.
    The default should be assumed to be a depth from the surface, as this is what SHETRAN returns, but this
    will load either depth or level without consideration of type.
    """
    levels = pd.read_csv(level_path, parse_dates=[0], dayfirst=True, skiprows=1, header=None, usecols=[0, 1])
    levels.columns = ['Date', 'Level']
    levels.set_index('Date', inplace=True)
    return levels


# --- Load and prepare SHETRAN and recorded flow data ---------
def load_groundwater_level_files(level_path, st=None, datum_adjustment=0):
    """
    Load in level data from SHETRAN or other sources
    :param level_path:
    :param st: start date (dd/mm/yyyy) used for loading the SHETRAN regular level file, which does not have date.
    :param datum_adjustment: a float or integer representing the groundwater of a borehole. Give this if the GW data is
            given as a level above a datum instead of a depth. The GW data will be deducted from this value to convert
            from level to depth. If data is already a depth below ground surface then leave or specify as 0. Units
            should match the data source (meter, feet etc.)
    :return :
    """
    if ('_regulartimestep' in level_path) or ('output_WaterTable_Element' in level_path):
        print('Loading SHETRAN groundwater output timestep flow output...')
        level = load_shetran_regular_groundater_level(level_path, st=st)
    else:
        print('Loading non-SHETRAN groundwater level data...')
        level = load_recorded_groundwater_level(level_path)
        if float(datum_adjustment) != 0:
            level.Level = float(datum_adjustment) - level.Level
    # else:
    #     print('Unsure which format to use...')

    level = round(level, 2)
    return level


# --- Plot Groundwater Level Data Interactively ----------------
def plot_groundwater_level_datasets(level_path_list: dict, sim_start_date=None, datum_adjustment=0):
    """
    :param level_path_list: Dictionary of labels and paths to SHETRAN output_watertable_element files
            and recorded level (depth) datasets.
    :param st: start date (dd/mm/yyyy) used for loading the SHETRAN regular level file, which does not have date.
    :param datum_adjustment: a float or integer representing the groundwater of a borehole. Give this if the GW data is
        given as a level above a datum instead of a depth. The GW data will be deducted from this value to convert
        from level to depth. If data is already a depth below ground surface then leave or specify as 0. Units
        should match the data source (meter, feet etc.)
    :return:

    This will plot an interactive line plot of flow data from different (correctly formatted) sources.
    Example:
        paths = {
        'NRFA 69035': 'myfolder/Dean Farm 90191310 - GW Depth.csv',
        'SHETRAN 69035': 'myfolder/Optimisation_Outputs/output_43018_GWlevel_sim_regulartimestep_No10.txt
        }
        plot_groundwater_level_datasets(level_path_list=paths)
    """

    # Set up figure:
    fig = go.Figure()

    # Run through each dictionary item.
    for sim_name, sim_data in level_path_list.items():
        # Load the flow.
        level = load_groundwater_level_files(sim_data, st=sim_start_date, datum_adjustment=datum_adjustment)

        # Build the plot.
        fig.add_trace(go.Scatter(x=level.index, y=level.Level, name=sim_name, opacity=0.8))

    # Update layout properties:
    fig.layout['xaxis'] = dict(title=dict(text='Date'))
    fig.layout['yaxis'] = dict(title=dict(text='Groundwater Level (mbgl)'))
    fig.update_layout(title_text="Depth to Groundwater")

    # Flip the y-axis so that it looks like a depth and add a line at ground level:
    fig.update_yaxes(autorange="reversed")
    fig.add_hline(y=0)

    return fig
