# SHETRAN-UK-Generic-Catchment-Setup-Scripts

This project contains the code required for setting up SHETRAN models for UK catchments. These can be produced using ascii masks of a range of resolutions.
The project also includes functions that can be used to analyse the simulations. 
Simulations can be set up from different gridded historical climate data (HadUK, CEH-GEAR, and CHESS). 
Timeseries datasets are not included in this repository. Code is included on how to build the static datasets (rasters and lookup tables) as well as some suggested parameters, although these will most likely need calibrating for different catchments.

## Scripts
- **01 - Build SHETRAN Input Datasets.ipynb:** This notebook can be used to generate the raster and lookup table datasets that are needed to setup the catchments. These can be made at a variety of resolutions. All input datasets are available online.
- **02 - Functions - Model Set Up.py:** Functions for setting up SHETRAN catchment models. These sample input datasets, build library files, and lots of useful functions for manipulating SHETRAN input files.
- **03 - SHETRAN UK Catchment Set Up Script.py:** This is the main script used for implementing the functions in the scripts above. This is setup for use on the Newcastle computing system, but should be transferable to other machines.
- **04 - Functions - Model Visualisation.py:** Functions that focus on post-processing of SHETRAN results - typically plotting calculating performance.
- **05 - Notebook - General Python for SHETRAN.ipynb:** An example notebook for dealing with SHETRAN files with examples from other packages. 
- **06 - Notebook - Visualisation.ipynb:** Some example visualisations of SHETRAN outputs. 
- **07 - Notebook - Merge Nested SHETRAN Models.ipynb:** A notebook showing how to take nested SHETRAN catchments and assimilate them into a larger model.

## Data
For information on the datasets used, see the Create SHETRAN Raster Data notebook and the resources below.

## Resources
- Physically-based modelling of UK river flows under climate change. Smith et al., 2024. https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2024.1468855/full
- Development of a system for automated setup of a physically-based, spatially-distributed hydrological model for catchments in Great Britain. Lewis et al., 2018. https://www.sciencedirect.com/science/article/pii/S1364815216311331
- A robust multi-purpose hydrological model for Great Britain. Lewis, 2016. https://theses.ncl.ac.uk/jspui/handle/10443/3290

## Tools
- SHETRAN model code: https://github.com/nclwater/Shetran-public
- SHETRAN Results Viewer: https://github.com/nclwater/shetran-results-viewer