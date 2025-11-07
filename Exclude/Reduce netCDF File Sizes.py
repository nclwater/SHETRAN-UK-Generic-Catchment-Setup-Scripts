import xarray as xr
import numpy as np

# Step 1: Load the large .nc file lazily
ds = xr.open_dataset(r"I:\CEH-GEAR downloads\1980.nc")  # Use chunking for large datasets

ds_cropped = ds.isel(time=slice(0, 50))

# Step 2: Access and round the rainfall_amount variable
ds_cropped["rainfall_amount"] = ds_cropped["rainfall_amount"].round(2)  # Round to 2 decimal places

# Step 3: Save the modified dataset to a new .nc file
ds_cropped.to_netcdf(r"I:\CEH-GEAR downloads\Rounded_1980.nc")
