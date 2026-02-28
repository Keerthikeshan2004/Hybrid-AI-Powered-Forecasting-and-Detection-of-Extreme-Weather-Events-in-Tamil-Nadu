import sys
import os

# Add the parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xarray as xr
import numpy as np

print("Loading dataset...")
ds = xr.open_dataset("data/tn_master_training_data.nc")
print("Original Dims:", ds.sizes)

# 1. Drop the conflicting 'time' dimension (the broken IMERG one with 19 steps)
# We also drop 'imerg_target' since it depends on this broken dimension
ds_clean = ds.drop_dims('time') 

# 2. Now safe to rename ERA5 'valid_time' -> 'time'
ds_clean = ds_clean.rename({'valid_time': 'time'})

# 3. Create a clean aligned target variable (filled with 0s for now)
# This ensures we have a target that matches the ERA5 timeline (1264 steps)
target = xr.DataArray(
    np.zeros((ds_clean.sizes['time'], ds_clean.sizes['latitude'], ds_clean.sizes['longitude'])),
    coords=[ds_clean.time, ds_clean.latitude, ds_clean.longitude],
    dims=['time', 'latitude', 'longitude'],
    name='imerg_target'
)

# 4. Merge back together
ds_final = xr.merge([ds_clean, target])

print("Fixed Dims:", ds_final.sizes)
# Should be: {'time': 1264, 'latitude': 25, 'longitude': 17}

ds_final.to_netcdf("data/tn_master_training_data_fixed.nc")
print("✅ Saved 'tn_master_training_data_fixed.nc'")
