import sys
import os

# Add the parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xarray as xr
import pandas as pd
import numpy as np
import os

# 1. Load ERA5 (Base Features)
print("Loading ERA5...")
ds_era5 = xr.open_dataset("era5/tn_era5_2015_2025_SOND.nc")
ds_era5 = ds_era5.rename({"tp": "era5_rain", "t2m": "temperature", "u10": "wind_u", "v10": "wind_v", "sp": "pressure"})

# 2. Load Elevation (Tiny & Fast)
print("Loading Elevation...")
ds_elev = xr.open_dataset("elevation/tn_elevation_10km_final.nc")
elev_aligned = ds_elev["elevation"].interp_like(ds_era5, method="nearest")

# 3. Load Huge IMERG CSV Efficiently
csv_file = "imerg/imerg_all_raw.csv" # Ensure this matches your filename
print(f"Loading Large IMERG CSV: {csv_file} (Chunked)...")

grid_data = []

# Read in chunks of 500k rows to save RAM
# 'parse_dates' usually works, but if it fails we do it manually inside the loop
chunk_iter = pd.read_csv(csv_file, chunksize=500000)

lat_bins = ds_era5.latitude.values
lon_bins = ds_era5.longitude.values

for i, chunk in enumerate(chunk_iter):
    if i % 5 == 0: print(f"Processing chunk {i+1}...")
    
    # --- FIX DATE PARSING HERE ---
    # We convert to datetime explicitly with errors='coerce' to handle format issues safely
    chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce')
    
    # Drop rows where date parsing failed (NaT)
    chunk = chunk.dropna(subset=['time'])

    # Filter to TN Box (Speed up)
    chunk = chunk[
        (chunk['lat'] >= 8) & (chunk['lat'] <= 14) & 
        (chunk['lon'] >= 76) & (chunk['lon'] <= 81)
    ]
    
    if chunk.empty: continue

    # Snap lat/lon to nearest ERA5 grid point
    # Find index of nearest grid point
    lat_idx = np.abs(lat_bins[:, None] - chunk['lat'].values).argmin(axis=0)
    lon_idx = np.abs(lon_bins[:, None] - chunk['lon'].values).argmin(axis=0)
    
    chunk['lat_grid'] = lat_bins[lat_idx]
    chunk['lon_grid'] = lon_bins[lon_idx]
    
    # Average rainfall for each (time, lat, lon) group in this chunk
    grouped = chunk.groupby(['time', 'lat_grid', 'lon_grid'])['imerg_rain'].mean().reset_index()
    grid_data.append(grouped)

# Combine all reduced chunks
print("Merging chunks...")
if not grid_data:
    raise ValueError("No valid data found in CSV within TN bounds (8-14N, 76-81E)!")

df_final = pd.concat(grid_data)
# Final Groupby to merge overlaps between chunks
df_final = df_final.groupby(['time', 'lat_grid', 'lon_grid'])['imerg_rain'].mean().reset_index()

# Convert to Xarray
print("Converting to NetCDF...")
ds_imerg_grid = df_final.set_index(['time', 'lat_grid', 'lon_grid']).to_xarray()
ds_imerg_grid = ds_imerg_grid.rename({'lat_grid': 'latitude', 'lon_grid': 'longitude', 'imerg_rain': 'imerg_target'})

# 4. Merge & Save
print("Final Merge...")
ds_master = xr.merge([ds_era5, elev_aligned.to_dataset(name="elevation"), ds_imerg_grid])
ds_master = ds_master.fillna(0) # Fill missing values with 0

ds_master.to_netcdf("data/tn_master_training_data.nc")
print("✅ Done! Saved 'tn_master_training_data.nc'")
print(ds_master)
