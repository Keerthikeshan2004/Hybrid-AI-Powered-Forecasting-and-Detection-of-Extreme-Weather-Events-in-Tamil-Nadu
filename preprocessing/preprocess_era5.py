import sys
import os

# Add the parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xarray as xr
import os

ERA5_DIR = "era5"
ACCUM_PATH = os.path.join(ERA5_DIR, "data_stream-oper_stepType-accum.nc")
INSTANT_PATH = os.path.join(ERA5_DIR, "data_stream-oper_stepType-instant.nc")
OUT_PATH = os.path.join(ERA5_DIR, "tn_era5_2015_2025_SOND.nc")

print("Opening accumulated (accum) file...")
ds_accum = xr.open_dataset(ACCUM_PATH)

print("Opening instantaneous (instant) file...")
ds_inst = xr.open_dataset(INSTANT_PATH)

print("ACCUM variables:", list(ds_accum.data_vars))
print("INSTANT variables:", list(ds_inst.data_vars))

accum_vars = ds_accum[["tp"]]
inst_vars = ds_inst[["t2m", "sp", "u10", "v10"]]

print("Merging datasets (override expver)...")
ds_merged = xr.merge(
    [accum_vars, inst_vars],
    compat="override"   # <── key line to fix expver conflict
)

# Optional: subset to Tamil Nadu
ds_merged = ds_merged.sel(latitude=slice(14, 8), longitude=slice(76, 81))

print(f"Saving merged dataset to {OUT_PATH} ...")
ds_merged.to_netcdf(OUT_PATH)

print("Done. Final dataset:")
print(ds_merged)
