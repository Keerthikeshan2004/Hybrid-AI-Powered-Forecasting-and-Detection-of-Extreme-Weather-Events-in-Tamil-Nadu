# prepare_training_data_v2.py

import sys
import os

# Add the parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling

LAT_MIN, LAT_MAX = 8.0, 14.0
LON_MIN, LON_MAX = 76.0, 81.0
RES = 0.1

lat_grid = np.arange(LAT_MIN, LAT_MAX, RES)
lon_grid = np.arange(LON_MIN, LON_MAX, RES)

target_grid = xr.DataArray(
    data=np.zeros((len(lat_grid), len(lon_grid))),
    coords={"y": lat_grid, "x": lon_grid},
    dims=("y", "x"),
    name="template",
).rio.write_crs("EPSG:4326")

srtm_path = "n11_e076_1arc_v3.tif"
srtm = rioxarray.open_rasterio(srtm_path)

print("Original SRTM:")
print(srtm)

# nodata handling
nodata_val = srtm.rio.nodata
if nodata_val is not None:
    valid_mask = srtm != nodata_val
    print("Valid pixel ratio:",
          float(valid_mask.sum() / valid_mask.size))
    srtm = srtm.where(valid_mask)

if srtm.rio.crs is None:
    srtm = srtm.rio.write_crs("EPSG:4326")

print("Reprojecting...")
srtm_resampled = srtm.rio.reproject_match(
    target_grid,
    resampling=Resampling.average,
)

# squeeze band, rename dims
srtm_resampled = srtm_resampled.squeeze("band", drop=True)
srtm_resampled = srtm_resampled.rename({"y": "latitude", "x": "longitude"})
srtm_resampled.name = "elevation"

out_path = "elevation/tn_elevation_10km_final.nc"
srtm_resampled.to_netcdf(out_path)
print(f"✅ Saved '{out_path}'")

print("Elevation stats over grid:")
print("min:", float(np.nanmin(srtm_resampled.values)))
print("max:", float(np.nanmax(srtm_resampled.values)))

# 1) Boolean mask of valid cells (not nodata)
valid_mask = srtm_resampled != -32767

# 2) Count valid cells
n_valid = int(valid_mask.sum().item())
print("Number of valid cells:", n_valid)

if n_valid > 0:
    # 3) Get indices of valid cells
    lat_idx, lon_idx = np.where(valid_mask.values)

    # Bounding box of valid area (using indices)
    lat_vals = srtm_resampled.latitude.values[lat_idx]
    lon_vals = srtm_resampled.longitude.values[lon_idx]

    print("Valid latitude range:", float(lat_vals.min()), "to", float(lat_vals.max()))
    print("Valid longitude range:", float(lon_vals.min()), "to", float(lon_vals.max()))

    # 4) Sample one valid cell in the middle of that cluster
    mid = len(lat_idx) // 2
    sample_lat = lat_vals[mid]
    sample_lon = lon_vals[mid]
    sample_val = srtm_resampled.sel(
        latitude=sample_lat,
        longitude=sample_lon,
        method="nearest"
    ).values
    print(f"Sample valid point ~({sample_lat:.3f}N, {sample_lon:.3f}E): {sample_val} m")
else:
    print("No valid cells found (all nodata).")
