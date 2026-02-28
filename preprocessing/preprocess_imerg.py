import sys
import os

# Add the parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime

IMERG_DIR = "imerg"
files = sorted(glob.glob(os.path.join(IMERG_DIR, "*.HDF5")))

all_frames = []

for fpath in files:
    fname = os.path.basename(fpath)
    print("Processing:", fname)

    # 1) Parse timestamp from filename
    parts = fname.split(".")
    date_block = parts[4]                  # e.g. '20251222-S000000-E002959'
    date_part = date_block.split("-")[0]   # '20251222'
    time_part = date_block.split("-")[1][1:]  # '000000'
    ts = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")

    # 2) Read variable and coordinates
    with h5py.File(fpath, "r") as h5:
        grid = h5["Grid"]

        # precipitation variable (V06 or V07)
        if "precipitationCal" in grid:
            rain = grid["precipitationCal"][()]   # 2D: (lat, lon)
        elif "precipitation" in grid:
            rain = grid["precipitation"][()]
        else:
            raise KeyError("No precipitation variable found in " + fname)

        lat = grid["lat"][()]   # 1D lat array
        lon = grid["lon"][()]   # 1D lon array

    # ensure arrays are numpy
    rain = np.array(rain)
    lat = np.array(lat)
    lon = np.array(lon)

    # 3) Build full lat/lon grid and flatten
    LON, LAT = np.meshgrid(lon, lat)  # shapes match rain

    df = pd.DataFrame({
        "time": ts,
        "lat": LAT.ravel(),
        "lon": LON.ravel(),
        "imerg_rain": rain.ravel()
    })
    all_frames.append(df)

# 4) Concatenate and save
if all_frames:
    full_df = pd.concat(all_frames, ignore_index=True)
    full_df.to_csv("imerg/imerg_all_raw.csv", index=False)
    print("Saved imerg_all_raw.csv with", len(full_df), "rows")
    print(full_df.head())
else:
    print("No IMERG files processed.")
