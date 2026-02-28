import sys
import os

# Add the parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rasterio
import matplotlib.pyplot as plt
import numpy as np

# 1. Open the file
file_path = "n11_e076_1arc_v3.tif"  # Rename this to match your actual downloaded file

with rasterio.open(file_path) as src:
    
    # 2. Print Basic Metadata
    print(f"Driver: {src.driver}")
    print(f"Size: {src.width} x {src.height}")
    print(f"Coordinate Reference System (CRS): {src.crs}")
    print(f"Bounds: {src.bounds}")
    
    # 3. Read the elevation data (Band 1)
    elevation_data = src.read(1)
    
    # Mask out "nodata" values (usually -32768 for SRTM voids)
    elevation_data = np.where(elevation_data == src.nodata, np.nan, elevation_data)

    # 4. Plot the Terrain
    plt.figure(figsize=(10, 10))
    plt.imshow(elevation_data, cmap='terrain')
    plt.colorbar(label='Elevation (meters)')
    plt.title(f"SRTM Elevation: {file_path}")
    plt.show()

    # 5. Function to get elevation at a specific Lat/Lon
    def get_elevation(lat, lon):
        # Convert lat/lon to row/col
        row, col = src.index(lon, lat)
        
        # Check if point is within file bounds
        if 0 <= row < src.height and 0 <= col < src.width:
            val = elevation_data[row, col]
            return val
        else:
            return None

    # Example: Check elevation for Wellington, TN (approx 11.32°N, 76.79°E)
    lat_point = 11.32
    lon_point = 76.79
    elev = get_elevation(lat_point, lon_point)
    print(f"Elevation at {lat_point}, {lon_point}: {elev} meters")
