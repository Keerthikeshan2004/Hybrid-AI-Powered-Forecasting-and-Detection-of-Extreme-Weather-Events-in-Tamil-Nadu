import sys
import os

# Add the parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import pandas as pd
import numpy as np
import time
import os

# 1. Define the Grid
# Tamil Nadu Bounds: West 76, South 8, East 81, North 14
# Resolution: 0.05 degrees (~5.5km)
lat_range = np.arange(8.0, 14.05, 0.05)
lon_range = np.arange(76.0, 81.05, 0.05)

coordinates = []
for lat in lat_range:
    for lon in lon_range:
        coordinates.append((lat, lon))

print(f"Total grid points to fetch: {len(coordinates)}")

# 2. Fetch Data using Raw JSON
chunk_size = 100 
elevation_data = []
url = "https://api.open-meteo.com/v1/elevation"

print("Starting download using JSON API...")

for i in range(0, len(coordinates), chunk_size):
    chunk = coordinates[i:i + chunk_size]
    
    # Prepare comma-separated strings for the API
    lats = [str(round(c[0], 4)) for c in chunk]
    lons = [str(round(c[1], 4)) for c in chunk]
    
    params = {
        "latitude": ",".join(lats),
        "longitude": ",".join(lons)
    }
    
    try:
        # Retry loop
        while True:
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                elevations = data.get("elevation", [])
                
                # Check if we got the right amount of data
                if len(elevations) != len(chunk):
                    print(f"Warning: Batch {i} size mismatch!")
                
                for j, elev in enumerate(elevations):
                    elevation_data.append({
                        "latitude": float(lats[j]),
                        "longitude": float(lons[j]),
                        "elevation": elev
                    })
                break # Success
                
            elif response.status_code == 429:
                print("Rate limit (429) hit. Waiting 60 seconds...")
                time.sleep(1)
                continue # Retry
                 
            else:
                print(f"Error {response.status_code}: {response.text}")
                break # Non-recoverable error
        
        print(f"Batch {i} to {i + len(chunk)} complete.")
        
        # Polite delay to prevent 429 errors
        time.sleep(1.0)
        
    except Exception as e:
        print(f"Critical error at batch {i}: {e}")
        break

# 3. Save Data
if elevation_data:
    df = pd.DataFrame(elevation_data)
    df.to_csv("tn_elevation_grid.csv", index=False)
    print(f"\nSuccess! Saved {len(df)} points to 'tn_elevation_grid.csv'")
    print(df.head())
else:
    print("No data collected.")
