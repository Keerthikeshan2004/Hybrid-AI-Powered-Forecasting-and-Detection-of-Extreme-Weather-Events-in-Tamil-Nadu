import xarray as xr
import numpy as np
import os

# --- PATH LOGIC ---
# Get the absolute directory where grid.py lives (NewTNWP/inference)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to the data folder located one level above 'inference'
# This maps to: C:\Users\keert\.vscode\TNWP\NewTNWP\data\tn_master_training_data_fixed.nc
STATIC_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "tn_master_training_data_fixed.nc")

# --- CONFIG ---
HISTORY_HOURS = 12 

def get_base_grid_from_dataset():
    """
    Loads the historical/base data from the NetCDF Dataset.
    """
    print(f"📡 [grid.py] Loading Base Grid from: {STATIC_DATA_PATH}")
    
    try:
        # Check if file exists before opening to provide a clear error message
        if not os.path.exists(STATIC_DATA_PATH):
            print(f"❌ [grid.py] CRITICAL ERROR: File not found at {STATIC_DATA_PATH}")
            return None

        ds = xr.open_dataset(STATIC_DATA_PATH)
        
        # Standardize Coordinate Names
        if 'latitude' in ds:
            lats = ds.latitude.values
            lons = ds.longitude.values
        else:
            lats = ds.lat.values
            lons = ds.lon.values
            
        elevation = ds.elevation.values
        
        # Get the latest frame of data from the dataset as a baseline
        base_rain = ds.era5_rain.values[-1] if 'era5_rain' in ds else np.zeros_like(elevation)
        base_temp = ds.temperature.values[-1] if 'temperature' in ds else np.zeros_like(elevation)
        
        print(f"   -> Grid Shape: {base_rain.shape} (Lat x Lon)")

        return {
            "lats": lats,          
            "lons": lons,          
            "elevation": elevation,
            "rain": base_rain,     
            "temp": base_temp      
        }
        
    except Exception as e:
        print(f"❌ [grid.py] Error loading dataset: {e}")
        return None

def find_nearest_index(array, value):
    """ Helper to map a real Lat/Lon to the Dataset Grid Index """
    return (np.abs(array - value)).argmin()