import ftplib
import gzip
import shutil
import numpy as np
import os
import csv
from datetime import datetime

# --- CONFIGURATION ---
FTP_HOST = "hokusai.eorc.jaxa.jp"
FTP_USER = "rainmap"       
FTP_PASS = "Niskur+1404"   

# Tamil Nadu Bounding Box
TN_LAT_MIN, TN_LAT_MAX = 8.0, 14.0
TN_LON_MIN, TN_LON_MAX = 76.0, 81.0

# GSMaP Grid Constants
N_LAT = 1200        
N_LON = 3600        
LAT_TOP = 60.0      
LON_LEFT = 0.0      
RES = 0.1           

# District Coordinates
DISTRICTS = {
    "Ariyalur": (11.14, 79.07),
    "Chengalpattu": (12.69, 79.97),
    "Chennai": (13.08, 80.27),
    "Coimbatore": (11.01, 76.95),
    "Cuddalore": (11.74, 79.76),
    "Dharmapuri": (12.13, 78.01),
    "Dindigul": (10.36, 77.98),
    "Erode": (11.34, 77.71),
    "Kallakurichi": (11.73, 78.96),
    "Kancheepuram": (12.83, 79.70),
    "Kanyakumari": (8.08, 77.53),
    "Karaikal": (10.92, 79.83),
    "Karur": (10.96, 78.07),
    "Krishnagiri": (12.51, 78.21),
    "Madurai": (9.92, 78.11),
    "Mayiladuthurai": (11.10, 79.65),
    "Nagapattinam": (10.76, 79.84),
    "Namakkal": (11.22, 78.16),
    "Nilgiris": (11.49, 76.73),
    "Perambalur": (11.23, 78.88),
    "Puducherry": (11.94, 79.80),
    "Pudukkottai": (10.37, 78.82),
    "Ramanathapuram": (9.35, 78.83),
    "Ranipet": (12.94, 79.33),
    "Salem": (11.66, 78.14),
    "Sivaganga": (9.84, 78.48),
    "Tenkasi": (8.95, 77.31),
    "Thanjavur": (10.78, 79.13),
    "Theni": (10.01, 77.51),
    "Thoothukudi": (8.76, 78.13),
    "Tiruchirappalli": (10.79, 78.70),
    "Tirunelveli": (8.71, 77.75),
    "Tirupathur": (12.49, 78.55),
    "Tiruppur": (11.10, 77.34),
    "Tiruvallur": (13.14, 79.90),
    "Tiruvannamalai": (12.22, 79.07),
    "Tiruvarur": (10.77, 79.63),
    "Vellore": (12.91, 79.13),
    "Viluppuram": (11.93, 79.49),
    "Virudhunagar": (9.58, 77.96),
    "Sri Lanka": (7.0, 82.0) 
}

def get_status_emoji(mm):
    if mm is None: return "No Data ❓"
    if mm < 0.1: return "Clear ☀️"
    if mm < 2.5: return "Light 🌦️"
    if mm < 7.5: return "Moderate 🌧️"
    return "HEAVY ⛈️"

def get_realtime_data(save_dir="gsmap_data"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ftp = None
    try:
        print(f"Connecting to {FTP_HOST}...")
        ftp = ftplib.FTP(FTP_HOST)
        ftp.login(FTP_USER, FTP_PASS)
        
        target_dir = "/now/latest"
        print(f"Changing directory to: {target_dir}")
        ftp.cwd(target_dir)
        
        files = ftp.nlst()
        data_files = [f for f in files if f.endswith(".dat.gz")]
        data_files.sort()
        
        if not data_files:
            print(f"No .dat.gz files found in {target_dir}")
            return None

        latest_file = data_files[-1]
        local_path = os.path.join(save_dir, latest_file)
        
        print(f"Found {len(data_files)} files. Downloading latest: {latest_file}...")
        
        with open(local_path, "wb") as f:
            ftp.retrbinary(f"RETR {latest_file}", f.write)
            
        print("Download complete.")
        return local_path

    except Exception as e:
        print(f"\n❌ FTP Error: {e}")
        return None
    finally:
        if ftp:
            try:
                ftp.quit()
            except:
                pass

def process_gsmap_binary(file_path):
    if not file_path: return None
    
    print(f"Processing: {os.path.basename(file_path)}")
    dat_path = file_path.replace(".gz", "")
    try:
        with gzip.open(file_path, 'rb') as f_in:
            with open(dat_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    except Exception as e:
        print(f"Error extracting gzip: {e}")
        return None
    
    try:
        raw_data = np.fromfile(dat_path, dtype='<f4')
        expected_size = N_LAT * N_LON
        if raw_data.size != expected_size:
            print(f"❌ Data size mismatch! Expected {expected_size}, got {raw_data.size}")
            return None
            
        grid = raw_data.reshape((N_LAT, N_LON))
        
        lat_idx_start = int((LAT_TOP - TN_LAT_MAX) / RES)
        lat_idx_end   = int((LAT_TOP - TN_LAT_MIN) / RES)
        lon_idx_start = int((TN_LON_MIN - LON_LEFT) / RES)
        lon_idx_end   = int((TN_LON_MAX - LON_LEFT) / RES)
        
        tn_data = grid[lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end]
        tn_data = np.maximum(tn_data, 0.0)
        
        print(f"--- TN Data Statistics ---")
        print(f"Shape: {tn_data.shape}")
        print(f"Max Rainfall: {np.max(tn_data):.2f} mm/hr")
        
        return tn_data

    except Exception as e:
        print(f"Error processing binary: {e}")
        return None

def get_city_rainfall(data, city_name, target_lat, target_lon):
    """
    Returns (rainfall_value, status_message)
    """
    row_idx = int((TN_LAT_MAX - target_lat) / RES)
    col_idx = int((target_lon - TN_LON_MIN) / RES)
    
    rows, cols = data.shape
    if 0 <= row_idx < rows and 0 <= col_idx < cols:
        return data[row_idx, col_idx]
    else:
        return None

def save_districts_csv(data, filename="dataset/tn_districts_rainfall.csv"):
    print(f"\nGenerating District Report: {filename}...")
    
    # Using utf-8-sig to ensure emojis display correctly in Excel
    with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['District', 'Latitude', 'Longitude', 'Rainfall_mm_hr', 'Condition'])
        
        print(f"\n--- District Rainfall Report ---")
        print(f"{'District':<20} | {'Rain (mm/hr)':<15} | {'Condition'}")
        print("-" * 55)
        
        for city, (lat, lon) in DISTRICTS.items():
            val = get_city_rainfall(data, city, lat, lon)
            
            # Use your logic to get the status string
            status = get_status_emoji(val)
            
            rain_str = f"{val:.4f}" if val is not None else "N/A"
            writer.writerow([city, lat, lon, rain_str, status])
            
            print(f"{city:<20} | {rain_str:<15} | {status}")
            
    print(f"✅ Saved district report to {filename}")

if __name__ == "__main__":
    # 1. Download
    file_path = get_realtime_data()
    
    # 2. Process
    if file_path:
        tn_rain_matrix = process_gsmap_binary(file_path)
        
        if tn_rain_matrix is not None:
            # Save NPY
            np.save("dataset/tn_rain_latest.npy", tn_rain_matrix)
            
            # Save District Report with Emojis
            save_districts_csv(tn_rain_matrix)
            
            print("\n🎉 All operations complete.")
