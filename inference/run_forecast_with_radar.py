import sys
import os
import torch
import numpy as np
import folium
from folium.plugins import HeatMap
import warnings
import csv
import re
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORT MODULES ---
import Cyclone_prediction
import heatmap
import grid       
import LAT_LON    
from models.hybrid_model import HybridCNNConvLSTM
from AI_agent_next_hr1 import HybridWeatherAgent

# [CHANGE 1] Import the Flood Detection Module
import Flood_detection

warnings.filterwarnings("ignore")

# --- CONFIG ---
MODEL_PATH = "dataset/tn_hybrid_model_2.pth"
JAXA_CSV_PATH = "dataset/tn_districts_rainfall.csv"
IMD_CSV_PATH = "dataset/tn_weather_fixed.csv"
OWM_KEY = "07e0902937281053b0c758bf275744f8"

# ==========================================
# 1. UTILITY FUNCTIONS
# ==========================================
def normalize_name(name):
    return name.lower().strip()

def clean_value(val):
    try:
        if not val or str(val).strip().upper() in ['--', 'NIL', 'NAN', 'NONE']: return 0.0
        clean = re.sub(r"[^\d\.]", "", str(val))
        return float(clean) if clean else 0.0
    except: return 0.0

def get_category_label(mm):
    if mm < 0.1: return "Clear ☀️"
    if mm < 2.5: return "Light 🌦️"
    if mm < 15.0: return "Medium 🌧️"
    if mm < 50.0: return "Heavy 🌧️"
    return "VIOLENT ⛈️"

# ==========================================
# 2. DATA LOADERS
# ==========================================
def load_jaxa_report(csv_path):
    print(f"\n📡 Loading JAXA Real-Time Data from {csv_path}...") 
    jaxa_data = {}
    if not os.path.exists(csv_path): return jaxa_data
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [x.strip() for x in reader.fieldnames]
            for row in reader:
                dist = row.get('District')
                rain_str = row.get('Rainfall_mm_hr', '0')
                if dist: jaxa_data[normalize_name(dist)] = clean_value(rain_str)
    except Exception as e: print(f"❌ Error reading JAXA CSV: {e}")
    return jaxa_data

def load_imd_report(csv_path):
    print(f"\n📄 Loading IMD Daily Report from {csv_path}...")
    imd_data = []
    if not os.path.exists(csv_path): return imd_data
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                imd_data.append({
                    "station": normalize_name(row.get('Station', '')),
                    "rain": clean_value(row.get('Rainfall', '0'))
                })
    except Exception as e: print(f"❌ Error reading IMD CSV: {e}")
    return imd_data

def get_imd_rain(district, imd_records):
    target = normalize_name(district)
    matches = [x['rain'] for x in imd_records if target in x['station']]
    return max(matches) if matches else 0.0

def calculate_input_rain(jaxa, imd, api_max):
    W_IMD, W_JAXA, W_API = 0.15, 0.5, 0.35
    if imd > 0.0:
        if jaxa > 0.0: return (imd * W_IMD + jaxa * W_JAXA + api_max * W_API) / (W_IMD + W_JAXA + W_API)
        else: return imd
    elif jaxa > 0.0:
        return (jaxa + api_max) / 2.0 * 0.8 if api_max > 0.0 else jaxa * 0.25 
    elif api_max > 0.0:
        return api_max * 0.15
    return 0.0

# ==========================================
# 3. MAPPING FUNCTION
# ==========================================
def generate_rainfall_map(api_points, flood_risk, cyclone_tracks, rain_grid, lats, lons):
    print("\n🗺 Generating Final Rainfall Map (tn_weather_heatmap.html)...")
    m = folium.Map(location=[11.12, 78.65], zoom_start=7, tiles='CartoDB dark_matter')
    
    # Rainfall Heat Layers
    rain_heat = []
    h, w = rain_grid.shape
    for y in range(h):
        for x in range(w):
            val = float(rain_grid[y, x])
            if val > 0.01: rain_heat.append([lats[y], lons[x], val])
    if rain_heat: HeatMap(rain_heat, radius=15, name="Rainfall").add_to(m)

    # District Markers
    for dist, data in api_points.items():
         folium.CircleMarker([data["lat"], data["lon"]], radius=2, color="cyan", popup=dist).add_to(m)

    # Flood Risks
    for dist, info in flood_risk:
        folium.Marker(
            [info['lat'], info['lon']], 
            popup=f"🌊 FLOOD RISK: {dist} ({info['rain']:.1f}mm)", 
            icon=folium.Icon(color="red", icon="warning", prefix='fa')
        ).add_to(m)

    # Cyclone Tracks
    for c in cyclone_tracks:
        folium.Marker([c["current_lat"], c["current_lon"]], popup=f"🌀 {c['name']} (Live)", icon=folium.Icon(color="purple", icon="cloud", prefix='fa')).add_to(m)
        folium.Marker([c["pred_lat"], c["pred_lon"]], popup=f"🔮 {c['name']} Forecast", icon=folium.Icon(color="orange", icon="arrow-right", prefix='fa')).add_to(m)
        folium.PolyLine([(c["current_lat"], c["current_lon"]), (c["pred_lat"], c["pred_lon"])], color="orange", dash_array="5,5").add_to(m)

    m.save("tn_weather_heatmap.html")
    print("✔ Saved: tn_weather_heatmap.html")

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def run_pipeline():
    print("="*51,"🚀 STARTING FINAL INTEGRATED SYSTEM","="*51)

    # --- PHASE 1: RAINFALL PREDICTION SYSTEM ---
    print("-"*51,"PHASE 1: RAINFALL PREDICTION SYSTEM","-"*51)
    
    base_data = grid.get_base_grid_from_dataset()
    if base_data is None: return
    jaxa_data = load_jaxa_report(JAXA_CSV_PATH)
    imd_records = load_imd_report(IMD_CSV_PATH)
    api_points = LAT_LON.get_all_api_data()

    print("🤖 Initializing AI Risk Agent...")
    try:
        risk_agent = HybridWeatherAgent(OWM_KEY, JAXA_CSV_PATH)
    except Exception as e:
        print(f"⚠️ Agent Init Failed: {e}")
        risk_agent = None

    print("\n⚡ Processing District Data & Fusing Sources...")
    h, w = base_data["rain"].shape
    fused_channels = np.zeros((6, h, w))
    # Fill base channels
    for i, k in enumerate(['rain', 'temp', 'pressure', 'u', 'v', 'elevation']):
        fused_channels[i] = base_data.get(k, np.zeros((h,w)))

    report_rows = []
    flood_risk_list = [] 
    sorted_districts = sorted(api_points.items())
    
    for district, data in sorted_districts:
        lat, lon = data['lat'], data['lon']
        norm_dist = normalize_name(district)
        
        # Fuse Data
        val_jaxa = jaxa_data.get(norm_dist, 0.0)
        val_imd = get_imd_rain(district, imd_records)
        val_api = max(data['om'].get('om_rain', 0), data['owm'].get('owm_rain', 0), data['rv'].get('rv_rain', 0))
        current_rain = calculate_input_rain(val_jaxa, val_imd, val_api)
        
        # Update Grid
        y_idx = grid.find_nearest_index(base_data["lats"], lat)
        x_idx = grid.find_nearest_index(base_data["lons"], lon)
        if current_rain > fused_channels[0, y_idx, x_idx]:
             fused_channels[0, y_idx, x_idx] = current_rain
        
        # AI Agent Prediction
        ai_msg, ai_rain_val = "UNKNOWN", 0.0
        if risk_agent:
            risk_result = risk_agent.predict_risk(lat, lon, external_current_rain=current_rain)
            ai_msg = risk_result.get('status', 'UNKNOWN')
            ai_rain_val = clean_value(risk_result.get('rain_rate', '0'))
        
        report_rows.append({
            "name": district, "current": current_rain, "pred_rain": val_api, 
            "ai_rain": ai_rain_val, "risk_msg": ai_msg, "coords": (y_idx, x_idx),
            "lat": lat, "lon": lon
        })

    # Hybrid Model Inference
    print(f"\n🧠 Running Hybrid Rainfall Model Inference...")
    model = HybridCNNConvLSTM(input_dim=6, hidden_dim=32, kernel_size=(3,3), num_layers=2)
    forecast_grid = fused_channels[0]
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            model.eval()
            with torch.no_grad():
                inp = torch.FloatTensor(fused_channels).unsqueeze(0).unsqueeze(0)
                pred = model(inp, future_steps=1)
                forecast_grid = np.maximum(pred[0, 0, 0, :, :].numpy(), 0.0)
        except Exception as e: print(f"⚠️ Rainfall Model Error: {e}")

    # Generate Main Report
    print("\n" + "=" * 140)
    print(f"{'DISTRICT':<18} | {'CURR(mm)':<8} | {'PRED(mm)':<8} | {'HYBRID(mm)':<10} | {'CURR_STAT':<15} | {'AI_FUT(mm)':<10} | {'AI_STAT':<15} | {'RISK MSG'}")
    print("=" * 140)

    for row in report_rows:
        y, x = row['coords']
        Hybrid_model_val = forecast_grid[y, x]
        if row['current'] == 0 and Hybrid_model_val < 0.1: Hybrid_model_val = 0.0

        status = get_category_label(row['current'])
        ai_status = get_category_label(row['ai_rain'])
        risk_msg = row['risk_msg']

        if Hybrid_model_val > 60:
            risk_msg = "🔴 FLOOD ALERT"
            flood_risk_list.append((row['name'], {"lat": row['lat'], "lon": row['lon'], "rain": Hybrid_model_val}))

        print(f"{row['name']:<18} | {row['current']:<3f} | {row['pred_rain']:<3f} | {Hybrid_model_val:<3f} | {status:<15} | {row['ai_rain']:<10.3f} | {ai_status:<15} | {risk_msg}")
    print("-" * 140)
    print("-" * 140)
    # --- PHASE 2: FLOOD DETECTION MODULE (Screenshots Output) ---
    # [CHANGE 2] Directly calling the Flood_detection module's logic
    # This ensures the output matches the screenshot exactly (Header + LSTM result)
    print("-"*53,"PHASE 2: FLOOD DETECTION MODULE","-"*54)
    try:
        Flood_detection.run_forecast()
    except Exception as e:
        print(f"⚠️ Flood Module Execution Failed: {e}")

    # --- PHASE 3: CYCLONE TRACKING MODULE ---
    print("-" * 140)
    print("-"*53,"PHASE 3: CYCLONE TRACKING MODULE","-"*53)
    cyclone_tracks = Cyclone_prediction.run_cyclone_system()

    # --- PHASE 4: GENERATE MAPS ---
    print("-" * 140)
    print("-"*58,"PHASE 4: GENERATE MAPS","-"*58)
    generate_rainfall_map(
        api_points, 
        flood_risk_list, 
        cyclone_tracks, 
        forecast_grid, 
        base_data["lats"], 
        base_data["lons"]
    )

    heatmap.run_temperature_map()
    print("\n✅ All Modules Executed Successfully.")

if __name__ == "__main__":
    run_pipeline()