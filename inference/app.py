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

# --- STREAMLIT IMPORTS ---
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORT MODULES ---
import Cyclone_prediction
import heatmap
import grid       
import LAT_LON    
from models.hybrid_model import HybridCNNConvLSTM
from AI_agent_next_hr import HybridWeatherAgent 
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
@st.cache_data(ttl=3600)
def load_jaxa_report(csv_path):
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
    except Exception as e: st.sidebar.error(f"Error reading JAXA CSV: {e}")
    return jaxa_data

@st.cache_data(ttl=3600)
def load_imd_report(csv_path):
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
    except Exception as e: st.sidebar.error(f"Error reading IMD CSV: {e}")
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
    m = folium.Map(location=[11.12, 78.65], zoom_start=7, tiles='CartoDB dark_matter')
    
    rain_heat = []
    h, w = rain_grid.shape
    for y in range(h):
        for x in range(w):
            val = float(rain_grid[y, x])
            if val > 0.01: rain_heat.append([lats[y], lons[x], val])
    if rain_heat: HeatMap(rain_heat, radius=15, name="Rainfall").add_to(m)

    for dist, data in api_points.items():
         folium.CircleMarker([data["lat"], data["lon"]], radius=2, color="cyan", popup=dist).add_to(m)

    for dist, info in flood_risk:
        folium.Marker(
            [info['lat'], info['lon']], 
            popup=f"🌊 FLOOD RISK: {dist} ({info['rain']:.1f}mm)", 
            icon=folium.Icon(color="red", icon="warning", prefix='fa')
        ).add_to(m)

    for c in cyclone_tracks:
        folium.Marker([c["current_lat"], c["current_lon"]], popup=f"🌀 {c['name']} (Live)", icon=folium.Icon(color="purple", icon="cloud", prefix='fa')).add_to(m)
        folium.Marker([c["pred_lat"], c["pred_lon"]], popup=f"🔮 {c['name']} Forecast", icon=folium.Icon(color="orange", icon="arrow-right", prefix='fa')).add_to(m)
        folium.PolyLine([(c["current_lat"], c["current_lon"]), (c["pred_lat"], c["pred_lon"])], color="orange", dash_array="5,5").add_to(m)

    map_path = "tn_live_weather.html"
    m.save(map_path)
    return map_path

# ==========================================
# 4. MAIN PIPELINE 
# ==========================================
def run_pipeline():
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.info("Phase 1: Hybrid Model & Data Fusion...")
    base_data = grid.get_base_grid_from_dataset()
    if base_data is None: 
        st.error("Grid base data failed to load.")
        return None, None, None, None

    jaxa_data = load_jaxa_report(JAXA_CSV_PATH)
    imd_records = load_imd_report(IMD_CSV_PATH)
    api_points = LAT_LON.get_all_api_data()
    progress_bar.progress(20)

    h, w = base_data["rain"].shape
    fused_channels = np.zeros((6, h, w))
    for i, k in enumerate(['rain', 'temp', 'pressure', 'u', 'v', 'elevation']):
        fused_channels[i] = base_data.get(k, np.zeros((h,w)))

    model = HybridCNNConvLSTM(input_dim=6, hidden_dim=32, kernel_size=(3,3), num_layers=2)
    hybrid_forecast_grid = np.zeros((h, w))
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            model.eval()
            with torch.no_grad():
                inp = torch.FloatTensor(fused_channels).unsqueeze(0).unsqueeze(0)
                pred = model(inp, future_steps=1)
                hybrid_forecast_grid = np.maximum(pred[0, 0, 0, :, :].numpy(), 0.0)
        except Exception as e: 
            st.sidebar.warning(f"Hybrid Model Error: {e}")
    progress_bar.progress(40)

    status_text.info("Phase 2: AI Agent Execution & Live Data Fusion...")
    try: risk_agent = HybridWeatherAgent(OWM_KEY, JAXA_CSV_PATH)
    except Exception as e: risk_agent = None

    report_rows = []
    flood_risk_list = [] 
    sorted_districts = sorted(api_points.items())
    
    for district, data in sorted_districts:
        lat, lon = data['lat'], data['lon']
        norm_dist = normalize_name(district)
        
        val_jaxa = jaxa_data.get(norm_dist, 0.0)
        val_imd = get_imd_rain(district, imd_records)
        val_api = max(data['om'].get('om_rain', 0), data['owm'].get('owm_rain', 0), data['rv'].get('rv_rain', 0))
        current_rain = calculate_input_rain(val_jaxa, val_imd, val_api)
        
        y_idx = grid.find_nearest_index(base_data["lats"], lat)
        x_idx = grid.find_nearest_index(base_data["lons"], lon)
        hybrid_trend_val = float(hybrid_forecast_grid[y_idx, x_idx])
        
        if current_rain > fused_channels[0, y_idx, x_idx]:
             fused_channels[0, y_idx, x_idx] = current_rain
        
        ai_msg, ai_rain_val = "UNKNOWN", 0.0
        if risk_agent:
            risk_result = risk_agent.predict_risk(lat, lon, external_current_rain=current_rain, hybrid_model_rain=hybrid_trend_val)
            ai_msg = risk_result.get('status', 'UNKNOWN')
            ai_rain_val = clean_value(risk_result.get('rain_rate', '0'))
        
        status = get_category_label(current_rain)
        ai_status = get_category_label(ai_rain_val)
        risk_msg = ai_msg

        if ai_rain_val > 60:
            risk_msg = "🔴 FLOOD ALERT"
            flood_risk_list.append((district, {"lat": lat, "lon": lon, "rain": ai_rain_val}))

        report_rows.append({
            "District": district, 
            "Current (mm)": round(current_rain, 2), 
            "API Pred (mm)": round(val_api, 2), 
            "Hybrid Trend (mm)": round(hybrid_trend_val, 2),
            "Current Status": status, 
            "AI Future (mm)": round(ai_rain_val, 3), 
            "AI Status": ai_status, 
            "Risk Message": risk_msg
        })
    progress_bar.progress(60)

    status_text.info("Phase 3 & 4: Flood and Cyclone Detection...")
    try: Flood_detection.run_forecast()
    except Exception as e: pass

    try: cyclone_tracks = Cyclone_prediction.run_cyclone_system()
    except Exception as e: cyclone_tracks = []
    progress_bar.progress(80)

    status_text.info("Phase 5: Generating Maps...")
    map_file = generate_rainfall_map(api_points, flood_risk_list, cyclone_tracks, hybrid_forecast_grid, base_data["lats"], base_data["lons"])
    
    try: heatmap.run_temperature_map()
    except Exception as e: pass

    df_report = pd.DataFrame(report_rows)
    
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    return df_report, map_file, flood_risk_list, cyclone_tracks

# ==========================================
# 5. STREAMLIT UI LAYOUT (MULTI-PAGE)
# ==========================================
def main():
    st.set_page_config(page_title="TN Weather Hub", layout="wide", page_icon="🌤️")
    
    # --- Initialize Session State Variables ---
    if "pipeline_run" not in st.session_state:
        st.session_state.pipeline_run = False
        st.session_state.df = None
        st.session_state.map_path = None
        st.session_state.flood_risks = []
        st.session_state.cyclones = []

    # --- Sidebar Navigation ---
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    menu = ["🏠 Control Center", "🌧️ Rainfall Dashboard", "🌊 Flood Risks", "🌀 Cyclone Tracker", "🗺️ Heatmaps"]
    choice = st.sidebar.radio("Go to:", menu)
    st.sidebar.markdown("---")
    st.sidebar.caption("Integrated Weather Prediction System | Tamil Nadu")

    # ==========================================
    # PAGE 1: CONTROL CENTER
    # ==========================================
    if choice == "🏠 Control Center":
        st.title("Hybrid AI-Powered Forecasting and Detection of Extreme Weather Events in Tamil Nadu")
        st.markdown("Welcome to the TN Weather Prediction System. Run the pipeline below to fetch live data, run the hybrid models, and update all dashboard modules.")
        
        st.markdown("### ⚙️ System Execution")
        
        # --- Create 3 columns to center the button and reduce its width ---
        col1, col2, col3 = st.columns([1, 2, 1]) 
        
        with col2: # Place the button in the middle column
            run_pressed = st.button("🚀 Run Integrated Pipeline", type="primary", use_container_width=True)
            
        if run_pressed:
            with st.spinner("Fetching data and running models... This may take a moment."):
                df, map_path, floods, cyclones = run_pipeline()
                if df is not None:
                    # Save everything to session state so other tabs can access it
                    st.session_state.df = df
                    st.session_state.map_path = map_path
                    st.session_state.flood_risks = floods
                    st.session_state.cyclones = cyclones
                    st.session_state.pipeline_run = True
                    st.success("✅ System updated successfully! Use the sidebar to view detailed reports.")
        # Show High-level summary if data exists
        if st.session_state.pipeline_run:
            st.markdown("### 📊 Live System Summary")
            df = st.session_state.df
            col1, col2, col3, col4 = st.columns(4)
            
            heavy_rain = df[df['Current Status'].str.contains("Heavy|VIOLENT")].shape[0]
            
            col1.metric("Districts Monitored", df.shape[0])
            col2.metric("Heavy Rain Alerts", heavy_rain)
            col3.metric("Active Flood Risks", len(st.session_state.flood_risks), delta_color="inverse")
            col4.metric("Active Cyclones", len(st.session_state.cyclones))
            
            st.info("👈 Navigate using the sidebar menu to see detailed breakdowns.")

    # ==========================================
    # PAGE 2: RAINFALL DASHBOARD
    # ==========================================
    elif choice == "🌧️ Rainfall Dashboard":
        st.title("🌧️ District Rainfall Analysis")
        if not st.session_state.pipeline_run:
            st.warning("⚠️ No data loaded. Please run the pipeline in the **Control Center** first.")
        else:
            st.markdown("Compare current observations against API predictions and Hybrid AI Model trends.")
            st.dataframe(
                st.session_state.df.drop(columns=['Risk Message']), 
                use_container_width=True, 
                hide_index=True,
                height=600
            )

    # ==========================================
    # PAGE 3: FLOOD RISKS
    # ==========================================
    elif choice == "🌊 Flood Risks":
        st.title("🌊 Flood Detection Module")
        if not st.session_state.pipeline_run:
            st.warning("⚠️ No data loaded. Please run the pipeline in the **Control Center** first.")
        else:
            df = st.session_state.df
            flood_df = df[df['Risk Message'].str.contains("FLOOD", na=False)]
            
            if flood_df.empty:
                st.success("✅ No severe flood risks detected in any district currently.")
            else:
                st.error(f"🚨 ALERT: {len(flood_df)} District(s) are showing high flood risk based on AI forecasts.")
                # Show only relevant columns for flood risks
                st.dataframe(
                    flood_df[['District', 'Current (mm)', 'AI Future (mm)', 'Risk Message']], 
                    use_container_width=True, 
                    hide_index=True
                )

    # ==========================================
    # PAGE 4: CYCLONE TRACKER
    # ==========================================
    elif choice == "🌀 Cyclone Tracker":
        st.title("🌀 Cyclone Tracking Module")
        if not st.session_state.pipeline_run:
            st.warning("⚠️ No data loaded. Please run the pipeline in the **Control Center** first.")
        else:
            cyclones = st.session_state.cyclones
            if not cyclones:
                st.success("✅ No active cyclone systems currently detected.")
            else:
                st.warning(f"⚠️ {len(cyclones)} Active System(s) Detected.")
                for c in cyclones:
                    with st.expander(f"System: {c['name']}", expanded=True):
                        col1, col2 = st.columns(2)
                        col1.write(f"**Current Location:** {c['current_lat']} N, {c['current_lon']} E")
                        col2.write(f"**Predicted Path:** {c['pred_lat']} N, {c['pred_lon']} E")

    # ==========================================
    # PAGE 5: HEATMAPS
    # ==========================================
    elif choice == "🗺️ Heatmaps":
        st.title("🗺️ Interactive Weather Heatmap")
        if not st.session_state.pipeline_run:
            st.warning("⚠️ No data loaded. Please run the pipeline in the **Control Center** first.")
        else:
            st.markdown("Visual representation of rainfall density, flood risks, and cyclone paths.")
            map_path = st.session_state.map_path
            
            if map_path and os.path.exists(map_path):
                with open(map_path, 'r', encoding='utf-8') as f:
                    html_data = f.read()
                # Embed the map into Streamlit
                components.html(html_data, height=700, scrolling=False)
            else:
                st.error("Map file could not be generated or found.")

if __name__ == "__main__":
    main()
#python -m streamlit run c:/Users/keert/.vscode/TNWP/NewTNWP/inference/app.py