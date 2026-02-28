import sys
import os
import torch
import numpy as np
import warnings
import csv
import re

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import Cyclone_prediction
import heatmap
import grid
import LAT_LON
from models.hybrid_model import HybridCNNConvLSTM
from AI_agent_next_hr import HybridWeatherAgent
import Flood_detection

warnings.filterwarnings("ignore")

MODEL_PATH = "dataset/tn_hybrid_model_2.pth"
JAXA_CSV_PATH = "dataset/tn_districts_rainfall.csv"
IMD_CSV_PATH = "dataset/tn_weather_fixed.csv"
OWM_KEY = "YOUR_API_KEY"


# ==========================
# UTILITIES
# ==========================
def normalize_name(name):
    return name.lower().strip()


def clean_value(val):
    try:
        if not val:
            return 0.0
        return float(re.sub(r"[^\d\.]", "", str(val)))
    except:
        return 0.0


def get_category_label(mm):
    if mm < 0.1: return "Clear ☀️"
    if mm < 2.5: return "Light 🌦️"
    if mm < 15.0: return "Medium 🌧️"
    if mm < 50.0: return "Heavy 🌧️"
    return "VIOLENT ⛈️"


def calculate_input_rain(jaxa, imd, api_max):
    W_IMD, W_JAXA, W_API = 0.15, 0.5, 0.35
    if imd > 0:
        if jaxa > 0:
            return (imd*W_IMD + jaxa*W_JAXA + api_max*W_API)/(W_IMD+W_JAXA+W_API)
        return imd
    elif jaxa > 0:
        return (jaxa + api_max)/2 * 0.8 if api_max > 0 else jaxa*0.25
    elif api_max > 0:
        return api_max * 0.15
    return 0.0


# ==========================
# MAIN PIPELINE
# ==========================
def run_pipeline():

    base_data = grid.get_base_grid_from_dataset()
    if base_data is None:
        return None

    jaxa_data = {}
    if os.path.exists(JAXA_CSV_PATH):
        with open(JAXA_CSV_PATH, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                jaxa_data[normalize_name(row['District'])] = clean_value(row['Rainfall_mm_hr'])

    imd_records = []
    if os.path.exists(IMD_CSV_PATH):
        with open(IMD_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                imd_records.append({
                    "station": normalize_name(row['Station']),
                    "rain": clean_value(row['Rainfall'])
                })

    api_points = LAT_LON.get_all_api_data()

    h, w = base_data["rain"].shape
    fused_channels = np.zeros((6, h, w))

    for i, k in enumerate(['rain','temp','pressure','u','v','elevation']):
        fused_channels[i] = base_data.get(k, np.zeros((h,w)))

    hybrid_forecast_grid = np.zeros((h,w))

    if os.path.exists(MODEL_PATH):
        model = HybridCNNConvLSTM(6,32,(3,3),2)
        model.load_state_dict(torch.load(MODEL_PATH,map_location='cpu'))
        model.eval()
        with torch.no_grad():
            inp = torch.FloatTensor(fused_channels).unsqueeze(0).unsqueeze(0)
            pred = model(inp,future_steps=1)
            hybrid_forecast_grid = np.maximum(pred[0,0,0,:,:].numpy(),0.0)

    try:
        risk_agent = HybridWeatherAgent(OWM_KEY, JAXA_CSV_PATH)
    except:
        risk_agent = None

    report_rows = []
    flood_risk_list = []

    for district,data in sorted(api_points.items()):

        lat,lon = data['lat'],data['lon']

        val_jaxa = jaxa_data.get(normalize_name(district),0.0)
        val_imd = 0.0
        val_api = max(data['om'].get('om_rain',0),
                      data['owm'].get('owm_rain',0),
                      data['rv'].get('rv_rain',0))

        current_rain = calculate_input_rain(val_jaxa,val_imd,val_api)

        y_idx = grid.find_nearest_index(base_data["lats"],lat)
        x_idx = grid.find_nearest_index(base_data["lons"],lon)
        hybrid_trend = float(hybrid_forecast_grid[y_idx,x_idx])

        if risk_agent:
            result = risk_agent.predict_risk(
                lat,lon,
                external_current_rain=current_rain,
                hybrid_model_rain=hybrid_trend
            )
            ai_rain = clean_value(result.get('rain_rate',0))
            risk_msg = result.get('status','NORMAL')
        else:
            ai_rain = 0
            risk_msg = "SAFE MODE"

        if ai_rain > 60:
            risk_msg = "🔴 FLOOD ALERT"
            flood_risk_list.append((district,ai_rain))

        report_rows.append({
            "name":district,
            "current":current_rain,
            "pred_rain":val_api,
            "trend":hybrid_trend,
            "ai_rain":ai_rain,
            "risk_msg":risk_msg
        })

    cyclone_tracks = Cyclone_prediction.run_cyclone_system()
    Flood_detection.run_forecast()
    heatmap.run_temperature_map()

    return report_rows,flood_risk_list,cyclone_tracks