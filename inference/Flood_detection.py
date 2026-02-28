import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests
import joblib
import os
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MODEL_PATH = "models/flood_lstm_model.pth"
SCALER_PATH = "models/flood_lstm_scaler.pkl"

# LSTM Hyperparameters (Must match the trained model)
SEQ_LENGTH = 7   
INPUT_SIZE = 3   
HIDDEN_SIZE = 64
NUM_LAYERS = 2

DISTRICTS = {
    "Ariyalur":       {"lat": 11.14, "lon": 79.07, "elev": 76.0},
    "Chengalpattu":   {"lat": 12.69, "lon": 79.97, "elev": 36.0},
    "Chennai":        {"lat": 13.08, "lon": 80.27, "elev": 6.0},
    "Coimbatore":     {"lat": 11.01, "lon": 76.95, "elev": 411.0},
    "Cuddalore":      {"lat": 11.74, "lon": 79.76, "elev": 6.0},
    "Dharmapuri":     {"lat": 12.13, "lon": 78.01, "elev": 462.0},
    "Dindigul":       {"lat": 10.36, "lon": 77.98, "elev": 280.0},
    "Erode":          {"lat": 11.34, "lon": 77.71, "elev": 183.0},
    "Kallakurichi":   {"lat": 11.73, "lon": 78.96, "elev": 100.0},
    "Kancheepuram":   {"lat": 12.83, "lon": 79.70, "elev": 83.0},
    "Kanyakumari":    {"lat": 8.08,  "lon": 77.53, "elev": 30.0},
    "Karaikal":       {"lat": 10.92, "lon": 79.83, "elev": 4.0},
    "Karur":          {"lat": 10.96, "lon": 78.07, "elev": 122.0},
    "Krishnagiri":    {"lat": 12.51, "lon": 78.21, "elev": 641.0},
    "Madurai":        {"lat": 9.92,  "lon": 78.11, "elev": 136.0},
    "Mayiladuthurai": {"lat": 11.10, "lon": 79.65, "elev": 10.0},
    "Nagapattinam":   {"lat": 10.76, "lon": 79.84, "elev": 9.0},
    "Namakkal":       {"lat": 11.22, "lon": 78.16, "elev": 218.0},
    "Nilgiris":       {"lat": 11.49, "lon": 76.73, "elev": 2240.0},
    "Perambalur":     {"lat": 11.23, "lon": 78.88, "elev": 143.0},
    "Puducherry":     {"lat": 11.94, "lon": 79.80, "elev": 3.0},
    "Pudukkottai":    {"lat": 10.37, "lon": 78.82, "elev": 100.0},
    "Ramanathapuram": {"lat": 9.35,  "lon": 78.83, "elev": 2.0},
    "Ranipet":        {"lat": 12.94, "lon": 79.33, "elev": 160.0},
    "Salem":          {"lat": 11.66, "lon": 78.14, "elev": 278.0},
    "Sivaganga":      {"lat": 9.84,  "lon": 78.48, "elev": 102.0},
    "Tenkasi":        {"lat": 8.95,  "lon": 77.31, "elev": 145.0},
    "Thanjavur":      {"lat": 10.78, "lon": 79.13, "elev": 57.0},
    "Theni":          {"lat": 10.01, "lon": 77.51, "elev": 300.0},
    "Thoothukudi":    {"lat": 8.76,  "lon": 78.13, "elev": 4.0},
    "Tiruchirappalli":{"lat": 10.79, "lon": 78.70, "elev": 88.0},
    "Tirunelveli":    {"lat": 8.71,  "lon": 77.75, "elev": 47.0},
    "Tirupathur":     {"lat": 12.49, "lon": 78.55, "elev": 387.0},
    "Tiruppur":       {"lat": 11.10, "lon": 77.34, "elev": 295.0},
    "Tiruvallur":     {"lat": 13.14, "lon": 79.90, "elev": 37.0},
    "Tiruvannamalai": {"lat": 12.22, "lon": 79.07, "elev": 171.0},
    "Tiruvarur":      {"lat": 10.77, "lon": 79.63, "elev": 5.0},
    "Vellore":        {"lat": 12.91, "lon": 79.13, "elev": 216.0},
    "Viluppuram":     {"lat": 11.93, "lon": 79.49, "elev": 45.0},
    "Virudhunagar":   {"lat": 9.58,  "lon": 77.96, "elev": 101.0}
}

# ==========================================
# 1. MODEL DEFINITION (MUST MATCH SAVED FILE)
# ==========================================
class FloodLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(FloodLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# ==========================================
# 2. LIVE PREDICTION ENGINE
# ==========================================
def get_past_7_days_weather(lat, lon):
    end = datetime.now().date()
    start = end - timedelta(days=SEQ_LENGTH-1)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "daily": "precipitation_sum", "timezone": "auto"}
    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        hist = data['daily']['precipitation_sum']
        if not hist: return [0.0]*SEQ_LENGTH
        if len(hist) < SEQ_LENGTH: return [0.0]*(SEQ_LENGTH - len(hist)) + hist
        return hist[-SEQ_LENGTH:]
    except: return [0.0]*SEQ_LENGTH

def run_forecast():
    # 1. Load Model & Scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ Model files not found! Please run the training script first.")
        return

    model = FloodLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
        
    model.eval()
    scaler = joblib.load(SCALER_PATH)

    print("\n🌍 LIVE FLOOD FORECAST (LSTM Time-Series)")
    

    flood_found = False
    header_printed = False

    for name, data in DISTRICTS.items():
        # Get Live Data
        rain_seq = get_past_7_days_weather(data['lat'], data['lon'])
        
        # =========== SIMULATION BLOCK (Uncomment to Test Alerts) ===========
        # if name == "Chennai":
        #    rain_seq = [10.0, 20.0, 50.0, 80.0, 120.0, 180.0, 250.0] 
        # ===================================================================

        total_rain = sum(rain_seq)
        
        # Prepare Input Sequence
        input_seq = []
        for r in rain_seq:
            d_proxy = 5.0 + (r * 2.0) 
            e_val = data['elev'] 
            input_seq.append([r, d_proxy, e_val])

        input_seq = np.array(input_seq)
        try: input_scaled = scaler.transform(input_seq)
        except: input_scaled = input_seq / [300, 500, 2000]

        # Model Inference
        input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0)
        with torch.no_grad():
            prob = model(input_tensor).item() * 100

        # Sanity Check
        if total_rain < 20.0: prob = 0.0
        elif total_rain < 50.0: prob = min(prob, 20.0)

        # Status Logic
        is_risky = False
        status = ""
        if total_rain > 150:
            is_risky = True
            status = "🚨 ACCUMULATION ALERT"
        elif prob > 60:
            is_risky = True
            status = "🟠 FLOOD WARNING"

        if is_risky:
            if not header_printed:
                print(f"{'DISTRICT':<15} | {'7-DAY RAIN':<15} | {'RISK %':<8} | {'STATUS'}")
                print("-" * 65)
                header_printed = True
            print(f"{name:<15} | {total_rain:<15.1f} | {prob:<8.0f} | {status}")
            flood_found = True

    if not flood_found:
        print("✅ No flood risks detected based on past 7 days pattern.")
    print("-" * 140)

if __name__ == "__main__":
    run_forecast()