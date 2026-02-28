import torch
import torch.nn as nn
import numpy as np
import requests
import xml.etree.ElementTree as ET
import re
import os
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MODEL_FILE = "dataset/cyclone_track_model.pth"
HISTORY_DB_FILE = "active_cyclones_db.json"
SEQ_LENGTH = 5 

# --- MODEL ARCHITECTURE ---
class CycloneTrackModel(nn.Module):
    def __init__(self):
        super(CycloneTrackModel, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

# --- DATABASE MANAGER ---
class StormHistoryManager:
    def __init__(self, db_file):
        self.db_file = db_file
        self.history = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f: return json.load(f)
            except: return {}
        return {}

    def _save_db(self):
        with open(self.db_file, 'w') as f: json.dump(self.history, f, indent=2)

    def update_storm(self, storm_name, lat, lon, wind, pressure):
        if storm_name not in self.history: self.history[storm_name] = []
        
        timestamp = datetime.now().isoformat()
        new_record = {"ts": timestamp, "data": [lat, lon, wind, pressure]}
        
        if self.history[storm_name]:
            if self.history[storm_name][-1]['data'] == [lat, lon, wind, pressure]: return

        self.history[storm_name].append(new_record)
        if len(self.history[storm_name]) > 10: self.history[storm_name].pop(0)
        self._save_db()

    def get_sequence(self, storm_name):
        if storm_name not in self.history: return None
        records = self.history[storm_name]
        raw_data = [r['data'] for r in records]
        
        if len(raw_data) >= SEQ_LENGTH:
            return np.array(raw_data[-SEQ_LENGTH:])
        else:
            needed = SEQ_LENGTH - len(raw_data)
            padded = [raw_data[0]] * needed + raw_data
            return np.array(padded)

    def get_history_length(self, storm_name):
        return len(self.history.get(storm_name, []))

# --- LIVE DATA FETCHER ---
class LiveCycloneFetcher:
    def __init__(self):
        self.rss_url = "https://www.gdacs.org/xml/rss.xml"
        self.bob_bounds = {'lat_min': 5.0, 'lat_max': 25.0, 'lon_min': 75.0, 'lon_max': 100.0}

    def estimate_pressure(self, wind_kmph):
        wind_kt = wind_kmph * 0.539957
        if wind_kt < 30: return 1004
        return 1010 - (wind_kt / 3.92) ** 2

    def fetch_active_cyclones(self):
        print("[Network] Scanning GDACS for active cyclones...")
        try:
            resp = requests.get(self.rss_url, timeout=10)
            root = ET.fromstring(resp.content)
            active_storms = []
            namespaces = {'gdacs': 'http://www.gdacs.org', 'georss': 'http://www.georss.org/georss'}

            for item in root.findall('./channel/item'):
                event_type = item.find('gdacs:eventtype', namespaces)
                if event_type is None or event_type.text != 'TC': continue

                point = item.find('georss:point', namespaces)
                if point is None: continue
                lat, lon = map(float, point.text.split())

                if (self.bob_bounds['lat_min'] <= lat <= self.bob_bounds['lat_max'] and
                    self.bob_bounds['lon_min'] <= lon <= self.bob_bounds['lon_max']):
                    
                    title = item.find('title').text
                    desc = item.find('description').text
                    name_match = re.search(r'TC\s([A-Z0-9-]+)', title)
                    name = name_match.group(1) if name_match else "UNKNOWN_STORM"
                    wind_match = re.search(r'(\d+)\s?km/h', desc)
                    wind = float(wind_match.group(1)) if wind_match else 50.0 
                    
                    active_storms.append({
                        "name": name, "lat": lat, "lon": lon, 
                        "wind": wind, "pressure": self.estimate_pressure(wind)
                    })
            return active_storms
        except Exception as e:
            print(f"[Error] GDACS Fetch Failed: {e}")
            return []

# --- AI AGENT ---
class AdvancedCycloneAgent:
    def __init__(self, model_path):
        self.device = torch.device('cpu')
        self.db = StormHistoryManager(HISTORY_DB_FILE)
        self.tracks = [] # Stores prediction results

        print(f"[System] Loading Hybrid Model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.scaler = checkpoint.get('scaler')
            self.model = CycloneTrackModel()
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
        except Exception as e:
            print(f"[CRITICAL] Model Initialization Failed: {e}")
            self.model = None

    def process_storm(self, storm_data):
        if not self.model: return
        name = storm_data['name']
        print(f"\n🌪️ PROCESSING STORM: {name}")
        
        self.db.update_storm(name, storm_data['lat'], storm_data['lon'], storm_data['wind'], storm_data['pressure'])
        sequence = self.db.get_sequence(name)
        
        if self.scaler: scaled_seq = self.scaler.transform(sequence)
        else: scaled_seq = sequence

        with torch.no_grad():
            pred_scaled = self.model(torch.FloatTensor(scaled_seq).unsqueeze(0))

        if self.scaler: pred = self.scaler.inverse_transform(pred_scaled.numpy())[0]
        else: pred = pred_scaled.numpy()[0]

        pred_lat, pred_lon = round(float(pred[0]), 2), round(float(pred[1]), 2)
        print(f"   🔮 FORECAST (T+6h): {pred_lat}N, {pred_lon}E | {round(float(pred[2]), 1)} km/h")

        self.tracks.append({
            "name": name,
            "current_lat": storm_data['lat'], "current_lon": storm_data['lon'],
            "pred_lat": pred_lat, "pred_lon": pred_lon
        })

# --- EXPORTED FUNCTION ---
def run_cyclone_system():
    """Fetches data, runs predictions, and returns track list."""
    fetcher = LiveCycloneFetcher()
    agent = AdvancedCycloneAgent(MODEL_FILE)
    
    storms = fetcher.fetch_active_cyclones()
    if storms:
        for storm in storms:
            agent.process_storm(storm)
    else:
        print("\n✅ No active cyclones in Bay of Bengal.")
    
    return agent.tracks

if __name__ == "__main__":
    run_cyclone_system()