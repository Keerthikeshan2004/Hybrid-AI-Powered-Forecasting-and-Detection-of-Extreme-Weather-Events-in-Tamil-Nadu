import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import requests
import os
import warnings
from scipy.spatial import cKDTree

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. DEEP LEARNING CORE
# ==========================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        c_next = torch.sigmoid(cc_f) * c_cur + torch.sigmoid(cc_i) * torch.tanh(cc_g)
        h_next = torch.sigmoid(cc_o) * torch.tanh(c_next)
        return h_next, c_next

class SpatialEncoder(nn.Module):
    def __init__(self):
        super(SpatialEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

class HybridRainfallModel(nn.Module):
    def __init__(self):
        super(HybridRainfallModel, self).__init__()
        self.spatial_encoder = SpatialEncoder()
        self.cell_list = nn.ModuleList([
            ConvLSTMCell(32, 32, 3, True),
            ConvLSTMCell(32, 32, 3, True)
        ])
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, hidden_states=None):
        b, seq_len, c, h, w = x.size()
        encoded = self.spatial_encoder(x.view(b * seq_len, c, h, w)).view(b, seq_len, 32, h, w)
        
        if hidden_states is None:
            hidden_states = [(torch.zeros(b, 32, h, w), torch.zeros(b, 32, h, w)) for _ in range(2)]
        
        cur_input = encoded
        for i, cell in enumerate(self.cell_list):
            h, c = hidden_states[i]
            output_seq = []
            for t in range(seq_len):
                h, c = cell(cur_input[:, t], (h, c))
                output_seq.append(h.unsqueeze(1))
            cur_input = torch.cat(output_seq, dim=1)
            
        return self.final_conv(cur_input[:, -1])

# ==========================================
# 2. ADAPTIVE AI AGENT (FINAL FIXED VERSION)
# ==========================================
class HybridWeatherAgent:
    def __init__(self, owm_key, csv_path, model_path="dataset/tn_hybrid_model_2.pth"):
        self.owm_key = owm_key
        self.tree = None
        self.jaxa_df = None
        self.district_stats = {}
        
        self._load_jaxa_data(csv_path)
        
        self.model = HybridRainfallModel()
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print("✅ AI Agent: Model Weights Loaded.")
            except Exception as e:
                print(f"⚠️ AI Agent Model Load Error: {e}")
        else:
            print(f"⚠️ AI Agent: Model file '{model_path}' not found. Using random weights.")
            self.model.eval()

    def _load_jaxa_data(self, csv_path):
        if not os.path.exists(csv_path): return
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip().lower() for c in df.columns]
            rain_col = next((c for c in df.columns if 'rain' in c), None)
            dist_col = next((c for c in df.columns if 'district' in c), None)
            lat_col = next((c for c in df.columns if 'lat' in c), None)
            lon_col = next((c for c in df.columns if 'lon' in c), None)

            if rain_col and dist_col:
                self.jaxa_df = df.rename(columns={rain_col: 'rainfall', dist_col: 'district'}).fillna(0)
                self.district_stats = self.jaxa_df.groupby('district')['rainfall'].agg(['mean', 'max', 'std']).to_dict('index')
                if lat_col and lon_col:
                    self.locations = df[[lat_col, lon_col]].drop_duplicates()
                    self.tree = cKDTree(self.locations.values)
                    self.loc_map = self.locations.index.tolist()
        except Exception: pass

    def _build_context_tensor(self, lat, lon, current_rain, hybrid_past_rain):
        t = torch.zeros(1, 3, 6, 32, 32)
        
        # 1. Get Climate Stats
        stats = {'mean': 5.0, 'max': 50.0, 'std': 2.0}
        if self.tree:
            try:
                dist, idx = self.tree.query([lat, lon], k=1)
                if dist < 0.5:
                    row_idx = self.loc_map[idx]
                    dist_name = self.jaxa_df.loc[row_idx, 'district']
                    if dist_name in self.district_stats:
                        stats = self.district_stats[dist_name]
            except: pass

        # 2. Normalize Inputs (WITH FLOAT FIX)
        MAX_R = 50.0
        # Convert numpy floats to python floats to avoid PyTorch TypeError
        val_t0 = float(min(1.0, (hybrid_past_rain * 0.8) / MAX_R)) # t-2 (Approx)
        val_t1 = float(min(1.0, hybrid_past_rain / MAX_R))         # t-1 (Hybrid Trend)
        val_t2 = float(min(1.0, current_rain / MAX_R))             # t   (Live Reality)

        norm_mean = min(1.0, stats['mean'] / 50.0)
        norm_max = min(1.0, stats['max'] / 100.0)
        norm_std = min(1.0, stats['std'] / 20.0)

        # 3. Fill Tensor
        t[:, 0, 0, :, :] = val_t0
        t[:, 1, 0, :, :] = val_t1
        t[:, 2, 0, :, :] = val_t2

        for time_step in range(3):
            t[:, time_step, 1, :, :] = norm_mean
            t[:, time_step, 2, :, :] = norm_max
            t[:, time_step, 3, :, :] = norm_std

        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32))
        t[:, :, 4, :, :] = torch.tensor(grid_x).float()
        t[:, :, 5, :, :] = torch.tensor(grid_y).float()
        
        return t

    def predict_risk(self, lat, lon, external_current_rain, hybrid_model_rain):
        """
        Combines Hybrid Trend + Live Reality with BIAS CORRECTION.
        Prevents 'dampening' and silences 'ghost rain' noise.
        """
        inp = self._build_context_tensor(lat, lon, external_current_rain, hybrid_model_rain)
        
        raw_ai_output = 0.0
        if self.model:
            try:
                with torch.no_grad():
                    out = self.model(inp)
                    raw_score = out[0, 0, 16, 16].item() 
                    raw_ai_output = max(0.0, raw_score * 50.0)
            except Exception as e:
                print(f"Inference Error: {e}")
                raw_ai_output = external_current_rain

        # --- INTELLIGENT BLENDING (Persistence Logic) ---
        final_prediction = raw_ai_output

        # CASE A: It is Raining NOW (> 0.1mm)
        if external_current_rain > 0.1:
            # Blend: 60% Reality (Persistence), 40% AI Adjustment
            # e.g., Input 3.0 -> (1.8 + 0.1) = ~1.9mm (Safe)
            final_prediction = (external_current_rain * 0.6) + (raw_ai_output * 0.4)
            
            # If AI sees a Rising Trend (Hybrid > Current), boost prediction
            if hybrid_model_rain > external_current_rain:
                final_prediction *= 1.15 # 15% Boost for storm growth
        
        # CASE B: It is DRY NOW (0.0mm) - FIX FOR "0.329" GHOST RAIN
        else:
            # Only predict rain if the Hybrid Trend explicitly sees a storm coming (>0.15mm)
            if hybrid_model_rain > 0.15:
                # Storm is approaching! Trust the AI.
                final_prediction = raw_ai_output
            else:
                # It's dry now AND the trend is dry. Force 0.0.
                final_prediction = 0.0

        return {
            "status": "NORMAL" if final_prediction < 5 else "ALERT",
            "rain_rate": round(final_prediction, 3)
        }