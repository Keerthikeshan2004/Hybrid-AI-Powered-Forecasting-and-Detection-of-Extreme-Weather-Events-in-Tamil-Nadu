import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import sys
import os

# Add the parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- CONFIG ---
DATA_FILE = "dataset/tn_extreme_weather/tn_cyclone_tracks_filtered.csv"
MODEL_SAVE_PATH = "dataset/cyclone_track_model.pth"
SEQ_LENGTH = 4  # Input: Past 4 steps (e.g., 24 hours if data is 6-hourly)
PRED_LENGTH = 1 # Output: Next 1 step (next 6 hours)

# --- 1. DATASET CLASS ---
class CycloneDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# --- 2. PREPROCESSING ---
def prepare_data(file_path):
    print(f"📂 Loading Data: {file_path}")
    df = pd.read_csv(file_path)
    
    # We need SID (Storm ID) to group tracks
    # Features: Lat, Lon, Wind Speed (WMO_WIND), Pressure (WMO_PRES)
    # Ensure numeric
    cols = ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # Normalize Data (0-1 range) for LSTM
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    
    sequences = []
    targets = []
    
    # Group by Storm ID (SID) to create valid tracks
    grouped = df.groupby('SID')
    
    for _, group in grouped:
        data = group[cols].values
        if len(data) < SEQ_LENGTH + PRED_LENGTH:
            continue
            
        for i in range(len(data) - SEQ_LENGTH - PRED_LENGTH):
            seq = data[i : i + SEQ_LENGTH]       # Past 4 steps
            target = data[i + SEQ_LENGTH]        # Next step (Lat, Lon, Wind, Pres)
            
            sequences.append(seq)
            targets.append(target)
            
    return np.array(sequences), np.array(targets), scaler

# --- 3. LSTM MODEL ---
class CycloneLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=4):
        super(CycloneLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take last output
        return out

# --- 4. TRAINING LOOP ---
def train_model():
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file missing: {DATA_FILE}")
        return

    # A. Prepare
    X, y, scaler = prepare_data(DATA_FILE)
    print(f"✅ Created {len(X)} training sequences.")
    
    dataset = CycloneDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # B. Model
    model = CycloneLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # C. Train
    print("\n🚀 Starting Training...")
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seqs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")

    # D. Save
    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler
    }, MODEL_SAVE_PATH)
    print(f"\n💾 Model Saved: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
