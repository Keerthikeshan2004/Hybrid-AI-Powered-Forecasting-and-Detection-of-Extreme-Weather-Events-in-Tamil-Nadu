import sys
import os

# Add the parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader

# === IMPORT THE MODEL HERE ===
from models.hybrid_model import HybridCNNConvLSTM

# ==========================================
# 1. Dataset Class
# ==========================================
class WeatherDataset(Dataset):
    def __init__(self, nc_file, seq_len=12, pred_len=6):
        print(f"Loading {nc_file}...")
        ds = xr.open_dataset(nc_file).fillna(0)
        
        elev_repeated = np.broadcast_to(
            ds['elevation'].values[None, ...], 
            ds['era5_rain'].shape
        )
        
        self.feature_data = np.stack([
            ds['era5_rain'].values,
            ds['temperature'].values,
            ds['pressure'].values,
            ds['wind_u'].values,
            ds['wind_v'].values,
            elev_repeated
        ], axis=1)
        
        self.target_data = ds['imerg_target'].values[:, None, :, :]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = self.feature_data.shape[0] - seq_len - pred_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.feature_data[idx : idx + self.seq_len]
        target_start = idx + self.seq_len
        y = self.target_data[target_start : target_start + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# ==========================================
# 2. Training Loop
# ==========================================
def train():
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 0.001
    
    print("Initializing Dataset...")
    dataset = WeatherDataset("data/tn_master_training_data_fixed.nc")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Initializing Hybrid Model...")
    # Using the imported class
    model = HybridCNNConvLSTM(input_dim=6, hidden_dim=32, kernel_size=(3,3), num_layers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train() # Important for CNN layers in hybrid model
        
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x, future_steps=6)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.6f}")
                
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.6f}")
        
    print("✅ Training Complete!")
    torch.save(model.state_dict(), "dataset/tn_hybrid_model.pth")
    print("Saved model to 'tn_hybrid_model.pth'")

if __name__ == "__main__":
    train()
