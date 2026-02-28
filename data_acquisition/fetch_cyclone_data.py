import requests
import pandas as pd
import xml.etree.ElementTree as ET
import os
from datetime import datetime
import sys

# --- CONFIG: TAMIL NADU FOCUSED ---
# 1. Cyclone: North Indian Ocean ONLY (Code: NI)
# This file is small (~26MB) and contains ONLY storms affecting India/Bay of Bengal
IBTRACS_NI_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.NI.list.v04r00.csv"

# 2. Real-Time Alerts: NDMA Sachet (Govt of India)
NDMA_RSS_FEED = "https://sachet.ndma.gov.in/cap/rss"

OUTPUT_DIR = "dataset/tn_extreme_weather"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_tn_cyclone_history():
    """
    Downloads ONLY North Indian Ocean cyclone tracks (Bay of Bengal).
    Filters specifically for storms near Tamil Nadu (Lat 8-15N, Lon 76-81E).
    """
    print(f"\n🌪️  Downloading North Indian Ocean Cyclone Data (Bay of Bengal)...")
    file_path = os.path.join(OUTPUT_DIR, "ibtracs_ni_raw.csv")
    
    try:
        # Download with timeout and progress
        with requests.get(IBTRACS_NI_URL, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f:
                dl = 0
                for chunk in r.iter_content(chunk_size=8192):
                    dl += len(chunk)
                    f.write(chunk)
                    if total_size:
                        done = int(50 * dl / total_size)
                        sys.stdout.write(f"\r   [{'=' * done}{' ' * (50-done)}] {dl//1024} KB")
                        sys.stdout.flush()
        
        print(f"\n✅ Download Complete. Filtering for Tamil Nadu proximity...")

        # FILTER: Keep only storms that came near TN coast
        df = pd.read_csv(file_path, skiprows=[1], low_memory=False)
        
        # Convert columns to numeric
        df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
        df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
        
        # Bounding Box for Tamil Nadu & Neighborhood
        # Lat: 5°N to 20°N, Lon: 75°E to 90°E (Bay of Bengal focus)
        tn_storms = df[
            (df['LAT'].between(5, 20)) & 
            (df['LON'].between(75, 90))
        ]
        
        # Save filtered list
        clean_path = os.path.join(OUTPUT_DIR, "tn_cyclone_tracks_filtered.csv")
        tn_storms.to_csv(clean_path, index=False)
        print(f"✅ Filtered Data Saved: {clean_path}")
        print(f"   (Contains {len(tn_storms['SID'].unique())} unique storms affecting the region)")
        
        return tn_storms

    except Exception as e:
        print(f"\n❌ Download Failed: {e}")
        return None

def fetch_tn_live_alerts():
    """
    Fetches real-time flood/cyclone alerts and filters STRICTLY for Tamil Nadu.
    """
    print(f"\n🚨 Checking Live Alerts (Tamil Nadu & Puducherry Only)...")
    alerts = []
    
    try:
        # NDMA Sachet Feed
        resp = requests.get(NDMA_RSS_FEED, timeout=10)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall('./channel/item'):
                title = item.find('title').text or ""
                desc = item.find('description').text or ""
                
                # STRICT KEYWORD FILTER
                keywords = ["Tamil Nadu", "Puducherry", "Chennai", "Karaikal", "Pamban", "Bay of Bengal"]
                if any(k.lower() in (title + desc).lower() for k in keywords):
                    alerts.append({
                        "event": title,
                        "desc": desc[:150] + "...",
                        "source": "NDMA"
                    })
    except Exception as e:
        print(f"⚠️ Alert Check Failed: {e}")
        
    if not alerts:
        print("✅ No active severe weather alerts for Tamil Nadu at this moment.")
    else:
        print(f"⚠️ FOUND {len(alerts)} ACTIVE ALERTS FOR TN:")
        for a in alerts:
            print(f"   🔴 {a['event']}")
            
    return alerts

if __name__ == "__main__":
    # 1. Download & Filter Historical Data (Run once)
    if not os.path.exists(os.path.join(OUTPUT_DIR, "tn_cyclone_tracks_filtered.csv")):
        download_tn_cyclone_history()
    else:
        print("\n✅ Historical Data already exists. Skipping download.")
        
    # 2. Check Live Alerts
    fetch_tn_live_alerts()
