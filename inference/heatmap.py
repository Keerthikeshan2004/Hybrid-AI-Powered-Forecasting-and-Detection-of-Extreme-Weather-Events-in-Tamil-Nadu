import requests
import pandas as pd
import folium
from folium.features import DivIcon

# --- DATA ---
DISTRICTS_DATA = [
  {"district": "Ariyalur", "lat": 11.14, "lon": 79.07},
  {"district": "Chengalpattu", "lat": 12.69, "lon": 79.97},
  {"district": "Chennai", "lat": 13.08, "lon": 80.27},
  {"district": "Coimbatore", "lat": 11.01, "lon": 76.95},
  {"district": "Cuddalore", "lat": 11.74, "lon": 79.76},
  {"district": "Dharmapuri", "lat": 12.13, "lon": 78.01},
  {"district": "Dindigul", "lat": 10.36, "lon": 77.98},
  {"district": "Erode", "lat": 11.34, "lon": 77.71},
  {"district": "Kallakurichi", "lat": 11.73, "lon": 78.96},
  {"district": "Kancheepuram", "lat": 12.83, "lon": 79.70},
  {"district": "Kanyakumari", "lat": 8.08, "lon": 77.53},
  {"district": "Karaikal", "lat": 10.92, "lon": 79.83},
  {"district": "Karur", "lat": 10.96, "lon": 78.07},
  {"district": "Krishnagiri", "lat": 12.51, "lon": 78.21},
  {"district": "Madurai", "lat": 9.92, "lon": 78.11},
  {"district": "Mayiladuthurai", "lat": 11.10, "lon": 79.65},
  {"district": "Nagapattinam", "lat": 10.76, "lon": 79.84},
  {"district": "Namakkal", "lat": 11.22, "lon": 78.16},
  {"district": "Nilgiris", "lat": 11.49, "lon": 76.73},
  {"district": "Perambalur", "lat": 11.23, "lon": 78.88},
  {"district": "Puducherry", "lat": 11.94, "lon": 79.80},
  {"district": "Pudukkottai", "lat": 10.37, "lon": 78.82},
  {"district": "Ramanathapuram", "lat": 9.35, "lon": 78.83},
  {"district": "Ranipet", "lat": 12.94, "lon": 79.33},
  {"district": "Salem", "lat": 11.66, "lon": 78.14},
  {"district": "Sivaganga", "lat": 9.84, "lon": 78.48},
  {"district": "Tenkasi", "lat": 8.95, "lon": 77.31},
  {"district": "Thanjavur", "lat": 10.78, "lon": 79.13},
  {"district": "Theni", "lat": 10.01, "lon": 77.51},
  {"district": "Thoothukudi", "lat": 8.76, "lon": 78.13},
  {"district": "Tiruchirappalli", "lat": 10.79, "lon": 78.70},
  {"district": "Tirunelveli", "lat": 8.71, "lon": 77.75},
  {"district": "Tirupathur", "lat": 12.49, "lon": 78.55},
  {"district": "Tiruppur", "lat": 11.10, "lon": 77.34},
  {"district": "Tiruvallur", "lat": 13.14, "lon": 79.90},
  {"district": "Tiruvannamalai", "lat": 12.22, "lon": 79.07},
  {"district": "Tiruvarur", "lat": 10.77, "lon": 79.63},
  {"district": "Vellore", "lat": 12.91, "lon": 79.13},
  {"district": "Viluppuram", "lat": 11.93, "lon": 79.49},
  {"district": "Virudhunagar", "lat": 9.58, "lon": 77.96},
  {"district": "Sri Lanka", "lat": 7.00, "lon": 82.00}
]

def get_color(temp):
    if temp < 25: return '#00b894'  # Cool
    elif 25 <= temp < 30: return '#fdcb6e'  # Moderate
    elif 30 <= temp < 35: return '#e17055'  # Warm
    else: return '#d63031'  # Hot

def run_temperature_map():
    print("\n🌡️ Generating Temperature Heatmap (Open-Meteo)...")
    
    lats = [str(d['lat']) for d in DISTRICTS_DATA]
    lons = [str(d['lon']) for d in DISTRICTS_DATA]
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": ",".join(lats),
        "longitude": ",".join(lons),
        "current": "temperature_2m"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Robust parsing for list vs single object response
        if isinstance(data, list):
            temps = [x['current']['temperature_2m'] for x in data]
        elif 'current' in data and isinstance(data['current']['temperature_2m'], list):
            temps = data['current']['temperature_2m']
        else:
            temps = [data['current']['temperature_2m']] * len(DISTRICTS_DATA)
            
    except Exception as e:
        print(f"⚠️ Temp API Error: {e}. Using placeholders.")
        temps = [30.0] * len(DISTRICTS_DATA)

    m = folium.Map(location=[10.8, 78.5], zoom_start=7, tiles="CartoDB positron")

    for i, d in enumerate(DISTRICTS_DATA):
        t = temps[i]
        c = get_color(t)
        
        folium.Circle(
            location=[d['lat'], d['lon']], radius=12000, color=c, fill=True, fill_color=c, fill_opacity=0.7,
            tooltip=f"{d['district']}: {t}°C"
        ).add_to(m)

        folium.map.Marker(
            [d['lat'], d['lon']],
            icon=DivIcon(
                icon_size=(150,36), icon_anchor=(75,10),
                html=f"""<div style="font-size: 10pt; font-weight: bold; color: black; text-align: center; text-shadow: 1px 1px 2px white;">{t}°C</div>"""
            )
        ).add_to(m)

    m.save("tn_live_weather.html")
    print("✔ Saved: tn_live_weather.html")

if __name__ == "__main__":
    run_temperature_map()