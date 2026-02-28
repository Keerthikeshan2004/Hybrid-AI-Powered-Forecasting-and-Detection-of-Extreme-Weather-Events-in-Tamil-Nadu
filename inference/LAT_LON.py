import requests
import openmeteo_requests
import requests_cache
import math
import io
import numpy as np
from PIL import Image
from retry_requests import retry

# --- CONFIG ---
OWM_API_KEY = "07e0902937281053b0c758bf275744f8"
RAINVIEWER_META = "https://api.rainviewer.com/public/weather-maps.json"

DISTRICTS = {
    "Ariyalur":(11.14,79.07),"Chengalpattu":(12.69,79.97),"Chennai":(13.08,80.27),
    "Coimbatore":(11.01,76.95),"Cuddalore":(11.74,79.76),"Dharmapuri":(12.13,78.01),
    "Dindigul":(10.36,77.98),"Erode":(11.34,77.71),"Kallakurichi":(11.73,78.96),
    "Kancheepuram":(12.83,79.70),"Kanyakumari":(8.08,77.53),"Karaikal":(10.92,79.83),
    "Karur":(10.96,78.07),"Krishnagiri":(12.51,78.21),"Madurai":(9.92,78.11),
    "Mayiladuthurai":(11.10,79.65),"Nagapattinam":(10.76,79.84),"Namakkal":(11.22,78.16),
    "Nilgiris":(11.49,76.73),"Perambalur":(11.23,78.88),"Puducherry":(11.94,79.80),
    "Pudukkottai":(10.37,78.82),"Ramanathapuram":(9.35,78.83),"Ranipet":(12.94,79.33),
    "Salem":(11.66,78.14),"Sivaganga":(9.84,78.48),"Tenkasi":(8.95,77.31),
    "Thanjavur":(10.78,79.13),"Theni":(10.01,77.51),"Thoothukudi":(8.76,78.13),
    "Tiruchirappalli":(10.79,78.70),"Tirunelveli":(8.71,77.75),"Tirupathur":(12.49,78.55),
    "Tiruppur":(11.10,77.34),"Tiruvallur":(13.14,79.90),"Tiruvannamalai":(12.22,79.07),
    "Tiruvarur":(10.77,79.63),"Vellore":(12.91,79.13),"Viluppuram":(11.93,79.49),
    "Virudhunagar":(9.58,77.96),"sri lanka":(7.0,82),
}

# --- 1. OPEN-METEO ---
def fetch_open_meteo_points():
    print(" -> [API] Querying Open-Meteo (Lat/Lon Mode)...")
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    lats = [coord[0] for coord in DISTRICTS.values()]
    lons = [coord[1] for coord in DISTRICTS.values()]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lats, "longitude": lons,
        "current": ["precipitation", "rain", "showers", "temperature_2m"],
        "timezone": "Asia/Kolkata"
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        results = {}
        for i, (name, _) in enumerate(DISTRICTS.items()):
            current = responses[i].Current()
            precip = current.Variables(0).Value()
            if precip == 0.0: precip = current.Variables(2).Value() # Fallback to showers
            
            results[name] = {
                "om_rain": precip,
                "om_temp": current.Variables(3).Value()
            }
        return results
    except Exception as e:
        print(f"❌ Open-Meteo Error: {e}")
        return {}

# --- 2. OPENWEATHERMAP ---
def fetch_owm_points():
    print(" -> [API] Querying OpenWeatherMap (Lat/Lon Mode)...")
    results = {}
    for name, (lat, lon) in DISTRICTS.items():
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
            resp = requests.get(url, timeout=2).json()
            
            rain = resp.get("rain", {}).get("1h", 0.0)
            desc = resp.get("weather", [{}])[0].get("description", "").lower()
            
            # Trace detection
            if rain == 0.0 and ("rain" in desc or "drizzle" in desc):
                rain = 0.1
                
            results[name] = {
                "owm_rain": rain,
                "owm_temp": resp["main"]["temp"]
            }
        except:
            results[name] = {"owm_rain": 0.0, "owm_temp": 0.0}
    return results

# --- 3. RAINVIEWER (UPDATED: Color -> dBZ -> mm/hr) ---
def latlon_to_pixel(lat, lon, z=6, tile_size=256):
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    scale = (2 ** z) * tile_size
    x = (lon + 180.0) / 360.0 * scale
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * scale
    return int(x // 256), int(y // 256), int(x % 256), int(y % 256)

def get_dbz_from_color(r, g, b):
    """
    Maps 'Universal Blue' (Scheme 2) colors to approx dBZ.
    Uses Euclidean distance to nearest key color.
    """
    # Key Colors in Universal Blue Scheme: (R, G, B, dBZ)
    key_colors = [
        (0, 0, 0, -32.0),       # Transparent/Black (No Rain)
        (140, 220, 240, 20.0),  # Light Blue (Drizzle)
        (0, 100, 255, 30.0),    # Medium Blue (Light-Mod)
        (0, 0, 200, 40.0),      # Dark Blue (Moderate)
        (255, 240, 0, 45.0),    # Yellow (Heavy)
        (255, 120, 0, 50.0),    # Orange (Very Heavy)
        (255, 0, 0, 55.0),      # Red (Violent)
        (200, 0, 200, 60.0)     # Magenta (Extreme)
    ]
    
    best_dbz = -32.0
    min_dist = float('inf')
    
    for kr, kg, kb, k_dbz in key_colors:
        dist = math.sqrt((r-kr)**2 + (g-kg)**2 + (b-kb)**2)
        if dist < min_dist:
            min_dist = dist
            best_dbz = k_dbz
            
    return best_dbz

def dbz_to_mmhr(dbz):
    """
    Marshall-Palmer Relation: Z = 200 * R^1.6
    R = (Z / 200)^(1/1.6)
    where Z = 10^(dBZ/10)
    """
    if dbz < 10: return 0.0
    
    Z = 10 ** (dbz / 10.0)
    R = (Z / 200.0) ** (1.0 / 1.6)
    return round(R, 2)

def fetch_rainviewer_points():
    print(" -> [API] Querying RainViewer (Color Analysis -> mm/hr)...")
    results = {}
    
    try:
        meta = requests.get(RAINVIEWER_META).json()
        ts = meta["radar"]["past"][-1]["time"]
        host = meta["host"]
    except:
        return {}

    tile_cache = {}
    
    for name, (lat, lon) in DISTRICTS.items():
        xt, yt, px, py = latlon_to_pixel(lat, lon)
        tile_key = (xt, yt)
        
        if tile_key not in tile_cache:
            # Using Scheme 2 (Universal Blue)
            url = f"{host}/v2/radar/{ts}/256/6/{xt}/{yt}/2/0_0.png"
            try:
                img = Image.open(io.BytesIO(requests.get(url).content)).convert("RGBA")
                tile_cache[tile_key] = img
            except:
                tile_cache[tile_key] = None
        
        img = tile_cache[tile_key]
        rv_rain_mm = 0.0
        
        if img:
            try:
                r, g, b, a = img.getpixel((px, py))
                if a > 0: # Only process if not transparent
                    dbz = get_dbz_from_color(r, g, b)
                    rv_rain_mm = dbz_to_mmhr(dbz)
            except:
                pass
                
        results[name] = {"rv_rain": rv_rain_mm}
        
    return results

# --- MASTER FETCH ---
def get_all_api_data():
    print("\n🌍 [LAT_LON.py] Starting Multi-Source API Fetch...")
    om_data = fetch_open_meteo_points()
    owm_data = fetch_owm_points()
    rv_data = fetch_rainviewer_points()
    
    final_data = {}
    
    for name, coords in DISTRICTS.items():
        final_data[name] = {
            "lat": coords[0],
            "lon": coords[1],
            "om": om_data.get(name, {}),
            "owm": owm_data.get(name, {}),
            "rv": rv_data.get(name, {})
        }
        
    print(f"✅ [LAT_LON.py] Data collected for {len(final_data)} districts.")
    return final_data
