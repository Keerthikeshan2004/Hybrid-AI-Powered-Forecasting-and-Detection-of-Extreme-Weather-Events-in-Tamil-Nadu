import openmeteo_requests
import requests_cache
import pandas as pd 
from retry_requests import retry

# Setup the Open-Meteo API client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": [13.08, 11.32, 12.97],  # Chennai, Wellington, Bangalore
    "longitude": [80.27, 76.79, 77.59],
    "hourly": ["temperature_2m", "rain", "surface_pressure", "cloud_cover", "wind_speed_10m"],
    "timezone": "Asia/Kolkata"
}

responses = openmeteo.weather_api(url, params=params)
response = responses[0]

# Process hourly data
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_rain = hourly.Variables(1).ValuesAsNumpy()

data = {
    # 1. Note: This creates UTC timestamps initially
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
    "temperature": hourly_temperature_2m,
    "rain": hourly_rain
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
#  STEP 1: Convert UTC data to your Local Time (IST)
# ---------------------------------------------------------
df['date'] = df['date'].dt.tz_convert('Asia/Kolkata')

# ---------------------------------------------------------
#  STEP 2: Get current OS time & round to nearest hour
# ---------------------------------------------------------
# 'floor("h")' rounds 18:25 down to 18:00 to match the API interval
current_time = pd.Timestamp.now(tz='Asia/Kolkata').floor('h')

# ---------------------------------------------------------
#  STEP 3: Filter the DataFrame for that specific row
# ---------------------------------------------------------
real_time_data = df[df['date'] == current_time]

print(f"Current System Time: {current_time}")
print(real_time_data)
