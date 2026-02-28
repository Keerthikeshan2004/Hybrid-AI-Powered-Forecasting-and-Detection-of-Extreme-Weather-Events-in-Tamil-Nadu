import requests
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
from datetime import datetime

def fetch_tn_weather_fixed():
    url = "https://mausam.imd.gov.in/imd_latest/contents/Todaysweather_mc.php?id=26"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"📡 Connecting to IMD Server...")
    
    try:
        response = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # FIX: Find ALL tables and select the one that actually contains weather data
        tables = soup.find_all('table')
        weather_table = None
        
        for t in tables:
            # Check if this table has the header "Max Temp" or "Station"
            if "Max Temp" in t.get_text() and "Station" in t.get_text():
                weather_table = t
                break
        
        if not weather_table:
            print("❌ Error: Could not locate the specific Weather Data table.")
            return None

        # Process the correct table
        rows = weather_table.find_all('tr')
        data = []
        
        for row in rows:
            cols = [ele.get_text(strip=True) for ele in row.find_all('td')]
            
            # FIX: STRICTLY allow only rows with exactly 8 columns
            # This filters out the "Date" row (1 column) and wrapper rows
            if len(cols) == 8:
                # Double check it's not the header row itself
                if cols[0] != "Station": 
                    data.append(cols)

        if not data:
            print("⚠️ Table found, but no data rows match the format.")
            return None

        # Define Headers
        headers_list = [
            "Station", "Max Temp", "Max Dep", "Min Temp", "Min Dep", 
            "RH 0830", "RH 1730", "Rainfall"
        ]
        
        df = pd.DataFrame(data, columns=headers_list)
        
        # Filter for your cities
        target_cities = [
            "Chennai", "Coimbatore", "Madurai", "Salem", "Tiruttani", 
            "Vellore", "Puducherry", "Karaikal", "Coonoor", "Trichy"
        ]
        
        df_filtered = df[df['Station'].str.contains('|'.join(target_cities), case=False, na=False)]
        
        return df_filtered

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    df = fetch_tn_weather_fixed()
    
    if df is not None and not df.empty:
        print("\n" + "="*80)
        print(f"🌤️  TAMIL NADU WEATHER REPORT | {datetime.now().strftime('%d-%b-%Y %I:%M %p')}")
        print("="*80)
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
        
        df.to_csv("dataset/tn_weather_fixed.csv", index=False)
        print("\n💾 Saved to tn_weather_fixed.csv")
    else:
        print("\n⚠️  No data found.")
