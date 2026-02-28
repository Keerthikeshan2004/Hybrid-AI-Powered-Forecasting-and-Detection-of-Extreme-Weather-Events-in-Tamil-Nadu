import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_pipeline import run_pipeline

st.set_page_config(layout="wide", page_title="TN Hybrid Weather System")

st.title("🌏 Tamil Nadu Integrated Hybrid Weather Intelligence System")

data = run_pipeline()

if data is None:
    st.error("Base grid data missing.")
    st.stop()

report_rows, flood_risk_list, cyclone_tracks = data

tab1,tab2,tab3,tab4 = st.tabs([
    "🌧️ Rainfall",
    "🌀 Cyclone",
    "🌊 Flood",
    "🔥 Heatmap"
])

# Rainfall
with tab1:
    df = pd.DataFrame([
        {
            "District": r["name"],
            "Current (mm)": round(r["current"],2),
            "Predicted API (mm)": round(r["pred_rain"],2),
            "Hybrid Trend (mm)": round(r["trend"],2),
            "AI Future Rain (mm)": round(r["ai_rain"],3),
            "Risk Status": r["risk_msg"]
        }
        for r in report_rows
    ])
    st.dataframe(df,use_container_width=True)

# Cyclone
with tab2:
    if cyclone_tracks:
        for c in cyclone_tracks:
            st.warning(
                f"🌀 {c['name']} | Live: ({c['current_lat']},{c['current_lon']}) "
                f"| Forecast: ({c['pred_lat']},{c['pred_lon']})"
            )
    else:
        st.success("No active cyclones.")

# Flood
with tab3:
    if flood_risk_list:
        for d,r in flood_risk_list:
            st.error(f"🔴 FLOOD ALERT: {d} ({r:.2f} mm)")
    else:
        st.success("No flood risks detected.")

# Heatmap
with tab4:
    if os.path.exists("tn_live_weather.html"):
        with open("tn_live_weather.html","r",encoding="utf-8") as f:
            st.components.v1.html(f.read(),height=700,scrolling=True)
    else:
        st.warning("Heatmap not generated.")