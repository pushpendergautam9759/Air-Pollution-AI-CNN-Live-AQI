"""
Air Pollution AI Dashboard: CNN image prediction + live model-based AQ data.
Run: streamlit run app.py
"""
import os
import requests
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Air Pollution AI", page_icon="🌫️", layout="wide")
st.title("🌫️ Air Pollution AI — Image + Live AQ Analysis")
st.caption("Educational prototype | CNN visual classification + CAMS-based air-quality data")

CLASSES = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
           "Unhealthy", "Very Unhealthy", "Hazardous"]
MODEL_PATH = "outputs/air_pollution_cnn.keras"

@st.cache_resource
def load_cnn():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

def geocode_city(city):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    response = requests.get(url, params={"name": city, "count": 1,
                                         "language": "en", "format": "json"},
                            timeout=20)
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        raise ValueError("City not found. Try a nearby larger city.")
    return results[0]

def get_air_quality(lat, lon):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat, "longitude": lon,
        "current": "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "past_hours": 24, "forecast_hours": 1, "timezone": "auto"
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def fmt(value, unit):
    if value is None:
        return "Not available"
    try:
        return f"{float(value):,.1f} {unit}"
    except (ValueError, TypeError):
        return "Not available"

def main():
    st.info("Upload an outdoor photo. Select a city to fetch model-based pollutant estimates for that location.")
    left, right = st.columns([1,1])
    with left:
        uploaded = st.file_uploader("Upload an outdoor image", type=["jpg","jpeg","png"])
        city = st.text_input("City / location", value="New Delhi")
        run_button = st.button("Analyse image and air quality", type="primary", use_container_width=True)

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

    if run_button:
        if uploaded is None:
            st.error("Please upload an image first.")
            return
        with st.spinner("Analysing image and retrieving air-quality data..."):
            # CNN result is a visual class prediction, not a pollutant measurement.
            model = load_cnn()
            if model is None:
                st.warning("CNN model is not trained yet. Run `python train.py` first. Live air data can still be checked.")
                visual_label, visual_conf = None, None
            else:
                resized = image.resize((128,128))
                arr = np.asarray(resized, dtype=np.float32) / 255.0
                scores = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
                idx = int(np.argmax(scores))
                visual_label, visual_conf = CLASSES[idx], float(scores[idx]) * 100

            place = geocode_city(city)
            aq = get_air_quality(place["latitude"], place["longitude"])
            current = aq.get("current", {})
            units = aq.get("current_units", {})
            hourly = aq.get("hourly", {})

        st.subheader("1. CNN visual classification")
        if visual_label is not None:
            c1,c2 = st.columns(2)
            c1.metric("Predicted visual class", visual_label)
            c2.metric("Model confidence", f"{visual_conf:.1f}%")
            st.caption("Confidence is the CNN's score for its predicted image class. It is not a probability that the air has that measured AQI.")
        else:
            st.warning("Train the CNN before using image prediction. The live AQ section below is available now.")

        st.subheader("2. Live air-quality data — API estimate")
        st.write(f"**Location:** {place['name']}, {place.get('admin1','')}, {place.get('country','')}")
        aqi = current.get("us_aqi")
        if aqi is not None:
            st.metric("US AQI (model-based)", f"{aqi:.0f}")
        else:
            st.warning("US AQI value was not returned by the provider.")

        pollutant_rows = [
            ("PM2.5", "pm2_5", "µg/m³"),
            ("PM10", "pm10", "µg/m³"),
            ("Carbon monoxide (CO)", "carbon_monoxide", "µg/m³"),
            ("Nitrogen dioxide (NO₂)", "nitrogen_dioxide", "µg/m³"),
            ("Sulphur dioxide (SO₂)", "sulphur_dioxide", "µg/m³"),
            ("Ozone (O₃)", "ozone", "µg/m³")
        ]
        cols = st.columns(3)
        for i,(label,key,unit) in enumerate(pollutant_rows):
            value = current.get(key)
            with cols[i % 3]:
                st.metric(label, fmt(value, unit))

        st.caption("Provider timestamp: " + str(current.get("time", "not supplied")) +
                   " | Time zone: " + str(aq.get("timezone", "not supplied")))
        st.subheader("3. Recent hourly pollutant values")
        if hourly.get("time"):
            chart_data = pd.DataFrame({"time": hourly["time"]})
            for key,label in [("pm2_5","PM2.5"),("pm10","PM10")]:
                if key in hourly:
                    chart_data[label] = hourly[key]
            chart_data = chart_data.set_index("time")
            st.line_chart(chart_data)
        else:
            st.write("Hourly history is not available in this response.")

        st.warning(
            "Interpretation: Open-Meteo provides CAMS atmospheric-composition model data "
            "(grid resolution varies by domain), not a direct reading from a nearby CPCB "
            "monitoring station. The displayed US AQI is not India's CPCB National AQI. "
            "Do not use this educational dashboard for medical, emergency or official decisions."
        )
        st.markdown("[Check official India AQI on CPCB](https://app.cpcbccr.com/ccr/)")
        st.markdown("[Open-Meteo Air Quality API documentation](https://open-meteo.com/en/docs/air-quality-api)")

if __name__ == "__main__":
    main()
