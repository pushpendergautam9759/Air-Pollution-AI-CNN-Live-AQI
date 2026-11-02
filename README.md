# Air Pollution AI — CNN + Live AQI Integration

## What this project does
1. User uploads an outdoor image.
2. A trained CNN predicts one of six visual pollution classes and returns its model confidence.
3. User enters a city.
4. The app geocodes the city and calls the Open-Meteo Air Quality API.
5. The dashboard displays model-based US AQI and pollutant concentrations (PM2.5, PM10, CO, NO2, SO2, O3), with recent hourly PM trends.

## Critical distinction
- CNN output is an image-based visual class prediction.
- Open-Meteo values are CAMS model-based atmospheric composition estimates, not direct CPCB station measurements.
- US AQI is not the Indian CPCB National AQI.
- A photograph cannot measure pollutant concentrations.
Use CPCB's official portal for official Indian AQI and never use this educational prototype for health/emergency decisions.

## Setup
1. Install Python 3.10 or 3.11.
2. Download/extract the image dataset. Follow dataset/DATASET_INFO.txt.
3. Install dependencies: `pip install -r requirements.txt`
4. Check DATASET_DIR at the top of train.py.
5. Train CNN: `python train.py`
6. Launch dashboard: `streamlit run app.py`
7. Optional CLI image prediction: `python predict.py --image sample.jpg`

The first training run requires the dataset files locally. The app needs internet access for city search and the live air-quality API.
