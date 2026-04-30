# Report Alignment Notes

This implementation uses only the assets available in this folder:

- `final_smart_model.h5`
- `final_smart_model.tflite`
- bundled CSV datasets
- Firebase Realtime Database credentials
- OpenWeather forecast API

## What Was Added To Move Closer To The Report

- Shared inference core in `smartfarm_core.py`.
- Physics-guided post-processing layer using a soil-water balance correction:
  `moisture_next = moisture_previous + precipitation * 0.8 - ET * 0.5`.
- Physics residual tracking between raw LSTM prediction and water-balance prediction.
- Physical violation check for impossible moisture rise during no-rain days.
- Temporal importance score for rainfall, ET, and heat stress events.
- Vegetation/crop stress proxy derived from available moisture and weather features.
- Local validation snapshot with RMSE, MAE, R2, and violation rate.
- Professional Streamlit dashboard with Forecast, Physics Layer, Model Evidence, and Firebase Report tabs.
- `decision_engine.py` now saves a richer physics-guided report, summary, forecast, and evaluation metrics to Firebase.

## Honest Technical Boundary

The current folder does not contain Sentinel-2 imagery, NDVI rasters, PyTorch training code, ConvLSTM, BiLSTM, attention layers, or a true physics-guided training loss. Because of that, the system should be presented as a practical, report-inspired implementation:

> A smart irrigation decision-support system using live IoT telemetry, weather forecast, a trained LSTM model, and a physics-guided soil-water balance correction layer.

## Demo Talking Points

- The model gives raw soil-moisture predictions.
- The physics layer constrains predictions using precipitation and evapotranspiration.
- The dashboard shows the residual between model prediction and physical balance.
- The violation rate indicates whether predictions break simple mass-conservation logic.
- The TFLite artifact supports the future-scope claim of edge deployment.
