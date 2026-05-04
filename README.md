# Precision Irrigation Scheduling

AI and Physics-Guided Smart Irrigation Decision Support for Precision Agriculture

## Project Identity

| Field | Details |
| --- | --- |
| Project ID | PCSE26-05 |
| Project Title | Precision Irrigation Scheduling |
| Institution | KIET Group of Institutions, Ghaziabad |
| Affiliated University | Dr. A.P.J. Abdul Kalam Technical University, Lucknow (AKTU) |
| Department | Computer Science & Engineering |
| Degree Programme | Bachelor of Technology (B.Tech) - Final Year Major Project |
| Academic Session | 2025-2026 |
| Submission Date | May 2026 |
| Supervisor | Dr. Himanshi Chaudhry |
| Team Members | Ashwani Baghel (2300290109002), Utkarsh Patel (2300290109015), Omendra Rajput (2300290109005), Vaibhav Singh (2300290109016) |

## Table of Contents

1. Executive Summary
2. Motivation and Problem Statement
3. Core Innovation: Physics-Guided Irrigation Intelligence
4. System Architecture
5. Key Results and Metrics
6. Crop-Aware Generalizability
7. Repository Structure
8. Quick Start
9. Dataset
10. Dependencies and Tech Stack
11. Responsible Deployment and Data Privacy
12. Research Paper and Documentation
13. Team and Acknowledgements
14. License and Citation

## 1. Executive Summary

Precision Irrigation Scheduling is an AI-assisted smart agriculture system designed to generate explainable irrigation recommendations using soil telemetry, weather forecast data, crop-specific water requirements, and a trained LSTM soil-moisture prediction model. The system converts sensor and weather data into a 5-day irrigation plan that can be understood by farmers, evaluators, and technical reviewers.

The project addresses a practical challenge in agriculture: irrigation is often performed using fixed schedules or manual judgement, which can lead to water wastage, crop stress, or delayed irrigation. Precision Irrigation Scheduling improves this process by combining machine learning with a physics-guided soil-water balance correction layer. This ensures that model predictions remain consistent with rainfall and evapotranspiration behavior.

The solution includes a Streamlit dashboard, Firebase Realtime Database integration, OpenWeather forecast support, trained Keras and TensorFlow Lite model artifacts, dataset fallback logic, and model diagnostics. The dashboard provides a main recommendation, current field metrics, crop-specific thresholds, a 5-day irrigation schedule, soil-moisture trend visualization, and an option to publish the final report to Firebase for mobile-app integration.

## 2. Motivation and Problem Statement

### 2.1 The Agricultural Gap

Efficient irrigation is one of the most important requirements for sustainable farming. Traditional irrigation practices often depend on fixed timing, visual inspection, or farmer experience. These approaches can be unreliable because soil moisture changes with temperature, humidity, rainfall, wind, solar radiation, and crop type.

Poor irrigation scheduling can cause:

- Excess water usage and increased farming cost.
- Crop stress due to under-irrigation.
- Waterlogging and drainage risk due to over-irrigation.
- Reduced yield quality in sensitive crops.
- Lack of real-time decision support for farmers.

### 2.2 The Prediction Gap

Machine-learning models can forecast soil moisture, but raw AI predictions may sometimes behave unrealistically. For example, soil moisture should not rise sharply on a dry day without rainfall. A purely data-driven model may still produce such inconsistencies if it is not constrained by physical field behavior.

| Problem | Practical Impact |
| --- | --- |
| Fixed irrigation schedule | Water may be applied even when the field is already wet. |
| Manual observation | Decisions depend heavily on experience and may miss hidden soil conditions. |
| Raw model prediction | AI output may not always follow real soil-water balance behavior. |
| Single-crop assumption | Different crops need different moisture thresholds. |
| Lack of dashboard evidence | Evaluators and farmers cannot easily verify why a decision was made. |

### 2.3 Proposed Solution

Precision Irrigation Scheduling solves these issues by combining:

- Live sensor telemetry through Firebase.
- OpenWeather current weather and 5-day forecast.
- LSTM-based soil-moisture prediction.
- Physics-guided correction using precipitation and evapotranspiration.
- Crop-specific irrigation thresholds.
- Streamlit dashboard visualization and diagnostics.
- Firebase publishing for mobile-app or cloud integration.

## 3. Core Innovation: Physics-Guided Irrigation Intelligence

The primary technical contribution of this project is a physics-guided irrigation forecasting pipeline. Instead of relying only on raw neural-network output, the system corrects predictions using a soil-water balance equation.

Core correction logic:

```text
next moisture = previous moisture + precipitation * 0.8 - evapotranspiration * 0.5
```

The system then blends the LSTM forecast with the physics-based projection. This helps the final output remain both data-driven and physically reasonable.

The pipeline performs the following steps:

1. Sensor Telemetry Intake - Reads soil moisture, temperature, and humidity from Firebase.
2. Weather Forecast Processing - Converts OpenWeather forecast into daily weather features.
3. LSTM Forecasting - Predicts future soil moisture using trained model inference.
4. Physics Correction - Applies rainfall and evapotranspiration based water-balance adjustment.
5. Crop Rule Overlay - Applies crop-specific irrigation and drainage thresholds.
6. Decision Generation - Produces "Irrigation required", "No action", or "Drainage alert".
7. Diagnostic Reporting - Shows RMSE, MAE, R2 score, residuals, and physical consistency.

## 4. System Architecture

| Layer | Component | Responsibility |
| --- | --- | --- |
| Layer 5 | Interface Layer | Streamlit dashboard, crop selection, visual reports, recommendation publishing |
| Layer 4 | Decision Layer | Irrigation rules, drainage alerts, crop thresholds, final action generation |
| Layer 3 | Physics Layer | Soil-water balance correction, evapotranspiration adjustment, residual tracking |
| Layer 2 | Prediction Layer | Keras LSTM model inference, scalers, TensorFlow Lite deployment artifact |
| Layer 1 | Data Layer | Firebase telemetry, OpenWeather forecast, 10-year historical dataset fallback |

```text
Sensor Data + Weather Forecast + Crop Profile
                  |
                  v
        LSTM Soil-Moisture Forecast
                  |
                  v
      Physics-Guided Water Balance Layer
                  |
                  v
     Crop-Specific Irrigation Decision
                  |
                  v
 Streamlit Dashboard + Firebase Prediction Output
```

Input: sensor values, crop profile, and weather forecast.

Output: 5-day irrigation recommendation, diagnostics, and Firebase-ready report.

Design invariant: every recommendation must be explainable through sensor values, weather conditions, model prediction, crop threshold, and physics-corrected soil-moisture behavior.

## 5. Key Results and Metrics

### 5.1 Model Performance Snapshot

Local validation was performed on the latest 120 samples from the main dataset.

| Metric | Raw LSTM | Physics-Corrected Output |
| --- | ---: | ---: |
| RMSE | 8.68 | 4.55 |
| MAE | 7.32 | 3.79 |
| R2 Score | Not primary | 0.952 |
| Physical Violation Rate | Not constrained | 0.0% |

The physics-guided correction reduced error and improved practical consistency. The 0.0% physical violation rate indicates that the corrected forecast did not produce unrealistic moisture rises during no-rain conditions in the validation snapshot.

### 5.2 Dashboard Decision Output

| Output Module | Description |
| --- | --- |
| Main Recommendation | Shows whether irrigation is required today or can be delayed. |
| Current Field Condition | Displays soil moisture, temperature, humidity, and irrigation-day count. |
| 5-Day Irrigation Plan | Shows daily forecast moisture, rainfall, ET, threshold, and action. |
| Soil Moisture Trend | Visualizes expected moisture movement across the forecast window. |
| Model Diagnostics | Compares raw LSTM output with physics-corrected output. |
| Firebase Report | Publishes recommendation output for mobile-app integration. |

### 5.3 Demonstration Scenarios

| Scenario | Input Condition | Expected System Decision |
| --- | --- | --- |
| Dry field | Soil moisture below crop threshold | Irrigation required |
| Normal field | Moisture within crop-safe range | No action |
| Wet field | Moisture above high-moisture threshold | Drainage alert |
| Rain expected | Forecast shows rainfall in next days | Irrigation may be delayed |
| High ET day | Heat and radiation increase water loss | Earlier irrigation warning |

## 6. Crop-Aware Generalizability

Precision Irrigation Scheduling is not limited to a single crop. It uses crop profiles so that thresholds and water demand can change based on the selected crop.

| Crop | Irrigation Threshold | Target Moisture | Crop Water Factor | Interpretation |
| --- | ---: | ---: | ---: | --- |
| General field | 40% | 75% | 1.00x | Baseline field profile |
| Wheat | 38% | 70% | 0.95x | Moderate water demand |
| Rice | 65% | 88% | 1.25x | High water demand |
| Maize | 45% | 74% | 1.05x | Medium-high demand |
| Tomato | 55% | 78% | 1.15x | Sensitive crop profile |
| Mustard | 34% | 65% | 0.82x | Lower water demand |
| Sugarcane | 52% | 82% | 1.20x | High water demand |

This makes the system adaptable for multiple agricultural use cases without retraining the model for every crop.

## 7. Repository Structure

```text
Final Year Project/
|
+-- README.md
+-- instruction_to_run.txt
+-- .env.example
+-- Certificate/
|   +-- 20260501_220601.PDF
+-- Literature/
|   +-- PRECISION IRRIGATION PROJECT - REFE.txt
|   +-- Research_Paper_FY26_05.pdf
+-- SmartFarmBackend/
    +-- app.py
    +-- smartfarm_core.py
    +-- decision_engine.py
    +-- dataset.py
    +-- converter.py
    +-- requirements.txt
    +-- REPORT_ALIGNMENT_NOTES.md
    +-- final_smart_model.h5
    +-- final_smart_model.tflite
    +-- ultra_realistic_dataset_10_years.csv
    +-- logical_dataset_10_years.csv
    +-- dummy_dataset_5_years.csv
```

## 8. Quick Start

```powershell
# 1. Open the project
cd "D:\Projects\Final Year Project"

# 2. Move to backend
cd SmartFarmBackend

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install dependencies if needed
pip install -r requirements.txt

# 5. Set required environment variables
$env:OPENWEATHER_API_KEY="your_openweather_api_key"
$env:FIREBASE_CREDENTIALS_PATH="predictive-sis-firebase-adminsdk-fbsvc-ad13aadf29.json"
$env:FIREBASE_DATABASE_URL="https://your-project-default-rtdb.asia-southeast1.firebasedatabase.app/"

# 6. Launch dashboard
streamlit run app.py --server.port 8501
```

The application will be available at:

```text
http://localhost:8501
```

Alternative command:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Console report:

```powershell
python decision_engine.py
```

See `instruction_to_run.txt` for the complete hardware, software, demo, and troubleshooting guide.

## 9. Dataset

The main dataset contains daily weather and soil-moisture records from 2015 to 2024.

| Dataset | Rows | Period | Primary Use |
| --- | ---: | --- | --- |
| `ultra_realistic_dataset_10_years.csv` | 3653 | 2015-2024 | Main runtime dataset |
| `logical_dataset_10_years.csv` | 3653 | 2015-2024 | Generated logical dataset |
| `dummy_dataset_5_years.csv` | 1827 | 2020-2024 | Demo and testing fallback |

Feature set:

- Maximum temperature
- Minimum temperature
- Average temperature
- Humidity
- Wind speed
- Solar radiation
- Evapotranspiration
- Precipitation
- Precipitation probability
- Precipitation cover
- Soil moisture

## 10. Dependencies and Tech Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| Programming Language | Python | Backend logic and model inference |
| Dashboard | Streamlit | Web interface and visualization |
| Machine Learning | TensorFlow, Keras | LSTM model loading and prediction |
| Data Processing | Pandas, NumPy | Dataset handling and numerical operations |
| Scaling and Metrics | Scikit-learn | MinMax scaling, RMSE, MAE, R2 |
| Cloud Database | Firebase Realtime Database | Sensor input and recommendation output |
| Cloud SDK | Firebase Admin SDK | Backend database access |
| Weather Data | OpenWeather API | Current weather and 5-day forecast |
| Deployment Artifact | TensorFlow Lite | Future mobile or edge inference |

Required Python packages are listed in:

```text
SmartFarmBackend/requirements.txt
```

## 11. Responsible Deployment and Data Privacy

This project is designed as an academic decision-support system. It does not directly control irrigation hardware in the current implementation. Final irrigation action should be verified by a human user, especially in real field conditions.

Responsible deployment principles:

- Firebase credentials and API keys must not be committed publicly.
- Sensor data should be protected through proper Firebase rules.
- Weather API keys should be stored in environment variables.
- The dashboard should be treated as a recommendation tool, not a fully autonomous controller.
- Any automatic pump-control extension should include fail-safe checks, manual override, and electrical safety review.

## 12. Research Paper and Documentation

The project is supported by research and reference material available in the `Literature/` folder.

| File | Description |
| --- | --- |
| `Literature/Research_Paper_FY26_05.pdf` | Project research paper document |
| `Literature/PRECISION IRRIGATION PROJECT - REFE.txt` | Reference and supporting notes |
| `SmartFarmBackend/REPORT_ALIGNMENT_NOTES.md` | Technical alignment notes for implementation boundaries |
| `instruction_to_run.txt` | Complete guide to run, explain, and demonstrate the project |

## 13. Team and Acknowledgements

Development Team:

| Name | Roll Number |
| --- | --- |
| Ashwani Baghel | 2300290109002 |
| Utkarsh Patel | 2300290109015 |
| Omendra Rajput | 2300290109005 |
| Vaibhav Singh | 2300290109016 |

Faculty Supervisor:

Dr. Himanshi Chaudhry

The team acknowledges the open-source communities behind Python, TensorFlow, Keras, Streamlit, Pandas, NumPy, Scikit-learn, Firebase Admin SDK, and OpenWeather API services that enabled the implementation of this project.

## 14. License and Citation

This project is submitted as a B.Tech Final Year Major Project for academic evaluation and non-commercial research use.

Suggested citation:

```bibtex
@techreport{2026precision_irrigation_scheduling,
  title       = {Precision Irrigation Scheduling},
  author      = {Ashwani Baghel, Utkarsh Patel, Omendra Rajput, Vaibhav Singh},
  year        = {2026},
  month       = {May},
  institution = {KIET Group of Institutions, Ghaziabad},
  type        = {B.Tech Final Year Major Project Report},
  note        = {Project ID: PCSE26-05}
}
```

Precision Irrigation Scheduling demonstrates how AI, IoT telemetry, weather forecasting, and physics-based reasoning can work together to support sustainable and explainable irrigation planning.
