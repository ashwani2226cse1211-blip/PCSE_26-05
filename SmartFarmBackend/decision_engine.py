import json
import os

import firebase_admin
from firebase_admin import credentials, db

from smartfarm_core import (
    DATASET_PATH,
    FIREBASE_CREDENTIALS_PATH,
    FIREBASE_DATABASE_URL,
    build_markdown_report,
    evaluate_model,
    fetch_current_weather,
    fallback_forecast_from_dataset,
    fetch_weather_forecast,
    get_crop_profile,
    load_runtime_resources,
    run_physics_guided_forecast,
    summarize_forecast,
)


def initialize_firebase():
    if not firebase_admin._apps:
        service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
        if service_account_json:
            cred = credentials.Certificate(json.loads(service_account_json))
        else:
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})


def read_current_sensor_data(resources):
    try:
        initialize_firebase()
        sensor_data = db.reference("sensorData").get() or {}
        return {
            "soil_moisture": float(sensor_data.get("soilPercent", 50.0)),
            "temperature": float(sensor_data.get("temperature", 25.0)),
            "humidity": float(sensor_data.get("humidity", 60.0)),
            "source": "Firebase live telemetry",
        }
    except Exception as error:
        latest = resources.dataset.iloc[-1]
        return {
            "soil_moisture": float(latest["soil_moisture"]),
            "temperature": float(latest["temp"]),
            "humidity": float(latest["humidity"]),
            "source": f"Dataset fallback: {error}",
        }


def read_forecast(resources):
    try:
        return fetch_weather_forecast()
    except Exception as error:
        forecast, source = fallback_forecast_from_dataset(resources.dataset)
        return forecast, f"{source}; weather API unavailable: {error}"


def read_current_weather(sensor):
    try:
        return fetch_current_weather()
    except Exception as error:
        weather = {
            "temperature": sensor["temperature"],
            "feels_like": sensor["temperature"],
            "humidity": sensor["humidity"],
            "windspeed": 0.0,
            "description": "Sensor fallback",
            "updated_at": "",
            "location": "field telemetry",
        }
        return weather, f"Sensor fallback; current weather unavailable: {error}"


def run_final_reporting_engine():
    crop_profile = get_crop_profile("Wheat")

    print("Loading trained model, dataset, and scalers...")
    resources = load_runtime_resources(DATASET_PATH)

    print("Reading sensor telemetry...")
    sensor = read_current_sensor_data(resources)

    print("Reading current weather...")
    current_weather, current_weather_source = read_current_weather(sensor)

    print("Preparing meteorological forecast...")
    forecast, forecast_source = read_forecast(resources)

    print("Running LSTM forecast with physics-guided correction...")
    rows = run_physics_guided_forecast(resources, sensor["soil_moisture"], forecast, crop_profile)
    summary = summarize_forecast(rows)
    report = build_markdown_report(sensor["soil_moisture"], rows, forecast_source)

    print("\n" + report)

    print("\nComputing local validation snapshot...")
    metrics = evaluate_model(resources, max_samples=120)
    print(
        "Evaluation: "
        f"raw RMSE={metrics['raw_rmse']:.2f}, "
        f"physics RMSE={metrics['physics_rmse']:.2f}, "
        f"R2={metrics['r2']:.2f}, "
        f"violation rate={metrics['violation_rate']:.1f}%"
    )

    try:
        initialize_firebase()
        db.reference("prediction_output").set(
            {
                "recommendation": report,
                "summary": summary,
                "crop_profile": {
                    "name": crop_profile.name,
                    "irrigation_threshold": crop_profile.irrigation_threshold,
                    "irrigate_to": crop_profile.irrigate_to,
                    "high_moisture_threshold": crop_profile.high_moisture_threshold,
                    "crop_coefficient": crop_profile.crop_coefficient,
                    "note": crop_profile.note,
                },
                "forecast": rows,
                "evaluation": metrics,
                "sensor_source": sensor["source"],
                "current_weather": current_weather,
                "current_weather_source": current_weather_source,
                "forecast_source": forecast_source,
            }
        )
        print("\nPhysics-guided recommendation saved to Firebase.")
    except Exception as error:
        print(f"\nFirebase save failed: {error}")


if __name__ == "__main__":
    run_final_reporting_engine()
