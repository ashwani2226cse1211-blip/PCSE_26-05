from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import sqrt
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
FIREBASE_CREDENTIALS_PATH = os.environ.get(
    "FIREBASE_CREDENTIALS_PATH",
    str(BASE_DIR / "predictive-sis-firebase-adminsdk-fbsvc-ad13aadf29.json"),
)
FIREBASE_DATABASE_URL = os.environ.get(
    "FIREBASE_DATABASE_URL",
    "https://predictive-sis-default-rtdb.asia-southeast1.firebasedatabase.app/",
)
WEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
FARM_LATITUDE = "28.6692"
FARM_LONGITUDE = "77.4538"
AI_MODEL_PATH = BASE_DIR / "final_smart_model.h5"
DATASET_PATH = BASE_DIR / "ultra_realistic_dataset_10_years.csv"

FEATURE_COLUMNS = [
    "tempmax",
    "tempmin",
    "temp",
    "humidity",
    "windspeed",
    "solarradiation",
    "Evapoatranspiration",
    "precip",
    "precipprob",
    "precipcover",
    "soil_moisture",
]

TARGET_MOISTURE = 40.0
IRRIGATE_UP_TO_MOISTURE = 75.0
HIGH_MOISTURE_THRESHOLD = 85.0
PHYSICS_WEIGHT = 0.58


@dataclass
class RuntimeResources:
    model: Any
    dataset: pd.DataFrame
    main_scaler: MinMaxScaler
    moisture_scaler: MinMaxScaler


@dataclass(frozen=True)
class CropProfile:
    name: str
    irrigation_threshold: float
    irrigate_to: float
    high_moisture_threshold: float
    crop_coefficient: float
    note: str


CROP_PROFILES = {
    "General field": CropProfile(
        name="General field",
        irrigation_threshold=TARGET_MOISTURE,
        irrigate_to=IRRIGATE_UP_TO_MOISTURE,
        high_moisture_threshold=HIGH_MOISTURE_THRESHOLD,
        crop_coefficient=1.00,
        note="Baseline profile matching the original model behavior.",
    ),
    "Wheat": CropProfile(
        name="Wheat",
        irrigation_threshold=38.0,
        irrigate_to=70.0,
        high_moisture_threshold=82.0,
        crop_coefficient=0.95,
        note="Moderate water demand; irrigation can wait slightly longer than high-demand crops.",
    ),
    "Rice": CropProfile(
        name="Rice",
        irrigation_threshold=65.0,
        irrigate_to=88.0,
        high_moisture_threshold=96.0,
        crop_coefficient=1.25,
        note="High water demand profile; recommends irrigation earlier and tolerates wetter soil.",
    ),
    "Maize": CropProfile(
        name="Maize",
        irrigation_threshold=45.0,
        irrigate_to=74.0,
        high_moisture_threshold=84.0,
        crop_coefficient=1.05,
        note="Medium-high water demand profile for regular field moisture.",
    ),
    "Tomato": CropProfile(
        name="Tomato",
        irrigation_threshold=55.0,
        irrigate_to=78.0,
        high_moisture_threshold=86.0,
        crop_coefficient=1.15,
        note="Sensitive crop profile; avoids letting soil moisture fall too low.",
    ),
    "Mustard": CropProfile(
        name="Mustard",
        irrigation_threshold=34.0,
        irrigate_to=65.0,
        high_moisture_threshold=78.0,
        crop_coefficient=0.82,
        note="Lower water demand profile; irrigation is delayed compared with vegetables and rice.",
    ),
    "Sugarcane": CropProfile(
        name="Sugarcane",
        irrigation_threshold=52.0,
        irrigate_to=82.0,
        high_moisture_threshold=90.0,
        crop_coefficient=1.20,
        note="High water demand profile; keeps the soil moisture target higher.",
    ),
}


def get_crop_profile(crop_name: str | None = None) -> CropProfile:
    if crop_name and crop_name in CROP_PROFILES:
        return CROP_PROFILES[crop_name]
    return CROP_PROFILES["General field"]


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(max(lower, min(upper, value)))


def estimate_et(temp: float, humidity: float, solarradiation: float) -> float:
    return clamp((temp / 10.0) + (solarradiation / 50.0) - (humidity / 20.0), 0.0, 12.0)


def estimate_solar(avg_temp: float, humidity: float, precip: float) -> float:
    cloud_penalty = min(110.0, precip * 6.0 + max(0.0, humidity - 72.0) * 1.2)
    heat_boost = max(0.0, avg_temp - 22.0) * 3.0
    return clamp(215.0 + heat_boost - cloud_penalty, 60.0, 350.0)


def water_balance_projection(previous_moisture: float, precip: float, et: float, crop_coefficient: float = 1.0) -> float:
    crop_et = et * crop_coefficient
    return clamp(previous_moisture + (precip * 0.8) - (crop_et * 0.5))


def crop_stress_proxy(moisture: float, temp: float, humidity: float, et: float) -> float:
    moisture_score = clamp((moisture - 25.0) / 60.0, 0.0, 1.0)
    heat_penalty = clamp((temp - 35.0) / 15.0, 0.0, 0.35)
    dry_air_penalty = clamp((45.0 - humidity) / 45.0, 0.0, 0.25)
    et_penalty = clamp((et - 6.0) / 10.0, 0.0, 0.25)
    return clamp(moisture_score - heat_penalty - dry_air_penalty - et_penalty, 0.05, 0.95)


def attention_score(day: dict[str, float]) -> float:
    rain_signal = min(day["precip"] / 25.0, 1.0) * 0.45
    et_signal = min(day["et"] / 8.0, 1.0) * 0.35
    heat_signal = min(max(day["temp"] - 30.0, 0.0) / 12.0, 1.0) * 0.20
    return clamp(rain_signal + et_signal + heat_signal, 0.0, 1.0)


def load_runtime_resources(dataset_path: str = DATASET_PATH) -> RuntimeResources:
    model = load_model(AI_MODEL_PATH)
    dataset = pd.read_csv(dataset_path, index_col="datetime")
    dataset = dataset[FEATURE_COLUMNS].copy()

    main_scaler = MinMaxScaler(feature_range=(0, 1))
    main_scaler.fit(dataset)

    moisture_scaler = MinMaxScaler(feature_range=(0, 1))
    moisture_scaler.fit(dataset[["soil_moisture"]])

    return RuntimeResources(
        model=model,
        dataset=dataset,
        main_scaler=main_scaler,
        moisture_scaler=moisture_scaler,
    )


def fetch_weather_forecast() -> tuple[list[dict[str, float]], str]:
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": FARM_LATITUDE,
        "lon": FARM_LONGITUDE,
        "appid": WEATHER_API_KEY,
        "units": "metric",
    }
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    payload = response.json()
    if "list" not in payload:
        raise ValueError(payload.get("message", "Weather API response did not include forecast data."))
    city = payload.get("city") or {}
    location = ", ".join(filter(None, [city.get("name"), city.get("country")])) or "configured farm"
    return process_openweather_payload(payload), f"OpenWeather forecast for {location}"


def fetch_current_weather() -> tuple[dict[str, Any], str]:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": FARM_LATITUDE,
        "lon": FARM_LONGITUDE,
        "appid": WEATHER_API_KEY,
        "units": "metric",
    }
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    payload = response.json()
    if "main" not in payload:
        raise ValueError(payload.get("message", "Weather API response did not include current data."))

    main = payload["main"]
    weather = payload.get("weather") or []
    wind = payload.get("wind") or {}
    timezone_seconds = int(payload.get("timezone") or 0)
    local_timezone = timezone(timedelta(seconds=timezone_seconds))
    updated_at = ""
    if payload.get("dt"):
        updated_at = datetime.fromtimestamp(payload["dt"], local_timezone).strftime("%Y-%m-%d %H:%M")

    location = ", ".join(filter(None, [payload.get("name"), payload.get("sys", {}).get("country")]))
    if not location:
        location = "configured farm"

    current = {
        "temperature": float(main["temp"]),
        "feels_like": float(main.get("feels_like", main["temp"])),
        "humidity": float(main["humidity"]),
        "windspeed": float(wind.get("speed", 0.0)),
        "description": str(weather[0].get("description", "Current conditions")).title()
        if weather
        else "Current conditions",
        "updated_at": updated_at,
        "location": location,
    }
    return current, f"OpenWeather current weather for {location}"


def process_openweather_payload(payload: dict[str, Any]) -> list[dict[str, float]]:
    timezone_seconds = int((payload.get("city") or {}).get("timezone") or 0)
    local_timezone = timezone(timedelta(seconds=timezone_seconds))
    daily: dict[str, dict[str, Any]] = {}
    for item in payload.get("list", []):
        if item.get("dt"):
            date = datetime.fromtimestamp(item["dt"], local_timezone).strftime("%Y-%m-%d")
        else:
            date = item["dt_txt"].split(" ")[0]
        if date not in daily:
            daily[date] = {
                "temp": [],
                "humidity": [],
                "wind": [],
                "precip": 0.0,
                "pop": [],
                "slots": 0,
                "wet_slots": 0,
            }
        daily[date]["temp"].append(float(item["main"]["temp"]))
        daily[date]["humidity"].append(float(item["main"]["humidity"]))
        daily[date]["wind"].append(float(item["wind"]["speed"]))
        daily[date]["pop"].append(float(item.get("pop", 0.0)))
        daily[date]["slots"] += 1
        slot_precip = 0.0
        if "rain" in item and "3h" in item["rain"]:
            slot_precip += float(item["rain"]["3h"])
        if "snow" in item and "3h" in item["snow"]:
            slot_precip += float(item["snow"]["3h"])
        daily[date]["precip"] = float(daily[date]["precip"]) + slot_precip
        if slot_precip > 0:
            daily[date]["wet_slots"] += 1

    forecast = []
    for date in sorted(daily):
        values = daily[date]
        temps = values["temp"]
        humidity_values = values["humidity"]
        wind_values = values["wind"]
        pop_values = values["pop"]
        avg_temp = float(np.mean(temps))
        avg_humidity = float(np.mean(humidity_values))
        avg_wind = float(np.mean(wind_values))
        precip = float(values["precip"])
        precipprob = float(max(pop_values) * 100.0) if pop_values else (100.0 if precip > 0 else 0.0)
        precipcover = float(values["wet_slots"] / max(1, values["slots"]) * 100.0)
        solar = estimate_solar(avg_temp, avg_humidity, precip)
        et = estimate_et(avg_temp, avg_humidity, solar)
        forecast.append(
            {
                "date": date,
                "tempmax": float(max(temps)),
                "tempmin": float(min(temps)),
                "temp": avg_temp,
                "humidity": avg_humidity,
                "windspeed": avg_wind,
                "solarradiation": solar,
                "Evapoatranspiration": et,
                "precip": precip,
                "precipprob": precipprob,
                "precipcover": precipcover,
                "et": et,
                "slot_count": int(values["slots"]),
            }
        )
    return forecast[:5]


def fallback_forecast_from_dataset(dataset: pd.DataFrame, days: int = 5) -> tuple[list[dict[str, float]], str]:
    sample = dataset.tail(days).copy()
    today = pd.Timestamp.today().normalize()
    forecast = []
    for i, (_, row) in enumerate(sample.iterrows()):
        day = {column: float(row[column]) for column in FEATURE_COLUMNS}
        day["date"] = (today + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        day["et"] = float(day["Evapoatranspiration"])
        forecast.append(day)
    return forecast, "Historical climatology fallback"


def build_model_input(day_data: dict[str, float], moisture: float) -> pd.DataFrame:
    row = {column: float(day_data[column]) for column in FEATURE_COLUMNS if column != "soil_moisture"}
    row["soil_moisture"] = float(moisture)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_raw_moisture(resources: RuntimeResources, day_data: dict[str, float], previous_moisture: float) -> float:
    input_df = build_model_input(day_data, previous_moisture)
    scaled = resources.main_scaler.transform(input_df)
    sequence = np.repeat(scaled, 30, axis=0).reshape(1, 30, len(FEATURE_COLUMNS))
    scaled_prediction = resources.model.predict(sequence, verbose=0)
    prediction = resources.moisture_scaler.inverse_transform(scaled_prediction)[0][0]
    return clamp(float(prediction))


def recommendation_for_moisture(moisture: float, crop_profile: CropProfile | None = None) -> tuple[str, str]:
    profile = crop_profile or get_crop_profile()
    if moisture > profile.high_moisture_threshold:
        return "Drainage alert", "Soil moisture is above safe range. Keep drainage path open."
    if moisture < profile.irrigation_threshold:
        return "Irrigation required", f"Irrigate until the field reaches nearly {profile.irrigate_to:.0f}% moisture."
    return "No action", f"Moisture is inside the {profile.name} operational range."


def run_physics_guided_forecast(
    resources: RuntimeResources,
    current_moisture: float,
    processed_forecast: list[dict[str, float]],
    crop_profile: CropProfile | None = None,
) -> list[dict[str, Any]]:
    profile = crop_profile or get_crop_profile()
    previous = clamp(current_moisture)
    rows = []
    for day in processed_forecast:
        raw_prediction = predict_raw_moisture(resources, day, previous)
        crop_et = float(day["et"]) * profile.crop_coefficient
        balance_prediction = water_balance_projection(previous, day["precip"], day["et"], profile.crop_coefficient)
        residual = raw_prediction - balance_prediction
        corrected = clamp((1.0 - PHYSICS_WEIGHT) * raw_prediction + PHYSICS_WEIGHT * balance_prediction)
        violation = day["precip"] < 0.2 and corrected > previous + 2.0
        action, advice = recommendation_for_moisture(corrected, profile)
        stress = crop_stress_proxy(corrected, day["temp"], day["humidity"], crop_et)

        rows.append(
            {
                "date": day["date"],
                "crop": profile.name,
                "raw_moisture": raw_prediction,
                "physics_moisture": balance_prediction,
                "moisture": corrected,
                "residual": residual,
                "violation": violation,
                "precip": day["precip"],
                "weather_et": day["et"],
                "et": crop_et,
                "temp": day["temp"],
                "humidity": day["humidity"],
                "windspeed": day["windspeed"],
                "solarradiation": day["solarradiation"],
                "stress_proxy": stress,
                "attention": attention_score({**day, "et": crop_et}),
                "action": action,
                "advice": advice,
                "irrigation_threshold": profile.irrigation_threshold,
                "irrigate_to": profile.irrigate_to,
                "high_moisture_threshold": profile.high_moisture_threshold,
                "crop_coefficient": profile.crop_coefficient,
            }
        )
        previous = corrected
    return rows


def summarize_forecast(rows: list[dict[str, Any]]) -> dict[str, Any]:
    irrigation_days = [row for row in rows if row["action"] == "Irrigation required"]
    drainage_days = [row for row in rows if row["action"] == "Drainage alert"]
    violations = [row for row in rows if row["violation"]]
    min_moisture = min(row["moisture"] for row in rows) if rows else 0.0
    max_residual = max(abs(row["residual"]) for row in rows) if rows else 0.0
    consistency = clamp(100.0 - (len(violations) * 20.0) - (max_residual * 1.2), 0.0, 100.0)
    return {
        "irrigation_days": len(irrigation_days),
        "drainage_days": len(drainage_days),
        "min_moisture": min_moisture,
        "max_residual": max_residual,
        "physics_consistency": consistency,
        "critical_day": min(rows, key=lambda item: item["moisture"])["date"] if rows else None,
        "violations": len(violations),
        "crop": rows[0].get("crop", "General field") if rows else "General field",
        "irrigation_threshold": rows[0].get("irrigation_threshold", TARGET_MOISTURE) if rows else TARGET_MOISTURE,
    }


def build_markdown_report(current_moisture: float, rows: list[dict[str, Any]], source: str) -> str:
    summary = summarize_forecast(rows)
    lines = [
        "Precision Irrigation Decision Report",
        "",
        f"Crop profile: {summary['crop']}",
        f"Current soil moisture: {current_moisture:.1f}%",
        f"Irrigation threshold: {summary['irrigation_threshold']:.1f}%",
        f"Forecast source: {source}",
        f"Physics consistency score: {summary['physics_consistency']:.1f}%",
        "Crop method: rule-based crop water-demand overlay on top of the trained soil-moisture model.",
        "",
        "Five-day recommendation:",
    ]

    for row in rows:
        label = datetime.strptime(row["date"], "%Y-%m-%d").strftime("%A, %b %d")
        lines.extend(
            [
                "",
                f"{label}",
                f"- Rain forecast: {row['precip']:.1f} mm",
                f"- Weather ET estimate: {row['weather_et']:.2f}",
                f"- Crop-adjusted ET estimate: {row['et']:.2f}",
                f"- LSTM raw moisture: {row['raw_moisture']:.1f}%",
                f"- Physics-corrected moisture: {row['moisture']:.1f}%",
                f"- Residual: {row['residual']:.2f}",
                f"- Recommendation: {row['action']} - {row['advice']}",
            ]
        )

    lines.extend(["", "Summary:"])
    if summary["irrigation_days"]:
        lines.append(
            f"Irrigation planning is required on {summary['irrigation_days']} day(s), "
            f"especially around {summary['critical_day']}."
        )
    elif summary["drainage_days"]:
        lines.append("No irrigation is required. Monitor drainage because moisture may stay high.")
    else:
        lines.append("No immediate irrigation is required. Continue monitoring the next forecast update.")
    return "\n".join(lines)


def evaluate_model(resources: RuntimeResources, max_samples: int = 120) -> dict[str, float]:
    data = resources.dataset.tail(max_samples + 31).reset_index(drop=True)
    y_true: list[float] = []
    sequences = []
    metadata = []

    for index in range(30, len(data) - 1):
        history = data.loc[index - 29 : index, FEATURE_COLUMNS]
        sequences.append(resources.main_scaler.transform(history))
        metadata.append((index, data.loc[index + 1, FEATURE_COLUMNS]))

    if not sequences:
        return {
            "samples": 0.0,
            "raw_rmse": 0.0,
            "physics_rmse": 0.0,
            "raw_mae": 0.0,
            "physics_mae": 0.0,
            "r2": 0.0,
            "violation_rate": 0.0,
        }

    batch = np.array(sequences).reshape(len(sequences), 30, len(FEATURE_COLUMNS))
    scaled_predictions = resources.model.predict(batch, verbose=0)
    raw_predictions = resources.moisture_scaler.inverse_transform(scaled_predictions).reshape(-1)

    raw_pred: list[float] = []
    corrected_pred: list[float] = []
    violations = 0

    for raw, (index, next_row) in zip(raw_predictions, metadata):
        previous = float(data.loc[index, "soil_moisture"])
        balance = water_balance_projection(
            previous,
            float(next_row["precip"]),
            float(next_row["Evapoatranspiration"]),
        )
        corrected = clamp((1.0 - PHYSICS_WEIGHT) * raw + PHYSICS_WEIGHT * balance)
        actual = float(next_row["soil_moisture"])

        if float(next_row["precip"]) < 0.2 and corrected > previous + 2.0:
            violations += 1

        y_true.append(actual)
        raw_pred.append(clamp(raw))
        corrected_pred.append(corrected)

    raw_rmse = sqrt(float(np.mean((np.array(y_true) - np.array(raw_pred)) ** 2)))
    corrected_rmse = sqrt(float(np.mean((np.array(y_true) - np.array(corrected_pred)) ** 2)))
    return {
        "samples": float(len(y_true)),
        "raw_rmse": raw_rmse,
        "physics_rmse": corrected_rmse,
        "raw_mae": float(mean_absolute_error(y_true, raw_pred)),
        "physics_mae": float(mean_absolute_error(y_true, corrected_pred)),
        "r2": float(r2_score(y_true, corrected_pred)),
        "violation_rate": float(violations / max(1, len(y_true)) * 100.0),
    }
