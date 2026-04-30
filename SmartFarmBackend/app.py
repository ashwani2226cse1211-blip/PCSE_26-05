from html import escape

import firebase_admin
import pandas as pd
import streamlit as st
from firebase_admin import credentials, db

from smartfarm_core import (
    CROP_PROFILES,
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


st.set_page_config(page_title="Smart Irrigation Advisor", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #f4f7ef;
        --panel: #ffffff;
        --ink: #172217;
        --muted: #5d6b5d;
        --line: #dce5d9;
        --green: #2f7d32;
        --green-soft: #e7f4e4;
        --amber: #9a6a00;
        --amber-soft: #fff3ce;
        --red: #a13c2f;
        --red-soft: #fde7e2;
        --blue: #236a8f;
        --blue-soft: #e3f1f7;
    }
    .stApp {
        background: var(--bg);
        color: var(--ink);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }
    header[data-testid="stHeader"] {
        background: rgba(244, 247, 239, 0.9);
    }
    h1, h2, h3, p, label, span {
        letter-spacing: 0;
    }
    .hero {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 26px 24px 22px;
        margin-top: 0.35rem;
        margin-bottom: 18px;
        overflow: visible;
    }
    .eyebrow {
        color: var(--green);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-title {
        color: var(--ink);
        font-size: clamp(1.9rem, 3.8vw, 2.9rem);
        line-height: 1.16;
        font-weight: 850;
        margin: 0 0 10px;
        padding-top: 2px;
    }
    .hero-copy {
        color: var(--muted);
        font-size: 1rem;
        max-width: 850px;
        line-height: 1.5;
    }
    .decision {
        border-radius: 8px;
        padding: 18px 20px;
        border: 2px solid var(--green);
        background: var(--green-soft);
        margin-bottom: 18px;
    }
    .decision.warn {
        border-color: #d59b1e;
        background: var(--amber-soft);
    }
    .decision.risk {
        border-color: #d66b55;
        background: var(--red-soft);
    }
    .decision-kicker {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .decision-title {
        color: var(--ink);
        font-size: clamp(1.6rem, 3.3vw, 2.4rem);
        font-weight: 850;
        line-height: 1.08;
        margin-bottom: 8px;
    }
    .decision-copy {
        color: var(--ink);
        font-size: 1rem;
        line-height: 1.5;
    }
    .metric-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px;
        min-height: 112px;
        margin-bottom: 14px;
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.86rem;
        margin-bottom: 8px;
    }
    .metric-value {
        color: var(--ink);
        font-size: clamp(1.55rem, 3vw, 2.15rem);
        font-weight: 850;
        line-height: 1.05;
    }
    .metric-note {
        color: var(--muted);
        font-size: 0.8rem;
        margin-top: 8px;
        line-height: 1.35;
    }
    .day-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 15px;
        min-height: 154px;
        margin-bottom: 14px;
    }
    .day-card.good { border-top: 5px solid var(--green); }
    .day-card.warn { border-top: 5px solid #d59b1e; }
    .day-card.risk { border-top: 5px solid #d66b55; }
    .day-date {
        color: var(--muted);
        font-size: 0.86rem;
        margin-bottom: 7px;
    }
    .day-action {
        color: var(--ink);
        font-size: 1.28rem;
        font-weight: 850;
        line-height: 1.15;
        margin-bottom: 8px;
    }
    .day-line {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.45;
    }
    .simple-note {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.45;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--green);
        background: var(--green);
        color: #ffffff;
        font-weight: 750;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_resources():
    return load_runtime_resources(DATASET_PATH)


@st.cache_data(ttl=900, show_spinner=False)
def get_evaluation_snapshot():
    resources = get_resources()
    return evaluate_model(resources, max_samples=120)


@st.cache_data(ttl=300, show_spinner=False)
def get_current_weather_snapshot():
    return fetch_current_weather()


@st.cache_data(ttl=900, show_spinner=False)
def get_forecast_snapshot():
    return fetch_weather_forecast()


def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})


def read_sensor_snapshot(resources):
    try:
        initialize_firebase()
        sensor_data = db.reference("sensorData").get() or {}
        return {
            "soil_moisture": float(sensor_data.get("soilPercent", 50.0)),
            "temperature": float(sensor_data.get("temperature", 25.0)),
            "humidity": float(sensor_data.get("humidity", 60.0)),
            "source": "Live sensor data",
        }
    except Exception as error:
        latest = resources.dataset.iloc[-1]
        return {
            "soil_moisture": float(latest["soil_moisture"]),
            "temperature": float(latest["temp"]),
            "humidity": float(latest["humidity"]),
            "source": f"Demo dataset fallback, live cloud data unavailable: {error}",
        }


def load_forecast(resources):
    try:
        return get_forecast_snapshot()
    except Exception as error:
        forecast, source = fallback_forecast_from_dataset(resources.dataset)
        return forecast, f"{source}; weather unavailable: {error}"


def load_current_weather(sensor):
    try:
        return get_current_weather_snapshot()
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


def farmer_action(action):
    if action == "Irrigation required":
        return "Irrigation needed", "warn"
    if action == "Drainage alert":
        return "Check drainage", "risk"
    return "No irrigation", "good"


def make_main_decision(rows, current_moisture, crop_profile):
    if current_moisture < crop_profile.irrigation_threshold:
        return (
            "Irrigation is needed today",
            f"Soil moisture is below the {crop_profile.name} threshold of {crop_profile.irrigation_threshold:.0f}%. "
            f"Run irrigation until the field reaches around {crop_profile.irrigate_to:.0f}% moisture.",
            "warn",
        )

    first_irrigation = next((row for row in rows if row["action"] == "Irrigation required"), None)
    if first_irrigation:
        label = pd.to_datetime(first_irrigation["date"]).strftime("%b %d")
        return (
            f"Irrigation will likely be needed around {label}",
            "The field is manageable right now, but the forecast shows soil moisture dropping soon.",
            "warn",
        )

    first_drainage = next((row for row in rows if row["action"] == "Drainage alert"), None)
    if first_drainage:
        label = pd.to_datetime(first_drainage["date"]).strftime("%b %d")
        return (
            f"Waterlogging risk around {label}",
            "Do not irrigate. Keep drainage channels clear and monitor the field.",
            "risk",
        )

    return (
        "No irrigation is needed in the next 5 days",
        f"Soil moisture is in the safe range for {crop_profile.name}. Keep monitoring sensor and weather updates.",
        "good",
    )


def metric_card(column, label, value, note):
    column.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(str(value))}</div>
            <div class="metric-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def day_card(column, row):
    action_label, tone = farmer_action(row["action"])
    date_label = pd.to_datetime(row["date"]).strftime("%d %b")
    column.markdown(
        f"""
        <div class="day-card {tone}">
            <div class="day-date">{escape(date_label)}</div>
            <div class="day-action">{escape(action_label)}</div>
            <div class="day-line">Expected moisture: <b>{row['moisture']:.1f}%</b></div>
            <div class="day-line">Rain chance amount: <b>{row['precip']:.1f} mm</b></div>
            <div class="day-line">Crop ET: <b>{row['et']:.1f}</b> | Threshold: <b>{row['irrigation_threshold']:.0f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def schedule_dataframe(rows):
    table_rows = []
    for row in rows:
        action_label, _ = farmer_action(row["action"])
        table_rows.append(
            {
                "Day": pd.to_datetime(row["date"]).strftime("%d %b"),
                "Decision": action_label,
                "Expected moisture": f"{row['moisture']:.1f}%",
                "Rain forecast": f"{row['precip']:.1f} mm",
                "Crop ET": f"{row['et']:.2f}",
                "Threshold": f"{row['irrigation_threshold']:.0f}%",
                "Reason": "Below crop threshold" if row["action"] == "Irrigation required" else "Moisture is in crop-safe range",
            }
        )
    return pd.DataFrame(table_rows)


def technical_dataframe(rows):
    table = pd.DataFrame(rows)
    table["date"] = pd.to_datetime(table["date"]).dt.strftime("%d %b")
    display = table[
        [
            "date",
            "raw_moisture",
            "physics_moisture",
            "moisture",
            "residual",
            "weather_et",
            "et",
            "attention",
            "stress_proxy",
        ]
    ].copy()
    display.columns = [
        "Date",
        "LSTM Raw %",
        "Water Balance %",
        "Final %",
        "Residual",
        "Weather ET",
        "Crop ET",
        "Temporal Weight",
        "Crop Stress Proxy",
    ]
    return display


def main():
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Smart Farm Irrigation Advisor</div>
            <div class="hero-title">Should the field be irrigated today?</div>
            <div class="hero-copy">
                This dashboard combines soil sensor readings, weather forecast, and the trained AI model
                to give a clear irrigation decision for the field.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_crop = st.sidebar.selectbox("Crop profile", list(CROP_PROFILES.keys()), index=1)
    crop_profile = get_crop_profile(selected_crop)
    st.sidebar.metric("Irrigation threshold", f"{crop_profile.irrigation_threshold:.0f}%")
    st.sidebar.metric("Crop water factor", f"{crop_profile.crop_coefficient:.2f}x")
    st.sidebar.caption(crop_profile.note)

    with st.spinner("Preparing recommendation from sensor, weather, and AI model..."):
        resources = get_resources()
        sensor = read_sensor_snapshot(resources)
        current_weather, current_weather_source = load_current_weather(sensor)
        forecast, forecast_source = load_forecast(resources)
        rows = run_physics_guided_forecast(resources, sensor["soil_moisture"], forecast, crop_profile)
        summary = summarize_forecast(rows)

    decision_title, decision_copy, decision_tone = make_main_decision(rows, sensor["soil_moisture"], crop_profile)
    st.markdown(
        f"""
        <div class="decision {decision_tone}">
            <div class="decision-kicker">Main Recommendation</div>
            <div class="decision-title">{escape(decision_title)}</div>
            <div class="decision-copy">{escape(decision_copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Current Field Condition")
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Soil moisture", f"{sensor['soil_moisture']:.1f}%", f"Irrigation below {crop_profile.irrigation_threshold:.0f}% for {crop_profile.name}")
    metric_card(c2, "Current temp", f"{current_weather['temperature']:.1f} C", current_weather["description"])
    metric_card(c3, "Current humidity", f"{current_weather['humidity']:.0f}%", current_weather["location"])
    metric_card(c4, "Irrigation days", int(summary["irrigation_days"]), f"{crop_profile.name} threshold: {crop_profile.irrigation_threshold:.0f}%")

    st.markdown(f"### 5-Day Irrigation Plan for {crop_profile.name}")
    day_columns = st.columns(5)
    for column, row in zip(day_columns, rows):
        day_card(column, row)

    st.dataframe(schedule_dataframe(rows), use_container_width=True, hide_index=True)

    st.markdown("### Soil Moisture Trend")
    moisture_chart = pd.DataFrame(rows)
    moisture_chart["date"] = pd.to_datetime(moisture_chart["date"])
    chart_data = moisture_chart.set_index("date")[["moisture"]].rename(
        columns={"moisture": "Expected soil moisture"}
    )
    chart_data["Irrigation threshold"] = 40.0
    st.line_chart(chart_data, height=300)
    st.markdown(
        "<div class='simple-note'>When forecast moisture goes below 40%, the system recommends irrigation.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Model Diagnostics"):
        st.write("Data sources")
        st.write(f"- Crop profile: {crop_profile.name}")
        st.write(f"- Crop rule layer: threshold {crop_profile.irrigation_threshold:.0f}%, crop water factor {crop_profile.crop_coefficient:.2f}x")
        st.write(f"- Sensor source: {sensor['source']}")
        st.write(f"- Current weather source: {current_weather_source}")
        if current_weather["updated_at"]:
            st.write(f"- Current weather updated at: {current_weather['updated_at']}")
        st.write(f"- Weather source: {forecast_source}")
        st.write("- AI model: trained Keras LSTM model")
        st.write("- Physics layer: soil-water balance correction")

        metrics = get_evaluation_snapshot()
        e1, e2, e3, e4 = st.columns(4)
        metric_card(e1, "Raw RMSE", f"{metrics['raw_rmse']:.2f}", "Original LSTM")
        metric_card(e2, "Physics RMSE", f"{metrics['physics_rmse']:.2f}", "Correction ke baad")
        metric_card(e3, "R2 Score", f"{metrics['r2']:.2f}", "Model fit")
        metric_card(e4, "Violation Rate", f"{metrics['violation_rate']:.1f}%", "Physical consistency")

        st.dataframe(
            technical_dataframe(rows).style.format(
                {
                    "LSTM Raw %": "{:.1f}",
                    "Water Balance %": "{:.1f}",
                    "Final %": "{:.1f}",
                    "Residual": "{:.2f}",
                    "ET": "{:.2f}",
                    "Temporal Weight": "{:.2f}",
                    "Crop Stress Proxy": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Publish Recommendation"):
        report = build_markdown_report(sensor["soil_moisture"], rows, forecast_source)
        st.code(report, language="markdown")
        if st.button("Publish to Mobile App"):
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
                        "current_weather": current_weather,
                        "forecast": rows,
                        "last_updated": pd.Timestamp.now().isoformat(),
                    }
                )
                st.success("Recommendation published for the mobile app.")
            except Exception as error:
                st.error(f"Publish failed: {error}")


if __name__ == "__main__":
    main()
