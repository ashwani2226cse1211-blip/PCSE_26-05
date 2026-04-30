# Smart Farm Irrigation Advisor

Final year project with a Streamlit backend, trained soil-moisture model, OpenWeather forecast integration, Firebase telemetry, and Android app source.

## Backend

```powershell
cd SmartFarmBackend
.\venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Required local environment:

- `OPENWEATHER_API_KEY`
- `FIREBASE_CREDENTIALS_PATH` pointing to your Firebase service-account JSON
- `FIREBASE_DATABASE_URL`

Firebase service-account JSON files, Android `google-services.json`, virtual environments, build outputs, and archives are intentionally not committed.
