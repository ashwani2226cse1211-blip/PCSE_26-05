import pandas as pd
import numpy as np
from datetime import timedelta

print("10 saal ka logical dataset banaya jaa raha hai...")

# --- Configuration ---
start_date = "2015-01-01"
end_date = "2024-12-31"
file_name = "logical_dataset_10_years.csv"

# --- 1. Date Range Banayein ---
dates = pd.date_range(start=start_date, end=end_date, freq='D')
n_days = len(dates)
df = pd.DataFrame({'datetime': dates})

# --- 2. Logical Mausam ke Patterns Banayein ---
day_of_year = df['datetime'].dt.dayofyear

# Base Temperature ka pattern (Seasonal)
temp_amplitude = 12
temp_baseline = 26
df['temp'] = temp_baseline + temp_amplitude * np.sin((day_of_year - 150) * 2 * np.pi / 365) + np.random.uniform(-2, 2, n_days)

# Baarish ka pattern (Monsoon mein zyada chance)
monsoon_mask = (df['datetime'].dt.month >= 6) & (df['datetime'].dt.month <= 9)
precip_prob = np.where(monsoon_mask, 0.4, 0.05)
is_raining = np.random.rand(n_days) < precip_prob
df['precip'] = np.where(is_raining, np.random.uniform(1, 40, n_days), 0)

# --- 3. Ek-Doosre se Jude Hue Features Banayein ---
# Ab hum features ko baarish aur dhoop ke hisaab se adjust karenge

# Humidity: Jab baarish hogi toh zyada, jab dhoop hogi toh kam
base_humidity = 55 + 30 * np.sin((day_of_year - 210) * 2 * np.pi / 365)
df['humidity'] = np.where(df['precip'] > 0, base_humidity + 15, base_humidity - 10) + np.random.uniform(-5, 5, n_days)
df['humidity'] = np.clip(df['humidity'], 20, 98)

# Solar Radiation: Jab baarish hogi toh kam, jab dhoop hogi toh zyada
base_solar = 200 + 100 * np.sin((day_of_year - 140) * 2 * np.pi / 365)
df['solarradiation'] = np.where(df['precip'] > 0, base_solar / 2, base_solar * 1.2) + np.random.uniform(-30, 30, n_days)
df['solarradiation'] = np.clip(df['solarradiation'], 50, 350)

# Windspeed, TempMax, TempMin
df['windspeed'] = np.clip(15 + 10 * np.sin((day_of_year - 120) * 2 * np.pi / 365) + np.random.uniform(-5, 5, n_days), 2, 40)
df['tempmax'] = df['temp'] + np.random.uniform(3, 6, n_days)
df['tempmin'] = df['temp'] - np.random.uniform(3, 6, n_days)

# --- 4. Logical Evapotranspiration aur Soil Moisture Banayein ---
# Ab hum in a saaf features se ET aur Soil Moisture banayenge
df['Evapoatranspiration'] = np.clip((df['temp'] / 10) + (df['solarradiation'] / 50) - (df['humidity'] / 20), 0.5, 10)

soil_moisture = np.zeros(n_days)
soil_moisture[0] = 60
for i in range(1, n_days):
    # Paani ka balance: (Pichhli nami + Baarish) - (Bhaap bankar udna)
    next_day_moisture = soil_moisture[i-1] + (df.loc[i, 'precip'] * 0.8) - (df.loc[i, 'Evapoatranspiration'] * 0.5)
    soil_moisture[i] = np.clip(next_day_moisture, 20, 95) # 20% se kam nahi, 95% se zyada nahi

df['soil_moisture'] = soil_moisture

# Baaki columns ko fill karein
df['precipprob'] = np.where(df['precip'] > 0, 100, 0)
df['precipcover'] = np.where(df['precip'] > 0, np.random.uniform(4, 20, n_days), 0)

# Final columns set karein
final_columns = ['datetime', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed', 
                 'solarradiation', 'Evapoatranspiration', 'precip', 'precipprob', 
                 'precipcover', 'soil_moisture']
df = df[final_columns]
df.set_index('datetime', inplace=True)

df.to_csv(file_name)

print(f"✅ Naya, logical dataset '{file_name}' naam se safaltapoorvak ban gaya hai.")
print("Aap is dataset par apne model ko dobara train kar sakte hain.")