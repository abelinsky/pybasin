import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# === 1. загрузка данных ===
df = pd.read_csv(
    "./model_output/without_anomalies/thermal_history_datafiles/model_data_comparison_VR_KDK-01_1-2-2026_ms0.csv"
)

# !!! проверь имена колонок !!!
# Обычно это что-то вроде:
# 'VR', 'sumF_model'
print(df.columns)

# === 2. фильтрация адекватных значений ===
df = df[(df["VR"] > 0.15) & (df["VR"] < 1.5)].copy()

x = df["sumF_model"].values
y = np.log(df["VR"].values)

# === 3. линейная регрессия ln(Ro) = a + b*sumF ===
A = np.vstack([np.ones_like(x), x]).T
a, b = np.linalg.lstsq(A, y, rcond=None)[0]

print("==== OPTIMAL Ro CALIBRATION ====")
print(f"ln(Ro) = {a:.3f} + {b:.3f} * sumF")
print(f"Ro = exp({a:.3f} + {b:.3f} * sumF)")

# === 4. оценка качества ===
Ro_pred = np.exp(a + b * x)

rmse = np.sqrt(mean_squared_error(df["VR"], Ro_pred))

print(f"RMSE = {rmse:.3f}")

# простой GOF (как в pybasin)
GOF = 1 - rmse / np.std(df["VR"].values)
print(f"GOF = {GOF:.2f}")
