"""
=========================================================
Battery SOH Prediction - Phase 4
=========================================================
Author: Atharva Pednekar
Description:
This script loads the trained model and predicts
battery State of Health (SOH) for new battery data.
=========================================================
"""

# ===== IMPORTS =====
import pandas as pd
import joblib
import os

# ===== FILE PATHS =====
MODEL_PATH = r"C:\Users\Atharva Pednekar\Pccoe project\AmpRenew\soh_xgboost_model.pkl"
INPUT_PATH = r"C:\Users\Atharva Pednekar\Pccoe project\AmpRenew\battery_features_engineered.csv"
OUTPUT_PATH = "predicted_battery_health.csv"

# ===== LOAD MODEL =====
print(" Loading trained model ...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully from", MODEL_PATH)

# ===== LOAD NEW DATA =====
print("\n Loading new data from:", INPUT_PATH)
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f" File not found: {INPUT_PATH}\n"
                            "Please make sure the CSV file exists at this location.")

df_new = pd.read_csv(INPUT_PATH)
print(" Data loaded successfully. Shape:", df_new.shape)

# ===== MATCH TRAINING FEATURES =====
expected_features = [
    'Voltage (V)', 'Current (A)', 'Battery Temp (°C)', 'Ambient Temp (°C)',
    'Charging Duration (min)', 'Degradation Rate (%)', 'Charging Mode',
    'Efficiency (%)', 'Battery Type', 'EV Model',
    'Optimal Charging Duration Class', 'temp_diff', 'voltage_current_ratio',
    'efficiency_drop', 'degradation_slope', 'eff_temp_ratio',
    'rolling_efficiency_mean_5', 'rolling_deg_rate_mean_5',
    'rolling_temp_mean_5', 'rolling_deg_slope_5'
]

missing = [col for col in expected_features if col not in df_new.columns]
if missing:
    raise ValueError(f" Missing columns in input data: {missing}")

# Filter to expected columns (ignore extras like 'SOC (%)')
X_new = df_new[expected_features]

# ===== MAKE PREDICTIONS =====
print("\n⚙ Running SOH predictions ...")
predicted_soh = model.predict(X_new)

# ===== SAVE RESULTS =====
df_new["Predicted_SOH"] = predicted_soh
df_new.to_csv(OUTPUT_PATH, index=False)
print(f"\n Predictions completed successfully!")
print(f" Results saved to: {OUTPUT_PATH}")