"""
=========================================================
Battery Health Prediction API
Phase 5 - FastAPI Deployment
=========================================================
Description:
This FastAPI app allows users to upload EV battery data
(.csv file) and get predicted State of Health (SOH)
and estimated Remaining Useful Life (RUL).
=========================================================
"""

# ===== IMPORTS =====
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import pandas as pd
import numpy as np
import joblib, os, io, matplotlib.pyplot as plt

# ===== INITIALIZE FASTAPI APP =====
app = FastAPI(title="EV Battery Health Predictor", version="1.0")

# ===== LOAD TRAINED MODEL =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjust this relative path to where your model actually is
MODEL_PATH = os.path.join (BASE_DIR, r"C:\Users\Atharva Pednekar\Pccoe project\AmpRenew\soh_xgboost_model.pkl")
print(" Loading trained model ...")
model = joblib.load(MODEL_PATH)
print(" Model loaded successfully!")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===== HOMEPAGE (UPLOAD FORM) =====
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Battery SOH Prediction</title>
        </head>
        <body style="font-family:Arial; margin:50px;">
            <h2> EV Battery Health Prediction</h2>
            <p>Upload a CSV file containing battery data to get predicted SOH (%) and Remaining Useful Life.</p>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept=".csv" required>
                <input type="submit" value="Upload and Predict">
            </form>
        </body>
    </html>
    """

# ===== PREDICTION ENDPOINT =====
# ===== PREDICTION ENDPOINT =====
# ===== PREDICTION ENDPOINT (robust, feature-alignment + auto-engineering) =====
@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # --- Save uploaded file ---
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # --- Read CSV ---
        df_new = pd.read_csv(file_path)
        print(f"📦 Received: {file.filename}, shape={df_new.shape}")

        # --- Basic checks / ordering ---
        # Ensure data is sorted by cycle if Charging Cycles exists
        if 'Charging Cycles' in df_new.columns:
            df_new = df_new.sort_values('Charging Cycles').reset_index(drop=True)

        # --- Expected features (from training + engineered) ---
        # Base features used during training (must match exactly the names used when training)
        base_features = [
             "Voltage (V)", "Current (A)", "Battery Temp (°C)",
            "Ambient Temp (°C)", "Charging Duration (min)", "Degradation Rate (%)",
            "Charging Mode", "Efficiency (%)", "Battery Type", "EV Model",
            "Optimal Charging Duration Class"
        ]

        # Engineered features your model expects (from the error trace)
        engineered_expected = [
            "temp_diff", "voltage_current_ratio", "efficiency_drop",
            "degradation_slope", "eff_temp_ratio",
            "rolling_efficiency_mean_5", "rolling_deg_rate_mean_5",
            "rolling_temp_mean_5", "rolling_deg_slope_5"
        ]

        expected_features = base_features + engineered_expected

        # --- Create missing raw base columns with safe defaults or warnings ---
        missing_raw = [c for c in base_features if c not in df_new.columns]
        if missing_raw:
            print("⚠️ training data did not have the following fields:", ", ".join(missing_raw))
            # Best-effort handling:
            for c in missing_raw:
                # If SOC missing, create a neutral placeholder (50) - strongly recommended to provide real SOC
                if c == "SOC (%)":
                    df_new["SOC (%)"] = 50.0
                    print("  → SOC (%) missing: filling placeholder 50.0 (recommend: provide real SOC values)")
                else:
                    # numeric placeholder = 0; categorical → create empty category
                    df_new[c] = 0
                    print(f"  → {c} missing: filling placeholder 0")

        # --- Basic engineered columns we can compute from available columns ---
        # temp_diff
        if "temp_diff" not in df_new.columns:
            if all(x in df_new.columns for x in ["Battery Temp (°C)", "Ambient Temp (°C)"]):
                df_new["temp_diff"] = df_new["Battery Temp (°C)"] - df_new["Ambient Temp (°C)"]
            else:
                df_new["temp_diff"] = 0.0

        # voltage_current_ratio (guard divide by zero)
        if "voltage_current_ratio" not in df_new.columns:
            if "Voltage (V)" in df_new.columns and "Current (A)" in df_new.columns:
                cur = df_new["Current (A)"].replace({0: np.nan})
                df_new["voltage_current_ratio"] = (df_new["Voltage (V)"] / cur).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                df_new["voltage_current_ratio"] = 0.0

        # efficiency_drop (current - previous)
        if "efficiency_drop" not in df_new.columns:
            if "Efficiency (%)" in df_new.columns:
                df_new["efficiency_drop"] = df_new["Efficiency (%)"].diff().fillna(0.0)
            else:
                df_new["efficiency_drop"] = 0.0

        # eff_temp_ratio = Efficiency / (Battery Temp + tiny epsilon)
        if "eff_temp_ratio" not in df_new.columns:
            if "Efficiency (%)" in df_new.columns and "Battery Temp (°C)" in df_new.columns:
                df_new["eff_temp_ratio"] = df_new["Efficiency (%)"] / (df_new["Battery Temp (°C)"] + 1e-6)
            else:
                df_new["eff_temp_ratio"] = 0.0

        # rolling means and rolling slopes (window=5), computed along sequence (per battery if battery_id exists)
        window = 5

        # attempt to group by battery_id or EV Model; fallback to whole dataframe sequence
        group_col = None
        if "battery_id" in df_new.columns:
            group_col = "battery_id"
        elif "EV Model" in df_new.columns:
            group_col = "EV Model"

        # helper for rolling slope: slope of y over index positions
        def rolling_slope(series):
            # compute slope using np.polyfit over the windowed values (x = 0..n-1)
            y = series.values
            if len(y) < 2 or np.all(np.isnan(y)):
                return 0.0
            x = np.arange(len(y))
            # if constant, slope = 0
            try:
                coef = np.polyfit(x, y, 1)
                return float(coef[0])
            except Exception:
                return 0.0

        # compute rolling features grouped if possible
        if group_col:
            df_new = df_new.sort_values([group_col, 'Charging Cycles'] if 'Charging Cycles' in df_new.columns else [group_col]).reset_index(drop=True)
            df_new["rolling_efficiency_mean_5"] = df_new.groupby(group_col)["Efficiency (%)"].transform(lambda s: s.rolling(window, min_periods=1).mean().fillna(0.0))
            df_new["rolling_deg_rate_mean_5"] = df_new.groupby(group_col)["Degradation Rate (%)"].transform(lambda s: s.rolling(window, min_periods=1).mean().fillna(0.0))
            df_new["rolling_temp_mean_5"] = df_new.groupby(group_col)["Battery Temp (°C)"].transform(lambda s: s.rolling(window, min_periods=1).mean().fillna(0.0))
            # rolling slopes using rolling.apply with window
            df_new["rolling_deg_slope_5"] = df_new.groupby(group_col)["Degradation Rate (%)"].transform(lambda s: s.rolling(window, min_periods=2).apply(lambda x: rolling_slope(pd.Series(x)), raw=False).fillna(0.0))
        else:
            # global rolling over the entire dataframe
            df_new["rolling_efficiency_mean_5"] = df_new["Efficiency (%)"].rolling(window, min_periods=1).mean().fillna(0.0) if "Efficiency (%)" in df_new.columns else 0.0
            df_new["rolling_deg_rate_mean_5"] = df_new["Degradation Rate (%)"].rolling(window, min_periods=1).mean().fillna(0.0) if "Degradation Rate (%)" in df_new.columns else 0.0
            df_new["rolling_temp_mean_5"] = df_new["Battery Temp (°C)"].rolling(window, min_periods=1).mean().fillna(0.0) if "Battery Temp (°C)" in df_new.columns else 0.0
            df_new["rolling_deg_slope_5"] = df_new["Degradation Rate (%)"].rolling(window, min_periods=2).apply(lambda x: rolling_slope(pd.Series(x))).fillna(0.0) if "Degradation Rate (%)" in df_new.columns else 0.0

        # degradation_slope (overall slope of Degradation Rate (%) vs cycles per group or global)
        if "degradation_slope" not in df_new.columns:
            if "Degradation Rate (%)" in df_new.columns:
                if group_col:
                    def group_deg_slope(g):
                        x = g.index.to_numpy()
                        y = g["Degradation Rate (%)"].to_numpy()
                        if len(y) < 2:
                            return np.zeros(len(y))
                        slope = np.polyfit(np.arange(len(y)), y, 1)[0]
                        return np.full(len(y), slope)
                    df_new["degradation_slope"] = np.concatenate([group_deg_slope(g) for _, g in df_new.groupby(group_col)])
                else:
                    # global slope
                    y = df_new["Degradation Rate (%)"].to_numpy()
                    if len(y) >= 2:
                        slope = np.polyfit(np.arange(len(y)), y, 1)[0]
                        df_new["degradation_slope"] = slope
                    else:
                        df_new["degradation_slope"] = 0.0
            else:
                df_new["degradation_slope"] = 0.0

        # --- Now ensure all expected engineered features exist (fill missing with 0) ---
        for col in engineered_expected:
            if col not in df_new.columns:
                df_new[col] = 0.0

        # --- Prepare model input: select and order features exactly as expected by the model ---
        # If you saved the exact feature list during training (recommended), load it instead.
        # Here we rely on `expected_features` constructed above.
        missing_for_model = [c for c in expected_features if c not in df_new.columns]
        if missing_for_model:
            # If there are still missing critical columns, return informative error
            msg = f"ERROR: still missing required features for model: {missing_for_model}"
            print(msg)
            # best-effort add zeros so model can still run but warn
            for c in missing_for_model:
                df_new[c] = 0.0
            print("Filled remaining missing features with zeros — prediction will proceed but accuracy may be impacted.")

        X_new = df_new[expected_features].copy()

        # --- Convert object (string) columns to numeric codes (consistent with training encoding) ---
        for col in X_new.columns:
            if X_new[col].dtype == 'object':
                print(f"🔠 Encoding categorical column for model: {col}")
                X_new[col] = X_new[col].astype('category').cat.codes

        # --- Final dtype check: ensure all numeric ---
        non_numeric = [c for c in X_new.columns if X_new[c].dtype == 'O']
        if non_numeric:
            print("Converting leftover object columns to numeric via category codes:", non_numeric)
            for c in non_numeric:
                X_new[c] = X_new[c].astype('category').cat.codes







        # --- Predict SOH ---
        predicted_soh = model.predict(X_new)
        df_new['Predicted SOH (%)'] = predicted_soh

        # --- RUL estimation and plotting (unchanged) ---
        rul_cycles, rul_years = None, None
        plot_path = None
        if 'Charging Cycles' in df_new.columns:
            df_new = df_new.sort_values('Charging Cycles')
            slope, intercept = np.polyfit(df_new['Charging Cycles'], df_new['Predicted SOH (%)'], 1)
            SOH_THRESHOLD = 70
            predicted_eol_cycle = (SOH_THRESHOLD - intercept) / slope
            current_cycle = df_new['Charging Cycles'].max()
            rul_cycles = max(predicted_eol_cycle - current_cycle, 0)
            rul_years = rul_cycles / 250
            df_new['RUL (cycles)'] = rul_cycles
            df_new['RUL (years)'] = rul_years

            plt.figure(figsize=(7, 4))
            plt.plot(df_new['Charging Cycles'], df_new['Predicted SOH (%)'], label='Predicted SOH')
            plt.axhline(70, color='r', linestyle='--', label='70% EOL Threshold')
            plt.xlabel("Charging Cycles")
            plt.ylabel("Predicted SOH (%)")
            plt.title("Predicted SOH Degradation Trend")
            plt.legend()
            plot_path = os.path.join(UPLOAD_DIR, "soh_plot.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

        # --- Save predicted output ---
        output_file = os.path.join(UPLOAD_DIR, "predicted_output.csv")
        df_new.to_csv(output_file, index=False)

        # --- Summary ---
        mean_soh = df_new["Predicted SOH (%)"].mean()
        soh_min = df_new["Predicted SOH (%)"].min()
        soh_max = df_new["Predicted SOH (%)"].max()
        summary_html = f"""
            <h3>📊 Summary</h3>
            <ul>
                <li>Average Predicted SOH: <b>{mean_soh:.2f}%</b></li>
                <li>SOH Range: <b>{soh_min:.2f}% - {soh_max:.2f}%</b></li>
                {f"<li>Estimated Remaining Life: <b>{rul_cycles:.0f} cycles (~{rul_years:.2f} years)</b></li>" if rul_cycles else ""}
            </ul>
        """

        html = f"""
        <html>
        <body style="font-family:Arial; margin:50px;">
            <h2>✅ Prediction Complete</h2>
            {summary_html}
            <p>Predicted results saved: <b>{output_file}</b></p>
            {'<p><img src="/plot" alt="SOH Plot" width="600"></p>' if plot_path else ''}
            <p><a href="/download">📥 Download Predicted CSV</a></p>
            <p><a href="/">⬅️ Go Back</a></p>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    except Exception as e:
        print(" Error during prediction:", e)
        return HTMLResponse(
            content=f"<h3 style='color:red;'>⚠️ Error: {str(e)}</h3><p>Please check your CSV and required fields.</p>",
            status_code=500
        )
# ===== PLOT ENDPOINT =====
@app.get("/plot")
def plot():
    plot_path = os.path.join(UPLOAD_DIR, "soh_plot.png")
    if os.path.exists(plot_path):
        return FileResponse(plot_path, media_type="image/png")
    else:
        return {"error": "Plot not generated"}