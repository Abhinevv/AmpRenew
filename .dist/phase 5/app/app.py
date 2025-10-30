"""
=========================================================
Battery Health Prediction API
Phase 5 - FastAPI Deployment
=========================================================
Author: Atharva Pednekar
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
print("🚀 Loading trained model ...")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

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
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Read CSV
    df_new = pd.read_csv(file_path)
    print(f"📦 Received file: {file.filename}, shape: {df_new.shape}")

    # Drop unused columns
    drop_cols = ['Charging Cycles'] if 'Charging Cycles' in df_new.columns else []
    X_new = df_new.drop(columns=drop_cols, errors='ignore')

    # Predict SOH
    predicted_soh = model.predict(X_new)
    df_new['Predicted SOH (%)'] = predicted_soh

    # RUL estimation (if cycles available)
    if 'Charging Cycles' in df_new.columns:
        df_new = df_new.sort_values('Charging Cycles')
        coeffs = np.polyfit(df_new['Charging Cycles'], df_new['Predicted SOH (%)'], 1)
        slope, intercept = coeffs
        SOH_THRESHOLD = 70
        predicted_EOL = (SOH_THRESHOLD - intercept) / slope
        current_cycle = df_new['Charging Cycles'].max()
        rul_cycles = predicted_EOL - current_cycle
        rul_years = rul_cycles / 250

        df_new['RUL (cycles)'] = rul_cycles
        df_new['RUL (years)'] = rul_years
    else:
        rul_cycles, rul_years = None, None

    # Save output
    output_file = os.path.join(UPLOAD_DIR, "predicted_output.csv")
    df_new.to_csv(output_file, index=False)

    # Generate a quick plot if cycles exist
    if 'Charging Cycles' in df_new.columns:
        plt.figure(figsize=(7, 4))
        plt.plot(df_new['Charging Cycles'], df_new['Predicted SOH (%)'], label='Predicted SOH', color='blue')
        plt.axhline(70, color='r', linestyle='--', label='70% EOL Threshold')
        plt.xlabel("Charging Cycles")
        plt.ylabel("Predicted SOH (%)")
        plt.title("Predicted SOH Degradation Trend")
        plt.legend()
        plot_path = os.path.join(UPLOAD_DIR, "soh_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    else:
        plot_path = None

    # Response HTML page
    html = f"""
    <html>
    <body style="font-family:Arial; margin:50px;">
        <h2>✅ Prediction Complete</h2>
        <p>Predicted results saved: <b>{output_file}</b></p>
        {'<p><img src="/plot" alt="SOH Plot" width="600"></p>' if plot_path else ''}
        <p><a href="/download">📥 Download Predicted CSV</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# ===== DOWNLOAD ENDPOINT =====
@app.get("/download")
def download():
    output_file = os.path.join(UPLOAD_DIR, "predicted_output.csv")
    return FileResponse(output_file, media_type="text/csv", filename="predicted_output.csv")

# ===== PLOT ENDPOINT =====
@app.get("/plot")
def plot():
    plot_path = os.path.join(UPLOAD_DIR, "soh_plot.png")
    if os.path.exists(plot_path):
        return FileResponse(plot_path, media_type="image/png")
    else:
        return {"error": "Plot not generated"}

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Battery Health Prediction API is running"}
