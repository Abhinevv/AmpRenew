

# ===== IMPORTS =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import warnings
import difflib
warnings.filterwarnings("ignore")

# ===== STEP 1: LOAD DATA =====
print("\n🚀 Loading dataset ...")
df = pd.read_csv("battery_features_engineered.csv")

# Normalize column names (strip whitespace)
df.columns = df.columns.str.strip()
print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head(3))

# ===== STEP 2: DETECT TARGET (SOH) =====
print("\n🧩 Preparing features and target ...")

# List of plausible target name variants
candidate_names = [
    "SOH (%)", "SOH(%)", "SOH %", "SOH", "State of Health", "State_of_Health", "state_of_health"
]

# If exact present, use it; else try to find a close match using difflib
target_col = None
for name in candidate_names:
    if name in df.columns:
        target_col = name
        break

if target_col is None:
    # perform fuzzy matching against dataframe column names
    cols = list(df.columns)
    best_matches = {}
    for cname in cols:
        # compare normalized lowercase forms
        for cand in candidate_names:
            score = difflib.SequenceMatcher(
                None, cname.lower().replace(" ", ""), cand.lower().replace(" ", "")
            ).ratio()
            best_matches[cname] = max(best_matches.get(cname, 0), score)

    # pick the column with highest similarity if above threshold (0.6)
    if best_matches:
        best_col = max(best_matches, key=best_matches.get)
        if best_matches[best_col] >= 0.60:
            print(f"⚠️ Did not find exact 'SOH (%)' header — using best match: '{best_col}' (score {best_matches[best_col]:.2f})")
            target_col = best_col

if target_col is None:
    # final fallback: show available columns and raise clear error
    print("\nAvailable columns:", df.columns.tolist())
    raise ValueError("❌ 'SOH (%)' column not found and no close match detected. "
                     "Please include a target column (e.g. 'SOH (%)' or 'SOH').")

# Now we have the target column
print(f"Using target column: '{target_col}'")
y = df[target_col]

# ===== STEP 2b: PREPARE FEATURES =====
# Drop non-feature columns only if they exist
drop_cols = ['Charging Cycles']  # keep this for RUL calculation if present
existing_drop_cols = [c for c in drop_cols if c in df.columns]
# Ensure we don't drop the target accidentally
cols_to_drop = [c for c in existing_drop_cols if c != target_col]

X = df.drop(columns=cols_to_drop + [target_col], errors='ignore')
print("Feature count:", X.shape[1])

# ===== STEP 3: TRAIN/TEST SPLIT =====
print("\n📊 Splitting dataset ...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ===== STEP 4: BASELINE MODEL - RANDOM FOREST =====
print("\n🌲 Training Random Forest (baseline) ...")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

print("\n🎯 Random Forest Performance:")
print(f"MAE  : {rf_mae:.4f}")
print(f"RMSE : {rf_rmse:.4f}")
print(f"R²   : {rf_r2:.4f}")

# ===== STEP 5: ADVANCED MODEL - XGBOOST =====
print("\n⚡ Training XGBoost model ...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_r2 = r2_score(y_test, y_pred_xgb)

print("\n🚀 XGBoost Performance:")
print(f"MAE  : {xgb_mae:.4f}")
print(f"RMSE : {xgb_rmse:.4f}")
print(f"R²   : {xgb_r2:.4f}")

# ===== STEP 6: HYPERPARAMETER TUNING (XGBOOST) =====
print("\n🔍 Running GridSearchCV for hyperparameter tuning (this may take a while)...")
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [300, 500],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9]
}

grid = GridSearchCV(
    estimator=xgb.XGBRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("\n✅ Best Hyperparameters Found:")
print(grid.best_params_)

# Evaluate tuned model
y_pred_best = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_best)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2 = r2_score(y_test, y_pred_best)

print("\n🔧 Tuned XGBoost Performance:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ===== STEP 7: VISUALIZATION =====
print("\n📈 Plotting Actual vs Predicted SOH ...")
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel(f"Actual {target_col}")
plt.ylabel("Predicted " + target_col)
plt.title("Actual vs Predicted SOH (Tuned XGBoost)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== STEP 8: RUL ESTIMATION =====
print("\n🔮 Estimating Remaining Useful Life (RUL) ...")
df_test = X_test.copy()
df_test['actual_soh'] = y_test.values
df_test['predicted_soh'] = y_pred_best

if 'Charging Cycles' in df.columns:
    # Use original df to get the cycle numbers for the test indices
    cycles = df.loc[df_test.index, 'Charging Cycles']
    df_test = df_test.assign(cycle=cycles.values)
    df_test = df_test.sort_values('cycle')

    SOH_THRESHOLD = 70  # End of life threshold

    # fit linear trend to predicted soh vs cycle
    coeffs = np.polyfit(df_test['cycle'], df_test['predicted_soh'], 1)
    slope, intercept = coeffs

    if abs(slope) < 1e-8:
        print("⚠️ Predicted SOH trend slope is ~0. Cannot reliably estimate EOL cycles (slope≈0).")
        predicted_cycle_EOL = np.nan
        rul_cycles = np.nan
        rul_years = np.nan
    else:
        predicted_cycle_EOL = (SOH_THRESHOLD - intercept) / slope
        current_cycle = df_test['cycle'].max()
        rul_cycles = predicted_cycle_EOL - current_cycle
        rul_years = rul_cycles / 250  # assuming 250 cycles per year

    print("\n🔋 Estimated Remaining Useful Life:")
    print(f"Cycles Remaining: {rul_cycles}")
    print(f"Years Remaining : {rul_years}")
else:
    print("⚠️ 'Charging Cycles' column missing, skipping RUL estimation.")

# ===== STEP 9: SAVE MODEL =====
print("\n💾 Saving final trained model ...")
joblib.dump(best_model, "soh_xgboost_model.pkl")
print("✅ Model saved as 'soh_xgboost_model.pkl'")

print("\n✅ Phase 3 completed successfully!")
