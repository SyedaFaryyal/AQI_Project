import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Ensure parent folder import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = "data/features.csv"
MODELS_DIR = "data/models"


def load_features(path=DATA_PATH):
    """Load features CSV (handles both 'time' and 'timestamp' columns)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run the feature pipeline first.")

    preview = pd.read_csv(path, nrows=1)
    time_col = "time" if "time" in preview.columns else "timestamp"

    df = pd.read_csv(path, parse_dates=[time_col])
    df = df.rename(columns={time_col: "time"})
    df = df.sort_values("time")
    return df


def train_and_evaluate(X_train, X_test, y_train, y_test, day_label):
    """Train model and compute metrics (works with all sklearn versions)."""
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Manual RMSE for old sklearn versions
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"📅 {day_label} -> RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
    return model


def main():
    print("🚀 Starting model training pipeline...")

    df = load_features()
    print(f"✅ Loaded {len(df)} rows from {DATA_PATH}")

    ignore_cols = ["time", "target_day1", "target_day2", "target_day3"]
    feature_cols = [c for c in df.columns if c not in ignore_cols and df[c].dtype != "object"]

    os.makedirs(MODELS_DIR, exist_ok=True)

    for day in [1, 2, 3]:
        target_col = f"target_day{day}"
        if target_col not in df.columns:
            print(f"⚠️ Skipping {target_col} (column not found).")
            continue

        sub_df = df.dropna(subset=[target_col])
        if sub_df.empty:
            print(f"⚠️ Not enough data for {target_col}. Skipping.")
            continue

        X = sub_df[feature_cols]
        y = sub_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = train_and_evaluate(X_train, X_test, y_train, y_test, f"Day {day}")
        model_path = os.path.join(MODELS_DIR, f"model_day{day}.pkl")
        joblib.dump(model, model_path)
        print(f"💾 Saved model to {model_path}")

    print("🏁 Training completed successfully!")


if __name__ == "__main__":
    main()
