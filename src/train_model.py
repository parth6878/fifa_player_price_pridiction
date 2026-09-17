from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor

from data_preprocessing import (
    clean_raw_data,
    get_feature_target,
    split_data,
    build_preprocessor
)

from model_evaluation import (
    XGB_PARAM_GRID,
    build_model_pipeline,
    evaluate_model,
    plot_predictions_vs_actual,
    save_model
)


# -----------------------------
# 1. File paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent   # the repo root, regardless of cwd
DATA_PATH = BASE_DIR / "data" / "fifa21 raw data v2.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "fifa_model.pkl"
PLOT_PATH = BASE_DIR / "artifacts" / "plots" / "predictions_vs_actual.png"

# Create folders if they don't exist
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 2. Load raw data
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)


# -----------------------------
# 3. Clean data
# -----------------------------

print("Cleaning data...")

df = clean_raw_data(df)


# -----------------------------
# 4. Separate features and target
# -----------------------------

X, y = get_feature_target(df)


# -----------------------------
# 5. Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = split_data(X, y)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# 6. Build preprocessor
# -----------------------------

preprocessor = build_preprocessor()


# -----------------------------
# 7. Create XGBoost model
#    using already-found best parameters
# -----------------------------

model = XGBRegressor(
    **XGB_PARAM_GRID,
    random_state=42
)


# -----------------------------
# 8. Combine preprocessing + model
# -----------------------------

pipeline = build_model_pipeline(preprocessor, model)


# -----------------------------
# 9. Train final model
# -----------------------------

print("Training model...")

pipeline.fit(X_train, y_train)


# -----------------------------
# 10. Evaluate model
# -----------------------------

print("\nModel Evaluation:")

results = evaluate_model(
    pipeline,
    X_test,
    y_test,
    log_target=True
)


# -----------------------------
# 11. Create prediction plot
# -----------------------------

fig = plot_predictions_vs_actual(
    pipeline,
    X_test,
    y_test,
    log_target=True
)

fig.savefig(PLOT_PATH)

print(f"Plot saved to {PLOT_PATH}")


# -----------------------------
# 12. Save trained model
# -----------------------------

save_model(pipeline, MODEL_PATH)
