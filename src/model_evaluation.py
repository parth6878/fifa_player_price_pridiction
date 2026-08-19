import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# path = Path("models/fifa_model.pkl")
# path.parent.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

XGB_PARAM_GRID ={'colsample_bytree': 1.0,
                'learning_rate': 0.1,
                'max_depth': 7,
                'n_estimators': 500,
                'subsample': 0.7}


def build_model_pipeline(preprocessor, model):
    """Combine the preprocessor with an estimator into one pipeline."""
    return Pipeline([("preprocessor", preprocessor), ("model", model)])



def evaluate_model(pipeline, X_test, y_test, log_target=True):
    """Print R2 / MAE / RMSE on the test set (MAE & RMSE in real € values)."""
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    if log_target:
        y_test_real = np.expm1(y_test)
        y_pred_real = np.expm1(y_pred)
    else:
        y_test_real, y_pred_real = y_test, y_pred

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = root_mean_squared_error(y_test_real, y_pred_real)

    print(f"R2:   {r2:.4f}")
    print(f"MAE:  €{mae:,.0f}")
    print(f"RMSE: €{rmse:,.0f}")

    return {"r2": r2, "mae": mae, "rmse": rmse}


def plot_predictions_vs_actual(pipeline, X_test, y_test, log_target=True):
    """Scatter plot of predicted vs actual values."""
    y_pred = pipeline.predict(X_test)

    if log_target:
        y_test_real = np.expm1(y_test)
        y_pred_real = np.expm1(y_pred)
    else:
        y_test_real, y_pred_real = y_test, y_pred

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test_real, y_pred_real, alpha=0.3)
    plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], "r--")
    plt.xlabel("Actual Value (€)")
    plt.ylabel("Predicted Value (€)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Predicted vs Actual Value — test set")
    plt.tight_layout()
    return plt.gcf()


def save_model(pipeline, path):
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


def load_model(path):
    return joblib.load(path)