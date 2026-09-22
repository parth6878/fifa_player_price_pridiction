from pathlib import Path
import sys
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib

# Ensure repo root is on sys.path so we can import from src
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_preprocessing import (
    FEATURE_COLS,
    NUMERIC_FEATURES,
    ONE_HOT_FEATURES,
    TARGET_ENC_FEATURES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# -------------------------------------------------------------------
# Model Loading at Startup
# -------------------------------------------------------------------
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "fifa_model.pkl"
model = None
known_clubs = []
valid_positions = [
    "CAM", "CB", "CDM", "CF", "CM", "GK", "LB", "LM",
    "LW", "LWB", "RB", "RM", "RW", "RWB", "ST"
]

def load_pipeline():
    global model, known_clubs
    if not MODEL_PATH.exists():
        logger.error("Model file not found at %s", MODEL_PATH)
        return False
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Successfully loaded model from %s", MODEL_PATH)

        # Extract clubs from TargetEncoder categories if available
        try:
            preprocessor = model.named_steps.get("preprocessor")
            if preprocessor:
                target_enc = preprocessor.named_transformers_["target_enc"].named_steps["encoder"]
                if hasattr(target_enc, "categories_") and len(target_enc.categories_) > 0:
                    known_clubs = sorted(list(target_enc.categories_[0]))
                    logger.info("Extracted %d clubs from model TargetEncoder", len(known_clubs))
        except Exception as exc:
            logger.warning("Could not extract club categories from pipeline: %s", exc)

        return True
    except Exception as exc:
        logger.error("Failed to load model from %s: %s", MODEL_PATH, exc)
        return False

# Attempt startup load
load_pipeline()

# Fallback clubs if not populated from pipeline
if not known_clubs:
    known_clubs = [
        "FC Barcelona", "Real Madrid", "Paris Saint-Germain", "Liverpool",
        "Manchester City", "FC Bayern München", "Juventus", "Chelsea",
        "Manchester United", "Atlético Madrid", "Borussia Dortmund", "Arsenal",
        "Tottenham Hotspur", "Inter", "Milan", "Ajax", "Napoli"
    ]

# -------------------------------------------------------------------
# Validation Rules
# -------------------------------------------------------------------
VALIDATION_BOUNDS = {
    "Age": (15, 45, "Age must be between 15 and 45 years."),
    "OVA": (40, 99, "Overall rating (OVA) must be between 40 and 99."),
    "POT": (40, 99, "Potential rating (POT) must be between 40 and 99."),
    "Attacking": (0, 500, "Attacking score must be between 0 and 500."),
    "Defending": (0, 500, "Defending score must be between 0 and 500."),
    "Power": (0, 500, "Power score must be between 0 and 500."),
    "Movement": (0, 500, "Movement score must be between 0 and 500."),
    "Goalkeeping": (0, 500, "Goalkeeping score must be between 0 and 500."),
    "Mentality": (0, 500, "Mentality score must be between 0 and 500."),
    "Skill": (0, 500, "Skill score must be between 0 and 500."),
    "Height": (120, 220, "Height must be between 120 cm and 220 cm."),
    "Weight": (40, 130, "Weight must be between 40 kg and 130 kg."),
}

def validate_input(data):
    """Validate request payload against required schema and realistic bounds."""
    if not isinstance(data, dict):
        return "Request body must be a JSON object."

    # Check for missing required features
    missing_fields = [f for f in FEATURE_COLS if f not in data]
    if missing_fields:
        return f"Missing required fields: {', '.join(missing_fields)}"

    # Validate numeric features
    for field, (min_val, max_val, err_msg) in VALIDATION_BOUNDS.items():
        val = data.get(field)
        if val is None:
            return f"Field '{field}' is required."
        try:
            num_val = float(val)
        except (ValueError, TypeError):
            return f"Field '{field}' must be a numeric value."
        if not (min_val <= num_val <= max_val):
            return err_msg

    # Validate categoricals
    foot = data.get("Preferred Foot")
    if foot not in ["Left", "Right"]:
        return "Preferred Foot must be either 'Left' or 'Right'."

    pos = data.get("Best Position")
    if pos not in valid_positions:
        return f"Best Position '{pos}' is invalid. Allowed: {', '.join(valid_positions)}."

    club = data.get("Club")
    if not isinstance(club, str) or not club.strip():
        return "Club must be a non-empty text string."

    return None

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/clubs", methods=["GET"])
def get_clubs():
    return jsonify({"clubs": known_clubs})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Trained model is not available. Ensure artifacts/models/fifa_model.pkl exists."
        }), 503

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid request. Expected application/json body."}), 400

    error_msg = validate_input(payload)
    if error_msg:
        return jsonify({"error": error_msg}), 422

    try:
        # Build single-row DataFrame in the exact column order of FEATURE_COLS
        row_dict = {}
        for col in FEATURE_COLS:
            val = payload[col]
            if col in NUMERIC_FEATURES:
                row_dict[col] = [float(val)]
            else:
                row_dict[col] = [str(val).strip()]

        df = pd.DataFrame(row_dict)[FEATURE_COLS]

        # Pipeline handles scaling and encoding internally
        log_pred = model.predict(df)[0]
        euro_value = float(np.expm1(log_pred))

        # Clamp lower bound to 0
        euro_value = max(0.0, euro_value)

        return jsonify({
            "predicted_value_eur": round(euro_value, 2)
        })
    except Exception as exc:
        logger.exception("Error during prediction inference: %s", exc)
        return jsonify({
            "error": f"Internal prediction error: {str(exc)}"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
