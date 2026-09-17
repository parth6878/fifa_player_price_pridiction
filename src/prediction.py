import pandas as pd
import numpy as np
from pathlib import Path
from model_evaluation import load_model
from data_preprocessing import FEATURE_COLS

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "fifa_model.pkl"

model = load_model(MODEL_PATH)

player = pd.DataFrame([{
    "POT": 95, "OVA": 86, "Age": 18,
    "Attacking": 450, "Defending": 80, "Power": 75,
    "Movement": 90, "Goalkeeping": 35, "Mentality": 90,
    "Skill": 5, "Height": 180, "Weight": 70,
    "Preferred Foot": "Left", "Best Position": "RW",
    "Club": "FC Barcelona",
}])[FEATURE_COLS]

pred = np.expm1(model.predict(player)[0])
print(f"€{pred:,.0f}")
