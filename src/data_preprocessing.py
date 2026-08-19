"""
data_preprocessing.py

Cleans the raw FIFA 21 dataset and builds the sklearn preprocessing
pipeline (scaling + encoding) used for model training.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.model_selection import train_test_split

TARGET_COL = "Value_log"

NUMERIC_FEATURES = [
    "POT", "OVA", "Age", "Attacking", "Defending", "Power",
    "Movement", "Goalkeeping", "Mentality", "Skill", "Height", "Weight",
]
ONE_HOT_FEATURES = ["Preferred Foot", "Best Position"]
TARGET_ENC_FEATURES = ["Club"]
FEATURE_COLS = NUMERIC_FEATURES + ONE_HOT_FEATURES + TARGET_ENC_FEATURES

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def clean_height(x):
    try:
        feet, inches = x.rstrip('"').split("'")
        return int(int(feet) * 30.48 + int(inches) * 2.54)
    except Exception:
        return int(x[0])


def clean_money(x):
    if pd.isna(x):
        return 0.0
    x = str(x).lstrip("€")
    try:
        if "M" in x:
            return float(x.rstrip("M")) * 1_000_000
        if "K" in x:
            return float(x.rstrip("K")) * 1_000
        return float(x)
    except ValueError:
        return 0.0


def clean_date(x):
    parts = x.split()
    month = MONTHS.get(parts[0], parts[0])
    day = parts[1].rstrip(",")
    year = parts[-1]
    return f"{day}-{month}-{year}"


def clean_raw_data(df):
    """Turn the raw FIFA 21 CSV into a tidy, fully-typed DataFrame."""
    df = df.copy()

    df = df.drop(columns=["photoUrl", "playerUrl", "LongName", "Positions", "Loan Date End", "BOV"])
    df = df.rename(columns={"↓OVA": "OVA"})

    df["Height"] = df["Height"].apply(clean_height)
    df["Weight"] = df["Weight"].apply(lambda x: int(x.split("k")[0].split("l")[0]))

    for col in ["Value", "Wage", "Release Clause"]:
        df[col] = df[col].apply(clean_money)
    df["Hits"] = df["Hits"].fillna(0).apply(clean_money)

    df["Club"] = df["Club"].astype(str).str.lstrip("\n").str.strip()

    for col in ["W/F", "SM", "IR"]:
        df[col] = df[col].apply(lambda x: int(str(x).rstrip("★")))

    df["Joined"] = pd.to_datetime(df["Joined"].apply(clean_date), format="%d-%m-%Y")

    df = df.drop_duplicates()
    df["Value_log"] = np.log1p(df["Value"])

    return df


def get_feature_target(df, feature_cols=FEATURE_COLS, target_col=TARGET_COL):
    return df[feature_cols].copy(), df[target_col].copy()


def split_data(x, y, test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def build_preprocessor():
    """ColumnTransformer: scales numeric features, encodes categoricals."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    one_hot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    target_enc_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", TargetEncoder(random_state=42)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("onehot", one_hot_pipeline, ONE_HOT_FEATURES),
        ("target_enc", target_enc_pipeline, TARGET_ENC_FEATURES),
    ])
