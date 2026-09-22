# FIFA Player Price Predictor

Estimate a football player's transfer market value (in €) from their FIFA
attributes — overall rating, potential, physical profile, club, position,
and performance stats — using a trained XGBoost regression model served
through a Flask API and a single-page web UI.

## What it does

You fill in a player's profile (club, position, preferred foot, age,
height/weight, overall/potential rating, and six composite performance
scores), and the app sends that profile to a trained regression pipeline,
which returns a predicted market value. The model was trained on
log-transformed prices (`log1p`), so predictions are inverse-transformed
with `expm1` before being shown, and clamped at €0.

## Features

- **Live player card preview** — a FIFA Ultimate Team–style card that
  updates in real time as you adjust stats, so you can see the profile
  you're about to submit.
- **Quick presets** — one-click example profiles (a wonderkid winger, an
  elite striker, a playmaker, and a top goalkeeper) to try the model
  without filling in every field by hand.
- **Club dropdown sourced from the model itself** — the `Club` field is
  populated live from `/api/clubs`, which reads the exact set of clubs the
  trained encoder knows about. This guarantees every submitted value is
  one the model can actually encode, instead of relying on free-text spelling
  matching a hidden list.
- **Live backend health indicator** — a small status dot in the header
  calls `/health` on load and shows whether the trained model is actually
  loaded, rather than silently failing on the first prediction.
- **Server-side validation** — every field has an explicit valid range or
  allowed set, enforced before the payload ever reaches the model (see
  [Input validation](#input-validation) below).

## Tech stack

| Layer      | Technology |
|------------|------------|
| Backend    | Python, Flask, Flask-CORS |
| ML model   | scikit-learn pipeline (`ColumnTransformer` + `TargetEncoder` + one-hot encoding) wrapping an XGBoost regressor |
| Model I/O  | joblib (`artifacts/models/fifa_model.pkl`) |
| Data prep  | pandas, numpy |
| Frontend   | Vanilla HTML/CSS/JS (no framework or build step) |
| Fonts      | Oswald (headings/scoreboard numerals), Inter (body text) — via Google Fonts |

## Project structure

```
fifa_player_price_pridiction/
├── app/                      # Flask application
│   ├── app.py                 # Routes, validation, model loading, inference
│   ├── templates/
│   │   └── index.html         # Single-page UI
│   └── static/
│       ├── css/style.css      # Styling
│       └── js/app.js          # Form logic, live card preview, API calls
├── src/
│   └── data_preprocessing.py  # FEATURE_COLS / NUMERIC_FEATURES /
│                               # ONE_HOT_FEATURES / TARGET_ENC_FEATURES
│                               # and any training-time preprocessing
├── artifacts/
│   └── models/
│       └── fifa_model.pkl     # Trained pipeline (preprocessing + XGBoost)
├── tests/                     # Test suite
├── requirements.txt
└── README.md
```


## Getting started

### Prerequisites

- Python 3.9+
- A trained model artifact at `artifacts/models/fifa_model.pkl` (produced
  by whatever training script lives under `src/`)

### Installation

```bash
git clone https://github.com/parth6878/fifa_player_price_pridiction.git
cd fifa_player_price_pridiction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running the app

```bash
cd app          # the folder containing app.py
python app.py
```

The server starts on `http://0.0.0.0:5000`. Open
`http://localhost:5000` in a browser.

If `artifacts/models/fifa_model.pkl` isn't found, the app still starts,
but `/predict` will return a `503` until a valid model is in place — check
`/health` to confirm the model loaded.

## Using the app

1. Pick a club from the dropdown (populated from the model's known clubs),
   a position, and a preferred foot.
2. Set age, height, weight, overall rating (OVA), and potential (POT).
3. Set the six composite performance scores — **Attacking, Skill,
   Movement, Power, Mentality, Defending, Goalkeeping** — each on a 0–500
   scale. These are aggregate category scores (not the individual 0–99
   sub-attributes), matching how the model was trained.
4. Click **Calculate market valuation** to get a predicted transfer value,
   along with the player's potential upside (POT − OVA) and a market tier
   label.

Prefer not to fill in every field yourself? Use one of the **Quick
presets** to load a full example profile, then tweak individual values.

## API reference

| Method | Route         | Description |
|--------|---------------|--------------|
| GET    | `/`           | Serves the web UI |
| GET    | `/api/clubs`  | Returns `{ "clubs": [...] }` — the list of clubs the loaded model recognizes |
| GET    | `/health`     | Returns model load status: `{ "status", "model_loaded", "model_path" }` |
| POST   | `/predict`    | Accepts a player profile as JSON, returns a predicted value |

### `POST /predict`

**Request body** (all fields required):

```json
{
  "Club": "FC Barcelona",
  "Best Position": "RW",
  "Preferred Foot": "Left",
  "Age": 18,
  "OVA": 86,
  "POT": 95,
  "Height": 180,
  "Weight": 70,
  "Attacking": 450,
  "Skill": 430,
  "Movement": 440,
  "Power": 380,
  "Mentality": 390,
  "Defending": 120,
  "Goalkeeping": 55
}
```

**Success response** (`200`):

```json
{ "predicted_value_eur": 84250000.0 }
```

**Error responses:**

- `400` — request body isn't valid JSON
- `422` — a field is missing, out of range, or invalid (see table below);
  response includes a human-readable `error` message
- `503` — the trained model isn't loaded

## Input validation

Enforced server-side in `app.py` before any field reaches the model:

| Field | Valid range / values |
|---|---|
| Age | 15 – 45 |
| OVA | 40 – 99 |
| POT | 40 – 99 |
| Height | 120 – 220 cm |
| Weight | 40 – 130 kg |
| Attacking, Skill, Movement, Power, Mentality, Defending, Goalkeeping | 0 – 500 |
| Preferred Foot | `Left` or `Right` |
| Best Position | one of `CAM, CB, CDM, CF, CM, GK, LB, LM, LW, LWB, RB, RM, RW, RWB, ST` |
| Club | non-empty string (the UI restricts this to clubs the model actually knows, via `/api/clubs`) |

## How the model works

- **Numeric features** (Age, OVA, POT, Height, Weight, and the seven
  composite scores) are passed through as-is or scaled inside the
  pipeline.
- **`Best Position` and `Preferred Foot`** are one-hot encoded.
- **`Club`** is target-encoded (`sklearn.preprocessing.TargetEncoder`),
  which is why the set of valid clubs is fixed to whatever the encoder saw
  during training — this is exactly the list served at `/api/clubs`.
- The regressor is **XGBoost**, trained on `log1p(price)`. At inference
  time, `app.py` applies `np.expm1` to the raw prediction and clamps the
  result at `0` before returning it.

## Running tests

```bash
pytest tests/
```




