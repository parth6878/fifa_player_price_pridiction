# FIFA 21 Player Value Prediction

Predicts a football player's market value (`Value`) from FIFA 21 attribute data using a tuned Random
Forest Regressor. Built end-to-end: raw data cleaning → EDA → feature engineering → model comparison →
hyperparameter tuning → evaluation.

## Results

| Metric | Score |
|---|---|
| R² (test set) | **0.978** |
| MAE (test set) | **€128,437** |
| RMSE (test set) | **€776,692** |

Random Forest outperformed Linear Regression by roughly 11x on cross-validated MAE (0.038 vs 0.455 on
the log-transformed target), reflecting the non-linear relationship between rating and value in the
underlying data.

## Dataset

- Source: `fifa21_raw_data_v2.csv` (FIFA 21 player attributes, ~19,000 players, 77 columns)
- Target: `Value` (player market value in €), modeled as `log1p(Value)` to correct for heavy right-skew
  (raw skew ≈ 8.0)

## Project structure

```
├── fifa21_raw_data_v2.csv        # raw input data
├── eda.ipynb                     # data cleaning + exploratory analysis
├── model.ipynb                   # feature prep, model training, tuning, evaluation
├── model.joblib                  # final trained Random Forest
├── club_encoder.joblib           # fitted TargetEncoder for the Club feature
├── training_columns.joblib       # exact column order expected by the model
└── README.md
```

## Workflow

> Note: preprocessing (encoding, target encoding) and the model are currently separate, manual steps
> in the notebook — not wrapped in an `sklearn.pipeline.Pipeline` object. See Future Improvements below.

### 1. Data cleaning
- Parsed mixed-format `Height`/`Weight` (dataset contains both metric `170cm`/`72kg` and imperial
  `5'11"`/`172lbs` rows — handled both).
- Converted `€1.2M` / `€500K` style strings (`Value`, `Wage`, `Release Clause`) to floats.
- Parsed star-rating columns (`W/F`, `SM`, `IR`) and `Joined` date.
- Verified no duplicate rows; addressed missing values (`Hits` filled with 0, `Loan Date End` dropped —
  ~95% missing and not needed for this target).

### 2. EDA — key findings
- **OVA vs Value is non-linear/exponential**, not linear — value stays low and flat until roughly OVA 80,
  then rises steeply for elite-rated players.
- **Age adds a premium on top of rating** — younger players at a given OVA are valued higher, reflecting
  resale/potential value.
- Composite attribute scores (Attacking, Defending, etc.) correlate strongly with the technical
  sub-attributes they're built from (e.g. Marking/Tackles with Defending, r > 0.9) — sub-attributes were
  excluded from modeling as redundant.
- Goalkeepers were excluded from outfield attribute-correlation checks (GKs score near-zero on
  Attacking/Defending, which distorts those relationships if included).
- **Club has strong, genuine predictive power for Value** — confirmed via proper cross-validated testing
  (target-encoded Club alone reaches R² ≈ 0.76; Nationality alone only reaches R² ≈ 0.22 and was dropped
  as largely redundant with Club).

### 3. Features used

| Type | Features |
|---|---|
| Numeric | `OVA`, `POT`, `Age`, `Attacking`, `Defending`, `Power`, `Movement`, `Goalkeeping`, `Mentality`, `Skill`, `Height`, `Weight` |
| Categorical (one-hot) | `Best Position`, `Preferred Foot` |
| Categorical (target-encoded) | `Club` (fit on training fold only, via `sklearn.preprocessing.TargetEncoder`) |

**Explicitly excluded** (leakage or redundancy):
- `Release Clause`, `Wage`, `BOV` — near-duplicates of the target or of `OVA`
- All raw sub-attributes underlying the composite scores (`Crossing`, `Finishing`, `Marking`, etc.)
- `PAC`/`SHO`/`PAS`/`DRI`/`DEF`/`PHY` — FIFA's own redundant 6-stat summary
- `Nationality` — weak standalone signal, redundant with `Club`

### 4. Modeling
- Train/test split: 80/20, `random_state=42`
- Baseline comparison: Linear Regression vs. Random Forest (5-fold CV, `neg_mean_absolute_error`)
- Hyperparameter tuning: `RandomizedSearchCV` (25 iterations, 5-fold CV) over `n_estimators`, `max_depth`,
  `min_samples_leaf`
- Best parameters: `n_estimators=700`, `max_depth=30`, `min_samples_leaf=1`
- Final evaluation on held-out test set, predictions converted back from log scale with `np.expm1()`

## Known limitations

- Three test-set predictions are severe outliers (near-zero predicted value for low/mid actual value
  players) — flagged for further investigation; not yet root-caused.
- Model has not been deployed as a service/API — currently notebook-only.
- `Club`/`Nationality` encodings reflect a single FIFA 21 snapshot; club valuations shift season to
  season, so periodic retraining would be needed for production use.

## Future improvements

| Area | Improvement |
|---|---|
| Root cause | Investigate the 3 severe test-set outliers (near-zero predicted value on low/mid-value players) |
| Code structure | Wrap encoding + model into a single `sklearn.pipeline.Pipeline` so preprocessing and training can't drift out of sync |
| Feature set | Try adding `Contract`/`Joined` as engineered features (e.g. years remaining, years at club) |
| Modeling | Try Gradient Boosting (XGBoost/LightGBM) as a stronger alternative to Random Forest |
| Validation | Re-tune with a full `GridSearchCV` around the current best `RandomizedSearchCV` parameters |
| Maintenance | Retrain periodically as new FIFA/FC data is released — Club/market valuations shift season to season |
| Presentation | Build a small web app (e.g. Streamlit) or a slide presentation to showcase the model interactively — let a user input player stats and see the predicted value, plus the key EDA charts |
| Deployment | Wrap the saved model + encoder behind a simple API endpoint for real use outside the notebook |

## Requirements

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Web Application

An interactive web application powered by Flask and an EA Sports FUT-inspired UI is available under `app/`.

### 1. Running the Web Application Locally

Ensure your environment has the required dependencies installed (e.g. `conda activate testenv`):

```bash
# From the project root:
python app/app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 2. Web Features
- **Profile & Potential**: Select Club (autocomplete over 682 clubs), Preferred Foot (`Left`/`Right`), Best Position (`ST`, `RW`, `CB`, `GK`, etc.), OVA, POT, and Age.
- **Physical & Technical Attributes**: Synchronized sliders and numeric boxes for Height, Weight, Attacking, Skill, Movement, Power, Mentality, Defending, Goalkeeping.
- **Live FUT Card Preview**: Instant visual feedback on player card ratings and stats.
- **Quick Presets**: 1-click presets for Wonderkid, Elite Striker, Playmaker, and Top Goalkeeper.
- **Prediction Display**: Formats predicted valuation as `€X,XXX,XXX` alongside tier insights.

### 3. API Endpoint: `POST /predict`

Accepts raw (unscaled/unencoded) attributes in JSON:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "POT": 95,
    "OVA": 86,
    "Age": 18,
    "Attacking": 450,
    "Defending": 80,
    "Power": 75,
    "Movement": 90,
    "Goalkeeping": 35,
    "Mentality": 90,
    "Skill": 5,
    "Height": 180,
    "Weight": 70,
    "Preferred Foot": "Left",
    "Best Position": "RW",
    "Club": "FC Barcelona"
  }'
```

**Response (200 OK):**
```json
{
  "predicted_value_eur": 82593744.0
}
```

**Validation Error (422 Unprocessable Entity):**
If an input is outside realistic bounds (e.g., `Age > 45` or `OVA > 99`):
```json
{
  "error": "Age must be between 15 and 45 years."
}
```

### 4. Running Automated Tests

```bash
python tests/test_api.py
```
