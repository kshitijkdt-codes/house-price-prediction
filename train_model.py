"""Train the XGBoost house price model and save it for the API.

Usage (from the backend/ folder):
    python train_model.py                 # full run with hyper-parameter search
    python train_model.py --quick         # skip the search (faster)
    python train_model.py --data path.csv # use a specific CSV

If backend/data/Hyderbad_House_price.csv exists it is used, otherwise the
synthetic sample file is used (see generate_sample_data.py).

What was fixed compared with the version in the report
------------------------------------------------------
* The old code used `rate_persqft` as an input. Price = rate x area, so the model
  was effectively being given the answer (data leakage) and a user can't know
  the rate before asking for a price. It is no longer used.
* The inputs now match what the website asks for: location, property type,
  BHK, area and building status (BHK and property type come from `title`).
* One consistent set of saved files that app.py can load.
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
REAL_DATA = HERE / "data" / "Hyderbad_House_price.csv"
SAMPLE_DATA = HERE / "data" / "sample_hyderabad_house_price.csv"
FEATURES = ["location_enc", "type_enc", "status_enc", "area_insqft", "bhk"]


def parse_title(title: str):
    """'3 BHK Independent House' -> (3, 'Independent House')."""
    match = re.search(r"(\d+)\s*BHK", str(title), re.I)
    bhk = int(match.group(1)) if match else np.nan
    rest = re.sub(r"\d+\s*BHK", "", str(title), flags=re.I).strip()
    return bhk, (rest.title() if rest else "Apartment")


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    for col in ("title", "location", "building_status"):
        df[col] = df[col].astype(str).str.strip()
    parsed = df["title"].apply(parse_title)
    df["bhk"] = [p[0] for p in parsed]
    df["property_type"] = [p[1] for p in parsed]
    # The real dataset also contains "Residential Plot" rows (about 1 in 4). They have no BHK and
    # their areas/rates are inconsistent, so this house-price model leaves them out.
    is_plot = df["property_type"].str.contains("plot", case=False)
    print(f"Leaving out {int(is_plot.sum())} plot rows (this model is for houses/flats)")
    df = df[~is_plot]
    df = df.dropna(subset=["price(L)", "area_insqft", "bhk", "location", "building_status"])
    df = df.drop_duplicates()  # the real data has ~100 exact duplicate listings
    df = df[(df["area_insqft"] > 0) & (df["price(L)"] > 0)]
    # Remove extreme price-per-sqft outliers (keep the middle 98%).
    low, high = df["rate_persqft"].quantile([0.01, 0.99])
    df = df[df["rate_persqft"].between(low, high)]
    df["bhk"] = df["bhk"].astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="skip hyper-parameter search")
    args = parser.parse_args()

    data_path = args.data or (REAL_DATA if REAL_DATA.exists() else SAMPLE_DATA)
    if not data_path.exists():
        raise SystemExit(f"No data found. Run `python generate_sample_data.py` or put the real CSV at {REAL_DATA}")
    is_sample = data_path.name.startswith("sample_")
    print(f"Using data: {data_path}" + ("  (SYNTHETIC SAMPLE - not real prices!)" if is_sample else ""))

    df = load_and_clean(data_path)
    print(f"{len(df)} rows after cleaning")

    encoders = {
        "location": LabelEncoder().fit(df["location"]),
        "type": LabelEncoder().fit(df["property_type"]),
        "status": LabelEncoder().fit(df["building_status"]),
    }
    df["location_enc"] = encoders["location"].transform(df["location"])
    df["type_enc"] = encoders["type"].transform(df["property_type"])
    df["status_enc"] = encoders["status"].transform(df["building_status"])

    # Prices are very right-skewed (a few Rs 20 Cr+ homes), so the model learns log(price)
    # and we convert back with expm1 for every score and for the API.
    X, y = df[FEATURES], df["price(L)"]
    X_train, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = np.log1p(y_train_raw)

    base = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
    if args.quick:
        model = XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.1,
                             max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        best_params, cv_rmse = {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6}, None
    else:
        search = RandomizedSearchCV(
            base,
            {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.03, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6, 8],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
            },
            n_iter=15, cv=5, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=1,
        )
        search.fit(X_train, y_train)
        model, best_params = search.best_estimator_, search.best_params_
        cv_rmse = float(-search.best_score_)

    pred = np.expm1(model.predict(X_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    rel_err = np.abs(pred - y_test) / np.maximum(y_test, 1e-6)
    rel_p80 = float(np.quantile(rel_err, 0.8))  # 80% of test predictions were within this fraction

    metrics = {
        "data_file": data_path.name,
        "is_synthetic_sample": is_sample,
        "rows_used": int(len(df)),
        "rmse_lakhs": round(rmse, 3),
        "mae_lakhs": round(mae, 3),
        "r2": round(r2, 4),
        "log_target": True,
        "cv_rmse_log_price": None if cv_rmse is None else round(cv_rmse, 4),
        "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()},
        "rel_error_p80": round(rel_p80, 4),
        "feature_importance": {f: round(float(i), 4) for f, i in zip(FEATURES, model.feature_importances_)},
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(encoders["location"], MODEL_DIR / "location_encoder.pkl")
    joblib.dump(encoders["type"], MODEL_DIR / "type_encoder.pkl")
    joblib.dump(encoders["status"], MODEL_DIR / "status_encoder.pkl")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Test RMSE: {rmse:.2f} L | MAE: {mae:.2f} L | R2: {r2:.3f}")
    print(f"Saved model + encoders + metrics.json to {MODEL_DIR}")


if __name__ == "__main__":
    main()
