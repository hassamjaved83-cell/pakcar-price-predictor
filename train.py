"""
Train the PakCar price model and report honest test-set metrics.

Usage:
    python train.py

Reads  data/processed/cleaned_cars.csv
Writes models/pak_car_price_xgboost.pkl   (sklearn Pipeline, predicts PKR)
       models/metrics.json                (held-out test metrics)
       models/feature_importance.csv
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "processed", "cleaned_cars.csv")
MODEL_DIR = os.path.join(ROOT, "models")

TARGET = "price_pkr"
RANDOM_STATE = 42

CATEGORICAL = [
    "registered_city",
    "color",
    "assembly",
    "body_type",
    "brand",
    "model",
    "variant",
]
NUMERIC = ["engine_cc", "year"]

# Columns that are text / identifiers, never model inputs
NON_FEATURES = {
    "title", TARGET, "last_updated", "ad_ref", "features", "url",
    "auction_grade", "chassis_no", "import_date",
}

# Remove obviously broken listings before training
MIN_PRICE = 300_000          # PKR
MAX_PRICE = 150_000_000      # PKR


def load_data():
    df = pd.read_csv(DATA_PATH)

    rows_total = len(df)
    df = df.dropna(subset=[TARGET, "year", "brand", "model"])
    df = df[(df[TARGET] >= MIN_PRICE) & (df[TARGET] <= MAX_PRICE)]

    feature_flags = [
        c for c in df.columns
        if c not in NON_FEATURES and c not in CATEGORICAL and c not in NUMERIC
    ]

    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df["engine_cc"] = pd.to_numeric(df["engine_cc"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[feature_flags] = df[feature_flags].fillna(0).astype(int)

    print(f"Rows in file      : {rows_total}")
    print(f"Rows used         : {len(df)}")
    print(f"Feature flags     : {len(feature_flags)}")

    return df, feature_flags


def build_model(feature_flags):
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3),
             CATEGORICAL),
            ("num", "passthrough", NUMERIC + feature_flags),
        ]
    )

    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Train on log(price) so a 10% error on a Mehran counts the same
    # as a 10% error on a Fortuner; predict() still returns PKR.
    regressor = TransformedTargetRegressor(
        regressor=xgb, func=np.log1p, inverse_func=np.expm1
    )

    return Pipeline([("prep", preprocess), ("model", regressor)])


def main():
    df, feature_flags = load_data()

    X = df[CATEGORICAL + NUMERIC + feature_flags]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipe = build_model(feature_flags)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    within_10 = float(np.mean(np.abs(pred - y_test) / y_test <= 0.10))

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "test_r2": round(float(r2_score(y_test, pred)), 4),
        "test_mae_pkr": int(mean_absolute_error(y_test, pred)),
        "test_mape_pct": round(
            float(mean_absolute_percentage_error(y_test, pred)) * 100, 2
        ),
        "test_within_10pct": round(within_10 * 100, 1),
        "cv5_r2_mean": round(float(cv_r2.mean()), 4),
        "cv5_r2_std": round(float(cv_r2.std()), 4),
    }

    # Refit on all data for the shipped model
    pipe.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, os.path.join(MODEL_DIR, "pak_car_price_xgboost.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    names = pipe.named_steps["prep"].get_feature_names_out()
    importances = pipe.named_steps["model"].regressor_.feature_importances_
    (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    )

    print("\nHeld-out test metrics")
    for k, v in metrics.items():
        print(f"  {k:<18}: {v}")


if __name__ == "__main__":
    main()
