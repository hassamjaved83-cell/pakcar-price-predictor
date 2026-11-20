"""
Helpers behind the app's "Why this price?" and "Similar cars" panels.

Kept free of Streamlit so they can be unit-tested (see tests/).
"""

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

CATEGORICAL = [
    "registered_city",
    "color",
    "assembly",
    "body_type",
    "brand",
    "model",
    "variant",
]

PRETTY = {
    "registered_city": "City",
    "color": "Colour",
    "assembly": "Assembly",
    "body_type": "Body type",
    "brand": "Brand",
    "model": "Model",
    "variant": "Variant",
    "engine_cc": "Engine",
    "year": "Year",
    "abs": "ABS",
    "drls": "DRLs",
    "led_headlights": "LED headlights",
    "rear_ac_vents": "Rear AC vents",
    "head_up_display_(hud)": "Head-up display",
    "tyre_pressure_monitoring_system_(tpms)": "Tyre pressure monitor",
    "360_camera": "360° camera",
}


def pretty_feature(name):
    if name in PRETTY:
        return PRETTY[name]
    return name.replace("_", " ").capitalize()


def load_metrics(root):
    path = os.path.join(root, "models", "metrics.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _original_column(encoded_name):
    """'cat__model_Corolla' -> 'model', 'num__abs' -> 'abs'."""
    kind, _, rest = encoded_name.partition("__")
    if kind == "cat":
        # Longest match first so e.g. 'body_type' wins over a shorter prefix
        for col in sorted(CATEGORICAL, key=len, reverse=True):
            if rest.startswith(col + "_"):
                return col
    return rest


def explain(pipe, input_df):
    """
    Per-car explanation using XGBoost's exact tree SHAP values.

    Returns (baseline_pkr, DataFrame[label, column, effect_pct, log_contrib])
    sorted by absolute effect. effect_pct is the % change in price this input
    causes compared with the model's average car.
    """
    prep = pipe.named_steps["prep"]
    booster = pipe.named_steps["model"].regressor_.get_booster()

    X = prep.transform(input_df)
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)[0]

    names = prep.get_feature_names_out()
    bias = float(contribs[-1])

    grouped = {}
    for name, value in zip(names, contribs[:-1]):
        col = _original_column(name)
        grouped[col] = grouped.get(col, 0.0) + float(value)

    row = input_df.iloc[0]
    records = []
    for col, value in grouped.items():
        raw = row.get(col)
        if col in CATEGORICAL:
            label = f"{pretty_feature(col)}: {raw}"
        elif col == "year":
            label = f"Year {int(raw)}"
        elif col == "engine_cc":
            label = f"Engine {int(raw)} cc"
        else:
            label = pretty_feature(col) if int(raw) == 1 else f"No {pretty_feature(col).lower()}"
        records.append(
            {
                "label": label,
                "column": col,
                "log_contrib": value,
                "effect_pct": (np.exp(value) - 1) * 100,
            }
        )

    out = pd.DataFrame(records)
    out = out.reindex(out["log_contrib"].abs().sort_values(ascending=False).index)
    baseline_pkr = float(np.expm1(bias))
    return baseline_pkr, out.reset_index(drop=True)


def similar_cars(df, car, limit=5):
    """Closest real listings: same brand + model, nearest year, same variant first."""
    pool = df[
        (df["brand"] == car["brand"])
        & (df["model"] == car["model"])
        & df["price_pkr"].notna()
    ].copy()

    if pool.empty:
        return pool

    pool["_score"] = (pool["year"] - car["year"]).abs() + (
        pool["variant"] != car["variant"]
    ) * 2
    pool = pool.sort_values(["_score", "price_pkr"]).head(limit)

    return pd.DataFrame(
        {
            "Variant": pool["variant"],
            "Year": pool["year"].astype(int),
            "City": pool["registered_city"],
            "Price (lakh)": (pool["price_pkr"] / 100_000).round(1),
        }
    ).reset_index(drop=True)


def default_specs(df, brand, model, variant):
    """Most common engine CC and body type for a variant (for auto-fill)."""
    rows = df[(df["brand"] == brand) & (df["model"] == model)]
    exact = rows[rows["variant"] == variant]
    if not exact.empty:
        rows = exact

    engine = rows["engine_cc"].dropna()
    body = rows["body_type"].dropna()

    return (
        int(engine.mode().iloc[0]) if not engine.empty else 1300,
        str(body.mode().iloc[0]) if not body.empty else None,
    )
